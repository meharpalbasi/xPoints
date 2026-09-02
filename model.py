#!/usr/bin/env python3
"""Shadow xPoints model — the backtest harness's pipeline, run for the next gameweek.

Trains on every completed player-gameweek since 2025/26 (the first season under
the current scoring rules), joined across seasons on FPL's stable player
`code`, and predicts every current player for the upcoming gameweek. Features
are the harness's as-of set: for the prediction row, everything derives from
matches already played, so the newest completed match is included exactly
once (the old script's prediction row was one gameweek stale).

Output: predictions_model.json in the same schema as predictions.json, with
source "xpoints_model", plus a per-gameweek archive predictions/gw{N}_model.json
that freezes at the deadline and is graded by score.py alongside ep_next on
identical rows. Nothing here touches what the clients read; ep_next stays the
champion until the scorecard says otherwise (ROADMAP.md promotion gate).
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from backtest import STATS, build_features, load_player_gameweeks
from baseline import (BOOTSTRAP_URL, FIXTURES_URL, POSITIONS, STATUS_MAP, archive_allowed,
                      event_deadline, fetch_json, next_gameweek, season_label, step_summary,
                      team_fixture_stats)
from fetch_data import fetch as fetch_season
from prediction_safety import write_predictions

SEASONS = ["2025-26", "2026-27"]
MODEL_VERSION = "xpoints-two-stage-blend-v2"
SOURCE = "xpoints_model"
PARAMS = dict(objective="reg:tweedie", tweedie_variance_power=1.3, n_estimators=300, max_depth=4,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=4)
OUT_PATH = Path("predictions_model.json")
META_PATH = Path("predictions_model_meta.json")
ARCHIVE_DIR = Path("predictions")


def season_index(season):
    return int(season[:4])


def load_season(season):
    """Player-gameweek rows for one season, keyed by stable player code."""
    pg = load_player_gameweeks(Path(f"data/merged_gw_{season}.csv"))
    codes = pd.read_csv(f"data/players_raw_{season}.csv", usecols=["id", "code"]).rename(columns={"id": "element"})
    pg = pg.merge(codes, on="element", how="inner")
    pg["season"] = season_index(season)
    pg["season_gw"] = pg["season"] * 100 + pg["GW"]
    return pg


def assemble_history(frames):
    """Concatenate seasons into one frame ordered by (code, season_gw)."""
    hist = pd.concat(frames, ignore_index=True)
    return hist.sort_values(["code", "season_gw"]).reset_index(drop=True)


def prediction_frame(bootstrap, fixtures, gw, season):
    """One row per current player for the target gameweek: deadline-known
    context filled in, every match stat NaN (unknown — that is the point)."""
    fx = team_fixture_stats(fixtures, gw)
    rows = []
    for p in bootstrap["elements"]:
        etype = p.get("element_type")
        if etype not in POSITIONS:
            continue
        t = fx.get(p["team"], {"count": 0, "difficulty": [], "home": 0})
        n = t["count"]
        row = {
            "element": p["id"], "code": p["code"], "GW": gw, "season": season,
            "season_gw": season * 100 + gw, "position": POSITIONS[etype], "position_id": etype,
            "team": p["team"], "name": p.get("web_name"), "value": p.get("now_cost"),
            "price": (p.get("now_cost") or 0) / 10.0, "fixture_count": n,
            "home_share": (t["home"] / n) if n else 0.0,
        }
        for stat in STATS + ["expected_goals_conceded"]:
            row[stat] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


PRICE_EDGES = [0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0, 20.0]
BLEND_WEIGHT = 0.5  # share of the conditional head; the rest is the price-implied expectation


def price_expectation(train_starters, targets):
    """Mean points of training-set STARTERS per (position, price bin) — the
    prior FPL's pricing encodes, made explicit. Falls back to the overall
    starter mean for bins with no history."""
    lookup = (train_starters.assign(_b=pd.cut(train_starters["price"], PRICE_EDGES))
              .groupby(["position_id", "_b"], observed=True)["total_points"].mean())
    overall = float(train_starters["total_points"].mean())
    bins = pd.cut(targets["price"], PRICE_EDGES)
    out = np.array([lookup.get((int(p), b), np.nan) for p, b in zip(targets["position_id"], bins)], dtype=float)
    return np.where(np.isnan(out), overall, out)


def train_and_predict(history, pred, params=PARAMS):
    """Two-stage, price-blended projection (the harness winner, ablation PR #19):

        xPoints = P(60+ minutes) x ( w * E[points | starts] + (1-w) * price_expectation )

    Stage 1 is a classifier on every completed row. Stage 2 is a Tweedie head
    trained ONLY on rows where the player started, monotone in price so it can
    refine price's ordering but never invert it, blended with the as-of
    price-implied expectation. Returns per-row components for explainability.
    """
    frame = pd.concat([history, pred], ignore_index=True).sort_values(["code", "season_gw"]).reset_index(drop=True)
    X = build_features(frame, key="code")
    cols = list(X.columns)
    y = frame["total_points"]
    is_train = y.notna().to_numpy()
    started = (frame["minutes"].fillna(0).to_numpy() >= 60) & is_train
    Xf = X.to_numpy(dtype=float)
    Xt = Xf[~is_train]

    clf = XGBClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                        learning_rate=params["learning_rate"], subsample=params["subsample"],
                        colsample_bytree=params["colsample_bytree"], random_state=params["random_state"],
                        n_jobs=params["n_jobs"], eval_metric="logloss")
    clf.fit(Xf[is_train], started[is_train].astype(int))
    p60 = clf.predict_proba(Xt)[:, 1]

    mono = tuple(1 if c == "price" else 0 for c in cols)
    reg = XGBRegressor(**params, monotone_constraints=mono)
    reg.fit(Xf[started], np.clip(y[started].to_numpy(dtype=float), 0, None))
    cond = np.clip(reg.predict(Xt), 0, None)

    targets = frame.loc[~is_train].reset_index(drop=True)
    pexp = price_expectation(frame.loc[started], targets)
    blend = BLEND_WEIGHT * cond + (1 - BLEND_WEIGHT) * pexp
    preds = p60 * blend
    components = {"p_start60": p60, "xp_if_start": blend, "cond_head": cond, "price_expect": pexp}
    return targets, preds, cols, int(is_train.sum()), components


def apply_availability(xp, status, chance, fixture_count):
    """Deadline-known availability, applied AFTER the model (never trained on).

    injured / suspended / unavailable -> 0; doubtful -> scaled by FPL's stated
    chance; blank gameweek -> 0.
    """
    if not fixture_count:
        return 0.0
    if status in ("i", "s", "u"):
        return 0.0
    if status == "d":
        return xp * ((chance if chance is not None else 100) / 100)
    return xp


def build_rows(bootstrap, targets, preds, gw, generated_at, season_text, components=None):
    teams = {t["id"]: t for t in bootstrap["teams"]}
    players = {p["id"]: p for p in bootstrap["elements"]}
    components = components or {}
    rows = []
    for i, ((_, t), raw) in enumerate(zip(targets.iterrows(), preds)):
        p = players[int(t["element"])]
        raw = float(max(raw, 0.0))
        xp = apply_availability(raw, p.get("status"), p.get("chance_of_playing_next_round"), int(t["fixture_count"]))
        rows.append({
            "player_id": p["id"], "player_code": p.get("code"), "web_name": p.get("web_name"),
            "team": p["team"], "team_name": teams.get(p["team"], {}).get("name", ""),
            "element_type": p["element_type"], "position": POSITIONS[p["element_type"]],
            "xPoints": round(xp, 4), "xPoints_raw": round(raw, 4),
            "fpl_ep_next": float(p.get("ep_next") or 0.0),
            "form": float(p.get("form") or 0.0),
            "selected_by_percent": str(p.get("selected_by_percent") or "0.0"),
            "status": p.get("status"), "status_numeric": STATUS_MAP.get(p.get("status"), 1),
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round"),
            "now_cost": p.get("now_cost"), "price_millions": round((p.get("now_cost") or 0) / 10, 1),
            "fixture_count": int(t["fixture_count"]),
            "home_proportion": round(float(t["home_share"]), 2),
            "generated_at": generated_at, "season": season_text, "gameweek": gw,
            "source": SOURCE, "model_version": MODEL_VERSION, "ordering": "model_desc,price_desc",
            **{k: round(float(v[i]), 4) for k, v in components.items()},
        })
    rows.sort(key=lambda r: (-r["xPoints"], -(r["now_cost"] or 0), r["player_id"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="use data/ as-is")
    args = ap.parse_args()

    if not args.no_fetch:
        for s in SEASONS:
            fetch_season(s, quiet=True)

    bootstrap = fetch_json(BOOTSTRAP_URL)
    fixtures = fetch_json(FIXTURES_URL)
    gw = next_gameweek(bootstrap["events"])
    deadline = event_deadline(bootstrap["events"], gw)
    season_text = season_label(bootstrap["events"])
    current = season_index(SEASONS[-1])
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    history = assemble_history([load_season(s) for s in SEASONS])
    pred = prediction_frame(bootstrap, fixtures, gw, current)
    targets, preds, features, n_train, components = train_and_predict(history, pred)
    rows = build_rows(bootstrap, targets, preds, gw, generated_at, season_text, components)

    expected_ids = [p["id"] for p in bootstrap["elements"] if p.get("element_type") in POSITIONS]
    warnings = write_predictions(OUT_PATH, rows, expected_ids, gw)
    META_PATH.write_text(json.dumps({
        "model_version": MODEL_VERSION, "trained_at": generated_at, "target_gameweek": gw,
        "seasons": SEASONS, "training_rows": n_train, "n_features": len(features),
        "features": features, "params": PARAMS,
        "design": "P(60+) x (0.5 * monotone-price Tweedie head trained on starters + 0.5 * price-implied expectation)",
        "harness": "backtests/2025-26_blend.json (walk-forward GW8-38: two_stage_blend MAE 0.920 all, "
                   "starter p@20 0.173, captain regret 10.71 — best of all candidates)",
    }, indent=2))

    if archive_allowed(dt.datetime.now(dt.timezone.utc), deadline):
        ARCHIVE_DIR.mkdir(exist_ok=True)
        (ARCHIVE_DIR / f"gw{gw}_model.json").write_text(OUT_PATH.read_text())
        archived = f"archived predictions/gw{gw}_model.json (deadline {deadline})"
    else:
        archived = f"GW{gw} deadline passed — gw{gw}_model.json stays frozen"

    top = rows[:3]
    positive = sum(1 for r in rows if r["xPoints"] > 0)
    print(f"🧪 shadow model GW{gw}: trained on {n_train} rows x {len(features)} features; "
          f"{len(rows)} predictions, {positive} positive; top: "
          + ", ".join(f"{r['web_name']} {r['xPoints']:.2f} (ep_next {r['fpl_ep_next']:.1f})" for r in top))
    for w in warnings:
        print(f"⚠️  anomaly: {w}")
    print(f"📁 {archived}")
    step_summary([f"- shadow model GW{gw}: {n_train} training rows, {len(features)} features, "
                  f"{positive} positive predictions; {archived}"] + [f"- ⚠️ {w}" for w in warnings])


if __name__ == "__main__":
    main()
