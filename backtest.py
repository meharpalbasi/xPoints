#!/usr/bin/env python3
"""Walk-forward backtest for xPoints candidates on one season of history.

For each target gameweek g, every feature is computed from gameweeks < g
only (as-of), the model is trained on all (player, GW < g) rows, and it
predicts every player with a row at g. Baselines are scored on identical
rows with the same metrics.py functions the published scorecard uses, so a
number here means the same thing as a number on the accuracy page.

This is the tool every model change is judged by. Nothing is promoted on a
single aggregate; per-gameweek results are kept so paired comparisons and
the detection-power statement can be made honestly.

Usage: python backtest.py [--first-target 8] [--out backtests/2025-26.json]
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from metrics import captain_regret, mae, mean, precision_at_k, spearman, stdev

DATA = Path("data/merged_gw_2025-26.csv")
STATS = ["total_points", "minutes", "expected_goals", "expected_assists", "bps", "bonus",
         "defensive_contribution", "starts", "goals_scored", "assists", "clean_sheets", "saves"]
WINDOWS = (1, 3, 5, 10)
POSITIONS = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
POPULATIONS = {
    "all": lambda m: np.ones(len(m), dtype=bool),
    "played": lambda m: m > 0,
    "starters": lambda m: m >= 60,
}


TEAM_WINDOWS = (5, 10)
TEAM_FEATURES = [f"{side}_{m}_r{w}" for side in ("own", "opp")
                 for m in ("gf", "ga", "xg", "xgc") for w in TEAM_WINDOWS]


def team_form_table(df, element_team):
    """As-of team form: per (team, GW) goals/xG for and against, shifted and
    rolled so the value at GW g only reflects matches before g."""
    m = df.copy()
    m["team_id"] = m["element"].map(element_team)
    m = m.dropna(subset=["team_id"])
    m["team_id"] = m["team_id"].astype(int)
    m["gf"] = np.where(m["was_home"] == 1, m["team_h_score"], m["team_a_score"])
    m["ga"] = np.where(m["was_home"] == 1, m["team_a_score"], m["team_h_score"])
    per_match = m.groupby(["team_id", "GW", "fixture"], as_index=False).agg(
        gf=("gf", "max"), ga=("ga", "max"),
        xg=("expected_goals", "sum"),            # team xG = sum of its players' xG
        xgc=("expected_goals_conceded", "max"),  # team xGC = the full-match value
    )
    per_gw = per_match.groupby(["team_id", "GW"], as_index=False)[["gf", "ga", "xg", "xgc"]].mean()
    per_gw = per_gw.sort_values(["team_id", "GW"]).reset_index(drop=True)
    g = per_gw.groupby("team_id")
    out = per_gw[["team_id", "GW"]].copy()
    for metric in ("gf", "ga", "xg", "xgc"):
        lagged = g[metric].shift(1)
        for w in TEAM_WINDOWS:
            out[f"{metric}_r{w}"] = lagged.groupby(per_gw["team_id"]).rolling(w, min_periods=1).mean() \
                .reset_index(level=0, drop=True)
    return out


def load_player_gameweeks(path=DATA, team_features=False):
    """One row per (element, GW): sum stats across DGW fixtures.

    With team_features=True, each row also carries its own team's and its
    opponent's as-of form (from team_form_table) — deadline-known context,
    since the opponent for GW g is fixed before g's deadline.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df["was_home"] = df["was_home"].astype(str).str.lower().eq("true").astype(int)
    for c in STATS + ["expected_goals_conceded", "team_h_score", "team_a_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    agg = {c: "sum" for c in STATS + ["expected_goals_conceded"]}
    agg.update({"was_home": "mean", "value": "first", "position": "first",
                "team": "first", "name": "first", "fixture": "count"})
    pg = df.groupby(["element", "GW"], as_index=False).agg(agg)
    pg = pg.rename(columns={"fixture": "fixture_count", "was_home": "home_share"})
    pg["position_id"] = pg["position"].map(POSITIONS).fillna(3).astype(int)
    pg["price"] = pg["value"] / 10.0
    pg = pg.sort_values(["element", "GW"]).reset_index(drop=True)
    if not team_features:
        return pg

    season = path.stem.split("_")[-1]
    raw = pd.read_csv(path.parent / f"players_raw_{season}.csv", usecols=["id", "team"])
    element_team = dict(zip(raw["id"], raw["team"]))
    form = team_form_table(df, element_team)
    pg["team_id"] = pg["element"].map(element_team)
    # Opponent(s) for the gameweek: mean of opponent ids' form across DGW fixtures
    opp = df.groupby(["element", "GW"], as_index=False)["opponent_team"].agg(list)
    own = form.rename(columns={c: f"own_{c}" for c in form.columns if c not in ("team_id", "GW")})
    pg = pg.merge(own, on=["team_id", "GW"], how="left")
    opp_rows = opp.explode("opponent_team").rename(columns={"opponent_team": "team_id"})
    opp_rows["team_id"] = pd.to_numeric(opp_rows["team_id"], errors="coerce")
    opp_rows = opp_rows.merge(form, on=["team_id", "GW"], how="left")
    opp_mean = opp_rows.groupby(["element", "GW"], as_index=False)[
        [c for c in form.columns if c not in ("team_id", "GW")]].mean()
    opp_mean = opp_mean.rename(columns={c: f"opp_{c}" for c in opp_mean.columns if c not in ("element", "GW")})
    pg = pg.merge(opp_mean, on=["element", "GW"], how="left")
    return pg.sort_values(["element", "GW"]).reset_index(drop=True)


def build_features(pg, key="element"):
    """As-of features: for a row at GW g, everything derives from GW < g.

    shift(1) then roll on the per-player, order-sorted frame guarantees the
    target gameweek never leaks into its own features. Deadline-known
    context (price, fixture count, home share, position) is used as-is.

    `key` is the player identity column: "element" within one season, or a
    stable player code when the frame spans seasons (the frame must already
    be sorted by key, then by a season-aware gameweek order).
    """
    g = pg.groupby(key)
    feats = pd.DataFrame(index=pg.index)
    for stat in STATS:
        lagged = g[stat].shift(1)
        for w in WINDOWS:
            feats[f"{stat}_r{w}"] = lagged.groupby(pg[key]).rolling(w, min_periods=1).mean() \
                .reset_index(level=0, drop=True)
        feats[f"{stat}_std"] = lagged.groupby(pg[key]).expanding().mean().reset_index(level=0, drop=True)
    # Exposure-adjusted rates through the previous gameweek
    mins_cum = g["minutes"].shift(1).groupby(pg[key]).cumsum()
    for stat in ("expected_goals", "expected_assists", "defensive_contribution", "bps", "total_points"):
        cum = g[stat].shift(1).groupby(pg[key]).cumsum()
        feats[f"{stat}_per90"] = np.where(mins_cum >= 90, cum / mins_cum * 90, np.nan)
    feats["games_played_std"] = g["minutes"].shift(1).groupby(pg[key]).cumcount()
    feats["played_last_gw"] = (g["minutes"].shift(1) > 0).astype(float)
    feats["started_last_gw"] = (g["starts"].shift(1) > 0).astype(float)
    # Deadline-known context
    feats["price"] = pg["price"]
    feats["fixture_count"] = pg["fixture_count"]
    feats["home_share"] = pg["home_share"]
    for pid in (1, 2, 3, 4):
        feats[f"pos_{pid}"] = (pg["position_id"] == pid).astype(float)
    return feats


def candidates(seed=0):
    return {
        "ridge": lambda: Ridge(alpha=10.0),
        "xgb_mse": lambda: XGBRegressor(objective="reg:squarederror", n_estimators=300, max_depth=4,
                                        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                        random_state=seed, n_jobs=4),
        "xgb_tweedie": lambda: XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.3,
                                            n_estimators=300, max_depth=4, learning_rate=0.05,
                                            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=4),
    }


def score_fold(preds, actual, minutes, k=20):
    out = {}
    for pop, keep in POPULATIONS.items():
        mask = keep(minutes)
        a = actual[mask].tolist()
        res = {"n": int(mask.sum())}
        for name, p in preds.items():
            pv = p[mask].tolist()
            m = {}
            if name not in ("price",):  # rank-only predictor has no meaningful MAE
                m["mae"] = mae(pv, a)
            if name != "zero" and len(a) >= 3:
                m["spearman"] = spearman(pv, a)
                m["p_at_20"] = precision_at_k(pv, a, k, draws=50)
                if pop == "all":
                    m["captain_regret"] = captain_regret(pv, a, draws=50)
            res[name] = m
        out[pop] = res
    return out


def run(first_target=8, last_target=38, out=Path("backtests/2025-26.json")):
    pg = load_player_gameweeks(team_features=True)
    X = build_features(pg)
    y = pg["total_points"].to_numpy(dtype=float)
    gw = pg["GW"].to_numpy()
    minutes = pg["minutes"].to_numpy()
    cols = list(X.columns)
    Xf = X.to_numpy(dtype=float)
    Xz = np.nan_to_num(Xf, nan=0.0)  # ridge cannot take NaN; trees can
    # Variant matrix: base features + own/opponent team form (as-of, deadline-known)
    Xo = np.hstack([Xf, pg[TEAM_FEATURES].to_numpy(dtype=float)])

    folds = []
    t0 = time.time()
    for g in range(first_target, last_target + 1):
        tr, te = gw < g, gw == g
        if te.sum() == 0:
            continue
        preds = {
            "zero": np.zeros(te.sum()),
            "career_mean": np.nan_to_num(X.loc[te, "total_points_std"].to_numpy(), nan=0.0),
            "last5_mean": np.nan_to_num(X.loc[te, "total_points_r5"].to_numpy(), nan=0.0),
            "price": X.loc[te, "price"].to_numpy(),
        }
        for name, make in candidates().items():
            model = make()
            if name == "ridge":
                model.fit(Xz[tr], y[tr]); preds[name] = model.predict(Xz[te])
            elif name == "xgb_tweedie":
                model.fit(Xf[tr], np.clip(y[tr], 0, None)); preds[name] = model.predict(Xf[te])
            else:
                model.fit(Xf[tr], y[tr]); preds[name] = model.predict(Xf[te])
        opp = candidates()["xgb_tweedie"]()
        opp.fit(Xo[tr], np.clip(y[tr], 0, None)); preds["xgb_tweedie_opp"] = opp.predict(Xo[te])
        folds.append({"gw": g, "n_train": int(tr.sum()), **score_fold(preds, y[te], minutes[te])})
        print(f"GW{g:2d} train={tr.sum():5d} test={te.sum():3d} | starters ρ: "
              + " ".join(f"{m}={folds[-1]['starters'][m].get('spearman', float('nan')):.3f}"
                         for m in ("last5_mean", "price", "ridge", "xgb_mse", "xgb_tweedie")))

    models = ["zero", "career_mean", "last5_mean", "price", "ridge", "xgb_mse", "xgb_tweedie", "xgb_tweedie_opp"]
    summary = {}
    for pop in POPULATIONS:
        summary[pop] = {}
        for m in models:
            metrics = {}
            for key in ("mae", "spearman", "p_at_20", "captain_regret"):
                vals = [f[pop][m][key] for f in folds if key in f[pop][m] and not math.isnan(f[pop][m][key])]
                if vals:
                    metrics[key] = {"mean": round(mean(vals), 4), "sd": round(stdev(vals), 4) if len(vals) > 1 else None}
            summary[pop][m] = metrics
    # Paired differences vs xgb_mse on starter Spearman: the promotion-relevant comparison
    paired = {}
    for m in ("ridge", "xgb_tweedie", "xgb_tweedie_opp", "last5_mean", "price"):
        d = [f["starters"][m]["spearman"] - f["starters"]["xgb_mse"]["spearman"] for f in folds]
        paired[m] = {"mean_diff": round(mean(d), 4), "se": round(stdev(d) / math.sqrt(len(d)), 4), "n_folds": len(d)}
    # The two comparisons that decide whether the opponent block earns its place
    paired_opp = {}
    for ref in ("xgb_tweedie", "price"):
        d = [f["starters"]["xgb_tweedie_opp"]["spearman"] - f["starters"][ref]["spearman"] for f in folds]
        paired_opp[f"vs_{ref}"] = {"mean_diff": round(mean(d), 4), "se": round(stdev(d) / math.sqrt(len(d)), 4)}

    result = {
        "season": "2025-26", "source": str(DATA), "first_target": first_target, "last_target": last_target,
        "n_features": len(cols), "features": cols, "runtime_s": round(time.time() - t0, 1),
        "populations": {"all": "every player with a row at g", "played": "minutes > 0 at g", "starters": "minutes >= 60 at g"},
        "summary": summary, "paired_vs_xgb_mse_starters_spearman": paired,
        "opponent_block_starters_spearman": paired_opp, "team_features": TEAM_FEATURES, "folds": folds,
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"\nwrote {out}  ({result['runtime_s']}s)")
    print(f"\n{'model':<12} {'MAE all':>8} {'ρ all':>7} {'ρ starters':>11} {'p@20 st':>8} {'captain regret':>15}")
    for m in models:
        s = summary
        f = lambda pop, key: (s[pop][m].get(key) or {}).get("mean")
        fmt = lambda v: "   —" if v is None else f"{v:.3f}"
        print(f"{m:<12} {fmt(f('all','mae')):>8} {fmt(f('all','spearman')):>7} {fmt(f('starters','spearman')):>11} "
              f"{fmt(f('starters','p_at_20')):>8} {fmt(f('all','captain_regret')):>15}")
    print("\npaired vs xgb_mse (starter Spearman, mean diff ± se):",
          {k: f"{v['mean_diff']:+.3f} ± {v['se']:.3f}" for k, v in paired.items()})
    print("opponent block (xgb_tweedie_opp) starter Spearman:",
          {k: f"{v['mean_diff']:+.3f} ± {v['se']:.3f}" for k, v in paired_opp.items()})
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-target", type=int, default=8)
    ap.add_argument("--out", default="backtests/2025-26.json")
    a = ap.parse_args()
    run(first_target=a.first_target, out=Path(a.out))
