#!/usr/bin/env python3
"""Score archived pre-deadline predictions against official FPL results.

For every predictions/gw{N}.json whose event FPL has marked `data_checked`
(bonus and defensive-contribution points settled), join actual points on
player_id and write:

  scores/gw{N}.json       metrics per population per predictor (immutable)
  scores/gw{N}_rows.csv   the joined rows, so every number is auditable
  scorecard.json          cumulative per-gameweek table + power statement

Populations are named on every number because the same predictions score
positive R^2 over all players and negative over players who actually
played; an unnamed population is an unfalsifiable claim.

Predictors scored on identical rows:
  ep_next   the archived xPoints values (currently FPL's ep_next)
  zero      predict 0 for everyone — the MAE floor on a zero-inflated target
  price     rank by now_cost (present in archives from GW3 onward)
  blend     rank by blend_rank (the within-position ep_next x price blend)

Stdlib only. A gameweek already scored is never rescored without --force.
"""
import argparse
import csv
import datetime as dt
import json
import math
import urllib.request
from pathlib import Path

from metrics import (
    bootstrap_spearman_ci, captain_regret, gameweeks_to_detect, mae, mean,
    precision_at_k, rmse, spearman, stdev,
)

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
LIVE_URL = "https://fantasy.premierleague.com/api/event/{gw}/live/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
ARCHIVE_DIR = Path("predictions")
SCORES_DIR = Path("scores")
SCORECARD_PATH = Path("scorecard.json")

POPULATIONS = {
    "all": ("every player in the archive", lambda minutes: True),
    "played": ("minutes > 0", lambda minutes: minutes > 0),
    "starters": ("minutes >= 60", lambda minutes: minutes >= 60),
}
K_LIST = (10, 20)
# Per-gameweek sd of starter Spearman measured over 2025/26 in the 28 Aug 2026
# research report; used as the prior until enough gameweeks are scored.
PRIOR_STARTER_SPEARMAN_SD = 0.091
DETECT_EFFECT = 0.05


def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def checked_events(events):
    """Events whose results are final: finished AND data_checked."""
    return {e["id"]: e for e in events if e.get("finished") and e.get("data_checked")}


def archived_gameweeks():
    out = {}
    for path in ARCHIVE_DIR.glob("gw*.json"):
        if path.stem.endswith("_model"):
            continue
        try:
            out[int(path.stem[2:])] = path
        except ValueError:
            continue
    return out


def live_stats(gw):
    data = fetch_json(LIVE_URL.format(gw=gw))
    return {
        int(e["id"]): {
            "minutes": int(e["stats"].get("minutes") or 0),
            "total_points": int(e["stats"].get("total_points") or 0),
        }
        for e in data.get("elements", [])
    }


def join_rows(pred_rows, live):
    joined, missing = [], 0
    for r in pred_rows:
        pid = int(r["player_id"])
        stats = live.get(pid)
        if stats is None:
            missing += 1
            continue
        joined.append({
            "player_id": pid,
            "web_name": r.get("web_name"),
            "position": r.get("position"),
            "xPoints": float(r["xPoints"]),
            "now_cost": r.get("now_cost"),
            "blend_rank": r.get("blend_rank"),
            "minutes": stats["minutes"],
            "actual": stats["total_points"],
        })
    return joined, missing


def _clean(x):
    return None if (isinstance(x, float) and math.isnan(x)) else round(x, 4)


def score_gameweek(pred_rows, live, seed=0, extra=None):
    """Pure: archived prediction rows + live stats -> metrics dict, joined rows.

    `extra` maps a predictor name to {player_id: value} for other archived
    files graded on the same rows — e.g. the shadow model's gw{N}_model.json.
    """
    joined, missing = join_rows(pred_rows, live)
    has_price = bool(joined) and all(j["now_cost"] is not None for j in joined)
    has_blend = bool(joined) and all(j["blend_rank"] is not None for j in joined)
    extra = extra or {}
    for name, values in extra.items():
        for j in joined:
            j[name] = float(values.get(j["player_id"], 0.0))

    def predictors(rows):
        p = {
            "ep_next": [j["xPoints"] for j in rows],
            "zero": [0.0] * len(rows),
        }
        if has_price:
            p["price"] = [float(j["now_cost"]) for j in rows]
        if has_blend:
            p["blend"] = [-float(j["blend_rank"]) for j in rows]  # lower rank = better
        for name in extra:
            p[name] = [j[name] for j in rows]
        return p

    result = {
        "n": {name: 0 for name in POPULATIONS},
        "missing_from_live": missing,
        "predictors_available": sorted(predictors(joined).keys()) if joined else [],
        "metrics": {},
    }
    for pop_name, (_, keep) in POPULATIONS.items():
        rows = [j for j in joined if keep(j["minutes"])]
        result["n"][pop_name] = len(rows)
        actual = [float(j["actual"]) for j in rows]
        pop_metrics = {}
        for name, pred in predictors(rows).items():
            m = {}
            if (name in ("ep_next", "zero") or name in extra) and rows:
                m["mae"] = _clean(mae(pred, actual))
                m["rmse"] = _clean(rmse(pred, actual))
            if name != "zero" and len(rows) >= 3:
                lo, hi = bootstrap_spearman_ci(pred, actual, seed=seed)
                m["spearman"] = _clean(spearman(pred, actual))
                m["spearman_ci95"] = [_clean(lo), _clean(hi)]
                for k in K_LIST:
                    m[f"precision_at_{k}"] = _clean(precision_at_k(pred, actual, k, seed=seed))
                if pop_name == "all":
                    m["captain_regret"] = _clean(captain_regret(pred, actual, seed=seed))
            pop_metrics[name] = m
        result["metrics"][pop_name] = pop_metrics
    return result, joined


def write_gameweek(gw, event, pred_rows, result, joined):
    SCORES_DIR.mkdir(exist_ok=True)
    first = pred_rows[0] if pred_rows else {}
    payload = {
        "gameweek": gw,
        "scored_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deadline": event.get("deadline_time"),
        "data_checked": True,
        "prediction": {
            "generated_at": first.get("generated_at"),
            "source": first.get("source"),
            "model_version": first.get("model_version"),
            "ordering": first.get("ordering"),
            "rows": len(pred_rows),
        },
        "populations": {k: v[0] for k, v in POPULATIONS.items()},
        **result,
    }
    (SCORES_DIR / f"gw{gw}.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    with (SCORES_DIR / f"gw{gw}_rows.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gameweek", "player_id", "web_name", "position", "xPoints",
                    "now_cost", "blend_rank", "minutes", "actual_points"])
        for j in sorted(joined, key=lambda j: -j["xPoints"]):
            w.writerow([gw, j["player_id"], j["web_name"], j["position"], j["xPoints"],
                        j["now_cost"], j["blend_rank"], j["minutes"], j["actual"]])
    return payload


def load_scores():
    out = {}
    for path in SCORES_DIR.glob("gw*.json"):
        if path.stem.endswith("_rows"):
            continue
        try:
            out[int(path.stem[2:])] = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
    return dict(sorted(out.items()))


def build_scorecard(scores):
    def get(s, pop, pred, key):
        return s["metrics"].get(pop, {}).get(pred, {}).get(key)

    rows = []
    for gw, s in scores.items():
        rows.append({
            "gameweek": gw,
            "deadline": s.get("deadline"),
            "generated_at": s["prediction"].get("generated_at"),
            "source": s["prediction"].get("source"),
            "n_all": s["n"]["all"],
            "n_played": s["n"]["played"],
            "n_starters": s["n"]["starters"],
            "ep_next_mae_all": get(s, "all", "ep_next", "mae"),
            "zero_mae_all": get(s, "all", "zero", "mae"),
            "ep_next_spearman_all": get(s, "all", "ep_next", "spearman"),
            "ep_next_spearman_starters": get(s, "starters", "ep_next", "spearman"),
            "ep_next_spearman_starters_ci95": get(s, "starters", "ep_next", "spearman_ci95"),
            "ep_next_precision_at_20_starters": get(s, "starters", "ep_next", "precision_at_20"),
            "ep_next_captain_regret": get(s, "all", "ep_next", "captain_regret"),
            "model_mae_all": get(s, "all", "model", "mae"),
            "model_spearman_starters": get(s, "starters", "model", "spearman"),
            "model_spearman_starters_ci95": get(s, "starters", "model", "spearman_ci95"),
            "model_precision_at_20_starters": get(s, "starters", "model", "precision_at_20"),
            "model_captain_regret": get(s, "all", "model", "captain_regret"),
            "price_spearman_starters": get(s, "starters", "price", "spearman"),
            "blend_spearman_starters": get(s, "starters", "blend", "spearman"),
            "blend_precision_at_20_starters": get(s, "starters", "blend", "precision_at_20"),
        })

    starter_sp = [r["ep_next_spearman_starters"] for r in rows if r["ep_next_spearman_starters"] is not None]
    paired = [(r["model_spearman_starters"] - r["ep_next_spearman_starters"]) for r in rows
              if r["model_spearman_starters"] is not None and r["ep_next_spearman_starters"] is not None]
    observed_sd = stdev(starter_sp) if len(starter_sp) >= 3 else float("nan")
    sd_used = observed_sd if not math.isnan(observed_sd) else PRIOR_STARTER_SPEARMAN_SD
    summary = {
        "scored_gameweeks": len(rows),
        "mean_ep_next_spearman_starters": _clean(mean(starter_sp)) if starter_sp else None,
        "model_vs_ep_next_starter_spearman": {
            "gameweeks": len(paired),
            "mean_diff": _clean(mean(paired)) if paired else None,
            "sd_diff": _clean(stdev(paired)) if len(paired) >= 2 else None,
        },
        "mean_ep_next_mae_all": _clean(mean([r["ep_next_mae_all"] for r in rows if r["ep_next_mae_all"] is not None])) if rows else None,
        "mean_zero_mae_all": _clean(mean([r["zero_mae_all"] for r in rows if r["zero_mae_all"] is not None])) if rows else None,
        "power": {
            "effect_to_detect": DETECT_EFFECT,
            "per_gameweek_sd_used": _clean(sd_used),
            "sd_source": "observed across scored gameweeks" if not math.isnan(observed_sd)
                         else f"prior from 2025/26 research ({PRIOR_STARTER_SPEARMAN_SD}); observed once >= 3 gameweeks scored",
            "gameweeks_needed_power_80": gameweeks_to_detect(DETECT_EFFECT, sd_used, power=0.8),
            "gameweeks_needed_power_90": gameweeks_to_detect(DETECT_EFFECT, sd_used, power=0.9),
            "formula": "n = ((z_{1-alpha/2} + z_power) * sd / effect)^2, one-sample normal approximation, alpha 0.05",
        },
    }
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": (
            "Each archived pre-deadline snapshot is joined to official FPL results "
            "(event/{gw}/live) once the event is data_checked. Metrics are reported per "
            "named population; ties in rankings are broken at random and averaged. "
            "Files under scores/ are immutable once written."
        ),
        "populations": {k: v[0] for k, v in POPULATIONS.items()},
        "summary": summary,
        "gameweeks": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rescore gameweeks already scored")
    parser.add_argument("--gw", type=int, help="score only this gameweek")
    args = parser.parse_args()

    bootstrap = fetch_json(BOOTSTRAP_URL)
    checked = checked_events(bootstrap["events"])
    archives = archived_gameweeks()
    existing = load_scores()

    for gw in sorted(archives):
        if args.gw and gw != args.gw:
            continue
        if gw in existing and not args.force:
            print(f"⏭  GW{gw}: already scored (use --force to redo)")
            continue
        if gw not in checked:
            print(f"⏳ GW{gw}: results not final yet (finished+data_checked) — skipping")
            continue
        pred_rows = json.loads(archives[gw].read_text())
        live = live_stats(gw)
        extra = {}
        model_path = ARCHIVE_DIR / f"gw{gw}_model.json"
        if model_path.exists():
            extra["model"] = {int(r["player_id"]): float(r["xPoints"]) for r in json.loads(model_path.read_text())}
        result, joined = score_gameweek(pred_rows, live, extra=extra)
        payload = write_gameweek(gw, checked[gw], pred_rows, result, joined)
        ep = payload["metrics"]
        print(f"✅ GW{gw}: n={payload['n']} | ep_next MAE(all) {ep['all']['ep_next'].get('mae')} "
              f"vs zero {ep['all']['zero'].get('mae')} | starters Spearman "
              f"{ep['starters']['ep_next'].get('spearman')} CI {ep['starters']['ep_next'].get('spearman_ci95')}")

    scorecard = build_scorecard(load_scores())
    SCORECARD_PATH.write_text(json.dumps(scorecard, indent=2, allow_nan=False))
    p = scorecard["summary"]["power"]
    print(f"📊 scorecard: {scorecard['summary']['scored_gameweeks']} gameweeks scored; "
          f"detecting +{p['effect_to_detect']} starter Spearman needs {p['gameweeks_needed_power_80']} GWs "
          f"(80% power) at sd {p['per_gameweek_sd_used']} [{p['sd_source']}]")


if __name__ == "__main__":
    main()
