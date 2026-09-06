"""Grade the site's frozen projections and expected minutes.

For every archived site/projections/gw{N}.json and every gameweek M inside
its horizon that FPL has marked data_checked, join the projection for M to
official results and score it exactly as score.py scores the feed: per
named population (all, played, starters), MAE, RMSE, rank correlation with a
bootstrap interval and precision at 10 and 20. The result is keyed by the
horizon week (M minus N): week 0 is the next-deadline projection, week 4 the
one made four deadlines earlier.

For every site/xmins/gw{N}.json whose gameweek is final: P(60+) against
whether the player played sixty minutes (Brier score, the base-rate Brier it
must beat, and a calibration table), and expected minutes against actual
minutes (MAE).

Outputs
  scores/site/gw{M}.json    every grade for gameweek M; entries never change
                            once written, later runs only add missing ones
  site_scorecard.json       the summary by horizon week and for minutes

    python score_site.py            # grade what is final and not yet graded
    python score_site.py --gw 4     # only gameweek 4
"""

import argparse
import datetime as dt
import json
import math
from pathlib import Path

from metrics import bootstrap_spearman_ci, mae, mean, precision_at_k, rmse, spearman
from score import BOOTSTRAP_URL, K_LIST, POPULATIONS, checked_events, fetch_json, live_stats

SITE_DIR = Path("site")
SCORES_DIR = Path("scores") / "site"
SCORECARD_PATH = Path("site_scorecard.json")
CALIBRATION_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)


def _clean(x):
    if x is None:
        return None
    return None if (isinstance(x, float) and math.isnan(x)) else round(x, 4)


def archived(kind):
    out = {}
    for path in (SITE_DIR / kind).glob("gw*.json"):
        try:
            out[int(path.stem[2:])] = path
        except ValueError:
            continue
    return dict(sorted(out.items()))


def load_scores():
    out = {}
    for path in SCORES_DIR.glob("gw*.json"):
        try:
            out[int(path.stem[2:])] = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Projections

def score_projection_rows(rows, live, target_gw, seed=0):
    """Pure: archived projection rows + live stats for `target_gw` -> metrics per population.

    A row with no projection for the target gameweek counts as zero (a blank
    is zero by construction). Rows absent from the live file are dropped and
    counted in `missing_from_live`.
    """
    joined, missing = [], 0
    key = str(target_gw)
    for r in rows:
        stats = live.get(int(r["player_id"]))
        if stats is None:
            missing += 1
            continue
        xp = r.get("xp") or {}
        pred = xp.get(key)
        joined.append({"pred": float(pred) if pred is not None else 0.0, "minutes": stats["minutes"], "actual": stats["total_points"]})
    metrics, n = {}, {}
    for pop, (_, keep) in POPULATIONS.items():
        sub = [j for j in joined if keep(j["minutes"])]
        n[pop] = len(sub)
        if len(sub) < 3:
            metrics[pop] = {}
            continue
        pred = [j["pred"] for j in sub]
        actual = [j["actual"] for j in sub]
        m = {"mae": _clean(mae(pred, actual)), "rmse": _clean(rmse(pred, actual)), "zero_mae": _clean(mae([0.0] * len(actual), actual))}
        lo, hi = bootstrap_spearman_ci(pred, actual, seed=seed)
        m["spearman"] = _clean(spearman(pred, actual))
        m["spearman_ci95"] = [_clean(lo), _clean(hi)]
        for k in K_LIST:
            if len(sub) >= k:
                m[f"precision_at_{k}"] = _clean(precision_at_k(pred, actual, k, seed=seed))
        metrics[pop] = m
    return {"n": n, "missing_from_live": missing, "metrics": metrics}


# ---------------------------------------------------------------------------
# Expected minutes

def score_xmins_rows(rows, live):
    """Pure: archived xMins rows + live stats -> Brier and calibration for P(60+), MAE for minutes."""
    pairs = []
    missing = 0
    for r in rows:
        stats = live.get(int(r["player_id"]))
        if stats is None:
            missing += 1
            continue
        p60 = r.get("p60")
        xm = r.get("xMins")
        if p60 is None or xm is None:
            continue
        pairs.append({"p60": float(p60), "xmins": float(xm), "minutes": stats["minutes"], "sixty": 1.0 if stats["minutes"] >= 60 else 0.0, "appeared": 1.0 if stats["minutes"] > 0 else 0.0, "p_appear": r.get("pAppear")})
    if not pairs:
        return {"n": 0, "missing_from_live": missing}
    n = len(pairs)
    base_rate = mean([p["sixty"] for p in pairs])
    brier = mean([(p["p60"] - p["sixty"]) ** 2 for p in pairs])
    brier_base = mean([(base_rate - p["sixty"]) ** 2 for p in pairs])
    buckets = []
    for lo, hi in zip(CALIBRATION_EDGES[:-1], CALIBRATION_EDGES[1:]):
        inside = [p for p in pairs if lo <= p["p60"] < hi]
        buckets.append({
            "from": lo, "to": min(hi, 1.0), "n": len(inside),
            "predicted": _clean(mean([p["p60"] for p in inside])) if inside else None,
            "actual": _clean(mean([p["sixty"] for p in inside])) if inside else None,
        })
    appear = [p for p in pairs if p["p_appear"] is not None]
    return {
        "n": n,
        "missing_from_live": missing,
        "base_rate_60": _clean(base_rate),
        "brier_60": _clean(brier),
        "brier_base_rate": _clean(brier_base),
        "brier_skill": _clean(1 - brier / brier_base) if brier_base else None,
        "minutes_mae": _clean(mean([abs(p["xmins"] - p["minutes"]) for p in pairs])),
        "minutes_mae_played": _clean(mean([abs(p["xmins"] - p["minutes"]) for p in pairs if p["minutes"] > 0])) if any(p["minutes"] > 0 for p in pairs) else None,
        "brier_appear": _clean(mean([(float(p["p_appear"]) - p["appeared"]) ** 2 for p in appear])) if appear else None,
        "calibration": buckets,
    }


# ---------------------------------------------------------------------------
# Files

def merge_gameweek(existing, additions):
    """Entries already written win; new snapshots and the minutes grade are added."""
    out = dict(existing or {})
    proj = dict(out.get("projections") or {})
    for snapshot, entry in (additions.get("projections") or {}).items():
        proj.setdefault(str(snapshot), entry)
    out["projections"] = proj
    if additions.get("xmins") and not out.get("xmins"):
        out["xmins"] = additions["xmins"]
    return out


def build_scorecard(scores):
    """Summaries by horizon week and for minutes, plus every per-gameweek row."""
    by_week = {}
    rows = []
    minutes_rows = []
    for gw, s in scores.items():
        for snapshot, entry in (s.get("projections") or {}).items():
            k = entry.get("horizon_week")
            m = entry.get("metrics") or {}
            row = {
                "gameweek": gw, "snapshot": int(snapshot), "horizon_week": k,
                "n_all": (entry.get("n") or {}).get("all"), "n_starters": (entry.get("n") or {}).get("starters"),
                "mae_all": m.get("all", {}).get("mae"), "zero_mae_all": m.get("all", {}).get("zero_mae"),
                "spearman_starters": m.get("starters", {}).get("spearman"), "spearman_starters_ci95": m.get("starters", {}).get("spearman_ci95"),
                "precision_at_20_starters": m.get("starters", {}).get("precision_at_20"),
            }
            rows.append(row)
            by_week.setdefault(k, []).append(row)
        if s.get("xmins") and s["xmins"].get("n"):
            x = s["xmins"]
            minutes_rows.append({"gameweek": gw, "n": x["n"], "brier_60": x.get("brier_60"), "brier_base_rate": x.get("brier_base_rate"), "brier_skill": x.get("brier_skill"), "minutes_mae": x.get("minutes_mae"), "minutes_mae_played": x.get("minutes_mae_played"), "base_rate_60": x.get("base_rate_60"), "calibration": x.get("calibration")})
    def avg(items, key):
        xs = [i[key] for i in items if i.get(key) is not None]
        return _clean(mean(xs)) if xs else None
    horizon = {}
    for k in sorted(by_week, key=lambda v: (v is None, v)):
        items = by_week[k]
        horizon[str(k)] = {"gameweeks": len(items), "mean_mae_all": avg(items, "mae_all"), "mean_zero_mae_all": avg(items, "zero_mae_all"), "mean_spearman_starters": avg(items, "spearman_starters"), "mean_precision_at_20_starters": avg(items, "precision_at_20_starters")}
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": (
            "The site's projections table and expected-minutes table are frozen at each deadline "
            "(site/). Once FPL marks a gameweek data_checked, every frozen projection for it is "
            "joined to official results (event/{gw}/live) and scored per named population like the "
            "feed; the horizon week is how many deadlines before the gameweek the projection was made. "
            "P(60+) is scored with the Brier score against playing sixty minutes, beside the base-rate "
            "Brier it must beat. Entries under scores/site/ never change once written."
        ),
        "populations": {k: v[0] for k, v in POPULATIONS.items()},
        "summary": {
            "gameweeks_graded": len(scores),
            "by_horizon_week": horizon,
            "minutes": {
                "gameweeks": len(minutes_rows),
                "mean_brier_60": avg(minutes_rows, "brier_60"),
                "mean_brier_base_rate": avg(minutes_rows, "brier_base_rate"),
                "mean_brier_skill": avg(minutes_rows, "brier_skill"),
                "mean_minutes_mae": avg(minutes_rows, "minutes_mae"),
                "mean_minutes_mae_played": avg(minutes_rows, "minutes_mae_played"),
                "latest_calibration": minutes_rows[-1].get("calibration") if minutes_rows else None,
                "latest_gameweek": minutes_rows[-1]["gameweek"] if minutes_rows else None,
            },
        },
        "gameweeks": rows,
        "minutes": minutes_rows,
    }


def grade(bootstrap, projections_archive, xmins_archive, existing, live_for, only_gw=None, now=None):
    """Pure apart from `live_for(gw)`: returns { gw: merged payload } for every gameweek that gained a grade."""
    checked = checked_events(bootstrap["events"])
    now = now or dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    updated = {}
    live_cache = {}

    def live(gw):
        if gw not in live_cache:
            live_cache[gw] = live_for(gw)
        return live_cache[gw]

    for snapshot_gw, path in projections_archive.items():
        table = json.loads(Path(path).read_text())
        for target in table.get("horizon") or []:
            target = int(target)
            if only_gw and target != only_gw:
                continue
            if target not in checked:
                continue
            current = updated.get(target) or existing.get(target) or {}
            if str(snapshot_gw) in (current.get("projections") or {}):
                continue
            result = score_projection_rows(table["rows"], live(target), target)
            entry = {"horizon_week": target - snapshot_gw, "scored_at": stamp, "snapshot_deadline": table.get("deadline"), "snapshot_archived_at": table.get("archived_at"), "model": table.get("model"), "feed": table.get("feed"), **result}
            merged = merge_gameweek(current, {"projections": {str(snapshot_gw): entry}})
            merged.setdefault("gameweek", target)
            merged.setdefault("deadline", checked[target].get("deadline_time"))
            updated[target] = merged
            print(f"✅ GW{target} from the GW{snapshot_gw} snapshot (week {target - snapshot_gw}): MAE(all) {result['metrics'].get('all', {}).get('mae')}, starters Spearman {result['metrics'].get('starters', {}).get('spearman')}")

    for gw, path in xmins_archive.items():
        if only_gw and gw != only_gw:
            continue
        if gw not in checked:
            continue
        current = updated.get(gw) or existing.get(gw) or {}
        if current.get("xmins"):
            continue
        table = json.loads(Path(path).read_text())
        result = score_xmins_rows(table["rows"], live(gw))
        entry = {"scored_at": stamp, "snapshot_archived_at": table.get("archived_at"), "model": table.get("model"), **result}
        merged = merge_gameweek(current, {"xmins": entry})
        merged.setdefault("gameweek", gw)
        merged.setdefault("deadline", checked[gw].get("deadline_time"))
        updated[gw] = merged
        print(f"✅ GW{gw} expected minutes: Brier {result.get('brier_60')} against base rate {result.get('brier_base_rate')}, minutes MAE {result.get('minutes_mae')}")
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, help="grade only this gameweek")
    args = parser.parse_args()
    bootstrap = fetch_json(BOOTSTRAP_URL)
    existing = load_scores()
    updated = grade(bootstrap, archived("projections"), archived("xmins"), existing, live_stats, only_gw=args.gw)
    if not updated and not existing:
        print("⏳ nothing to grade yet")
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    for gw, payload in updated.items():
        (SCORES_DIR / f"gw{gw}.json").write_text(json.dumps(payload, indent=1, allow_nan=False))
    scorecard = build_scorecard(load_scores())
    SCORECARD_PATH.write_text(json.dumps(scorecard, indent=2, allow_nan=False))
    print(f"📊 site scorecard: {scorecard['summary']['gameweeks_graded']} gameweeks, horizon weeks {list(scorecard['summary']['by_horizon_week'])}, minutes graded {scorecard['summary']['minutes']['gameweeks']}")


if __name__ == "__main__":
    main()
