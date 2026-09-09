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
# The promotion gate: the shadow model may replace ep_next only after this many
# consecutive paired gameweeks say so (ROADMAP: six shadow gameweeks minimum).
PROMOTION_WINDOW = 6


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


def availability_factor(status, chance, fixture_count):
    """The deadline-known availability rule model.py applies after the model:
    injured / suspended / unavailable -> 0; doubtful -> FPL's stated chance;
    blank gameweek -> 0. Applied to the archived P(60+) so the probability
    graded is the one a manager would have been shown."""
    if not fixture_count:
        return 0.0
    if status in ("i", "s", "u"):
        return 0.0
    if status == "d":
        return (chance if chance is not None else 100) / 100
    return 1.0


def model_probabilities(model_rows):
    """{player_id: P(60+ minutes)} from an archived model file, availability applied.
    Empty for archives written before the model carried its components."""
    out = {}
    for r in model_rows:
        p = r.get("p_start60")
        if p is None:
            continue
        out[int(r["player_id"])] = float(p) * availability_factor(
            r.get("status"), r.get("chance_of_playing_next_round"), r.get("fixture_count", 1))
    return out


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


def score_gameweek(pred_rows, live, seed=0, extra=None, probs=None):
    """Pure: archived prediction rows + live stats -> metrics dict, joined rows.

    `extra` maps a predictor name to {player_id: value} for other archived
    files graded on the same rows — e.g. the shadow model's gw{N}_model.json.
    `probs` maps a name to {player_id: P(60+ minutes)}: probabilities graded
    with a Brier score against minutes >= 60, beside the Brier of predicting
    the population's base rate for everyone.
    """
    joined, missing = join_rows(pred_rows, live)
    has_price = bool(joined) and all(j["now_cost"] is not None for j in joined)
    has_blend = bool(joined) and all(j["blend_rank"] is not None for j in joined)
    extra = extra or {}
    for name, values in extra.items():
        for j in joined:
            j[name] = float(values.get(j["player_id"], 0.0))
    probs = probs or {}
    for name, values in probs.items():
        for j in joined:
            j[name] = values.get(j["player_id"])

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
        for name in probs:
            pairs = [(float(j[name]), 1.0 if j["minutes"] >= 60 else 0.0) for j in rows if j.get(name) is not None]
            if not pairs:
                continue
            base = sum(y for _, y in pairs) / len(pairs)
            pop_metrics[name] = {
                "n": len(pairs),
                "brier": _clean(sum((p - y) ** 2 for p, y in pairs) / len(pairs)),
                "base_rate": _clean(base),
                "base_rate_brier": _clean(sum((base - y) ** 2 for _, y in pairs) / len(pairs)),
            }
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


def promotion_gate(rows, window=PROMOTION_WINDOW):
    """Whether the shadow model has earned the feed, and if not, why not.

    Over the last `window` graded gameweeks with a paired model row, all of:
    at least `window` such gameweeks; the model's starter rank correlation
    ahead of ep_next's on average, with the paired 95% interval clear of
    zero; and its starter precision at 20 at least ep_next's. The scorecard
    says ready; a person flips the feed. Nothing switches on its own.
    """
    paired = [r for r in rows
              if r.get("model_spearman_starters") is not None and r.get("ep_next_spearman_starters") is not None]
    recent = paired[-window:]
    n = len(recent)
    diffs = [r["model_spearman_starters"] - r["ep_next_spearman_starters"] for r in recent]
    mean_diff = mean(diffs) if diffs else None
    se = stdev(diffs) / math.sqrt(n) if n >= 2 else None
    ci = [mean_diff - 1.96 * se, mean_diff + 1.96 * se] if se is not None and not math.isnan(se) else None
    p_model = [r["model_precision_at_20_starters"] for r in recent if r.get("model_precision_at_20_starters") is not None]
    p_feed = [r["ep_next_precision_at_20_starters"] for r in recent if r.get("ep_next_precision_at_20_starters") is not None]
    reasons = []
    if n < window:
        reasons.append(f"{n} of {window} paired gameweeks graded")
    if mean_diff is None or mean_diff <= 0:
        reasons.append("starter rank match not ahead of ep_next on average")
    elif ci is None or ci[0] <= 0:
        reasons.append("paired 95% interval for starter rank match includes zero")
    if p_model and p_feed and mean(p_model) < mean(p_feed):
        reasons.append("starter precision at 20 below ep_next")
    return {
        "ready": n >= window and not reasons,
        "window_gameweeks": window,
        "gameweeks_in_window": n,
        "gameweeks": [r["gameweek"] for r in recent],
        "starter_spearman_diff": {
            "mean": _clean(mean_diff) if mean_diff is not None else None,
            "se": _clean(se) if se is not None and not math.isnan(se) else None,
            "ci95": [_clean(ci[0]), _clean(ci[1])] if ci else None,
        },
        "precision_at_20_starters": {
            "model_mean": _clean(mean(p_model)) if p_model else None,
            "ep_next_mean": _clean(mean(p_feed)) if p_feed else None,
        },
        "reasons": reasons,
        "rule": (f"over the last {window} paired gameweeks: model starter Spearman ahead of ep_next "
                 "with the paired 95% interval above zero, and starter precision at 20 at least ep_next's; "
                 "the feed is switched by a person, never by this file"),
    }


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
            "model_p60_brier_all": get(s, "all", "model_p60", "brier"),
            "model_p60_base_rate_brier_all": get(s, "all", "model_p60", "base_rate_brier"),
            "price_spearman_starters": get(s, "starters", "price", "spearman"),
            "blend_spearman_starters": get(s, "starters", "blend", "spearman"),
            "blend_precision_at_20_starters": get(s, "starters", "blend", "precision_at_20"),
        })

    starter_sp = [r["ep_next_spearman_starters"] for r in rows if r["ep_next_spearman_starters"] is not None]
    paired = [(r["model_spearman_starters"] - r["ep_next_spearman_starters"]) for r in rows
              if r["model_spearman_starters"] is not None and r["ep_next_spearman_starters"] is not None]
    observed_sd = stdev(starter_sp) if len(starter_sp) >= 3 else float("nan")
    sd_used = observed_sd if not math.isnan(observed_sd) else PRIOR_STARTER_SPEARMAN_SD
    gate = promotion_gate(rows)
    summary = {
        "scored_gameweeks": len(rows),
        "promotion_ready": gate["ready"],
        "promotion": gate,
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
        extra, probs = {}, {}
        model_path = ARCHIVE_DIR / f"gw{gw}_model.json"
        if model_path.exists():
            model_rows = json.loads(model_path.read_text())
            extra["model"] = {int(r["player_id"]): float(r["xPoints"]) for r in model_rows}
            p60 = model_probabilities(model_rows)
            if p60:
                probs["model_p60"] = p60
        result, joined = score_gameweek(pred_rows, live, extra=extra, probs=probs)
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
    gate = scorecard["summary"]["promotion"]
    print(f"🚦 promotion gate: {'READY' if gate['ready'] else 'not ready'} "
          f"({gate['gameweeks_in_window']}/{gate['window_gameweeks']} paired gameweeks"
          + (f"; {'; '.join(gate['reasons'])}" if gate["reasons"] else "") + ")")


if __name__ == "__main__":
    main()
