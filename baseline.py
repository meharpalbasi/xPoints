#!/usr/bin/env python3
"""
ep_next baseline predictions — the safety net under script.py.

Emits predictions.json from FPL's own `ep_next` (expected points, next
gameweek) with CURRENT-season element IDs. It exists because the XGBoost
model trains on current-season history only, so it cannot run pre-season
or in the first ~5 gameweeks — and the previous behaviour (keep serving
the last file the model wrote) shipped 2025/26 element IDs into 2026/27,
where FPL had reassigned ~99% of them to different players.

Output contract: same path, same array shape, and a superset of every
field the consumers read (fpl-app xPoints/optimize/live-hero routes and
fpl-analyst-mobile's XPointsPrediction type). Adds provenance per row:
generated_at / season / source / model_version.

Also writes predictions/gw{N}.json — a per-gameweek snapshot that freezes
at each deadline, so predictions can later be scored against actual
points. (predictions.json itself is overwritten daily; before this file
existed, every prediction the project ever made was destroyed.)

Stdlib only, deliberately: this is the code that runs when the ML stack
is the thing that broke.

Usage:
    python baseline.py                 # full run: fetch, build, write both files
    python baseline.py --archive-only  # just copy existing predictions.json to
                                       # predictions/gw{N}.json (run after a
                                       # successful script.py so model output
                                       # gets archived too)
"""
import argparse
import datetime as dt
import json
import shutil
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
# Mirrors script.py's status encoding so downstream consumers see the same scale.
STATUS_MAP = {"a": 4, "d": 2, "i": 0, "s": 0, "u": 1}

MODEL_VERSION = "ep-next-baseline-1"
OUT_PATH = Path("predictions.json")
ARCHIVE_DIR = Path("predictions")


def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def next_gameweek(events):
    """The gameweek these predictions are for: is_next, else is_current,
    else the first unfinished event (covers every API state incl. pre-season)."""
    for ev in events:
        if ev.get("is_next"):
            return ev["id"]
    for ev in events:
        if ev.get("is_current"):
            return ev["id"]
    unfinished = [ev["id"] for ev in events if not ev.get("finished")]
    return min(unfinished) if unfinished else None


def season_label(events):
    """e.g. '2026/27', derived from the GW1 deadline year."""
    first = min(events, key=lambda ev: ev["id"])
    year = int(first["deadline_time"][:4])
    return f"{year}/{str(year + 1)[-2:]}"


def team_fixture_stats(fixtures, gw):
    """Per-team stats for the target gameweek: fixture count, mean
    difficulty, home share. Handles DGWs (multiple rows) and BGWs (none)."""
    stats = defaultdict(lambda: {"count": 0, "difficulty": [], "home": 0})
    for f in fixtures:
        if f.get("event") != gw:
            continue
        h, a = f["team_h"], f["team_a"]
        stats[h]["count"] += 1
        stats[h]["home"] += 1
        stats[h]["difficulty"].append(f.get("team_h_difficulty") or 3)
        stats[a]["count"] += 1
        stats[a]["difficulty"].append(f.get("team_a_difficulty") or 3)
    return stats


def build_rows(bootstrap, fixtures, gw):
    teams = {t["id"]: t for t in bootstrap["teams"]}
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    season = season_label(bootstrap["events"])
    fx = team_fixture_stats(fixtures, gw)

    rows = []
    for p in bootstrap["elements"]:
        etype = p.get("element_type")
        if etype not in POSITIONS:
            # Managers (element_type 5, added by FPL for assistant-manager
            # chips) and anything else FPL invents never belong in an
            # outfield-points file. The 2025/26 file's top row was a manager.
            continue
        t = fx.get(p["team"], {"count": 0, "difficulty": [], "home": 0})
        n = t["count"]
        ep = float(p.get("ep_next") or 0.0)
        rows.append({
            "player_id": p["id"],
            "player_code": p.get("code"),  # stable across seasons — the join key predictions.json should always have carried
            "web_name": p.get("web_name"),
            "team": p["team"],
            "team_name": teams.get(p["team"], {}).get("name", ""),
            "element_type": etype,
            "position": POSITIONS[etype],
            "xPoints": round(ep, 4),
            "xPoints_raw": round(ep, 4),
            "form": float(p.get("form") or 0.0),
            "selected_by_percent": str(p.get("selected_by_percent") or "0.0"),
            "status_numeric": STATUS_MAP.get(p.get("status"), 1),
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round"),
            "fixture_count": n,
            "avg_difficulty": round(sum(t["difficulty"]) / n, 2) if n else 0.0,
            "fixture_difficulty": round(sum(t["difficulty"]) / n, 2) if n else 0.0,
            "home_proportion": round(t["home"] / n, 2) if n else 0.0,
            "generated_at": generated_at,
            "season": season,
            "gameweek": gw,
            "source": "fpl_ep_next",
            "model_version": MODEL_VERSION,
        })

    rows.sort(key=lambda r: r["xPoints"], reverse=True)
    return rows


def sanity_check(rows):
    """Refuse to overwrite a working file with a broken one. Every failure
    here should page someone (the workflow alerts on non-zero exit)."""
    problems = []
    if len(rows) < 400:
        problems.append(f"only {len(rows)} rows (expected ~550-800)")
    if rows and rows[0]["xPoints"] < 1.0:
        problems.append(f"max xPoints {rows[0]['xPoints']} < 1.0 — ep_next looks unpopulated")
    nonzero = sum(1 for r in rows if r["xPoints"] > 0)
    if nonzero < 100:
        problems.append(f"only {nonzero} players with xPoints > 0")
    bad_types = [r["player_id"] for r in rows if r["element_type"] not in POSITIONS]
    if bad_types:
        problems.append(f"non-player element_types leaked: {bad_types[:5]}")
    return problems


def archive(gw):
    if gw is None:
        print("⚠️  no target gameweek resolvable — skipping archive")
        return
    ARCHIVE_DIR.mkdir(exist_ok=True)
    dest = ARCHIVE_DIR / f"gw{gw}.json"
    shutil.copyfile(OUT_PATH, dest)
    print(f"📁 archived {OUT_PATH} -> {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-only", action="store_true",
                        help="copy existing predictions.json to predictions/gw{N}.json and exit")
    args = parser.parse_args()

    bootstrap = fetch_json(BOOTSTRAP_URL)
    gw = next_gameweek(bootstrap["events"])

    if args.archive_only:
        if not OUT_PATH.exists():
            sys.exit("predictions.json does not exist — nothing to archive")
        archive(gw)
        return

    fixtures = fetch_json(FIXTURES_URL)
    rows = build_rows(bootstrap, fixtures, gw)

    problems = sanity_check(rows)
    if problems:
        for p in problems:
            print(f"❌ sanity check failed: {p}")
        sys.exit(1)

    OUT_PATH.write_text(json.dumps(rows, indent=2))
    nonzero = sum(1 for r in rows if r["xPoints"] > 0)
    print(f"✅ wrote {len(rows)} predictions for GW{gw} "
          f"({nonzero} non-zero, top: {rows[0]['web_name']} {rows[0]['xPoints']:.2f})")
    archive(gw)


if __name__ == "__main__":
    main()
