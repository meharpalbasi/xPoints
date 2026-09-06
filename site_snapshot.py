"""Freeze the site's own numbers at each deadline so they can be graded.

Alongside the xPoints feed (predictions/gw{N}.json), this archives two tables
the site computes for itself:

  site/projections/gw{N}.json   expected points per player per gameweek over
                                the planner's horizon, from /api/fpl/projections
  site/xmins/gw{N}.json         expected minutes, P(start), P(60+) and the
                                role per player, from /api/fpl/xmins

Both files may only be (re)written while gameweek N's deadline is still in
the future, so the last hourly run before the deadline is the one that
freezes them. After the deadline they stay as they were and score_site.py
grades them against official results.

    python site_snapshot.py             # archive for the next deadline
    python site_snapshot.py --out DIR   # write under DIR instead (a dry run)
"""

import argparse
import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path

from baseline import BOOTSTRAP_URL, archive_allowed, event_deadline, fetch_json, next_gameweek

SITE_BASE_URL = os.environ.get("SITE_BASE_URL", "https://fplanaly.st")
PROJECTIONS_PATH = "/api/fpl/projections?horizon=5"
XMINS_PATH = "/api/fpl/xmins"
SITE_DIR = Path("site")
MIN_ROWS = 400
HEADERS = {"User-Agent": "xPoints-archive/1.0 (+https://github.com/meharpalbasi/xPoints)", "Accept": "application/json"}


def fetch_site(path, base=SITE_BASE_URL):
    req = urllib.request.Request(base + path, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def shape_projections(table, gw):
    """The compact archive of the projections table for gameweek `gw`, or a list of problems."""
    problems = []
    if _num(table.get("gameweek")) != gw:
        problems.append(f"projections target gameweek {table.get('gameweek')}, expected {gw}")
    horizon = [int(g) for g in table.get("horizon") or []]
    if not horizon or horizon[0] != gw:
        problems.append(f"projections horizon {horizon} does not start at {gw}")
    players = table.get("players") or {}
    rows = []
    for pid, p in players.items():
        xp = p.get("xp") or {}
        rows.append({
            "player_id": int(pid),
            "player_code": p.get("code"),
            "web_name": p.get("web_name"),
            "team": p.get("team"),
            "element_type": p.get("element_type"),
            "rate": _num(p.get("rate")),
            "rate_source": p.get("rateSource"),
            "xp": {str(g): _num(xp.get(str(g), xp.get(g))) for g in horizon},
        })
    if len(rows) < MIN_ROWS:
        problems.append(f"only {len(rows)} projection rows")
    positive = sum(1 for r in rows if (r["xp"].get(str(gw)) or 0) > 0)
    if rows and positive < len(rows) * 0.3:
        problems.append(f"only {positive} of {len(rows)} rows project above zero for gameweek {gw}")
    rows.sort(key=lambda r: r["player_id"])
    feed = table.get("feed") or {}
    payload = {
        "gameweek": gw,
        "horizon": horizon,
        "model": table.get("model"),
        "minutes_model": (table.get("minutes") or {}).get("model"),
        "feed": {"available": feed.get("available"), "gameweek": feed.get("gameweek"), "label": feed.get("label"), "generated_at": feed.get("generatedAt")},
        "site_generated_at": table.get("generatedAt"),
        "rows": rows,
    }
    return payload, problems


def shape_xmins(table, gw):
    problems = []
    if _num(table.get("gameweek")) != gw:
        problems.append(f"xmins target gameweek {table.get('gameweek')}, expected {gw}")
    rows = []
    for p in table.get("players") or []:
        rows.append({
            "player_id": int(p["id"]),
            "player_code": p.get("code"),
            "web_name": p.get("web_name"),
            "team": p.get("team"),
            "element_type": p.get("element_type"),
            "status": p.get("status"),
            "xMins": _num(p.get("xMins")),
            "pStart": _num(p.get("pStart")),
            "p60": _num(p.get("p60")),
            "pAppear": _num(p.get("pAppear")),
            "fixtures": p.get("fixtures"),
            "availability": _num(p.get("availability")),
            "evidence": p.get("evidence"),
            "role": p.get("role"),
        })
    if len(rows) < MIN_ROWS:
        problems.append(f"only {len(rows)} xmins rows")
    if rows and all(r["p60"] in (None, 0) for r in rows):
        problems.append("every P(60+) is zero or missing")
    rows.sort(key=lambda r: r["player_id"])
    payload = {
        "gameweek": gw,
        "model": table.get("model"),
        "gameweeks_used": table.get("gameweeksUsed"),
        "site_generated_at": table.get("generatedAt"),
        "rows": rows,
    }
    return payload, problems


def signature(payload):
    """Rows only, so an unchanged table is not rewritten every hour for a new timestamp."""
    return json.dumps(payload.get("rows"), sort_keys=True)


def write_archive(kind, gw, payload, deadline_iso, out_dir=SITE_DIR, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    if not archive_allowed(now, deadline_iso):
        print(f"🔒 GW{gw} deadline {deadline_iso} has passed; site/{kind}/gw{gw}.json stays frozen")
        return False
    dest = Path(out_dir) / kind / f"gw{gw}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            if signature(json.loads(dest.read_text())) == signature(payload):
                print(f"⏸  site/{kind}/gw{gw}.json unchanged, not rewritten")
                return False
        except (ValueError, OSError):
            pass
    payload = {**payload, "archived_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "deadline": deadline_iso}
    dest.write_text(json.dumps(payload, indent=1, allow_nan=False))
    print(f"📁 archived site/{kind}/gw{gw}.json ({len(payload['rows'])} rows, deadline {deadline_iso})")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(SITE_DIR), help="directory to write under (default: site)")
    parser.add_argument("--base", default=SITE_BASE_URL, help="site base URL")
    args = parser.parse_args()

    bootstrap = fetch_json(BOOTSTRAP_URL)
    gw = next_gameweek(bootstrap["events"])
    if gw is None:
        sys.exit("no target gameweek resolvable")
    deadline = event_deadline(bootstrap["events"], gw)

    failures = 0
    for kind, path, shaper in (("projections", PROJECTIONS_PATH, shape_projections), ("xmins", XMINS_PATH, shape_xmins)):
        try:
            table = fetch_site(path, args.base)
        except Exception as exc:  # noqa: BLE001 - any fetch failure is reported, never raised
            print(f"❌ {kind}: fetch failed: {exc}")
            failures += 1
            continue
        payload, problems = shaper(table, gw)
        if problems:
            for p in problems:
                print(f"❌ {kind}: {p}")
            failures += 1
            continue
        write_archive(kind, gw, payload, deadline, out_dir=args.out)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
