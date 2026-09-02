#!/usr/bin/env python3
"""Download vaastav's FPL history for a season into data/ (never committed).

Two files per season:
  merged_gw.csv   one row per player-fixture with points, minutes, xG, DEFCON...
  players_raw.csv the season's element id -> stable player `code` mapping, which
                  is the ONLY safe join key across seasons (element ids are
                  reassigned every year)

2025/26 is the first season under defensive-contribution scoring, so earlier
seasons' total_points are not comparable and are not used. The merged file's
`xP` column is FPL's post-hoc ep_this, populated in ~11 of 38 gameweeks; it is
never used.
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/"
FILES = {"merged_gw": "gws/merged_gw.csv", "players_raw": "players_raw.csv"}


def fetch(season, quiet=False):
    Path("data").mkdir(exist_ok=True)
    for name, rel in FILES.items():
        dest = Path("data") / f"{name}_{season}.csv"
        with urllib.request.urlopen(BASE.format(season=season) + rel, timeout=60) as r:
            dest.write_bytes(r.read())
        if not quiet:
            print(f"wrote {dest} ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    for season in sys.argv[1:] or ["2025-26", "2026-27"]:
        fetch(season)
