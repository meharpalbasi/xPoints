#!/usr/bin/env python3
"""Download vaastav's per-gameweek FPL history for a season into data/.

Training data is not committed; this makes it reproducible. 2025/26 is the
first season under the current scoring rules (defensive contributions), so it
is the only season used for training — earlier seasons have incomparable
total_points. Note the file's `xP` column is FPL's post-hoc ep_this, populated
in only 11 of 38 gameweeks; it is never used here.
"""
import sys
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"


def fetch(season):
    Path("data").mkdir(exist_ok=True)
    dest = Path("data") / f"merged_gw_{season}.csv"
    with urllib.request.urlopen(URL.format(season=season), timeout=60) as r:
        dest.write_bytes(r.read())
    print(f"wrote {dest} ({dest.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    for season in sys.argv[1:] or ["2025-26"]:
        fetch(season)
