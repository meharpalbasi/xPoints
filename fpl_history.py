#!/usr/bin/env python3
"""The season in progress, straight from FPL's own API.

vaastav's history lags the live season by days: on 9 September 2026 it held
GW1 while GW3 was final and carried no transfer data at all, so the model was
training without its newest completed matches. FPL's element-summary endpoint
has every completed row per player with the same columns as merged_gw.csv
(points, minutes, xG, starts, price, ownership, transfers), so the current
season is built from it directly. Only gameweeks FPL has marked finished are
kept: the row FPL creates for a fixture before kick-off is not history.

Cached under data/ (never committed) and refreshed when the cache is older
than a few hours or lacks the newest finished gameweek.
"""
import concurrent.futures as cf
import datetime as dt
from pathlib import Path

import pandas as pd

from baseline import POSITIONS, fetch_json

SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{id}/"
STAT_FIELDS = ["minutes", "total_points", "goals_scored", "assists", "clean_sheets", "goals_conceded",
               "bonus", "bps", "saves", "starts", "expected_goals", "expected_assists",
               "expected_goal_involvements", "expected_goals_conceded", "defensive_contribution",
               "value", "selected", "transfers_in", "transfers_out", "transfers_balance",
               "team_h_score", "team_a_score"]
COLUMNS = ["name", "position", "team", "element", "GW", "fixture", "opponent_team", "kickoff_time",
           "was_home"] + STAT_FIELDS
CACHE_MAX_AGE = dt.timedelta(hours=6)


def finished_gameweeks(events):
    return {e["id"] for e in events if e.get("finished")}


def player_rows(player, history, finished, team_names):
    """merged_gw-shaped rows for one player: completed gameweeks only."""
    rows = []
    for h in history:
        if h.get("round") not in finished:
            continue
        row = {"name": player.get("web_name"), "position": POSITIONS[player["element_type"]],
               "team": team_names.get(player["team"], ""), "element": player["id"], "GW": h["round"],
               "fixture": h.get("fixture"), "opponent_team": h.get("opponent_team"),
               "kickoff_time": h.get("kickoff_time"), "was_home": bool(h.get("was_home"))}
        for f in STAT_FIELDS:
            row[f] = h.get(f)
        rows.append(row)
    return rows


def season_rows(bootstrap, fetch=fetch_json, workers=8):
    """Every player's completed gameweeks this season, as one DataFrame."""
    finished = finished_gameweeks(bootstrap["events"])
    team_names = {t["id"]: t["name"] for t in bootstrap["teams"]}
    players = [p for p in bootstrap["elements"] if p.get("element_type") in POSITIONS]

    def one(p):
        return player_rows(p, fetch(SUMMARY_URL.format(id=p["id"])).get("history", []), finished, team_names)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for rs in pool.map(one, players) for r in rs]
    return pd.DataFrame(rows, columns=COLUMNS)


def cache_is_fresh(path, newest_finished, now=None, max_age=CACHE_MAX_AGE):
    if not path.exists():
        return False
    now = now or dt.datetime.now(dt.timezone.utc)
    age = now - dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
    if age > max_age:
        return False
    try:
        cached = pd.read_csv(path, usecols=["GW"])
    except (ValueError, OSError):
        return False
    have = int(cached["GW"].max()) if len(cached) else 0
    return have >= newest_finished


def load_or_fetch(season, bootstrap, cache_dir=Path("data"), use_cache_only=False, fetch=fetch_json):
    """The season's completed rows, from the cache when it is fresh."""
    path = Path(cache_dir) / f"fpl_history_{season}.csv"
    newest = max(finished_gameweeks(bootstrap["events"]), default=0)
    if path.exists() and (use_cache_only or cache_is_fresh(path, newest)):
        return pd.read_csv(path)
    df = season_rows(bootstrap, fetch=fetch)
    Path(cache_dir).mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df
