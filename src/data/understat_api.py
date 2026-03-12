"""Understat data fetching module for xPoints.

Fetches player-level and team-level advanced stats from Understat via the
understatapi library. Maps Understat names to FPL API IDs using fuzzy matching.
Caches results to data/ directory as JSON.

Player stats: xG, xA, npxG, xGChain, xGBuildup, key_passes, shots
Team stats: xG, xGA, Deep, PPDA (att + def) per match
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from fuzzywuzzy import fuzz, process
from understatapi import UnderstatClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_TTL_HOURS = 6  # re-fetch if cache older than this

# Understat uses slightly different team names than FPL
TEAM_NAME_OVERRIDES: dict[str, str] = {
    "Manchester United": "Man Utd",
    "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Tottenham": "Spurs",
    "Nottingham Forest": "Nott'm Forest",
    "Leicester": "Leicester",
    "Ipswich": "Ipswich",
}

# Reverse mapping (FPL short name -> Understat name)
FPL_TO_UNDERSTAT_TEAM: dict[str, str] = {v: k for k, v in TEAM_NAME_OVERRIDES.items()}

CURRENT_SEASON = "2025"  # Understat uses the start year of the season


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(name: str) -> Path:
    """Return path for a named cache file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{name}.json"


def _cache_is_fresh(path: Path, ttl_hours: float = CACHE_TTL_HOURS) -> bool:
    """Check whether a cache file exists and is younger than *ttl_hours*."""
    if not path.exists():
        return False
    age_s = time.time() - path.stat().st_mtime
    return age_s < ttl_hours * 3600


def _read_cache(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _write_cache(path: Path, data: Any) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _get_fpl_bootstrap() -> dict:
    """Fetch FPL bootstrap-static and cache it."""
    cache = _cache_path("fpl_bootstrap")
    if _cache_is_fresh(cache, ttl_hours=1):
        return _read_cache(cache)
    resp = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    _write_cache(cache, data)
    return data


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------

def build_fpl_player_lookup() -> pd.DataFrame:
    """Return a DataFrame of FPL players with id, web_name, first_name, second_name, team."""
    bs = _get_fpl_bootstrap()
    players = pd.DataFrame(bs["elements"])[
        ["id", "web_name", "first_name", "second_name", "team"]
    ]
    teams = pd.DataFrame(bs["teams"])[["id", "name", "short_name"]]
    players = players.merge(
        teams.rename(columns={"id": "team", "name": "team_name", "short_name": "team_short"}),
        on="team",
        how="left",
    )
    players["full_name"] = players["first_name"] + " " + players["second_name"]
    return players


def fuzzy_match_player(
    understat_name: str,
    fpl_players: pd.DataFrame,
    team_hint: Optional[str] = None,
    threshold: int = 70,
) -> Optional[int]:
    """Match an Understat player name to an FPL player ID via fuzzy matching.

    Args:
        understat_name: Player name from Understat.
        fpl_players: DataFrame from build_fpl_player_lookup().
        team_hint: Optional Understat team name to narrow search.
        threshold: Minimum fuzzy score to accept a match.

    Returns:
        FPL player ID or None if no match found.
    """
    candidates = fpl_players.copy()

    # Narrow by team if hint provided
    if team_hint:
        fpl_team = TEAM_NAME_OVERRIDES.get(team_hint, team_hint)
        team_filter = candidates[
            (candidates["team_name"] == fpl_team)
            | (candidates["team_short"] == fpl_team)
            | (candidates["team_name"] == team_hint)
        ]
        if not team_filter.empty:
            candidates = team_filter

    # Try matching against full_name, then web_name
    choices = candidates["full_name"].tolist()
    result = process.extractOne(understat_name, choices, scorer=fuzz.token_sort_ratio)

    if result and result[1] >= threshold:
        idx = candidates[candidates["full_name"] == result[0]].index[0]
        return int(candidates.loc[idx, "id"])

    # Fallback to web_name
    choices_web = candidates["web_name"].tolist()
    result_web = process.extractOne(understat_name, choices_web, scorer=fuzz.token_sort_ratio)
    if result_web and result_web[1] >= threshold:
        idx = candidates[candidates["web_name"] == result_web[0]].index[0]
        return int(candidates.loc[idx, "id"])

    return None


def map_understat_team_to_fpl(understat_team: str) -> Optional[int]:
    """Map an Understat team name to an FPL team ID.

    Returns:
        FPL team ID or None.
    """
    bs = _get_fpl_bootstrap()
    teams = pd.DataFrame(bs["teams"])
    fpl_name = TEAM_NAME_OVERRIDES.get(understat_team, understat_team)

    match = teams[
        (teams["name"] == fpl_name)
        | (teams["short_name"] == fpl_name)
        | (teams["name"] == understat_team)
    ]
    if not match.empty:
        return int(match.iloc[0]["id"])

    # Fuzzy fallback
    choices = teams["name"].tolist()
    result = process.extractOne(understat_team, choices, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= 70:
        return int(teams[teams["name"] == result[0]].iloc[0]["id"])
    return None


# ---------------------------------------------------------------------------
# Main data class
# ---------------------------------------------------------------------------

class UnderstatData:
    """Fetches and caches Understat player and team data.

    Usage:
        ud = UnderstatData()
        player_df = ud.get_player_match_stats()
        team_df = ud.get_team_match_stats()
    """

    def __init__(self, season: str = CURRENT_SEASON, league: str = "EPL"):
        self.season = season
        self.league = league
        self._fpl_players: Optional[pd.DataFrame] = None

    @property
    def fpl_players(self) -> pd.DataFrame:
        if self._fpl_players is None:
            self._fpl_players = build_fpl_player_lookup()
        return self._fpl_players

    # ------------------------------------------------------------------
    # Player-level stats
    # ------------------------------------------------------------------

    def fetch_league_players(self) -> list[dict]:
        """Fetch all player season summaries from Understat for the league/season."""
        cache = _cache_path(f"understat_players_{self.league}_{self.season}")
        if _cache_is_fresh(cache):
            return _read_cache(cache)

        with UnderstatClient() as client:
            raw = client.league(league=self.league).get_player_data(season=self.season)

        # raw is list of dicts
        players = raw if isinstance(raw, list) else list(raw.values())
        _write_cache(cache, players)
        return players

    def fetch_player_matches(self, understat_player_id: int) -> list[dict]:
        """Fetch per-match stats for a single player from Understat.

        Args:
            understat_player_id: Understat's numeric player ID.

        Returns:
            List of match-level stat dicts.
        """
        cache = _cache_path(f"understat_player_{understat_player_id}_{self.season}")
        if _cache_is_fresh(cache):
            return _read_cache(cache)

        try:
            with UnderstatClient() as client:
                raw = client.player(player=understat_player_id).get_match_data(season=self.season)
            matches = raw if isinstance(raw, list) else list(raw.values())
            _write_cache(cache, matches)
            return matches
        except Exception as e:
            print(f"[understat] Error fetching player {understat_player_id}: {e}")
            return []

    def get_player_match_stats(self, min_minutes: int = 200) -> pd.DataFrame:
        """Fetch per-match stats for all qualifying players, mapped to FPL IDs.

        Filters to players with >= min_minutes total minutes in the season.

        Returns:
            DataFrame with columns: fpl_id, understat_id, player_name, team,
            date, xG, xA, npxG, xGChain, xGBuildup, key_passes, shots, minutes
        """
        cache = _cache_path(f"understat_player_matches_{self.league}_{self.season}")
        if _cache_is_fresh(cache):
            return pd.DataFrame(_read_cache(cache))

        league_players = self.fetch_league_players()

        # Filter by minutes
        qualifying = [
            p for p in league_players
            if float(p.get("time", 0)) >= min_minutes
        ]
        print(f"[understat] {len(qualifying)} players with >= {min_minutes} mins")

        all_rows: list[dict] = []
        for i, p in enumerate(qualifying):
            pid = int(p["id"])
            pname = p["player_name"]
            pteam = p.get("team_title", "")

            fpl_id = fuzzy_match_player(pname, self.fpl_players, team_hint=pteam)

            matches = self.fetch_player_matches(pid)
            for m in matches:
                all_rows.append({
                    "fpl_id": fpl_id,
                    "understat_id": pid,
                    "player_name": pname,
                    "team": pteam,
                    "date": m.get("date", ""),
                    "round": m.get("round", m.get("week", "")),
                    "h_team": m.get("h_team", ""),
                    "a_team": m.get("a_team", ""),
                    "xG": float(m.get("xG", 0)),
                    "xA": float(m.get("xA", 0)),
                    "npxG": float(m.get("npxG", 0)),
                    "xGChain": float(m.get("xGChain", 0)),
                    "xGBuildup": float(m.get("xGBuildup", 0)),
                    "key_passes": int(m.get("key_passes", 0)),
                    "shots": int(m.get("shots", 0)),
                    "minutes": int(m.get("time", 0)),
                })

            # Rate-limit: small delay between players
            if (i + 1) % 10 == 0:
                print(f"[understat] Fetched {i + 1}/{len(qualifying)} players...")
                time.sleep(1)

        _write_cache(cache, all_rows)
        return pd.DataFrame(all_rows)

    # ------------------------------------------------------------------
    # Team-level stats
    # ------------------------------------------------------------------

    def fetch_team_stats(self, understat_team_name: str) -> list[dict]:
        """Fetch per-match team stats from Understat.

        Args:
            understat_team_name: Team name as used by Understat (e.g. "Manchester City").

        Returns:
            List of match-level stat dicts with xG, xGA, ppda, deep etc.
        """
        cache = _cache_path(
            f"understat_team_{understat_team_name.replace(' ', '_')}_{self.season}"
        )
        if _cache_is_fresh(cache):
            return _read_cache(cache)

        try:
            with UnderstatClient() as client:
                # Get team ID first
                league_teams = client.league(league=self.league).get_team_data(
                    season=self.season
                )

            # league_teams is a dict keyed by team name or list
            if isinstance(league_teams, dict):
                team_data = league_teams.get(understat_team_name, [])
            else:
                team_data = [
                    t for t in league_teams
                    if t.get("title", "") == understat_team_name
                ]

            if isinstance(team_data, list):
                _write_cache(cache, team_data)
                return team_data
            else:
                result = list(team_data) if team_data else []
                _write_cache(cache, result)
                return result

        except Exception as e:
            print(f"[understat] Error fetching team {understat_team_name}: {e}")
            return []

    def get_team_match_stats(self) -> pd.DataFrame:
        """Fetch per-match team stats for all EPL teams, mapped to FPL team IDs.

        Returns:
            DataFrame with: fpl_team_id, team_name, date, h_a,
            xG, xGA, scored, conceded, deep, ppda_att, ppda_def, result
        """
        cache = _cache_path(f"understat_team_matches_{self.league}_{self.season}")
        if _cache_is_fresh(cache):
            return pd.DataFrame(_read_cache(cache))

        # Get all teams from league overview
        with UnderstatClient() as client:
            league_teams = client.league(league=self.league).get_team_data(
                season=self.season
            )

        # Determine team names
        if isinstance(league_teams, dict):
            team_names = list(league_teams.keys())
        else:
            team_names = list({t.get("title", "") for t in league_teams})

        all_rows: list[dict] = []
        for tname in team_names:
            fpl_team_id = map_understat_team_to_fpl(tname)
            matches = self.fetch_team_stats(tname)

            for m in matches:
                # Understat team match data structure varies; handle both formats
                if isinstance(m, dict):
                    h_a = m.get("h_a", m.get("side", ""))
                    xG = float(m.get("xG", 0))
                    xGA = float(m.get("xGA", 0))
                    scored = int(m.get("scored", m.get("goals", 0)))
                    conceded = int(m.get("missed", m.get("against", 0)))
                    deep = int(m.get("deep", 0))

                    ppda = m.get("ppda", {})
                    if isinstance(ppda, dict):
                        ppda_att = float(ppda.get("att", 0))
                        ppda_def = float(ppda.get("def", 0))
                    else:
                        ppda_att = 0.0
                        ppda_def = 0.0

                    all_rows.append({
                        "fpl_team_id": fpl_team_id,
                        "team_name": tname,
                        "date": m.get("date", m.get("datetime", "")),
                        "h_a": h_a,
                        "xG": xG,
                        "xGA": xGA,
                        "scored": scored,
                        "conceded": conceded,
                        "deep": deep,
                        "ppda_att": ppda_att,
                        "ppda_def": ppda_def,
                        "result": m.get("result", ""),
                    })

            time.sleep(0.5)  # Rate limit between teams

        _write_cache(cache, all_rows)
        return pd.DataFrame(all_rows)

    # ------------------------------------------------------------------
    # ID mapping cache
    # ------------------------------------------------------------------

    def get_player_id_map(self) -> dict[int, Optional[int]]:
        """Return mapping of Understat player ID -> FPL player ID.

        Uses cached league player data + fuzzy matching.
        """
        cache = _cache_path(f"id_map_player_{self.league}_{self.season}")
        if _cache_is_fresh(cache):
            return {int(k): v for k, v in _read_cache(cache).items()}

        league_players = self.fetch_league_players()
        mapping: dict[int, Optional[int]] = {}
        for p in league_players:
            pid = int(p["id"])
            pname = p["player_name"]
            pteam = p.get("team_title", "")
            fpl_id = fuzzy_match_player(pname, self.fpl_players, team_hint=pteam)
            mapping[pid] = fpl_id

        _write_cache(cache, mapping)
        return mapping


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ud = UnderstatData()

    print("=== Team match stats ===")
    team_df = ud.get_team_match_stats()
    print(f"Rows: {len(team_df)}")
    if not team_df.empty:
        print(team_df.head(10).to_string(index=False))

    print("\n=== Player match stats (top 20 by xG) ===")
    player_df = ud.get_player_match_stats(min_minutes=500)
    print(f"Rows: {len(player_df)}")
    if not player_df.empty:
        top = player_df.groupby("player_name")["xG"].sum().sort_values(ascending=False).head(20)
        print(top.to_string())
