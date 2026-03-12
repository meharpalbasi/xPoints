"""FPL API data fetching — bootstrap, player histories, fixtures."""

import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://fantasy.premierleague.com/api"


def fetch_bootstrap() -> Dict[str, Any]:
    """Fetch the bootstrap-static endpoint (players, teams, events)."""
    resp = requests.get(f"{BASE_URL}/bootstrap-static/", timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_dataframes(
    bootstrap: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (players_df, teams_df, events_df) from bootstrap data."""
    players_df = pd.DataFrame(bootstrap["elements"])
    teams_df = pd.DataFrame(bootstrap["teams"])
    events_df = pd.DataFrame(bootstrap["events"])
    return players_df, teams_df, events_df


def detect_next_gameweek(events_df: pd.DataFrame) -> int:
    """Detect the next unfinished gameweek whose deadline is in the future."""
    events_df = events_df.copy()
    events_df["deadline_time"] = pd.to_datetime(events_df["deadline_time"], utc=True)
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    future = events_df[
        (events_df["finished"] == False) & (events_df["deadline_time"] > now_utc)
    ]
    if not future.empty:
        return int(future["id"].min())
    return int(events_df[events_df["finished"] == False]["id"].min())


def fetch_fixtures() -> pd.DataFrame:
    """Fetch all fixtures for the season."""
    resp = requests.get(f"{BASE_URL}/fixtures/", timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df = df.rename(columns={"id": "fixture_id"})
    return df


def fetch_player_history(player_id: int) -> Optional[pd.DataFrame]:
    """Fetch per-GW history for a single player."""
    resp = requests.get(f"{BASE_URL}/element-summary/{player_id}/", timeout=30)
    if resp.status_code != 200:
        return None
    history = resp.json().get("history", [])
    if not history:
        return None
    df = pd.DataFrame(history)
    df["player_id"] = player_id
    return df


def fetch_player_fixtures(player_id: int) -> Optional[pd.DataFrame]:
    """Fetch upcoming fixtures for a single player."""
    resp = requests.get(f"{BASE_URL}/element-summary/{player_id}/", timeout=30)
    if resp.status_code != 200:
        return None
    fixtures = resp.json().get("fixtures", [])
    if not fixtures:
        return None
    df = pd.DataFrame(fixtures)
    df["player_id"] = player_id
    return df


def fetch_all_player_data(
    player_ids: List[int],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Fetch history and upcoming fixtures for all players.

    Returns
    -------
    history_df : pd.DataFrame
        Concatenated per-GW history for every player.
    upcoming_df : pd.DataFrame | None
        Concatenated upcoming fixtures, or None if empty.
    """
    histories: List[pd.DataFrame] = []
    upcoming: List[pd.DataFrame] = []

    for pid in tqdm(player_ids, desc="Fetching player data"):
        url = f"{BASE_URL}/element-summary/{pid}/"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            continue
        j = resp.json()

        hist = j.get("history", [])
        if hist:
            hdf = pd.DataFrame(hist)
            hdf["player_id"] = pid
            histories.append(hdf)

        fxts = j.get("fixtures", [])
        if fxts:
            fdf = pd.DataFrame(fxts)
            fdf["player_id"] = pid
            upcoming.append(fdf)

    if not histories:
        raise ValueError("No historical data found. Check the API or season status.")

    history_df = pd.concat(histories, ignore_index=True)
    upcoming_df = pd.concat(upcoming, ignore_index=True) if upcoming else None
    return history_df, upcoming_df


def enrich_players_df(players_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns (status_numeric, chance_of_playing, form) to players_df."""
    df = players_df.copy()
    df["chance_of_playing_next_round"] = (
        pd.to_numeric(df["chance_of_playing_next_round"], errors="coerce").fillna(100.0)
    )
    status_map = {"a": 4.0, "d": 2.0, "i": 0.0, "s": 0.0, "u": 1.0}
    df["status_numeric"] = df["status"].map(status_map).fillna(1.0)
    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0.0)
    return df


def merge_player_metadata(
    history_df: pd.DataFrame, players_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge player-level metadata into the history dataframe."""
    merge_cols = [
        "id",
        "element_type",
        "team",
        "web_name",
        "selected_by_percent",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "status_numeric",
        "chance_of_playing_next_round",
        "form",
    ]
    available = [c for c in merge_cols if c in players_df.columns]
    df = history_df.merge(players_df[available], left_on="player_id", right_on="id", how="left")
    df.sort_values(["player_id", "round"], inplace=True)
    return df
