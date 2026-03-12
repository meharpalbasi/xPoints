"""Feature engineering pipeline with position-specific feature sets."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import (
    CONTEXT_FEATURES,
    POSITION_MAP,
    POSITION_RAW_STATS,
    ROLLING_WINDOWS,
)


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric, filling NaN with 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _rolling_feature_name(stat: str, window: int) -> str:
    """Canonical name for a rolling feature."""
    return f"{stat}_roll_{window}"


def create_rolling_features(
    df: pd.DataFrame,
    stats: List[str],
    windows: Optional[List[int]] = None,
    group_col: str = "player_id",
) -> pd.DataFrame:
    """Create shifted rolling features for given stats.

    For each stat, we first shift(1) within each player to avoid data leakage,
    then compute rolling sums (or means for minutes) over each window.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by [group_col, 'round'].
    stats : list[str]
        Raw stat columns to roll.
    windows : list[int] | None
        Rolling window sizes (defaults to config ROLLING_WINDOWS).
    group_col : str
        Grouping column (player_id).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with new rolling columns appended.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    df = df.copy()
    df.sort_values([group_col, "round"], inplace=True)

    for stat in stats:
        if stat not in df.columns:
            df[stat] = 0.0
        else:
            df[stat] = _safe_numeric(df[stat])

        lag_col = f"_lag_{stat}"
        df[lag_col] = df.groupby(group_col)[stat].shift(1)

        for w in windows:
            col_name = _rolling_feature_name(stat, w)
            agg = "mean" if stat == "minutes" else "sum"
            df[col_name] = (
                df.groupby(group_col)[lag_col]
                .rolling(w, min_periods=1)
                .agg(agg)
                .reset_index(level=0, drop=True)
            )

        df.drop(columns=[lag_col], inplace=True)

    return df


def get_feature_columns(position: str, windows: Optional[List[int]] = None) -> List[str]:
    """Return the full list of feature column names for a position.

    Parameters
    ----------
    position : str
        One of 'GK', 'DEF', 'MID', 'FWD'.
    windows : list[int] | None
        Rolling windows (defaults to config).

    Returns
    -------
    list[str]
        Ordered feature column names.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    raw_stats = POSITION_RAW_STATS[position]
    rolling_cols = [
        _rolling_feature_name(stat, w) for stat in raw_stats for w in windows
    ]
    return rolling_cols + CONTEXT_FEATURES


def build_feature_matrix(
    history_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full feature-engineered dataframe for all positions.

    Steps:
    1. Compute rolling features for the union of all position stats.
    2. Attach fixture difficulty + home/away indicator.
    3. Set target column.

    Parameters
    ----------
    history_df : pd.DataFrame
        Player history with metadata already merged.
    fixtures_df : pd.DataFrame
        Season fixtures (with fixture_id column).

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe ready for splitting by position.
    """
    # Collect the superset of stats across all positions
    all_stats: set = set()
    for stats_list in POSITION_RAW_STATS.values():
        all_stats.update(stats_list)

    df = create_rolling_features(history_df, sorted(all_stats))

    # Fixture difficulty
    df["fixture_difficulty"] = df.apply(
        lambda row: _get_difficulty(row, fixtures_df), axis=1
    )
    df["home_dummy"] = df["was_home"].astype(int)

    # Ensure context features are numeric
    df["selected_by_percent"] = _safe_numeric(df["selected_by_percent"])
    df["status_numeric"] = _safe_numeric(df["status_numeric"])
    df["chance_of_playing_next_round"] = _safe_numeric(
        df["chance_of_playing_next_round"]
    )
    df["form"] = _safe_numeric(df["form"])

    # Target
    df["target_points"] = df["total_points"]

    # Position label
    df["position"] = df["element_type"].map(POSITION_MAP)

    return df


def _get_difficulty(row: pd.Series, fixtures_df: pd.DataFrame) -> float:
    """Look up fixture difficulty for a history row."""
    fix_id = row.get("fixture")
    was_home = row.get("was_home")
    match_info = fixtures_df.loc[fixtures_df["fixture_id"] == fix_id]
    if match_info.empty:
        return np.nan
    if was_home:
        return float(match_info["team_h_difficulty"].values[0])
    return float(match_info["team_a_difficulty"].values[0])


def build_prediction_features(
    history_df: pd.DataFrame,
    players_df: pd.DataFrame,
    upcoming_df: Optional[pd.DataFrame],
    next_gw: int,
) -> pd.DataFrame:
    """Build feature rows for next-GW predictions.

    Takes the latest rolling features per player from history and attaches
    upcoming fixture info (difficulty, home/away, DGW/BGW fixture count).

    Parameters
    ----------
    history_df : pd.DataFrame
        Feature-engineered history (output of build_feature_matrix).
    players_df : pd.DataFrame
        Enriched player metadata.
    upcoming_df : pd.DataFrame | None
        Upcoming fixtures per player.
    next_gw : int
        The gameweek to predict.

    Returns
    -------
    pd.DataFrame
        One row per player with all feature columns populated.
    """
    # Latest row per player
    latest = (
        history_df.groupby("player_id")
        .apply(lambda g: g.loc[g["round"].idxmax()])
        .reset_index(drop=True)
    )

    # Attach DGW/BGW info
    if upcoming_df is not None:
        next_fix = upcoming_df[upcoming_df["event"] == next_gw]
        fixture_count = next_fix.groupby("player_id").size().reset_index(name="fixture_count")
        avg_diff = next_fix.groupby("player_id")["difficulty"].mean().reset_index(name="avg_difficulty")
        home_prop = next_fix.groupby("player_id")["is_home"].mean().reset_index(name="home_proportion")
        dgw = fixture_count.merge(avg_diff, on="player_id", how="outer").merge(
            home_prop, on="player_id", how="outer"
        )
        latest = latest.merge(dgw, on="player_id", how="left")
        latest["fixture_count"] = latest["fixture_count"].fillna(0)
        latest["fixture_difficulty"] = latest.get("avg_difficulty", pd.Series(3.0)).fillna(3.0)
        latest["home_dummy"] = latest.get("home_proportion", pd.Series(0.5)).fillna(0.5)
    else:
        latest["fixture_count"] = 1
        latest["fixture_difficulty"] = 3.0
        latest["home_dummy"] = 0.5

    latest["position"] = latest["element_type"].map(POSITION_MAP)
    latest["selected_by_percent"] = _safe_numeric(latest["selected_by_percent"])

    return latest
