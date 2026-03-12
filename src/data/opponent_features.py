"""Opponent feature engineering module for xPoints.

Builds the Xo (opponent) feature layer from the OpenFPL paper.
For each upcoming fixture, looks up the opponent team's rolling stats:
xGA, goals_conceded, PPDA, Deep, CS_rate over multiple time horizons.

Rolling windows: [1, 3, 5, 10, 38] matches (per OpenFPL specification).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .understat_api import UnderstatData, _cache_path, _cache_is_fresh, _write_cache, _read_cache

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLING_WINDOWS = [1, 3, 5, 10, 38]

# Stats to compute rolling features for
OPPONENT_STATS = [
    "xGA",          # expected goals against (how leaky is the defence?)
    "conceded",     # actual goals conceded
    "ppda_ratio",   # PPDA ratio (att/def) — pressing intensity
    "deep",         # deep completions allowed
    "cs_flag",      # clean sheet flag (0/1)
    "xG",           # opponent's own xG (attacking threat)
]


class OpponentFeatures:
    """Builds opponent feature vectors from Understat team-level data.

    For each fixture, computes rolling averages of the opponent's defensive
    and attacking stats, creating the Xo feature layer.

    Usage:
        of = OpponentFeatures()
        features_df = of.build_opponent_features()
    """

    def __init__(self, season: str = "2025", league: str = "EPL"):
        self.season = season
        self.league = league
        self._understat = UnderstatData(season=season, league=league)
        self._team_df: Optional[pd.DataFrame] = None

    @property
    def team_df(self) -> pd.DataFrame:
        """Lazy-load team match stats from Understat."""
        if self._team_df is None:
            self._team_df = self._understat.get_team_match_stats()
            self._prepare_team_df()
        return self._team_df

    def _prepare_team_df(self) -> None:
        """Add derived columns and sort by date."""
        df = self._team_df
        if df.empty:
            return

        # Parse date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.sort_values(["team_name", "date"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # PPDA ratio: att / def (higher = more pressing)
        df["ppda_ratio"] = np.where(
            df["ppda_def"] > 0,
            df["ppda_att"] / df["ppda_def"],
            0.0,
        )

        # Clean sheet flag
        df["cs_flag"] = (df["conceded"] == 0).astype(int)

        self._team_df = df

    def compute_rolling_stats(self, team_name: str) -> pd.DataFrame:
        """Compute rolling stats for a single team across all windows.

        Args:
            team_name: Understat team name.

        Returns:
            DataFrame indexed by match date with rolling feature columns.
        """
        df = self.team_df
        team_matches = df[df["team_name"] == team_name].copy()

        if team_matches.empty:
            return pd.DataFrame()

        team_matches.sort_values("date", inplace=True)
        team_matches.reset_index(drop=True, inplace=True)

        result = team_matches[["date", "team_name", "fpl_team_id", "h_a"]].copy()

        for stat in OPPONENT_STATS:
            if stat not in team_matches.columns:
                continue
            for w in ROLLING_WINDOWS:
                col_name = f"opp_{stat}_roll_{w}"
                # Use shift(1) so we only use data available BEFORE the match
                result[col_name] = (
                    team_matches[stat]
                    .shift(1)
                    .rolling(window=w, min_periods=1)
                    .mean()
                )

        return result

    def build_all_team_rolling(self) -> pd.DataFrame:
        """Compute rolling stats for ALL teams and concatenate.

        Returns:
            DataFrame with rolling opponent stats per team per match date.
        """
        cache = _cache_path(f"opponent_rolling_{self.league}_{self.season}")
        if _cache_is_fresh(cache):
            return pd.DataFrame(_read_cache(cache))

        df = self.team_df
        if df.empty:
            print("[opponent] No team data available.")
            return pd.DataFrame()

        team_names = df["team_name"].unique()
        frames: list[pd.DataFrame] = []

        for tname in team_names:
            team_rolling = self.compute_rolling_stats(tname)
            if not team_rolling.empty:
                frames.append(team_rolling)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        _write_cache(cache, result.to_dict(orient="records"))
        return result

    def build_opponent_features(
        self,
        fixtures: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build opponent feature vectors for a set of fixtures.

        If fixtures is provided, it should have columns:
            - opponent_team (Understat team name or FPL team ID)
            - date (match date)

        If fixtures is None, returns the full rolling stats table
        which can be joined to fixture data downstream.

        Args:
            fixtures: Optional DataFrame with fixture info.

        Returns:
            DataFrame with opponent rolling features.
        """
        all_rolling = self.build_all_team_rolling()

        if fixtures is None or all_rolling.empty:
            return all_rolling

        # If fixtures provided, merge opponent stats
        # Assume fixtures has 'opponent_team' and 'date' matching team_name/date
        # Must join on BOTH team name AND date to avoid many-to-many row explosion
        if "date" not in fixtures.columns or "date" not in all_rolling.columns:
            # Fallback if date column missing — but warn about potential duplication
            print("[opponent] WARNING: 'date' column missing, joining on team name only (may duplicate rows)")
            merged = fixtures.merge(
                all_rolling,
                left_on=["opponent_team"],
                right_on=["team_name"],
                how="left",
                suffixes=("", "_opp"),
            )
        else:
            # Ensure both date columns are datetime for proper matching
            fixtures = fixtures.copy()
            fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
            all_rolling["date"] = pd.to_datetime(all_rolling["date"], errors="coerce")
            merged = fixtures.merge(
                all_rolling,
                left_on=["opponent_team", "date"],
                right_on=["team_name", "date"],
                how="left",
                suffixes=("", "_opp"),
            )
        return merged

    def get_opponent_vector(
        self, team_name: str, as_of_date: Optional[str] = None
    ) -> dict[str, float]:
        """Get the latest opponent feature vector for a specific team.

        Args:
            team_name: Understat team name.
            as_of_date: Optional date string (YYYY-MM-DD). Uses latest if None.

        Returns:
            Dict of feature name -> value.
        """
        all_rolling = self.build_all_team_rolling()
        team_data = all_rolling[all_rolling["team_name"] == team_name].copy()

        if team_data.empty:
            return {}

        if as_of_date:
            team_data = team_data[team_data["date"] <= as_of_date]

        if team_data.empty:
            return {}

        latest = team_data.iloc[-1]
        feature_cols = [c for c in latest.index if c.startswith("opp_")]
        return {col: float(latest[col]) if pd.notna(latest[col]) else 0.0 for col in feature_cols}

    def get_feature_columns(self) -> list[str]:
        """Return the list of opponent feature column names.

        Useful for building the feature matrix in the training pipeline.
        """
        cols: list[str] = []
        for stat in OPPONENT_STATS:
            for w in ROLLING_WINDOWS:
                cols.append(f"opp_{stat}_roll_{w}")
        return cols


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    of = OpponentFeatures()

    print("=== Building opponent rolling features ===")
    rolling_df = of.build_all_team_rolling()
    print(f"Shape: {rolling_df.shape}")

    if not rolling_df.empty:
        print(f"\nFeature columns ({len(of.get_feature_columns())}):")
        for col in of.get_feature_columns():
            print(f"  {col}")

        # Show latest vector for a sample team
        sample_team = rolling_df["team_name"].iloc[0]
        print(f"\nLatest opponent vector for {sample_team}:")
        vec = of.get_opponent_vector(sample_team)
        for k, v in vec.items():
            print(f"  {k}: {v:.3f}")
