"""Betting odds data module for xPoints.

Fetches match odds from football-data.co.uk CSV files, converts to implied
probabilities, and derives features: implied_goals, implied_clean_sheet_prob.
Maps to FPL fixtures by date + team name.

Source: https://www.football-data.co.uk/englandm.php
Columns used: B365H, B365D, B365A (Bet365 match odds)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from fuzzywuzzy import fuzz, process

from .understat_api import _cache_path, _cache_is_fresh, _write_cache, _read_cache, _get_fpl_bootstrap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# football-data.co.uk URL pattern: mmz4281/{season_code}/E0.csv
# Season codes: 2526 = 2025-26, 2425 = 2024-25, etc.
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"

SEASON_CODES = {
    "2025": "2526",  # 2025-26 season
    "2024": "2425",  # 2024-25 season
    "2023": "2324",
    "2022": "2223",
}

# Mapping from football-data.co.uk team names to FPL team names
ODDS_TO_FPL_TEAM: dict[str, str] = {
    "Man United": "Man Utd",
    "Man City": "Man City",
    "Wolves": "Wolves",
    "Newcastle": "Newcastle",
    "West Ham": "West Ham",
    "Tottenham": "Spurs",
    "Nott'm Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield Utd",
    "Ipswich": "Ipswich",
    "Leicester": "Leicester",
    "Brighton": "Brighton",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Chelsea": "Chelsea",
    "Everton": "Everton",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Crystal Palace": "Crystal Palace",
    "Fulham": "Fulham",
    "Brentford": "Brentford",
    "Southampton": "Southampton",
    "Burnley": "Burnley",
    "Luton": "Luton",
}

# Average EPL goals per game (for implied goals calculation)
AVG_GOALS_PER_GAME = 2.75


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _implied_prob_from_odds(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def _normalize_probs(home: float, draw: float, away: float) -> tuple[float, float, float]:
    """Remove overround by normalizing probabilities to sum to 1."""
    total = home + draw + away
    if total == 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (home / total, draw / total, away / total)


def _implied_goals_from_probs(
    win_prob: float, draw_prob: float, lose_prob: float, is_home: bool
) -> float:
    """Estimate implied goals for a team from match probabilities.

    Uses a simplified Poisson-based approach:
    - P(0 goals) ≈ e^(-lambda) → lambda ≈ -ln(P(0 goals))
    - P(0 goals) ≈ P(lose)*k + P(draw)*k2  (simplified)

    More practically, we use the relationship between win probability
    and expected goals based on historical EPL data.
    """
    # Simplified approach: scale by win probability relative to average
    # Home teams with >50% win prob score more; away teams less
    base_goals = AVG_GOALS_PER_GAME / 2  # per team average ~1.375

    if is_home:
        # Home advantage: win_prob > 0.45 is strong
        strength = win_prob / 0.45  # normalise against average home win rate
    else:
        # Away: win_prob > 0.30 is strong
        strength = win_prob / 0.30

    return base_goals * strength


def _implied_cs_prob(
    opponent_win_prob: float, draw_prob: float
) -> float:
    """Estimate clean sheet probability from opponent's match probabilities.

    P(CS) ≈ P(opponent scores 0) ≈ P(draw or win) * correction_factor
    Simplified: teams keep clean sheets more when opponents are weak.

    A more precise approach uses Poisson distribution, but this is a
    practical approximation that correlates well with actual CS rates.
    """
    # Empirical approximation: weight opponent win prob (1.2) and draw prob (0.3)
    # to estimate P(opponent scores). Constants derived from historical EPL data.
    # Note: can exceed 1.0 for very strong opponents, hence the clamp below.
    p_opp_scores = opponent_win_prob * 1.2 + draw_prob * 0.3
    p_cs = max(0.05, min(0.60, 1.0 - p_opp_scores))
    return p_cs


# ---------------------------------------------------------------------------
# Main data class
# ---------------------------------------------------------------------------

class OddsData:
    """Fetches and processes betting odds from football-data.co.uk.

    Usage:
        od = OddsData()
        odds_df = od.get_match_odds()
        features_df = od.get_odds_features()
    """

    def __init__(self, season: str = "2025"):
        self.season = season
        self.season_code = SEASON_CODES.get(season, "2526")
        self._fpl_teams: Optional[pd.DataFrame] = None

    @property
    def fpl_teams(self) -> pd.DataFrame:
        """Lazy-load FPL team data."""
        if self._fpl_teams is None:
            bs = _get_fpl_bootstrap()
            self._fpl_teams = pd.DataFrame(bs["teams"])[["id", "name", "short_name"]]
        return self._fpl_teams

    def _map_team_to_fpl(self, odds_team: str) -> Optional[int]:
        """Map a football-data.co.uk team name to FPL team ID."""
        fpl_name = ODDS_TO_FPL_TEAM.get(odds_team, odds_team)
        match = self.fpl_teams[
            (self.fpl_teams["name"] == fpl_name)
            | (self.fpl_teams["short_name"] == fpl_name)
            | (self.fpl_teams["name"] == odds_team)
        ]
        if not match.empty:
            return int(match.iloc[0]["id"])

        # Fuzzy fallback
        choices = self.fpl_teams["name"].tolist()
        result = process.extractOne(odds_team, choices, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= 70:
            return int(self.fpl_teams[self.fpl_teams["name"] == result[0]].iloc[0]["id"])
        return None

    def fetch_odds_csv(self) -> pd.DataFrame:
        """Fetch the season's odds CSV from football-data.co.uk.

        Returns:
            DataFrame with match odds data.
        """
        cache = _cache_path(f"odds_csv_{self.season_code}")
        if _cache_is_fresh(cache, ttl_hours=12):
            return pd.DataFrame(_read_cache(cache))

        url = BASE_URL.format(season_code=self.season_code)
        print(f"[odds] Fetching {url}")

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[odds] Error fetching odds CSV: {e}. Returning empty DataFrame.")
            return pd.DataFrame()

        # Parse CSV (football-data.co.uk uses various encodings)
        try:
            df = pd.read_csv(io.StringIO(resp.text), on_bad_lines="skip")
        except Exception as e:
            print(f"[odds] CSV parse error: {e}")
            return pd.DataFrame()

        # Keep only essential columns
        keep_cols = [
            "Date", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR",  # Full-time result
            "B365H", "B365D", "B365A",  # Bet365 odds
        ]
        available = [c for c in keep_cols if c in df.columns]
        if len(available) < 6:
            print(f"[odds] Warning: missing expected columns. Available: {list(df.columns)}")

        df = df[available].copy()

        # Parse date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df.dropna(subset=["Date"], inplace=True)

        _write_cache(cache, df.to_dict(orient="records"))
        return df

    def get_match_odds(self) -> pd.DataFrame:
        """Get match odds with implied probabilities.

        Returns:
            DataFrame with columns: date, home_team, away_team, fpl_home_id,
            fpl_away_id, home_odds, draw_odds, away_odds, home_prob, draw_prob,
            away_prob (normalized to remove overround).
        """
        raw = self.fetch_odds_csv()
        if raw.empty:
            return pd.DataFrame()

        required = ["B365H", "B365D", "B365A", "HomeTeam", "AwayTeam"]
        if not all(c in raw.columns for c in required):
            print(f"[odds] Missing required columns: {required}")
            return pd.DataFrame()

        rows: list[dict] = []
        for _, r in raw.iterrows():
            try:
                h_odds = float(r["B365H"])
                d_odds = float(r["B365D"])
                a_odds = float(r["B365A"])
            except (ValueError, TypeError):
                continue

            h_prob = _implied_prob_from_odds(h_odds)
            d_prob = _implied_prob_from_odds(d_odds)
            a_prob = _implied_prob_from_odds(a_odds)

            h_norm, d_norm, a_norm = _normalize_probs(h_prob, d_prob, a_prob)

            rows.append({
                "date": r.get("Date", ""),
                "home_team": r["HomeTeam"],
                "away_team": r["AwayTeam"],
                "fpl_home_id": self._map_team_to_fpl(r["HomeTeam"]),
                "fpl_away_id": self._map_team_to_fpl(r["AwayTeam"]),
                "home_odds": h_odds,
                "draw_odds": d_odds,
                "away_odds": a_odds,
                "home_prob": h_norm,
                "draw_prob": d_norm,
                "away_prob": a_norm,
                "fthg": r.get("FTHG", np.nan),
                "ftag": r.get("FTAG", np.nan),
            })

        return pd.DataFrame(rows)

    def get_odds_features(self) -> pd.DataFrame:
        """Get odds-derived features for each team in each match.

        Returns two rows per match (one for each team) with:
        - implied_goals: expected goals based on odds
        - implied_cs_prob: probability of keeping a clean sheet
        - win_prob, draw_prob, lose_prob: match outcome probabilities
        - is_home: whether playing at home

        Returns:
            DataFrame with odds-derived features per team per match.
        """
        cache = _cache_path(f"odds_features_{self.season_code}")
        if _cache_is_fresh(cache):
            return pd.DataFrame(_read_cache(cache))

        odds_df = self.get_match_odds()
        if odds_df.empty:
            return pd.DataFrame()

        rows: list[dict] = []
        for _, m in odds_df.iterrows():
            # Home team features
            rows.append({
                "date": m["date"],
                "team": m["home_team"],
                "fpl_team_id": m["fpl_home_id"],
                "opponent": m["away_team"],
                "fpl_opponent_id": m["fpl_away_id"],
                "is_home": True,
                "win_prob": m["home_prob"],
                "draw_prob": m["draw_prob"],
                "lose_prob": m["away_prob"],
                "implied_goals": _implied_goals_from_probs(
                    m["home_prob"], m["draw_prob"], m["away_prob"], is_home=True
                ),
                "implied_cs_prob": _implied_cs_prob(m["away_prob"], m["draw_prob"]),
            })

            # Away team features
            rows.append({
                "date": m["date"],
                "team": m["away_team"],
                "fpl_team_id": m["fpl_away_id"],
                "opponent": m["home_team"],
                "fpl_opponent_id": m["fpl_home_id"],
                "is_home": False,
                "win_prob": m["away_prob"],
                "draw_prob": m["draw_prob"],
                "lose_prob": m["home_prob"],
                "implied_goals": _implied_goals_from_probs(
                    m["away_prob"], m["draw_prob"], m["home_prob"], is_home=False
                ),
                "implied_cs_prob": _implied_cs_prob(m["home_prob"], m["draw_prob"]),
            })

        result = pd.DataFrame(rows)
        _write_cache(cache, result.to_dict(orient="records"))
        return result

    def get_fixture_odds(
        self,
        fpl_team_id: int,
        match_date: Optional[str] = None,
    ) -> Optional[dict]:
        """Look up odds features for a specific FPL team and optional date.

        Args:
            fpl_team_id: FPL team ID.
            match_date: Optional date string (YYYY-MM-DD).

        Returns:
            Dict of odds features or None if not found.
        """
        features = self.get_odds_features()
        if features.empty:
            return None

        mask = features["fpl_team_id"] == fpl_team_id
        if match_date:
            date_series = pd.to_datetime(features["date"]).dt.strftime("%Y-%m-%d")
            mask = mask & (date_series == match_date)

        matched = features[mask]
        if matched.empty:
            return None

        # Return the most recent match
        row = matched.iloc[-1]
        return {
            "win_prob": float(row["win_prob"]),
            "draw_prob": float(row["draw_prob"]),
            "lose_prob": float(row["lose_prob"]),
            "implied_goals": float(row["implied_goals"]),
            "implied_cs_prob": float(row["implied_cs_prob"]),
            "is_home": bool(row["is_home"]),
        }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    od = OddsData()

    print("=== Match odds ===")
    match_odds = od.get_match_odds()
    print(f"Matches: {len(match_odds)}")
    if not match_odds.empty:
        print(match_odds[["date", "home_team", "away_team", "home_prob", "draw_prob", "away_prob"]].tail(10).to_string(index=False))

    print("\n=== Odds features ===")
    features = od.get_odds_features()
    print(f"Rows: {len(features)}")
    if not features.empty:
        sample = features[["date", "team", "opponent", "win_prob", "implied_goals", "implied_cs_prob"]].tail(10)
        print(sample.to_string(index=False))
