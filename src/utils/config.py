"""Configuration: feature sets per position, hyperparameter grids, constants."""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Position codes (FPL element_type)
# ---------------------------------------------------------------------------
POSITION_MAP: Dict[int, str] = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_IDS: Dict[str, int] = {v: k for k, v in POSITION_MAP.items()}
POSITIONS: List[str] = ["GK", "DEF", "MID", "FWD"]

# ---------------------------------------------------------------------------
# Rolling windows (per OpenFPL Table 2)
# ---------------------------------------------------------------------------
ROLLING_WINDOWS: List[int] = [1, 3, 5, 10, 38]

# ---------------------------------------------------------------------------
# Raw stat columns used per position (before rolling)
# ---------------------------------------------------------------------------
GK_RAW_STATS: List[str] = [
    "saves",
    "penalties_saved",
    "goals_conceded",
    "clean_sheets",
    "assists",
    "influence",
    "creativity",
    "minutes",
    "bps",
    "bonus",
    "yellow_cards",
    "red_cards",
]

# Base outfield stats shared across DEF/MID/FWD
_OUTFIELD_BASE: List[str] = [
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "influence",
    "creativity",
    "threat",
    "bps",
    "bonus",
    "minutes",
    "yellow_cards",
    "red_cards",
]

# Position-specific feature sets — defenders emphasise defensive stats,
# forwards emphasise attacking stats, midfielders get both.
DEF_RAW_STATS: List[str] = _OUTFIELD_BASE + [
    "clean_sheets",
    "goals_conceded",
    "expected_goals_conceded",
]

MID_RAW_STATS: List[str] = _OUTFIELD_BASE + [
    "clean_sheets",
    "goals_conceded",
]

FWD_RAW_STATS: List[str] = _OUTFIELD_BASE  # no defensive stats for forwards

POSITION_RAW_STATS: Dict[str, List[str]] = {
    "GK": GK_RAW_STATS,
    "DEF": DEF_RAW_STATS,
    "MID": MID_RAW_STATS,
    "FWD": FWD_RAW_STATS,
}

# ---------------------------------------------------------------------------
# Context features (non-rolling, added to every position)
# ---------------------------------------------------------------------------
CONTEXT_FEATURES: List[str] = [
    "fixture_difficulty",
    "home_dummy",
    "selected_by_percent",
    "status_numeric",
    "chance_of_playing_next_round",
    "form",
]

# ---------------------------------------------------------------------------
# Sample-weight bins per position (per OpenFPL)
# ---------------------------------------------------------------------------
SAMPLE_WEIGHT_BINS: Dict[str, int] = {
    "GK": 2,
    "DEF": 3,
    "MID": 4,
    "FWD": 3,
}

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
XGB_PARAM_GRID: Dict[str, List] = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

RF_PARAM_GRID: Dict[str, List] = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 3],
}

# ---------------------------------------------------------------------------
# Return-category thresholds (per OpenFPL)
# ---------------------------------------------------------------------------
RETURN_CATEGORIES: Dict[str, tuple] = {
    "Zeros": (None, 0),       # points <= 0
    "Blanks": (1, 2),         # 1 <= points <= 2
    "Tickers": (3, 5),        # 3 <= points <= 5
    "Haulers": (6, None),     # points >= 6
}

# ---------------------------------------------------------------------------
# Walk-forward CV defaults
# ---------------------------------------------------------------------------
MIN_TRAIN_GAMEWEEKS: int = 6
TEST_GAMEWEEKS: int = 1
