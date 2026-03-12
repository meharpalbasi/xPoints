"""Training pipeline: separate XGBoost + RandomForest per position."""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRegressor

from src.data.features import get_feature_columns
from src.utils.config import (
    POSITIONS,
    POSITION_IDS,
    RF_PARAM_GRID,
    SAMPLE_WEIGHT_BINS,
    XGB_PARAM_GRID,
)
from src.utils.validation import WalkForwardCV

# Type alias for trained model artifacts
ModelArtifacts = Dict[str, Dict[str, Any]]
# Structure: { "GK": {"xgb": model, "rf": model, "features": [...] }, ... }


def _compute_sample_weights(
    y: pd.Series, position: str
) -> np.ndarray:
    """Compute sample weights using KBinsDiscretizer + compute_sample_weight.

    Bins the target into position-specific number of bins (entropy strategy),
    then computes 'balanced' sample weights so under-represented return
    categories get higher weight.

    Parameters
    ----------
    y : pd.Series
        Target values.
    position : str
        Position code (GK/DEF/MID/FWD).

    Returns
    -------
    np.ndarray
        Per-sample weights.
    """
    n_bins = SAMPLE_WEIGHT_BINS[position]
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    bins = kbd.fit_transform(y.values.reshape(-1, 1)).ravel().astype(int)
    weights = compute_sample_weight("balanced", bins)
    return weights


def train_position_models(
    df: pd.DataFrame,
    position: str,
    target_col: str = "target_points",
    cv_folds: int = 3,
) -> Dict[str, Any]:
    """Train XGBoost + RandomForest for a single position.

    Uses walk-forward CV for hyperparameter search, and applies
    sample weighting via KBinsDiscretizer.

    Parameters
    ----------
    df : pd.DataFrame
        Training data (already filtered to this position).
    position : str
        Position code.
    target_col : str
        Name of the target column.
    cv_folds : int
        Max number of walk-forward folds to use in GridSearchCV.

    Returns
    -------
    dict
        Keys: 'xgb', 'rf', 'features'.
    """
    feature_cols = get_feature_columns(position)
    available = [c for c in feature_cols if c in df.columns]

    model_df = df.dropna(subset=available + [target_col]).copy()
    if model_df.empty:
        raise ValueError(f"No training data for position {position}")

    X = model_df[available]
    y = model_df[target_col]
    sample_weights = _compute_sample_weights(y, position)

    # Walk-forward CV object
    wfcv = WalkForwardCV()
    n_splits = wfcv.get_n_splits(model_df)
    if n_splits < 1:
        raise ValueError(f"Not enough gameweeks to validate for {position}")

    # Limit folds for speed
    folds = list(wfcv.split(model_df))[-cv_folds:]

    # --- XGBoost ---
    xgb = XGBRegressor(random_state=42, objective="reg:squarederror", verbosity=0)
    xgb_search = GridSearchCV(
        estimator=xgb,
        param_grid=XGB_PARAM_GRID,
        cv=folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    xgb_search.fit(X, y, sample_weight=sample_weights)
    best_xgb = xgb_search.best_estimator_
    print(f"  [{position}] XGB best params: {xgb_search.best_params_}")

    # --- Random Forest ---
    rf = RandomForestRegressor(random_state=42)
    rf_search = GridSearchCV(
        estimator=rf,
        param_grid=RF_PARAM_GRID,
        cv=folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    rf_search.fit(X, y, sample_weight=sample_weights)
    best_rf = rf_search.best_estimator_
    print(f"  [{position}] RF  best params: {rf_search.best_params_}")

    return {
        "xgb": best_xgb,
        "rf": best_rf,
        "features": available,
    }


def train_all_positions(
    df: pd.DataFrame,
    target_col: str = "target_points",
) -> ModelArtifacts:
    """Train models for all four positions.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature-engineered training dataframe (all positions).
    target_col : str
        Target column name.

    Returns
    -------
    ModelArtifacts
        Mapping of position -> trained model dict.
    """
    artifacts: ModelArtifacts = {}

    for pos in POSITIONS:
        pos_id = POSITION_IDS[pos]
        pos_df = df[df["element_type"] == pos_id].copy()
        if pos_df.empty:
            print(f"  [{pos}] No data — skipping")
            continue
        print(f"Training {pos} models ({len(pos_df)} rows)...")
        artifacts[pos] = train_position_models(pos_df, pos, target_col)

    return artifacts
