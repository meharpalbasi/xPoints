"""Ensemble logic: mean of XGB + RF predictions per position."""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.train import ModelArtifacts

logger = logging.getLogger(__name__)


def ensemble_predict(
    X: pd.DataFrame,
    models: Dict[str, object],
    feature_cols: List[str],
) -> np.ndarray:
    """Predict using the ensemble (mean of XGB + RF).

    With exactly 2 models, median == mean, so we call it what it is.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    models : dict
        Must contain 'xgb' and 'rf' trained estimators.
    feature_cols : list[str]
        Feature columns the models were trained on.

    Returns
    -------
    np.ndarray
        Mean predictions.

    Raises
    ------
    ValueError
        If any expected feature columns are missing from X.
    """
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        raise ValueError(
            f"ensemble_predict: {len(missing)} feature columns missing from input: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    X_sub = X[feature_cols]

    preds_xgb = models["xgb"].predict(X_sub)
    preds_rf = models["rf"].predict(X_sub)

    # Mean of the two predictions per sample
    return (preds_xgb + preds_rf) / 2.0


def ensemble_predict_all(
    df: pd.DataFrame,
    artifacts: ModelArtifacts,
) -> pd.DataFrame:
    """Run ensemble predictions for all positions.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'position' column and all feature columns.
    artifacts : ModelArtifacts
        Trained model artifacts per position.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with 'xPoints_raw' column added.
    """
    df = df.copy()
    df["xPoints_raw"] = np.nan

    for pos, models in artifacts.items():
        mask = df["position"] == pos
        if mask.sum() == 0:
            continue
        preds = ensemble_predict(
            df.loc[mask], models, models["features"]
        )
        df.loc[mask, "xPoints_raw"] = preds

    return df
