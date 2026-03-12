"""Ensemble logic: median of XGB + RF predictions per position."""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.train import ModelArtifacts


def ensemble_predict(
    X: pd.DataFrame,
    models: Dict[str, object],
    feature_cols: List[str],
) -> np.ndarray:
    """Predict using the ensemble (median of XGB + RF).

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
        Median predictions.
    """
    available = [c for c in feature_cols if c in X.columns]
    X_sub = X[available]

    preds_xgb = models["xgb"].predict(X_sub)
    preds_rf = models["rf"].predict(X_sub)

    # Median of the two predictions per sample
    stacked = np.vstack([preds_xgb, preds_rf])
    return np.median(stacked, axis=0)


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
