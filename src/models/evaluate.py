"""Evaluation with per-position metrics and per-return-category breakdown."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.ensemble import ensemble_predict
from src.models.train import ModelArtifacts
from src.utils.config import (
    POSITIONS,
    POSITION_IDS,
    RETURN_CATEGORIES,
)
from src.utils.validation import WalkForwardCV


def _category_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Compute MAE/RMSE per return category.

    Categories per OpenFPL:
    - Zeros: points <= 0
    - Blanks: 1-2
    - Tickers: 3-5
    - Haulers: >= 6
    """
    results: Dict[str, Dict[str, float]] = {}
    for cat, (lo, hi) in RETURN_CATEGORIES.items():
        if lo is None:
            mask = y_true <= hi
        elif hi is None:
            mask = y_true >= lo
        else:
            mask = (y_true >= lo) & (y_true <= hi)

        n = int(mask.sum())
        if n == 0:
            results[cat] = {"n": 0, "mae": None, "rmse": None}
            continue

        mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
        rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
        results[cat] = {"n": n, "mae": round(mae, 4), "rmse": round(rmse, 4)}

    return results


def evaluate_position(
    y_true: np.ndarray, y_pred: np.ndarray, position: str
) -> Dict[str, Any]:
    """Compute full evaluation metrics for one position.

    Parameters
    ----------
    y_true : np.ndarray
        Actual points.
    y_pred : np.ndarray
        Predicted points.
    position : str
        Position code.

    Returns
    -------
    dict
        Evaluation results.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)

    return {
        "position": position,
        "n": len(y_true),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "spearman_corr": round(float(spearman_corr), 4),
        "spearman_p": round(float(spearman_p), 6),
        "categories": _category_metrics(y_true, y_pred),
    }


def walk_forward_evaluate(
    df: pd.DataFrame,
    artifacts: ModelArtifacts,
    target_col: str = "target_points",
    min_train_gws: int = 6,
) -> Dict[str, Any]:
    """Post-hoc evaluation using pre-trained models on walk-forward folds.

    ⚠️ WARNING: This is NOT true walk-forward / out-of-sample validation.
    The models in `artifacts` were trained on the FULL dataset (all gameweeks).
    Each "test" fold was part of the training data, so these metrics will be
    over-optimistic compared to genuine out-of-sample performance.

    This function is useful for:
    - Checking predictions are sensible across time
    - Per-category (Zeros/Blanks/Tickers/Haulers) error breakdown
    - Sanity-checking model fit

    For true out-of-sample metrics, retrain models per fold (expensive).

    Parameters
    ----------
    df : pd.DataFrame
        Full feature-engineered dataframe.
    artifacts : ModelArtifacts
        Pre-trained models per position (trained on full data).
    target_col : str
        Target column.
    min_train_gws : int
        Minimum training gameweeks for walk-forward splits.

    Returns
    -------
    dict
        Evaluation results with overall and per-position breakdowns.
        Includes an 'in_sample_warning' flag.
    """
    wfcv = WalkForwardCV(min_train_gws=min_train_gws)

    all_y_true: List[float] = []
    all_y_pred: List[float] = []
    per_position: Dict[str, Dict[str, List[float]]] = {
        pos: {"y_true": [], "y_pred": []} for pos in POSITIONS
    }

    for train_idx, test_idx in wfcv.split(df):
        test_fold = df.iloc[test_idx]

        for pos, models in artifacts.items():
            pos_id = POSITION_IDS[pos]
            pos_test = test_fold[test_fold["element_type"] == pos_id]
            if pos_test.empty:
                continue

            feature_cols = models["features"]
            available = [c for c in feature_cols if c in pos_test.columns]
            pos_clean = pos_test.dropna(subset=available + [target_col])
            if pos_clean.empty:
                continue

            y_true = pos_clean[target_col].values
            y_pred = ensemble_predict(pos_clean, models, feature_cols)

            all_y_true.extend(y_true.tolist())
            all_y_pred.extend(y_pred.tolist())
            per_position[pos]["y_true"].extend(y_true.tolist())
            per_position[pos]["y_pred"].extend(y_pred.tolist())

    # Overall metrics
    y_t = np.array(all_y_true)
    y_p = np.array(all_y_pred)

    results: Dict[str, Any] = {
        "in_sample_warning": (
            "These metrics are in-sample: models were trained on all data "
            "including the test folds. Do not treat as out-of-sample estimates."
        ),
        "overall": evaluate_position(y_t, y_p, "ALL") if len(y_t) > 0 else {},
        "per_position": {},
    }

    for pos in POSITIONS:
        yt = np.array(per_position[pos]["y_true"])
        yp = np.array(per_position[pos]["y_pred"])
        if len(yt) > 0:
            results["per_position"][pos] = evaluate_position(yt, yp, pos)

    return results
