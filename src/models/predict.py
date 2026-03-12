"""Prediction pipeline: build next-GW predictions using trained models."""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.features import build_prediction_features
from src.models.ensemble import ensemble_predict_all
from src.models.train import ModelArtifacts
from src.utils.config import POSITION_MAP


def predict_next_gameweek(
    history_df: pd.DataFrame,
    players_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    upcoming_df: Optional[pd.DataFrame],
    next_gw: int,
    artifacts: ModelArtifacts,
) -> pd.DataFrame:
    """Generate xPoints predictions for the next gameweek.

    Parameters
    ----------
    history_df : pd.DataFrame
        Feature-engineered history (output of build_feature_matrix).
    players_df : pd.DataFrame
        Enriched player metadata.
    teams_df : pd.DataFrame
        Team data (id, name).
    upcoming_df : pd.DataFrame | None
        Upcoming fixtures per player.
    next_gw : int
        Gameweek number to predict.
    artifacts : ModelArtifacts
        Trained model artifacts per position.

    Returns
    -------
    pd.DataFrame
        Predictions sorted by xPoints descending.
    """
    pred_df = build_prediction_features(
        history_df, players_df, upcoming_df, next_gw
    )

    # Run ensemble
    pred_df = ensemble_predict_all(pred_df, artifacts)

    # DGW/BGW adjustment
    if "fixture_count" in pred_df.columns:
        pred_df["xPoints"] = pred_df["xPoints_raw"] * pred_df["fixture_count"]
    else:
        pred_df["xPoints"] = pred_df["xPoints_raw"]

    # Zero out BGW / unavailable players
    mask_zero = (
        (pred_df.get("fixture_count", pd.Series(1)) == 0)
        | (pred_df["chance_of_playing_next_round"] == 0)
    )
    pred_df.loc[mask_zero, "xPoints"] = 0.0

    # Map team names
    team_map = dict(zip(teams_df["id"], teams_df["name"]))
    pred_df["team_name"] = pred_df["team"].map(team_map)

    pred_df.sort_values("xPoints", ascending=False, inplace=True)

    return pred_df
