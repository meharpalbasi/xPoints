"""Walk-forward cross-validation splitter by gameweek."""

from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

from src.utils.config import MIN_TRAIN_GAMEWEEKS, TEST_GAMEWEEKS


class WalkForwardCV:
    """Walk-forward splitter: train on GW 1..N, test on GW N+1, slide forward.

    Parameters
    ----------
    min_train_gws : int
        Minimum number of training gameweeks before the first split.
    test_gws : int
        Number of gameweeks in each test fold (default 1).
    """

    def __init__(
        self,
        min_train_gws: int = MIN_TRAIN_GAMEWEEKS,
        test_gws: int = TEST_GAMEWEEKS,
    ) -> None:
        self.min_train_gws = min_train_gws
        self.test_gws = test_gws

    def split(
        self, df: pd.DataFrame, gw_col: str = "round"
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) arrays respecting gameweek boundaries.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain *gw_col* with integer gameweek identifiers.
        gw_col : str
            Column name for the gameweek number.

        Yields
        ------
        train_idx, test_idx : np.ndarray
            Integer position indices into *df*.
        """
        sorted_gws: List[int] = sorted(df[gw_col].unique())

        for i in range(self.min_train_gws, len(sorted_gws)):
            test_end = min(i + self.test_gws, len(sorted_gws))
            train_gws = sorted_gws[:i]
            test_gws = sorted_gws[i:test_end]
            if not test_gws:
                break
            train_mask = df[gw_col].isin(train_gws)
            test_mask = df[gw_col].isin(test_gws)
            yield (
                np.where(train_mask)[0],
                np.where(test_mask)[0],
            )

    def get_n_splits(self, df: pd.DataFrame, gw_col: str = "round") -> int:
        """Return the number of folds."""
        n_gws = df[gw_col].nunique()
        return max(0, n_gws - self.min_train_gws)
