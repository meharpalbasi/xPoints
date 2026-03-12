"""xPoints data source modules.

Provides standalone data fetching and feature engineering for:
- Understat player/team advanced stats
- Opponent feature vectors
- Betting odds implied probabilities
"""

from .understat_api import UnderstatData
from .opponent_features import OpponentFeatures
from .odds_data import OddsData

__all__ = ["UnderstatData", "OpponentFeatures", "OddsData"]
