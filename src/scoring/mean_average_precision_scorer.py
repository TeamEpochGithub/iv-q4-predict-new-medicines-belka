"""Module that implements mean average precision scorer as a scorer."""
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from src.scoring.scorer import Scorer


class MeanAveragePrecisionScorer(Scorer):
    """Class for mean average precision."""

    def __call__(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], **kwargs: Any) -> float:
        """Calculate the average precision score from sklearn.metrics.

        :param y_true: True labels
        :param y_pred: Predicted values
        :return: Score
        """
        return average_precision_score(y_true, y_pred)
