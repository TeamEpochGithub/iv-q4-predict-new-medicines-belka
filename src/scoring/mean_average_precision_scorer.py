from sklearn.metrics import average_precision_score
from src.scoring.scorer import Scorer
import numpy as np
from typing import Any


class MeanAveragePrecisionScorer(Scorer):

    def __call__(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], **kwargs: Any) -> float:
        return average_precision_score(y_true, y_pred)
