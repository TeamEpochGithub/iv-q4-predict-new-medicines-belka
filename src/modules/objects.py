"""Objects used in for the training and transformation pipelines."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from torch import nn

from src.typing.xdata import XData


@dataclass
class TrainPredictObj:
    """Object passed during training and prediction."""

    x_data: XData
    y_predictions: npt.NDArray[np.float_] | None = None
    model: nn.Module | None = None


@dataclass
class TrainObj:
    """Object passed only during training."""

    y_labels_original: npt.NDArray[np.int8]
    y_labels_modified: npt.NDArray[np.int_] | npt.NDArray[np.float_] | None = None
