"""Module to augment atom encoding."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.model.training.training_block import TrainingBlock

from src.modules.logging.logger import Logger


@dataclass
class PairwiseX(TrainingBlock, Logger):
    """Group X in pairs."""

    def train(self, X: npt.NDArray[Any], y: Any) -> tuple[npt.NDArray[Any], Any]:  # noqa: ANN401
        """Group X in pairs.

        :param X: Input array
        :param y: Labels
        :return:
        """
        if X is None:
            raise ValueError("There is no SMILE information for the molecules, can't convert to ECFP")

        # Reshape X and y to be pairwise and include the similiarity
        shape = tuple([X.shape[0], 2] + ([X.shape[1]] if len(X.shape) > 1 else []))
        X_temp = np.empty(shape, dtype=X.dtype)
        for idx in range(len(X)):
            if idx == len(X) - 1:
                X_temp[idx] = np.concatenate((np.expand_dims(X[idx], axis=0), np.expand_dims(X[0], axis=0)), axis=0)
                continue

            X_temp[idx] = np.concatenate((np.expand_dims(X[idx], axis=0), np.expand_dims(X[idx + 1], axis=0)), axis=0)

        return X_temp, y

    @property
    def is_augmentation(self) -> bool:
        """Return if block is augmentation block.

        :return: is_block
        """
        return False
