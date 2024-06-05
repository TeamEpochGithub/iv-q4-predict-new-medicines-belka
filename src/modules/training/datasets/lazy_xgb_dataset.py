"""Module for the lazy xgboost dataset."""
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import xgboost as xgb
from epochalyst.pipeline.model.training.training_block import TrainingBlock


@dataclass
class LazyXGBDataset:
    """Dataset to load data in batches for xgboost dmatrices.

    :param steps: The training block steps to apply to a batch.
    :param chunk_size: Size of chunk
    """

    steps: list[TrainingBlock]
    chunk_size: int = 10000

    def get_iterator(self, X: npt.NDArray[np.string_] | list[str], y: npt.NDArray[np.int_] | list[int]) -> Generator[xgb.DMatrix, None, None]:
        """Get the batch _iterator.

        :param X: Input data
        :param y: Labels
        :return: Generator
        """
        return self._iterator(X, y)

    def _calculate_steps(
        self,
        X: npt.NDArray[np.string_] | list[str],
        y: npt.NDArray[np.int_] | list[int],
    ) -> tuple[npt.NDArray[np.string_] | list[str], npt.NDArray[np.int_] | list[int]]:
        """Calculate the data using training steps provided.

        :param X: Input x data
        :param y: Input y data
        :return: Transformed data
        """
        for step in self.steps:
            X, y = step.train(X, y)

        return X, y

    def _iterator(self, X: npt.NDArray[np.string_] | list[str], y: npt.NDArray[np.int_] | list[int]) -> Generator[xgb.DMatrix, None, None]:
        """Iterate over data and yield a dmatrix.

        :param X: Input data
        :param y: Label data
        :return: DMatrix
        """
        index = 0
        while index < len(X):
            X_subset = X[index : index + self.chunk_size]
            y_subset = y[index : index + self.chunk_size]
            index += self.chunk_size
            x_processed, y_processed = self._calculate_steps(X_subset, y_subset)
            yield xgb.DMatrix(x_processed, label=y_processed)
