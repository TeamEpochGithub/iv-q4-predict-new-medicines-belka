"""Class to split by BB."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import KFold

from src.typing.xdata import XData
from src.utils.logger import logger


@dataclass
class NormalSplitter:
    """Class to split dataset by building blocks.

    :param n_splits: Number of splits
    """

    n_splits: int = 5
    protein_level: bool = False

    def split(self, X: XData, y: npt.NDArray[np.int8]) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        """Split X and y into train and test indices.

        :param X: The Xdata
        :param y: Labels
        :return: List of indices
        """
        splits = []
        logger.debug(f"Starting splitting with size:{len(y)}")
        kf = KFold(n_splits=self.n_splits, random_state=None, shuffle=False)

        kf_splits = kf.split(X.building_blocks)
        for train_index, test_index in kf_splits:
            splits.append((train_index, test_index))

        if self.protein_level:
            new_splits = []
            for split in splits:
                train_indices, test_indices = split

                train_indices = np.array(train_indices)
                test_indices = np.array(test_indices)

                # For each index it should be multiplied by 3 and then 0, 1, 2 should be added to it and then the indices should be added to the new split
                new_train_indices = []
                new_test_indices = []
                for index in train_indices:
                    new_train_indices.extend([index * 3 + i for i in range(3)])
                for index in test_indices:
                    new_test_indices.extend([index * 3 + i for i in range(3)])

                new_splits.append((np.array(new_train_indices), np.array(new_test_indices)))
            splits = new_splits

        return splits
