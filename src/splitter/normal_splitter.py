"""Class to split into train test."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import KFold

from src.typing.xdata import XData
from src.utils.logger import logger

from .base import Splitter


@dataclass
class NormalSplitter(Splitter):
    """Class to split dataset into train test.

    :param n_splits: Number of splits
    :param indices_for_flattened_data: If data is flattened to per protein, indices should be processed
    """

    n_splits: int = 5
    indices_for_flattened_data: bool = False

    def split(
        self,
        X: XData | None,
        y: npt.NDArray[np.int8] | None,
        _cache_path: Path,
    ) -> (
        list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]
        | tuple[list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]], npt.NDArray[np.int64], npt.NDArray[np.int64]]
    ):
        """Split X and y into train and test indices.

        :param X: The Xdata
        :param y: Labels
        :return: List of indices
        """
        if X is None or y is None:
            raise TypeError("X or y cannot be None - Caching not implemented for NormalSplitter")

        splits = []
        logger.debug(f"Starting splitting with size:{len(y)}")
        kf = KFold(n_splits=self.n_splits, random_state=None, shuffle=False)

        kf_splits = kf.split(X.encoded_rows, y)
        for train_index, test_index in kf_splits:
            splits.append((train_index, test_index))

        if self.indices_for_flattened_data:
            new_splits = []
            for idx, split in enumerate(splits):
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

                # Log some information about each split
                train_percentage_brd4 = 100 * len(y[:, 0][train_indices][y[:, 0][train_indices] == 1]) / len(y[train_indices])
                train_percentage_hsa = 100 * len(y[:, 1][train_indices][y[:, 1][train_indices] == 1]) / len(y[train_indices])
                train_percentage_seh = 100 * len(y[:, 2][train_indices][y[:, 2][train_indices] == 1]) / len(y[train_indices])

                test_percentage_brd4 = 100 * len(y[:, 0][test_indices][y[:, 0][test_indices] == 1]) / len(y[test_indices])
                test_percentage_hsa = 100 * len(y[:, 1][test_indices][y[:, 1][test_indices] == 1]) / len(y[test_indices])
                test_percentage_seh = 100 * len(y[:, 2][test_indices][y[:, 2][test_indices] == 1]) / len(y[test_indices])

                logger.info(
                    (
                        f"Split: {idx} -- "
                        f"{train_percentage_brd4:.2f}-{test_percentage_brd4:.2f}% BRD4 binds -- "
                        f"{train_percentage_hsa:.2f}-{test_percentage_hsa:.2f}% HSA binds -- "
                        f"{train_percentage_seh:.2f}-{test_percentage_seh:.2f}% sEH binds"
                    ),
                )
            splits = new_splits

        return splits

    @property
    def includes_test(self) -> bool:
        """Check if the splitter also generates a test set."""
        return False
