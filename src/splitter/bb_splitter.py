"""Class to split by BB."""
from dataclasses import dataclass
import time
import numpy as np
import numpy.typing as npt

from src.typing.xdata import XData


@dataclass
class BBSplitter:
    """Class to split dataset by building blocks.

    :param n_splits: Number of splits
    """

    n_splits: int = 5

    def split(self, X: XData, y: npt.NDArray[np.int8]) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        """Split X and y into train and test indices.

        :param X: The Xdata
        :param y: Labels
        :return: List of indices
        """
        bb1_values = range(len(X.bb1))

        if len(X.building_blocks) != len(y):
            raise ValueError("X is not equal to y")

        # bb2_values = range(len(X.bb2))
        # bb3_values = range(len(X.bb3))

        # Split the data into n_splits
        # Create splits where bb1_values are divided into n_splits, bb2_values are divided into n_splits, and bb3_values are divided into n_splits

        # Train split would have all the values of bb1, bb2, and bb3 and the test split would have all the other values of bb1, bb2, and bb3

        bb1_values_split = np.array_split(bb1_values, self.n_splits)
        # bb2_values_split = np.array_split(bb2_values, self.n_splits)
        # bb3_values_split = np.array_split(bb3_values, self.n_splits)

        splits = []
        for i in range(self.n_splits):
            split_bb1_values = bb1_values_split[i]
            # split_bb2_values = bb2_values_split[i]
            # split_bb3_values = bb3_values_split[i]

            # Split the data into train and test
            # Train split would have all the values of bb1, bb2, and bb3 and the test split would have all the other values of bb1, bb2, and bb3
            # X_test = X.building_blocks[
            # np.isin(X.building_blocks[:, 0], split_bb1_values) & np.isin(X.building_blocks[:, 1], split_bb2_values) & np.isin(X.building_blocks[:, 2], split_bb3_values)]
            # X_train = X.building_blocks[
            # ~np.isin(X.building_blocks[:, 0], split_bb1_values) & ~np.isin(X.building_blocks[:, 1], split_bb2_values) & ~np.isin(X.building_blocks[:, 2], split_bb3_values)]
            curr_time = time.time()
            X_train = np.where(np.isin(X.building_blocks[:, 0], split_bb1_values))[0]
            X_test = np.where(~np.isin(X.building_blocks[:, 0], split_bb1_values))[0]
            splits.append((X_train, X_test))
            print(f"Time = {time.time() - curr_time}")

        return splits
