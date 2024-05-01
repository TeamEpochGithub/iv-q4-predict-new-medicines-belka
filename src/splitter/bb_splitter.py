"""Class to split by BB."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.typing.xdata import XData


@dataclass
class BBSplitter:
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
            X_train = np.where(~np.isin(X.building_blocks[:, 0], split_bb1_values))[0]
            X_test = np.where(np.isin(X.building_blocks[:, 0], split_bb1_values))[0]
            splits.append((X_train, X_test))

        if self.protein_level:
            # If protein level is true then the for each index in the split, the split is divided into 3 splits based on the protein
            # This means if train indices are [1, 3, 5] it should become [1, 2, 3, 10, 11, 12, 20, 21, 22] if the protein level is true
            # This is done for each split
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
