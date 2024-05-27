"""Class to split by BB."""
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.typing.xdata import XData
from src.utils.logger import logger

from .base import Splitter


@dataclass
class BBSplitter(Splitter):
    """Class to split dataset by building blocks.

    :param n_splits: Number of splits
    :param bb_to_split_by: Building blocks to split by
    """

    n_splits: int = 5
    indices_for_flattened_data: bool = False
    bb_to_split_by: list[int] = field(default_factory=lambda: [1, 1, 1])

    def split(
        self,
        X: XData | None,
        y: npt.NDArray[np.int8] | None,
        cache_path: Path | None = None,
    ) -> (
        list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]
        | tuple[list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]], npt.NDArray[np.int64], npt.NDArray[np.int64]]
    ):
        """Split X and y into train and validation sets.

        :param X: The Xdata
        :param y: Labels
        :return: List of indices
        """
        # Load the splits if they exist
        if cache_path is not None and cache_path.exists():
            with open(cache_path, "rb") as f:
                logger.info(f"Loading splits from {cache_path}")
                return pickle.load(f)  # noqa: S301

        if X is None or y is None:
            raise TypeError("X or y cannot be None if no cache is available")

        bb1_values = range(len(X.bb1_smiles)) if X.bb1_smiles is not None else [0]
        bb2_values = range(len(X.bb2_smiles)) if X.bb2_smiles is not None else [0]
        bb3_values = range(len(X.bb3_smiles)) if X.bb3_smiles is not None else [0]

        if len(X.encoded_rows) != len(y):
            raise ValueError("X is not equal to y")

        # Split the data into n_splits
        # Create splits where bb1_values are divided into n_splits, bb2_values are divided into n_splits, and bb3_values are divided into n_splits
        # Train split would have all the values of bb1, bb2, and bb3 and the validation split would have all the other values of bb1, bb2, and bb3
        bb1_values_split = np.array_split(bb1_values, self.n_splits)
        bb2_values_split = np.array_split(bb2_values, self.n_splits)
        bb3_values_split = np.array_split(bb3_values, self.n_splits)

        splits = []
        for i in range(self.n_splits):
            split_bb1_values = bb1_values_split[i]
            split_bb2_values = bb2_values_split[i]
            split_bb3_values = bb3_values_split[i]

            X_train, X_test = self.create_train_test(X, split_bb1_values, split_bb2_values, split_bb3_values)
            splits.append((X_train, X_test))

        if self.indices_for_flattened_data:
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

        # Pickle splits
        logger.debug(f"Finished splitting with size:{len(y)}")
        if cache_path is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)

        return splits

    def create_train_test(
        self,
        X: XData,
        split_bb1_values: npt.NDArray[np.int16],
        split_bb2_values: npt.NDArray[np.int16],
        split_bb3_values: npt.NDArray[np.int16],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Split data into train and test.

        :param X: XData class
        :param split_bb1_values: The split of building block 1
        :param split_bb2_values: The split of building block 2
        :param split_bb3_values: The split of building block 3
        :return: X_train, X_test
        """
        # Train split would have all the values of bb1, bb2, and bb3 and the test split would have all the other values of bb1, bb2, and bb3
        train_bb1_bool = np.isin(X.encoded_rows[:, 0], split_bb1_values)
        if self.bb_to_split_by[0] == 0:
            train_bb1_bool = train_bb1_bool & False
        train_bb2_bool = np.isin(X.encoded_rows[:, 1], split_bb2_values)
        if self.bb_to_split_by[1] == 0:
            train_bb2_bool = train_bb2_bool & False
        train_bb3_bool = np.isin(X.encoded_rows[:, 2], split_bb3_values)
        if self.bb_to_split_by[2] == 0:
            train_bb3_bool = train_bb3_bool & False
        X_train = np.where(~train_bb1_bool & ~train_bb2_bool & ~train_bb3_bool)[0]

        if self.bb_to_split_by[0] == 0:
            train_bb1_bool = train_bb1_bool | True
        if self.bb_to_split_by[1] == 0:
            train_bb2_bool = train_bb2_bool | True
        if self.bb_to_split_by[2] == 0:
            train_bb3_bool = train_bb3_bool | True

        X_test = np.where(train_bb1_bool & train_bb2_bool & train_bb1_bool)[0]

        return X_train, X_test

    @property
    def includes_test(self) -> bool:
        """Check if the splitter also generates a test set."""
        return False
