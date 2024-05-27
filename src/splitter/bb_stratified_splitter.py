"""Class to split into stratified multi label train test."""
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import tqdm

from src.splitter.bb_splitter import BBSplitter
from src.typing.xdata import XData
from src.utils.logger import logger

from .base import Splitter


@dataclass
class BBStratifiedSplitter(Splitter):
    """Class to split dataset into stratified multi label split.

    :param n_splits: Number of splits
    :param indices_for_flattened_data: If data is flattened to per protein, indices should be processed
    """

    n_splits: int = 5
    test_size: float = 0.2

    def split(
        self,
        X: XData | None,
        y: npt.NDArray[np.int8] | None,
        cache_path: Path,
    ) -> (
        list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]
        | tuple[list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]], npt.NDArray[np.int64], npt.NDArray[np.int64]]
    ):
        """Split X and y into train and test indices.

        :param X: The Xdata
        :param y: Labels
        :return: List of indices
        """
        # Load the splits if they exist
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                logger.info(f"Loading splits from {cache_path}")
                splits, train_validation_indices, test_indices = pickle.load(f)  # noqa: S301
                logger.info(f"Train/Validation/Test Set Size: {len(splits[0][0]):,} / {len(splits[0][1]):,} / {len(test_indices):,}")
                return splits, train_validation_indices, test_indices

        if X is None or y is None:
            raise TypeError("X or y cannot be None if no cache is available")

        # Creating a Test set
        logger.info("Creating a test set")
        bb_splitter = BBSplitter(n_splits=(int(1 / self.test_size)), bb_to_split_by=[1, 1, 1])
        train_validation_indices, test_indices = bb_splitter.split(X, y)[0]
        logger.info(f"Train/Validation/Test Set Size: {len(splits[0][0]):,} / {len(splits[0][1]):,} / {len(test_indices):,}")

        # Splitting the rest into train and validation sets
        logger.info("Splitting the training set into Train/Validation sets")
        splits = []
        kf = MultilabelStratifiedKFold(n_splits=self.n_splits)
        kf_splits = kf.split(X.encoded_rows[train_validation_indices], y[train_validation_indices])
        for train_indices, test_indices in tqdm(kf_splits, total=self.n_splits, desc="Creating splits"):
            # Reindex the train and test indices
            train_indices_reindexed = train_validation_indices[train_indices]
            test_indices_reindexed = train_validation_indices[test_indices]
            splits.append((train_indices_reindexed, test_indices_reindexed))

        # Pickle the splits
        logger.info(f"Saving splits to {cache_path}")
        if not cache_path.parent.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((splits, train_validation_indices, test_indices), f, protocol=pickle.HIGHEST_PROTOCOL)

        return splits, train_validation_indices, test_indices

    @property
    def includes_test(self) -> bool:
        """Check if the splitter also generates a test set."""
        return True
