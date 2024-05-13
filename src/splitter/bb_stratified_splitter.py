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


@dataclass
class BBStratifiedSplitter:
    """Class to split dataset into stratified multi label split.

    :param n_splits: Number of splits
    :param indices_for_flattened_data: If data is flattened to per protein, indices should be processed
    """

    n_splits: int = 5

    def split(self, X: XData, y: npt.NDArray[np.int8], cache_path: Path) -> tuple[list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Split X and y into train and test indices.

        :param X: The Xdata
        :param y: Labels
        :return: List of indices
        """
        splits = []

        logger.debug(f"Starting splitting with size:{len(y)}")

        # Load the splits if they exist
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                logger.info(f"Loading splits from {cache_path}")
                return pickle.load(f)  # noqa: S301

        bb_splitter = BBSplitter(n_splits=self.n_splits, bb_to_split_by=[1, 1, 1])
        train_bb, val = bb_splitter.split(X, y)[0]
        logger.info(f"Train/Val: {len(train_bb)}/{len(val)}")

        kf = MultilabelStratifiedKFold(n_splits=self.n_splits)

        kf_splits = kf.split(X.building_blocks[train_bb], y[train_bb])
        for train_index, test_index in tqdm(kf_splits, total=self.n_splits, desc="Creating splits"):
            splits.append((train_index, test_index))

        # Pickle the splits
        logger.debug(f"Finished splitting with size:{len(y)}")
        with open(cache_path, "wb") as f:
            pickle.dump((splits, train_bb, val), f, protocol=pickle.HIGHEST_PROTOCOL)

        return splits, train_bb, val
