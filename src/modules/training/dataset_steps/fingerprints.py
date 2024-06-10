"""Module to augment atom encoding."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.model.training.training_block import TrainingBlock

from src.modules.logging.logger import Logger


@dataclass
class Fingerprints(TrainingBlock, Logger):
    """Fingerprint generation in dataset."""

    NUM_FUTURES: int = 100
    MIN_CHUNK_SIZE: int = 10000
    fingerprint: Any = None

    def train(self, X: npt.NDArray[np.string_], y: npt.NDArray[np.int_]) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Replace some atoms with similar ones.

        :param X: Input array
        :param y: Labels
        """
        if X is None:
            raise ValueError("There is no SMILE information for the molecules, can't convert to ECFP")
        self.log_to_terminal("Creating Fingerprint for molecules")
        chunk_size = len(X) // self.NUM_FUTURES
        chunk_size = max(chunk_size, self.MIN_CHUNK_SIZE)
        result = self.fingerprint.fit_transform(X=X, batch_size=chunk_size, n_jobs=-1, verbose=1)

        return result, y

    @property
    def is_augmentation(self) -> bool:
        """Return if block is augmentation block.

        :return: is_block
        """
        return False
