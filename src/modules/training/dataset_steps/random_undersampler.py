"""Module to perform undersampling on the molecule smiles."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst._core._caching._cacher import CacheArgs
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from numpy.random import default_rng


@dataclass
class RandomUndersampler(TrainingBlock):
    """Module to perform undersampling on the molecule smiles."""

    majority_sampling: float = 0.2

    def train(
        self,
        x: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        _cache_args: CacheArgs | None = None,
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Perform undersampling on the molecule smiles.

        :param X: array containing the smile strings
        :param y: array containing the labels
        """
        # Extract the indices in the majority class
        mask = np.all(y == [0, 0, 0], axis=1)
        indices = np.where(mask)[0]

        rng = default_rng()

        # Compute the indices of the samples to remove
        num_to_remove = int(len(indices) * (1 - self.majority_sampling))
        remove_indices = rng.choice(indices, num_to_remove, replace=False)

        # Create a mask to keep all the randomly selected majority samples
        final_mask = np.ones(len(x), dtype=bool)
        final_mask[remove_indices] = False

        return x[final_mask], y[final_mask]

    @property
    def is_augmentation(self) -> bool:
        """Check if augmentation is enabled."""
        return True
