"""Perform oversampling by swapping the building blocks."""

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst._core._caching._cacher import CacheArgs
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from numpy.random import default_rng


@dataclass
class SwapOversampling(TrainingBlock):
    """Perform oversampling by swapping the building blocks."""

    p_swap: float = 0.2

    def train(
        self,
        x: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        _cache_args: CacheArgs | None = None,
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.uint8]]:
        """Perform oversampling by swapping the building blocks.

        :param x: array containing the smile strings
        :param y: array containing the protein labels
        """
        rng = default_rng()

        # Generate random probabilities for each molecule
        random_probs = rng.random(len(y))

        # Swap the building blocks for the chosen molecules
        for idx in range(y.shape[0]):
            if random_probs[idx] < self.p_swap:
                random.shuffle(x[idx])

        return x, y

    @property
    def is_augmentation(self) -> bool:
        """Check if augmentation is enabled."""
        return True
