"""Perform oversampling by swapping the building blocks of the minority."""

from dataclasses import dataclass
from typing import Any
import random
import numpy as np
import numpy.typing as npt
from epochalyst._core._caching._cacher import CacheArgs
from epochalyst.pipeline.model.training.training_block import TrainingBlock

@dataclass
class BlockOversampling(TrainingBlock):
    """Perform oversampling by swapping the building blocks of the minority."""

    p_swap: float = 0.2
    def train(
            self,
            x: npt.NDArray[np.str_],
            y: npt.NDArray[np.uint8],
            _cache_args: CacheArgs | None = None,
            **train_args: Any,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        # Generate random probabilities for each molecule
        random_probs = np.random.rand(len(y))

        # Extract the indices of the binding molecules
        indices = [i for i in range(y.shape[0]) if not np.array_equal(y[i], [0, 0, 0])]

        for idx in range(y.shape[0]):
            if random_probs[idx] < self.p_swap:
                random.shuffle(x[idx])

        return x, y
    @property
    def is_augmentation(self) -> bool:
        """Check if augmentation is enabled."""
        return True
