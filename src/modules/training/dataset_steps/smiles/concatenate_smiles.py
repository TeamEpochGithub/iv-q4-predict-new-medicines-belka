"""Module to concatenate the building block smiles into one string."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst._core._caching._cacher import CacheArgs
from epochalyst.pipeline.model.training.training_block import TrainingBlock

@dataclass
class ConcatenateSmiles(TrainingBlock):
    """Module to concatenate the building block smiles into one string."""
    def train(
            self,
            x: npt.NDArray[np.str_],
            y: npt.NDArray[np.uint8],
            _cache_args: CacheArgs | None = None,
            **train_args: Any,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        # concatenate the building block smiles
        x = [''.join(molecule) for molecule in x]

        return np.array(x), y
    @property
    def is_augmentation(self) -> bool:
        """Check if augmentation is enabled."""
        return False


