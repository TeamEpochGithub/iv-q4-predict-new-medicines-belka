"""Module to perform undersampling on the molecule smiles."""

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from epochalyst._core._caching._cacher import CacheArgs
from epochalyst.pipeline.model.training.training_block import TrainingBlock

path = "data/shrunken/train_dicts/BBs_dict_reverse_1.p"
BBs_dict_reverse_1 = joblib.load(path)

smile_75 = BBs_dict_reverse_1[75]
smile_76 = BBs_dict_reverse_1[76]


@dataclass
class ProteinUndersampler(TrainingBlock):
    """Module to perform undersampling on the molecule smiles."""

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
        # Create the mask to remove the common BB
        final_mask = np.all((x != smile_75) & (x != smile_76), axis=1)

        return x[final_mask], y[final_mask]

    @property
    def is_augmentation(self) -> bool:
        """Check if augmentation is enabled."""
        return True
