"""Custom sequential class for augmentations."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst._core._caching._cacher import CacheArgs
from epochalyst.pipeline.model.training.training_block import TrainingBlock

ENCODING = {
    "l": 1,
    "y": 2,
    "@": 3,
    "3": 4,
    "H": 5,
    "S": 6,
    "F": 7,
    "C": 8,
    "r": 9,
    "s": 10,
    "/": 11,
    "c": 12,
    "o": 13,
    "+": 14,
    "I": 15,
    "5": 16,
    "(": 17,
    "2": 18,
    ")": 19,
    "9": 20,
    "i": 21,
    "#": 22,
    "6": 23,
    "8": 24,
    "4": 25,
    "=": 26,
    "1": 27,
    "O": 28,
    "[": 29,
    "D": 30,
    "B": 31,
    "]": 32,
    "N": 33,
    "7": 34,
    "n": 35,
    "-": 36,
    ".": 37,
    "\\": 38,
    "%": 39,
}


@dataclass
class SmilesEncoder(TrainingBlock):
    """Encode the SMILES string into categorical data."""

    def train(
        self,
        x: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        _cache_args: CacheArgs | None = None,
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Encode the SMILES string into categorical data."""
        return self._encode_smiles(x), y

    def _encode_smiles(self, smiles: npt.NDArray[np.str_]) -> npt.NDArray[np.uint8]:
        """Encode the SMILE strings into categorical data."""
        smiles_flattend = smiles.flatten()
        smiles_new = np.empty((len(smiles_flattend), 142), dtype=np.uint8)

        for smiles_idx in range(len(smiles_flattend)):
            tmp = [ENCODING[i] for i in smiles_flattend[smiles_idx]]
            tmp = tmp + [0] * (142 - len(tmp))
            smiles_new[smiles_idx] = np.array(tmp).astype(np.uint8)

        return smiles_new.reshape((*smiles.shape, -1))

    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return False
