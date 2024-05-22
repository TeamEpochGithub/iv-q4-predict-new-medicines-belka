"""Custom sequential class for augmentations."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst._core._caching._cacher import CacheArgs
from epochalyst.pipeline.model.training.training_block import TrainingBlock

from src.modules.transformation.encoder.smile_atom_encoder import (
    MAX_ENC_SIZE_BB1,
    MAX_ENC_SIZE_BB2,
    MAX_ENC_SIZE_BB3,
    MAX_ENC_SIZE_MOLECULE,
    encode_block,
    encode_block_1,
    encode_molecule,
)


@dataclass
class SmilesAtomEncoder(TrainingBlock):
    """Encode an atom based on enc dictionary.

    Supports SMILES Molecues and Buildings Blocks.
    :param max_enc_size_molecule: The maximum encoding size for molecules
    :param max_enc_size_bb: The maximum encoding size for building blocks
    """

    max_enc_size_molecule: int = MAX_ENC_SIZE_MOLECULE
    max_enc_size_bb: int | None = None

    def __post_init__(self) -> None:
        """Set the maximum encoding size for building blocks."""
        if self.max_enc_size_bb is None:
            self.max_enc_size_bb = max(MAX_ENC_SIZE_BB1, MAX_ENC_SIZE_BB2, MAX_ENC_SIZE_BB3)

    def train(
        self,
        x: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        _cache_args: CacheArgs | None = None,
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Encode an atom based on enc dictionary."""
        if self.max_enc_size_bb is None:
            raise ValueError("The maximum encoding size for building blocks must be set.")

        # Only molecules
        if x.ndim == 1:
            return np.array([encode_molecule(i, self.max_enc_size_molecule) for i in x]), y

        # Only building blocks
        if x.shape[1] == 3:
            result = np.empty((*x.shape, self.max_enc_size_bb), dtype=np.uint8)
            result[:, 0] = np.array([encode_block_1(j, self.max_enc_size_bb) for j in x[:, 0]])
            result[:, 1] = np.array([encode_block(j, self.max_enc_size_bb) for j in x[:, 1]])
            result[:, 2] = np.array([encode_block(j, self.max_enc_size_bb) for j in x[:, 2]])
            return result, y

        # Molecules and building blocks
        result = np.empty((*x.shape, self.max_enc_size_molecule), dtype=np.uint8)
        result[:, 0] = np.array([encode_molecule(j, self.max_enc_size_molecule) for j in x[:, 0]])
        result[:, 1] = np.array([encode_block_1(j, self.max_enc_size_molecule) for j in x[:, 1]])
        result[:, 2] = np.array([encode_block(j, self.max_enc_size_molecule) for j in x[:, 2]])
        result[:, 3] = np.array([encode_block(j, self.max_enc_size_molecule) for j in x[:, 3]])
        return result, y

    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return False
