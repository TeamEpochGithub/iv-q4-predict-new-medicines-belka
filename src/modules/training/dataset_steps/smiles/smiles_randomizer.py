"""Custom sequential class for augmentations."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst._core._caching._cacher import CacheArgs
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from rdkit import Chem  # type: ignore[import]


@dataclass
class SmilesRandomizer(TrainingBlock):
    """Randomize the SMILES string to produce an equivalent SMILES but different SMILES string.

    :param p: Probability of applying the augmentation.
    """

    p: float = 0.5

    def __post_init__(self) -> None:
        """Set up the randomizer."""
        self.rand = np.random.Generator(np.random.PCG64())

    def train(
        self,
        x: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        _cache_args: CacheArgs | None = None,
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.uint8]]:
        """Randomize the SMILES string."""
        return self._randomize_smiles(x), y

    def _randomize_smiles(self, smiles: npt.NDArray[np.str_]) -> npt.NDArray[np.str_]:
        """Randomize the SMILES string."""
        smiles_flattend = smiles.flatten()
        for i in range(len(smiles_flattend)):
            if self.rand.random() > self.p:
                continue
            mol = Chem.MolFromSmiles(smiles_flattend[i])
            smiles_flattend[i] = Chem.MolToSmiles(mol, doRandom=True)

        return smiles_flattend.reshape(smiles.shape)

    @property
    def is_augmentation(self) -> bool:
        """Check if augmentation is enabled."""
        return True
