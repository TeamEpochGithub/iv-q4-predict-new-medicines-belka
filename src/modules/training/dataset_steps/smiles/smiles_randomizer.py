"""Custom sequential class for augmentations."""

from dataclasses import dataclass, field
from inspect import signature
from typing import Any

from epochalyst._core._caching._cacher import CacheArgs
import torch
from rdkit import Chem
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from src.typing.xdata import XData, DataRetrieval
import numpy as np
import numpy.typing as npt


@dataclass
class SmilesRandomizer(TrainingBlock):
    """Randomize the SMILES string to produce an equivalent SMILES but different SMILES string.
    
    :param p: Probability of applying the augmentation.
    """

    p: float = 0.5

    def train(self, x: Any, y: Any, cache_args: CacheArgs | None = None, **train_args: Any) -> tuple[Any, Any]:
        return self._randomize_smiles(x), y
    
    def _randomize_smiles(self, smiles: npt.NDArray[np.str_]) -> npt.NDArray[np.str_]:
        """Randomize the SMILES string"""

        smiles_flattend = smiles.flatten()
        for i in range(len(smiles_flattend)):
            if np.random.rand() > self.p:
                continue
            mol = Chem.MolFromSmiles(smiles_flattend[i])
            smiles_flattend[i] = Chem.MolToSmiles(mol, doRandom=True)

        return smiles_flattend.reshape(smiles.shape)

    @property
    def is_augmentation(self) -> bool:
        return True