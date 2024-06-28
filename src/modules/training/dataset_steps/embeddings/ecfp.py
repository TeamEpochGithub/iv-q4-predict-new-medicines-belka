"""Module to augment atom encoding."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from rdkit import DataStructs  # type: ignore[import]

from src.modules.chemistry_functions.ecfp import EcfpReturnType, convert_smile_array_parallel, convert_smiles_array
from src.modules.logging.logger import Logger


@dataclass
class ECFP(TrainingBlock, Logger):
    """Fingerprint generation in dataset."""

    bits: int = 128
    radius: int = 2
    use_features: bool = False

    def train(self, X: npt.NDArray[np.bytes_], y: npt.NDArray[np.int_]) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int_]]:
        """Replace some atoms with similar ones.

        :param X: Input array
        :param y: Labels
        """
        if X is None:
            raise ValueError("There is no SMILE information for the molecules, can't convert to ECFP")
        self.log_to_terminal("Creating Fingerprint for molecules")
        result = convert_smile_array_parallel(smiles_array=X, bits=self.bits, radius=self.radius, use_features=self.use_features, desc="Creating ECFP for molecules")

        return result, y

    @property
    def is_augmentation(self) -> bool:
        """Return if block is augmentation block.

        :return: is_block
        """
        return False


@dataclass
class ECFPLabel(TrainingBlock, Logger):
    """Fingerprint generation in dataset for labels."""

    bits: int = 128
    radius: int = 2
    use_features: bool = False

    def train(self, X: npt.NDArray[np.bytes_], y: npt.NDArray[np.int_]) -> tuple[npt.NDArray[np.bytes_], npt.NDArray[np.int_]]:
        """Replace some atoms with similar ones.

        :param X: Input array
        :param y: Labels
        """
        if X is None:
            raise ValueError("There is no SMILE information for the molecules, can't convert to ECFP")

        result = convert_smiles_array(smiles_array=X, bits=self.bits, radius=self.radius, use_features=self.use_features, return_type=EcfpReturnType.NP_UNPACKED)

        # Make each value of y a tuple of result and original y: (4096, 3) + (4096, y) = (4096, 2, 3/y)
        if y is not None:
            labels = np.concatenate((y, result), axis=-1, dtype=np.int8)
        else:
            labels = None

        return X, labels

    @property
    def is_augmentation(self) -> bool:
        """Return if block is augmentation block.

        :return: is_block
        """
        return False


@dataclass
class ECFPPairwiseSimiliarity(TrainingBlock, Logger):
    """Similiartity calculation in dataset for labels."""

    bits: int = 128
    radius: int = 2
    use_features: bool = False
    multi_processing: bool = field(default=False, init=True, repr=False, compare=False)

    def train(self, X: npt.NDArray[np.bytes_], y: npt.NDArray[np.int_]) -> tuple[npt.NDArray[np.bytes_], npt.NDArray[np.floating[Any]]]:
        """Replace some atoms with similar ones.

        :param X: Input array
        :param y: Labels
        :return:
        """
        if X is None:
            raise ValueError("There is no SMILE information for the molecules, can't convert to ECFP")

        if y is None:
            return X, y

        # Calculate the ECFP for each molecule
        ecfp_array = convert_smiles_array(smiles_array=X, bits=self.bits, radius=self.radius, use_features=self.use_features, return_type=EcfpReturnType.RDKIT)

        # Reshape X and y to be pairwise and include the similiarity
        y_temp = np.empty((y.shape[0], y.shape[1] * 2 + 1), dtype=np.float32)
        for idx in range(len(X)):
            if idx == len(X) - 1:
                similiarity = DataStructs.TanimotoSimilarity(ecfp_array[idx], ecfp_array[0])
                y_temp[idx] = np.concatenate((y[idx], y[0], np.expand_dims(np.array(similiarity), axis=0)), axis=0)
                continue

            similiarity = DataStructs.TanimotoSimilarity(ecfp_array[idx], ecfp_array[idx + 1])
            y_temp[idx] = np.concatenate((y[idx], y[idx + 1], np.expand_dims(np.array(similiarity), axis=0)), axis=0)

        return X, y_temp

    @property
    def is_augmentation(self) -> bool:
        """Return if block is augmentation block.

        :return: is_block
        """
        return False
