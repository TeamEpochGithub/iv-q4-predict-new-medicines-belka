"""Module to augment atom encoding."""
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.model.training.training_block import TrainingBlock

from src.modules.chemistry_functions.ecfp_parallel import convert_smile_array_parallel, convert_smiles_array_not_packed
from src.modules.logging.logger import Logger


@dataclass
class ECFP(TrainingBlock, Logger):
    """Fingerprint generation in dataset."""

    bits: int = 128
    radius: int = 2
    use_features: bool = False

    def train(self, X: npt.NDArray[np.string_], y: npt.NDArray[np.int_]) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int_]]:
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
    multi_processing: bool = field(default=False, init=True, repr=False, compare=False)

    def train(self, X: npt.NDArray[np.string_], y: npt.NDArray[np.int_]) -> tuple[npt.NDArray[np.string_], npt.NDArray[np.int_]]:
        """Replace some atoms with similar ones.

        :param X: Input array
        :param y: Labels
        """
        if X is None:
            raise ValueError("There is no SMILE information for the molecules, can't convert to ECFP")
        if self.multi_processing:
            result = convert_smile_array_parallel(smiles_array=X, bits=self.bits, radius=self.radius, use_features=self.use_features, desc="Creating ECFP for molecules")
        else:
            result = convert_smiles_array_not_packed(smiles_array=X, bits=self.bits, radius=self.radius, use_features=self.use_features)

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
