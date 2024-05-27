"""Module to turn smiles into images."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import Draw  # type: ignore[import-not-found]


@dataclass
class SmilesToImage(TrainingBlock):
    """Turn smiles into images.

    :param img_size: The Image size of the molecules
    """

    img_size: int = 224

    def train(
        self,
        X: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        """Turn molecule smiles into an img representation.

        :param X: The smiles strings
        :param y: The labels for the molecules
        :return: Image
        """
        images = [self.smiles_to_image(smiles, self.img_size) for smiles in X]

        return np.array(images), y

    @staticmethod
    def smiles_to_image(smiles: str, img_size: int) -> npt.NDArray[np.float32]:
        """Turn a molecule smiles into an image.

        :param smiles: String representation of molecule
        :param img_size: The size of the image to output
        :return: Image array
        """
        mol = Chem.MolFromSmiles(smiles)
        return Draw.MolToImage(mol, size=(img_size, img_size))

    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return False
