"""Transforms the building blocks into images."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import Draw  # type: ignore[import-not-found]
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData


@dataclass
class BlocksToImage(VerboseTransformationBlock):
    """Transforms the building blocks into images.

    :param img_width: The width of the images representing the molecules
    :param img_heigth: The width of the images representing the molecules
    """

    img_width: int = 120
    img_heigth: int = 50

    def convert_smiles(self, smiles: npt.NDArray[np.str_]) -> npt.NDArray[np.float32]:
        """Transform the smiles into PIL images.

        :param smiles: Array containing the smile strings
        """
        images = []
        size = (self.img_width, self.img_heigth)
        for smile in tqdm(smiles, total=len(smiles), desc="Computing the images of the blocks"):
            mol = Chem.MolFromSmiles(smile)
            images.append(Draw.MolToImage(mol, size=size))
        return np.array(images)

    def custom_transform(self, X: XData) -> XData:
        """Transform the building blocks into images."""
        if X.bb1_smiles is None or X.bb2_smiles is None or X.bb3_smiles is None:
            raise ValueError("Missing embedding representation of the building block")

        X.bb1_embedding = self.convert_smiles(X.bb1_smiles)
        X.bb2_embedding = self.convert_smiles(X.bb2_smiles)
        X.bb3_embedding = self.convert_smiles(X.bb3_smiles)

        return X
