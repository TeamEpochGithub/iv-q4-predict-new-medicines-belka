"""Module that adds one hot encoding of protein to XData and flattens y."""
from src.modules.training.verbose_training_block import VerboseTrainingBlock
import numpy.typing as npt
import numpy as np
from src.typing.xdata import XData


class OneHotProtein(VerboseTrainingBlock):
    """Add the one hot encoding of protein to XData molecule smiles at the start."""

    def custom_train(self, x: XData, y: npt.NDArray[np.int8]) -> tuple[XData, npt.NDArray[np.int8]]:
        """Add one hot encoding of protein to XData and flatten y.

        :param x: XData
        :param y: The labels
        :return: New XData and flattened labels"""
        return self.add_protein_to_xdata(x), y.flatten()

    def custom_predict(self, x: XData) -> XData:
        """Add one hot encoding of protein to XData.

        :param x: XData
        :return: XData
        """
        return self.add_protein_to_xdata(x)

    def add_protein_to_xdata(self, x: XData) -> XData:
        """Add protein to XData.

        :param x: XData
        :return: XData
        """
        BRD4 = np.array([1, 0, 0])
        HSA = np.array([0, 1, 0])
        sEH = np.array([0, 0, 1])
        protein_onehot = [BRD4, HSA, sEH]

        # Check if molecule smiles are np arrays otherwise encoding cannot be appended
        if not isinstance(x.molecule_smiles[0], np.ndarray):
            raise ValueError("Molecule data is not numpy arrays, cannot add protein encoding")

        x.molecule_smiles = np.array([np.concatenate((protein, mol)) for mol, protein in zip(x.molecule_smiles, protein_onehot)])

        return x
