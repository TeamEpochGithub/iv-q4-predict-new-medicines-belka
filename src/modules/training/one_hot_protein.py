"""Module that adds one hot encoding of protein to XData and flattens y."""
import numpy as np
import numpy.typing as npt

from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.typing.xdata import XData


class OneHotProtein(VerboseTrainingBlock):
    """Add the one hot encoding of protein to XData molecule smiles at the start."""

    def custom_train(self, x: XData, y: npt.NDArray[np.int8]) -> tuple[XData, npt.NDArray[np.int8]]:
        """Add one hot encoding of protein to XData and flatten y.

        :param x: XData
        :param y: The labels
        :return: New XData and flattened labels
        """
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
            raise TypeError("Molecule data is not numpy arrays, cannot add protein encoding")

        result = []
        for mol in x.molecule_smiles:
            for protein in protein_onehot:
                result.append(np.concatenate((protein, mol)))
        x.molecule_smiles = np.array(result)

        return x
