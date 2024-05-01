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

        x.retrieval = "ECFP"

        # Check if molecule_ecfp exists
        if x.molecule_ecfp is None:
            raise ValueError("Molecule ECFP representation does not exist.")

        result = [np.concatenate((protein, mol)) for mol in x.molecule_ecfp for protein in protein_onehot]
        x.molecule_ecfp = result

        return x
