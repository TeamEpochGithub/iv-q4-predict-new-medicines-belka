"""Module that adds one hot encoding of protein to XData and flattens y."""
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.typing.xdata import XData


class OneHotProteinBB(VerboseTrainingBlock):
    """Add the one hot encoding of protein to XData building block ecfp at the start."""

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

        x.retrieval = "ECFP_BB"

        result = [np.concatenate((protein, x[i][0], x[i][1], x[i][2])) for i in tqdm(range(len(x.building_blocks)), desc="Concatenating Protein to building_blocks") for protein in protein_onehot]
        x.molecule_ecfp = result

        return x
