"""Module that adds one hot encoding of protein to XData and flattens y."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.typing.xdata import DataRetrieval, XData


@dataclass
class OneHotProteinBB(VerboseTrainingBlock):
    """Add the one hot encoding of protein to XData building block ecfp at the start.

    :param data: The data to use
    """

    data: list[str] = field(default_factory=lambda: ["ECFP_BB"])

    def __post_init__(self) -> None:
        """Post init method."""
        if self.data[0] != "ECFP_BB":
            raise ValueError("Currently, only 'ECFP_BB' is suported.")

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

    def add_protein_to_xdata(self, X: XData) -> XData:
        """Add protein to XData.

        :param x: XData
        :return: XData
        """
        BRD4 = np.array([1, 0, 0])
        HSA = np.array([0, 1, 0])
        sEH = np.array([0, 0, 1])
        protein_onehot = [BRD4, HSA, sEH]

        X.retrieval = DataRetrieval.ECFP_BB

        result = np.array(
            [np.concatenate((protein, X[i][0], X[i][1], X[i][2])) for i in tqdm(range(len(X)), desc="Concatenating Protein to building_blocks") for protein in protein_onehot],
        )
        X.molecule_ecfp = result

        return X
