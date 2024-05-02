"""Module that adds one hot encoding of protein to XData and flattens y."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.typing.xdata import XData

NUM_FUTURES = 100
MIN_CHUNK_SIZE = 1000


class OneHotProtein(VerboseTrainingBlock):
    """Add the one hot encoding of protein to XData molecule smiles at the start.

    :param num_futures: The number of futures to use
    :param min_chunk_size: The minimum chunk size
    """

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

    @staticmethod
    def _concatenate_protein(chunk: list[Any]) -> list[npt.NDArray[Any]]:
        """Concatenate protein to molecule.

        :param protein: Protein one hot encoding
        :param mol: Molecule ECFP
        :return: Concatenated protein and molecule
        """
        BRD4 = np.array([1, 0, 0])
        HSA = np.array([0, 1, 0])
        sEH = np.array([0, 0, 1])
        protein_onehot = [BRD4, HSA, sEH]

        return [np.concatenate((protein, mol)) for mol in chunk for protein in protein_onehot]

    def add_protein_to_xdata(self, x: XData) -> XData:
        """Add protein to XData.

        :param x: XData
        :return: XData
        """
        x.retrieval = "ECFP"

        # Check if molecule_ecfp exists
        if x.molecule_ecfp is None:
            raise ValueError("Molecule ECFP representation does not exist.")

        result = []
        chunk_size = len(x.molecule_ecfp) // NUM_FUTURES
        chunk_size = max(chunk_size, MIN_CHUNK_SIZE)
        chunks = [x.molecule_ecfp[i : i + chunk_size] for i in range(0, len(x.molecule_ecfp), chunk_size)]

        with ProcessPoolExecutor() as executor:
            self.log_to_terminal("Creating futures for One Hot Protein concatenation.")
            futures = [executor.submit(self._concatenate_protein, chunk) for chunk in chunks]
            for future in tqdm(futures, total=len(futures), desc="Concatenating Protein to molecules"):
                result.extend(future.result())

        x.molecule_ecfp = result

        return x
