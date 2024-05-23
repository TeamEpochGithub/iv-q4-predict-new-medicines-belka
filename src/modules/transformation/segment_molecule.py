"""Converts the molecule smiles into a sequence of consecutive tokens."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData


@dataclass
class SegmentMolecule(VerboseTransformationBlock):
    """Converts the molecule smiles into a sequence of consecutive tokens.

    param window_size: the size of each token in the sequence
    """

    window_size: int = 6
    padding_size: int = 50
    chunk_size: int = 10

    @staticmethod
    def segment_molecule(smiles: npt.NDArray[np.str_], window_size: int, padding_size: int) -> npt.NDArray[np.str_]:
        """Perform the convolution operation on each smile.

        param smiles: list containing the smile molecules
        """
        sequences = []
        for smile in smiles:
            # remove the branches from the smile
            new_smile = smile.replace("(", "").replace(")", "")

            # Extract n-grams from the sequence
            length = len(new_smile) - window_size + 1
            sequence = [" ".join(smile[i : i + window_size]) for i in range(length)]

            # Pad the sequence with special token
            sequences.append(sequence + ["PAD"] * (padding_size - len(sequence)))

        return np.array(sequences)

    def parallel_segment(self, smiles: npt.NDArray[np.str_], padding_size: int) -> npt.NDArray[np.str_]:
        """Perform the convolution operation using multiprocessing.

        param smiles: list containing the smile molecules
        """
        # Divide the smiles molecules into chunks
        chunks = [smiles[i : i + self.chunk_size] for i in range(0, len(smiles), self.chunk_size)]

        # Initialize the multiprocessing with the chunks
        results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.segment_molecule, smiles=chunk, window_size=self.window_size, padding_size=padding_size) for chunk in chunks]

            # Perform the multiprocessing on the chunks
            desc = "Perform convolution on the smiles"
            for future in tqdm(futures, total=len(futures), desc=desc):
                results.extend(future.result())

        return np.array(results)

    def custom_transform(self, data: XData) -> XData:
        """Compute the embeddings of the molecules in training.

        param data: the training or test set
        """
        # Check whether the building blocks are present
        if data.bb1_smiles is None or data.bb2_smiles is None or data.bb3_smiles is None:
            raise ValueError("There is no SMILE information for the molecules")

        # Perform the convolutional operation on each block
        data.bb1_smiles = self.parallel_segment(data.bb1_smiles, 60)
        data.bb2_smiles = self.parallel_segment(data.bb2_smiles, 40)
        data.bb3_smiles = self.parallel_segment(data.bb3_smiles, 50)

        return data
