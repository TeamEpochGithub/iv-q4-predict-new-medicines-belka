"""Converts the molecule smiles into a sequence of consecutive tokens."""

from dataclasses import dataclass
from rdkit import Chem
import re
import numpy.typing as npt
import numpy as np
from tqdm import tqdm
import joblib
from src.typing.xdata import XData
from concurrent.futures import ProcessPoolExecutor
from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock

@dataclass
class SegmentMolecule(VerboseTransformationBlock):
    """Converts the molecule smiles into a sequence of consecutive tokens.
    param window_size: the size of each atom combinations"""

    window_size: int = 6
    padding_size: int = 150
    chunk_size: int = 10000

    @staticmethod
    def segment_molecule(smiles: list[str],window_size:int,padding_size:int) -> npt.NDArray[np.str_]:
        """perform the convolution operation on each smile
        param smile: the smile string of the molecule
        """
        sequences = []
        for smile in smiles:
            # remove the branches from the smile
            smile = smile.replace('(', '').replace(')', '')

            # Extract n-grams from the sequence
            length = len(smile) - window_size + 1
            sequence = [" ".join(smile[i:i + window_size]) for i in range(length)]

            # Pad the sequence with special token
            sequences.append(sequence + ['PAD'] * (padding_size - len(sequence)))

        return sequences

    def custom_transform(self, data: XData) -> XData:
        """Compute the embeddings of the molecules in training.
        param data: the training or test set
        """
        # Check whether molecule smiles are present
        if data.molecule_smiles is None:
            raise ValueError("There is no SMILE information for the molecules")

        # Divide the smiles molecules into chunks
        smiles = list(data.molecule_smiles)
        chunks = [smiles[i : i + self.chunk_size] for i in range(0, len(smiles), self.chunk_size)]

        # Initialize the multiprocessing with the chunks
        results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.segment_molecule,smiles=chunk,window_size=self.window_size,padding_size=self.padding_size) for chunk in chunks]

            # Perform the multiprocessing on the chunks
            desc = "Perform convolution on the smiles"
            for future in tqdm(futures, total=len(futures), desc=desc):
                results.extend(future.result())

        data.molecule_smiles = np.array(results)
        return data

