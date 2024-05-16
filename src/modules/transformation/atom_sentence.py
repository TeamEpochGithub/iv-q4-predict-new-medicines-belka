"""Converts the molecule smiles into a sequence of atom combinations."""
#%%
from dataclasses import dataclass
from rdkit import Chem
import numpy.typing as npt
import numpy as np
from tqdm import tqdm
import joblib
from src.typing.xdata import XData
from concurrent.futures import ProcessPoolExecutor
from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock

@dataclass
class AtomSentence(VerboseTransformationBlock):
    """Converts the molecule smiles into a sequence of atom combinations.
    param window_size: the size of each atom combinations"""

    window_size: int = 6
    padding_size: int = 140
    def segment_molecule(self, smile: str) -> npt.NDArray[np.str_]:
        """Transform the molecule into a sequence of tokens.
        param smile: the smile string of the molecule
        """

        # Convert the smile to the molecule object
        mol = Chem.MolFromSmiles(smile)

        # Extract the atoms from the molecule
        tokens = [atom.GetSymbol() for atom in mol.GetAtoms()]

        # Extract n-grams from the sequence
        length = len(tokens) - self.window_size + 1
        sequence = [" ".join(tokens[i:i + self.window_size]) for i in range(length)]

        # Pad the sequence with special token
        return np.array(sequence + ['PAD'] * (self.padding_size - len(sequence)))


    def custom_transform(self, data: XData) -> XData:
        """Compute the embeddings of the molecules in training.
        param data: the training or test set
        """
        # Check whether molecule smiles are present
        if data.molecule_smiles is None:
            raise ValueError("There is no SMILE information for the molecules")

        # Extract the molecule smile from XData
        desc = "compute the embeddings of the molecule"
        tqdm_smiles = tqdm(list(data.molecule_smiles), desc=desc)

        encoded = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(self.segment_molecule)(smile) for smile in tqdm_smiles)
        encoded = np.stack(encoded)

        data.molecule_smiles = encoded
        return data

