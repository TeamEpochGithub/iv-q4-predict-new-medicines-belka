from torch.utils.data import Dataset
from src.typing.xdata import XData
from dataclasses import dataclass
import torch
import numpy as np
import numpy.typing as npt



@dataclass
class MoleculeText(Dataset):
    """Transforms the molecule into a sentence of atoms."""

    def __init__(self, X: XData, labels: npt.NDArray[np.float32]):
        """Initialize the torch dataset with smiles and labels

        param X: XData containing the smiles and tokenizer
        param y: array containing the molecule labels"""

        self.smiles = X.molecule_smiles
        self.labels = labels
        self.tokenizer = X.tokenizer

    def __len__(self):
        """Returns the total number of molecules in the dataset."""
        return len(self.smiles)

    def __getitem__(self, idx):
        """ Extract the sequence and labels for a molecule.
        param idx: index of the molecule in the dataset"""

        # convert the smiles into a sentence
        sentence = self.tokenizer(self.smiles[idx])

        # Convert the sequence and the labels to tensors
        sentence = torch.tensor(sentence)
        label = torch.tensor(self.labels[idx])

        return sentence, label
