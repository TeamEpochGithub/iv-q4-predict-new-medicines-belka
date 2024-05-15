from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from src.typing.xdata import XData

@dataclass
class MoleculeText(Dataset):
    """Transforms the molecule into a sequence of strings."""

    def __init__(self, X: XData, labels):
        """Initialize the torch dataset with smiles and labels

        param X: XData containing the smiles and tokenizer
        param y: array containing the molecule labels"""

        self.smiles = X
        self.labels = labels

    def __len__(self):
        """Returns the total number of molecules in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """ Extract the sequence and labels for a molecule."""

        # Convert the sequence and the labels to tensors
        sequence = torch.tensor(self.sequences[idx])
        label = torch.tensor(self.labels[idx])

        return sequence, label