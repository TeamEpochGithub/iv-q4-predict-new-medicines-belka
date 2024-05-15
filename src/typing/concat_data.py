from torch.utils.data import DataLoader, Dataset
from src.typing.xdata import  XData


class MoleculeText(Dataset):
    """Transforms the molecule into a sequence of strings."""

    def __init__(self, sequences, labels):
        """Initialize the torch dataset with the sequences

        param sequences: array containing the molecule tokens
        param labels: array containing the molecule labels"""

        self.sequences = sequences
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