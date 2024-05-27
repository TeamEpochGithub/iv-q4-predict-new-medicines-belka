"""Module containing custom GraphDataParallel class."""
from torch_geometric.nn import DataParallel


class GraphDataParallel(DataParallel):
    """Custom graph DataParallel class."""

    def __repr__(self) -> str:
        """Return the representation of the module. This is to get the same hash for the model with and without DataParallel."""
        return self.module.__repr__()
