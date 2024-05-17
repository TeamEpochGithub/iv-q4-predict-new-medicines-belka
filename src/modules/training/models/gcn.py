"""Module containing GCN model."""
import torch
from torch import nn
from torch_geometric.data import Data  # type: ignore[import-not-found]
from torch_geometric.nn import GCNConv, global_mean_pool  # type: ignore[import-not-found]


class GCNModel(nn.Module):
    """GCN model for graph classification.

    Each graph represents a molecule with nodes and edges.

    Parameters
    ----------
    num_node_features: int
        Number of features each node has.
    n_classes: int
        Number of classes for the classification task.
    """

    def __init__(self, num_node_features: int, n_classes: int) -> None:
        """Initialize the GCN model."""
        super().__init__()
        hidden_dim = 256
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the GCN model."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
