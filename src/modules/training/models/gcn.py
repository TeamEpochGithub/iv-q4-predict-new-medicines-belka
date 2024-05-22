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

    def __init__(self, num_node_features: int, n_classes: int, hidden_dim: int = 64) -> None:
        """Initialize the GCN model."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)

        self.pool = global_mean_pool

        # Dense and Dropout layers
        self.fc1 = nn.Linear(hidden_dim * 4, 1024)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(512, n_classes)

        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the GCN model."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)

        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        return self.fc4(x)
