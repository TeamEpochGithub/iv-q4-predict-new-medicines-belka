"""Module containing GCN model."""
import torch
from torch import nn
from torch_geometric.data import Data  # type: ignore[import-not-found]
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing  # type: ignore[import-not-found]
from torch_geometric.utils import add_self_loops  # type: ignore[import-not-found]

class MPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNN, self).__init__(aggr='add')  # "Add" aggregation (can be "mean" or "max")
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.edge_update = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Start propagating messages
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]
        return self.lin(x_j)

    def update(self, aggr):
        # aggr has shape [N, out_channels]
        return self.edge_update(aggr)


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
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the GCN model."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout2(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        return self.fc2(x)
