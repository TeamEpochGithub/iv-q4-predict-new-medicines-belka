""" Module containing GCN model"""
import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

class GCNModel(nn.Module):
    """GCN model for graph classification.

    Each graph represents a molecule with nodes and edges.

    Parameters:
    ----------
        hidden_dim: Dimensionality of hidden layers.
        n_classes: Number of classes for the classification task.
        num_node_features: Number of features each node has.
    """

    def __init__(self, num_node_features: int, n_classes: int):
        super(GCNModel, self).__init__()
        hidden_dim = 256
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # First Conv
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        # Second Graph Convolution
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        # Global Pooling (mean pooling) for each batch
        x = global_mean_pool(x, batch)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
