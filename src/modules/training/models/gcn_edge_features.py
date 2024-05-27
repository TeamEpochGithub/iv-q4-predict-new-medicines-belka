"""GCN Edge Features Module."""

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import NNConv, global_mean_pool


class GCNWithEdgeFeatures(nn.Module):
    """GCN Model with Edge Features."""

    def __init__(self, num_node_features: int, num_edge_features: int, n_classes: int, hidden_dim: int = 64) -> None:
        """Initialize the GCN model."""
        super().__init__()
        self.hidden_dim = hidden_dim

        edge_net1 = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim * num_node_features),
            nn.ReLU(),
            nn.Linear(hidden_dim * num_node_features, hidden_dim * num_node_features),
        )

        edge_net2 = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim * 2),
        )

        edge_net3 = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim * hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim * 2, 128 * hidden_dim * 4),
        )

        self.conv1 = NNConv(num_node_features, hidden_dim, edge_net1, aggr="mean")
        self.conv2 = NNConv(hidden_dim, hidden_dim * 2, edge_net2, aggr="mean")
        self.conv3 = NNConv(hidden_dim * 2, hidden_dim * 4, edge_net3, aggr="mean")

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
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = self.relu(self.conv2(x, edge_index, edge_attr))
        x = self.relu(self.conv3(x, edge_index, edge_attr))

        x = global_mean_pool(x, batch)

        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        return self.fc4(x)
