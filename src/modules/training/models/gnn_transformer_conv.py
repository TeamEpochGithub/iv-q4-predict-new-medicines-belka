"""GCN with Transformer Convolutions Module."""

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool

from src.modules.training.dataset_steps.graphs.smiles_to_graph import unpack_atom_features, unpack_edge_features


class GNNTransformerModel(torch.nn.Module):
    """Transformer GCN Module.

    :param num_node_features: Number of features per node
    :param num_edge_features: Number of edge features per node
    :param n_classes:  Number of classes to predict
    """

    def __init__(self, num_node_features: int, num_edge_features: int, n_classes: int, hidden_dim: int = 32, out_features: int = 1024, dropout: float = 0.2) -> None:
        """Initialize the GCN model.

        :param num_node_features: Number of features per node
        :param num_edge_features: Number of edge features per node
        :param n_classes:  Number of classes to predict
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.conv1 = TransformerConv(in_channels=num_node_features, out_channels=hidden_dim, heads=4, concat=False,
                                     edge_dim=num_edge_features, dropout=dropout)
        self.conv2 = TransformerConv(in_channels=hidden_dim, out_channels=hidden_dim * 2, heads=4, concat=False,
                                     edge_dim=num_edge_features, dropout=dropout)

        self.pool = global_mean_pool

        self.fc1 = nn.Linear(hidden_dim * 2, out_features)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_features, out_features)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(out_features, out_features // 2)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(out_features // 2, n_classes)

        # Define activation function
        self.relu = nn.ReLU()

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the GCN model.

        :param data: Input data
        :return Output data
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Unpack Node and Edge Attributes
        x = unpack_atom_features(x).to(torch.float)
        edge_attr = unpack_edge_features(edge_attr).to(torch.float)

        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = self.relu(self.conv2(x, edge_index, edge_attr))

        # Pooling layer
        x = self.pool(x, batch)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)

        return self.fc4(x)
