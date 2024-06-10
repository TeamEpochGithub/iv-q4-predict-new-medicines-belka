"""GCN with Transformer Convolutions Module."""

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool



class GNNTransformerModel(torch.nn.Module):
    """Transformer GCN Module.

    :param num_node_features: Number of features per node
    :param num_edge_features: Number of edge features per node
    :param n_classes:  Number of classes to predict
    """

    def __init__(self, num_node_features: int, num_edge_features: int, n_classes: int, hidden_dim: int = 32, out_features: int = 1024, dropout : float = 0.1) -> None:
        """Initialize the GCN model.

        :param num_node_features: Number of features per node
        :param num_edge_features: Number of edge features per node
        :param n_classes:  Number of classes to predict
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.conv1 = TransformerConv(in_channels=num_node_features, out_channels=hidden_dim, heads=4, concat=False, edge_dim=num_edge_features, dropout=dropout)
        self.conv2 = TransformerConv(in_channels=hidden_dim, out_channels=hidden_dim * 2, heads=4, concat=False, edge_dim=num_edge_features, dropout=dropout)

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

        edge_attr = unpack_bits(edge_attr.to(torch.uint8)).float()

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

def unpack_bits(tensor: torch.Tensor, dim: int = -1, mask: int = 0b00000001, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    """Unpack bits tensor into bits tensor."""
    return f_unpackbits(tensor, dim=dim, mask=mask, dtype=dtype)


def f_unpackbits(
    tensor: torch.Tensor,
    dim: int = -1,
    mask: int = 0b00000001,
    shape: tuple[int, ...] | None = None,
    out: torch.Tensor | None = None,
    dtype: torch.dtype = torch.uint8,
) -> torch.Tensor:
    """Unpack bits tensor into bits tensor."""
    dim = dim if dim >= 0 else dim + tensor.dim()
    shape_, nibbles, nibble = packshape(tensor.shape, dim=dim, mask=mask, pack=False)
    shape = shape if shape is not None else shape_
    out = out if out is not None else torch.empty(shape, device=tensor.device, dtype=dtype)

    if shape[dim] % nibbles == 0:
        shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype=torch.uint8, device=tensor.device)
        shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
        return torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out=out)

    for i in range(nibbles):
        shift = nibble * i  # type: ignore[assignment]
        sliced_output = tensor_dim_slice(out, dim, slice(i, None, nibbles))
        sliced_input = tensor.narrow(dim, 0, sliced_output.shape[dim])
        torch.bitwise_and(sliced_input >> shift, mask, out=sliced_output)
    return out


def packshape(shape: tuple[int, ...], dim: int = -1, mask: int = 0b00000001, *, pack: bool = True) -> tuple[tuple[int, ...], int, int]:
    """Define pack shape."""
    dim = dim if dim >= 0 else dim + len(shape)
    bits = 8
    nibble = 1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else 0
    nibbles = bits // nibble
    shape = (shape[:dim] + (int(np.ceil(shape[dim] / nibbles)),) + shape[dim + 1 :]) if pack else (shape[:dim] + (shape[dim] * nibbles,) + shape[dim + 1 :])
    return shape, nibbles, nibble


def tensor_dim_slice(tensor: torch.Tensor, dim: int, dim_slice: slice) -> torch.Tensor:
    """Slices a tensor for packing."""
    return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None),) + (dim_slice,)]

