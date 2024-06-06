"""GCN Edge Features Module."""

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import NNConv, global_mean_pool


class GCNWithEdgeFeatures(nn.Module):
    """GCN Model with Edge Features."""

    def __init__(self, num_node_features: int, num_edge_features: int, n_classes: int, hidden_dim: int = 32) -> None:
        """Initialize the GCN model."""
        super().__init__()
        self.hidden_dim = hidden_dim

        # Define edge networks
        self.edge_net1 = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim * num_node_features),
            nn.ReLU(),
            nn.Linear(hidden_dim * num_node_features, hidden_dim * num_node_features),
        )

        self.edge_net2 = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim * hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim * 2),
        )

        self.conv1 = NNConv(num_node_features, hidden_dim, self.edge_net1, aggr="mean")
        self.conv2 = NNConv(hidden_dim, hidden_dim * 2, self.edge_net2, aggr="mean")
        self.pool = global_mean_pool

        self.fc1 = nn.Linear(hidden_dim * 2, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(256, n_classes)

        # Define activation function
        self.relu = nn.ReLU()

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the GCN model."""
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
