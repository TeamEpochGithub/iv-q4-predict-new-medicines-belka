"""Module to turn smile representation of molecule into graph representation."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from rdkit import Chem  # type: ignore[import-not-found]
from torch_geometric.data import Data


@dataclass
class SmilesToChemGraph(TrainingBlock):
    """Turn smile representation into graph."""

    use_bond_attributes: bool = True

    def train(
        self,
        x: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        **train_args: Any,
    ) -> tuple[list[Data], None]:
        """Transform smile input into graph.

        :param x: The x molecule data
        :param y: The binding data
        :return: List of molecule graphs
        """
        return self._smiles_to_graph(x, y), None

    @staticmethod
    def _smile_to_graph(smile: str, label: npt.NDArray[np.uint8] | None = None, *, use_bond_attributes: bool = True) -> Data:
        """Create the torch graph from the smile format.

        :param smile: list containing the smile format
        :param use_bond_attributes: Use the bond attributes in the graph
        :return: list containing the atom and bond attributes
        """
        atom_attributes = torch.from_numpy(_atom_attribute(smile)).float()
        bond_index, bond_attributes = _bond_attribute(smile)
        bond_index_torch = torch.from_numpy(bond_index).long().t().contiguous()
        bond_attributes_packed = pack_bits(bond_attributes)
        if use_bond_attributes:
            bond_attributes_torch = torch.from_numpy(bond_attributes_packed).to(torch.uint8)
        if label is not None:
            label_torch = torch.from_numpy(label).int()

        return Data(
            x=atom_attributes,
            edge_index=bond_index_torch,
            edge_attr=bond_attributes_torch,
            y=label_torch if label is not None else None,
        )

    def _smiles_to_graph(self, smiles: npt.NDArray[np.str_], y: npt.NDArray[np.uint8] | None = None) -> list[Data]:
        """Transform smile to graph representation.

        :param smiles: The smiles to transform
        """
        graphs = []
        for i, smile in enumerate(smiles):
            curr_y = y[i] if y is not None else None
            graphs.append(self._smile_to_graph(smile, curr_y, use_bond_attributes=self.use_bond_attributes))

        return graphs

    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return False


def _atom_attribute(smile: str) -> npt.NDArray[np.float32]:
    """Extract the atom attribute from the smile.

    param smile: the molecule string format
    return: tensor containing the atom feature
    """
    # Extract the atoms from the smile
    mol = Chem.MolFromSmiles(smile)
    atoms = mol.GetAtoms()

    # Extract the attributes in the atom
    atom_features = [[atom.GetAtomicNum(), atom.GetDegree(), atom.GetHybridization(), atom.GetIsotope()] for atom in atoms]

    return np.array(atom_features, dtype=np.float32)


def _bond_attribute(smile: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.uint8]]:
    """Extract the bond attribute from the smile.

    param smile: the molecule string format
    return: tuple containing the edge index and edge features
    """
    mol = Chem.MolFromSmiles(smile)
    bonds = mol.GetBonds()

    edge_features = []
    edge_index = []

    for bond in bonds:
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((start, end))
        edge_index.append((end, start))

        edge_features.append([int(bond.GetIsConjugated()), int(bond.IsInRing())])
        edge_features.append([int(bond.GetIsConjugated()), int(bond.IsInRing())])

    edge_index_np = np.array(edge_index, dtype=np.int64)
    edge_features_np = np.array(edge_features, dtype=np.uint8)

    return edge_index_np, edge_features_np


def pack_bits(tensor: npt.NDArray[np.uint8], dim: int = -1, mask: int = 0b00000001) -> npt.NDArray[np.uint8]:
    """Pack bits of the input numpy array."""
    shape, nibbles, nibble = packshape(tensor.shape, dim=dim, mask=mask, pack=True)
    return np.packbits(tensor, axis=dim, bitorder="little")


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
