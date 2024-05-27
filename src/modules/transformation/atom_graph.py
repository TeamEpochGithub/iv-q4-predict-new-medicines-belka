"""Compute the graph representation of the molecule."""
import math
from dataclasses import dataclass
from typing import Tuple

import joblib
import numpy as np
import numpy.typing as npt
import torch
from rdkit import Chem  # type: ignore[import-not-found]
from torch_geometric.data import Data
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData

NUM_FUTURES = 100
MIN_CHUNK_SIZE = 1000

@dataclass
class AtomGraph(VerboseTransformationBlock):
    """Create a torch geometric graph from the molecule.

    param convert_molecule: whether to convert the molecule
    param convert_building_blocks: whether to convert the building blocks
    """

    convert_molecule: bool = True
    convert_building_blocks: bool = True

    @staticmethod
    def _torch_graph(smile: str) -> Data:
        """Create the torch graph from the smile format.

        param smile: list containing the smile format
        return: Data object containing the atom and bond attributes
        """
        atom_attributes = torch.from_numpy(_atom_attribute(smile)).float()
        bond_index, bond_attributes = _bond_attribute(smile)
        bond_index_torch = torch.from_numpy(bond_index).long().t().contiguous()
        bond_attributes_packed = pack_bits(bond_attributes)
        bond_attributes_torch = torch.from_numpy(bond_attributes_packed).to(torch.uint8)
        return Data(x=atom_attributes, edge_index=bond_index_torch, edge_attr=bond_attributes_torch)

    def custom_transform(self, data: XData) -> XData:
        """Create a torch geometric graph from the molecule."""
        # Compute the embeddings for each molecule
        if self.convert_molecule and data.molecule_smiles is not None:
            molecule_graphs = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(self._torch_graph)(smile) for smile in tqdm(data.molecule_smiles, desc="Creating atomic graph for molecules")
            )
            data.molecule_graph = molecule_graphs

        # Compute the embeddings for each block
        if self.convert_building_blocks and data.bb1_smiles is not None and data.bb2_smiles is not None and data.bb3_smiles is not None:
            data.bb1_graph = joblib.Parallel(n_jobs=-1)(joblib.delayed(self._torch_graph)(smile) for smile in tqdm(data.bb1_smiles, desc="Creating atomic graph for bb1"))
            data.bb2_graph = joblib.Parallel(n_jobs=-1)(joblib.delayed(self._torch_graph)(smile) for smile in tqdm(data.bb2_smiles, desc="Creating atomic graph for bb2"))
            data.bb3_graph = joblib.Parallel(n_jobs=-1)(joblib.delayed(self._torch_graph)(smile) for smile in tqdm(data.bb3_smiles, desc="Creating atomic graph for bb3"))

        return data

def _atom_attribute(smile: str) -> npt.NDArray[np.float32]:
    """Extract the atom attribute from the smile.

    param smile: the molecule string format
    return: array containing the atom features
    """
    mol = Chem.MolFromSmiles(smile)
    atoms = mol.GetAtoms()

    atom_features = [[atom.GetAtomicNum(), atom.GetDegree(), int(atom.GetHybridization()), int(atom.GetIsotope())] for atom in atoms]

    return np.array(atom_features, dtype=np.float32)

def _bond_attribute(smile: str) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.uint8]]:
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

def pack_bits(tensor: npt.NDArray[np.uint8], dim: int = -1, mask: int = 0b00000001, dtype=torch.uint8) -> npt.NDArray[np.uint8]:
    """Pack bits of the input numpy array."""
    shape, nibbles, nibble = packshape(tensor.shape, dim=dim, mask=mask, dtype=dtype, pack=True)
    packed_tensor = np.packbits(tensor, axis=dim, bitorder='little')
    return packed_tensor

def unpack_bits(tensor: torch.Tensor, dim: int = -1, mask: int = 0b00000001, dtype=torch.uint8) -> torch.Tensor:
    """Unpack bits from the input tensor."""
    return F_unpackbits(tensor, dim=dim, mask=mask, dtype=dtype)

def packshape(shape, dim: int = -1, mask: int = 0b00000001, dtype=torch.uint8, pack=True):
    dim = dim if dim >= 0 else dim + len(shape)
    bits = 8  # We're working with torch.uint8
    nibble = 1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else 0
    assert nibble <= bits and bits % nibble == 0
    nibbles = bits // nibble
    shape = (shape[:dim] + (int(np.ceil(shape[dim] / nibbles)),) + shape[dim + 1:]) if pack else (
            shape[:dim] + (shape[dim] * nibbles,) + shape[dim + 1:])
    return shape, nibbles, nibble

def F_unpackbits(tensor, dim: int = -1, mask: int = 0b00000001, shape=None, out=None, dtype=torch.uint8):
    dim = dim if dim >= 0 else dim + tensor.dim()
    shape_, nibbles, nibble = packshape(tensor.shape, dim=dim, mask=mask, dtype=tensor.dtype, pack=False)
    shape = shape if shape is not None else shape_
    out = out if out is not None else torch.empty(shape, device=tensor.device, dtype=dtype)
    assert out.shape == shape

    if shape[dim] % nibbles == 0:
        shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype=torch.uint8, device=tensor.device)
        shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
        return torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out=out)
    else:
        for i in range(nibbles):
            shift = nibble * i
            sliced_output = tensor_dim_slice(out, dim, slice(i, None, nibbles))
            sliced_input = tensor.narrow(dim, 0, sliced_output.shape[dim])
            torch.bitwise_and(sliced_input >> shift, mask, out=sliced_output)
    return out

def tensor_dim_slice(tensor, dim, dim_slice):
    return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None),) + (dim_slice,)]