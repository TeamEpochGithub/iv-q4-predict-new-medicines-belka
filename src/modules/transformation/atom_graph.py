"""Compute the graph representation of the molecule."""

from dataclasses import dataclass

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
    def _torch_graph(smile: str) -> npt.NDArray[np.float32]:
        """Create the torch graph from the smile format.

        param smile: list containing the smile format
        return: list containing the atom and bond attributes
        """
        atom_attributes = torch.from_numpy(_atom_attribute(smile)).float()
        bond_index, bond_attributes = _bond_attribute(smile)
        bond_index_torch = torch.from_numpy(bond_index).long().t().contiguous()
        bond_attributes_torch = torch.from_numpy(bond_attributes).float()
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
    return: tensor containing the atom feature
    """
    # Extract the atoms from the smile
    mol = Chem.MolFromSmiles(smile)
    atoms = mol.GetAtoms()

    # Extract the attributes in the atom
    atom_features = [[atom.GetAtomicNum(), atom.GetDegree(), atom.GetHybridization(), atom.GetIsotope()] for atom in atoms]

    return np.array(atom_features)


def _bond_attribute(smile: str) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Extract the bond attribute from the smile.

    param smile: the molecule string format
    return: tensor containing the edge feature
    """
    # Extract the bonds from the smile
    mol = Chem.MolFromSmiles(smile)
    bonds = mol.GetBonds()

    # Extract the attributes and the edge index
    edge_features = []
    edge_index = []

    for bond in bonds:
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((start, end))
        edge_index.append((end, start))

        edge_features.append([int(bond.GetIsConjugated()), int(bond.IsInRing())])
        edge_features.append([int(bond.GetIsConjugated()), int(bond.IsInRing())])

    return np.array(edge_index), np.array(edge_features)
