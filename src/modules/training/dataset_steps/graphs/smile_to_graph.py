"""Module to turn smile representation of molecule into graph representation."""
from dataclasses import dataclass
from typing import Any

from epochalyst.pipeline.model.training.training_block import TrainingBlock
from rdkit import Chem  # type: ignore[import-not-found]
from torch_geometric.data import Data
import numpy as np
import numpy.typing as npt
import torch


@dataclass
class SmileToGraph(TrainingBlock):
    """Turn smile representation into graph"""

    use_bond_atttributes: bool = False

    def train(
        self,
        x: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        **train_args: Any
    ) -> list[Data]:
        """Transform smile input into graph.

        :param x: The x molecule data
        :param y: The binding data
        :return: List of molecule graphs
        """
        return self._smiles_to_graph(x, y), None

    @staticmethod
    def _smile_to_graph(smile: str, label: npt.NDArray[np.uint8] | None = None, use_bond_atttributes: bool = False) -> Data:
        """Create the torch graph from the smile format.

        :param smile: list containing the smile format
        :param use_bond_atttributes: Use the bond attributes in the graph
        :return: list containing the atom and bond attributes
        """
        atom_attributes = torch.from_numpy(_atom_attribute(smile)).float()
        bond_index, bond_attributes = _bond_index_attr(smile, use_bond_atttributes)
        bond_index = torch.from_numpy(bond_index).long().t().contiguous()
        if use_bond_atttributes:
            bond_attributes = torch.from_numpy(bond_attributes).float()
        if label is not None:
            label = torch.from_numpy(label).int()

        graph = Data(x=atom_attributes, edge_index=bond_index, edge_attr=bond_attributes if use_bond_atttributes else None, y=label if label is not None else None)
        return graph

    def _smiles_to_graph(self, smiles: npt.NDArray[np.str_], y: npt.NDArray[np.uint8] | None = None) -> list[Data]:
        """Transform smile to graph representation.

        :param smiles: The smiles to transform"""
        graphs = []
        for i, smile in enumerate(smiles):
            curr_y = y[i] if y is not None else None
            graphs.append(self._smile_to_graph(smile, curr_y, self.use_bond_atttributes))

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

    return np.array(atom_features)


def _bond_index_attr(smile: str, get_bond_attributes: bool = False) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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

        if get_bond_attributes:
            edge_features.append([int(bond.GetIsConjugated()), int(bond.IsInRing())])
            edge_features.append([int(bond.GetIsConjugated()), int(bond.IsInRing())])

    return np.array(edge_index), np.array(edge_features)
