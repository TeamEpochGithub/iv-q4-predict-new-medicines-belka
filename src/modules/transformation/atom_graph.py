"""Compute the graph representation of the molecule."""

import numpy as np
import torch
import numpy.typing as npt
from typing import Any
from rdkit import Chem
from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock

class AtomGraph(VerboseTransformationBlock):
    """Create a torch geometric graph from the molecule."""

    def atom_attribute(self, smile: str)-> torch.Tensor:
        """Extract the atom attribute from the smile

        param smile: the molecule string format
        return: tensor containing the atom feature"""

        # extract the atoms from the smile
        mol = Chem.MolFromSmiles(smile)
        atoms = mol.GetAtoms()

        # extract the attributes in the atom
        atom_features = []
        for atom in atoms:
            attribute = [atom.GetAtomicNum(), atom.GetDegree()]
            atom_features.append(attribute)

        return torch.tensor(atom_features, dtype=torch.float)

    def bond_attribute(self, smile: str) -> torch.Tensor:
        """Extract the bond attribute from the smile.

        param smile: the molecule string format
        return: tensor containing the edge feature"""

        # extract the bonds from the smile
        mol = Chem.MolFromSmiles(smile)
        bonds = mol.GetAtoms()

        # extract the attributes and the edge index
        edge_features = []
        edge_index = []

        for bond in bonds:
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.append((start, end))
            edge_index.append((end, start))

            attribute = []


