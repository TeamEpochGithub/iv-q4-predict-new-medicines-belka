"""Compute the graph representation of the molecule."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from rdkit import Chem  # type: ignore[import-not-found]
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
    def _torch_graph(smiles: npt.NDArray[np.str_]) -> list[list[Any]]:
        """Create the torch graph from the smile format.

        param smile: list containing the smile format
        return: list containing the atom and bond attributes
        """
        graphs = []
        for smile in smiles:
            # Extract the atom attributes from the smile
            atom_feature = _atom_attribute(smile)

            # Extract the edge attributes and indices
            edge_index, edge_feature = _bond_attribute(smile)
            graphs.append([atom_feature, edge_index, edge_feature])

        return graphs

    def _parallel_graph(self, smiles: npt.NDArray[np.str_], desc: str) -> list[Any]:
        """Compute the torch graph using multiprocessing.

        param smiles: list containing the smiles of the molecules
        param desc: message to be shown during the process
        """
        # define the maximum chunk size
        chunk_size = len(smiles) // NUM_FUTURES
        chunk_size = max(chunk_size, MIN_CHUNK_SIZE)

        # Divide the smiles molecules into chunks
        chunks = [smiles[i : i + chunk_size] for i in range(0, len(smiles), chunk_size)]

        # Initialize the multiprocessing with the chunks
        results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._torch_graph, chunk) for chunk in chunks]

            # Perform the multiprocessing on the chunks
            for future in tqdm(futures, total=len(futures), desc=desc):
                results.extend(future.result())

        return results

    def custom_transform(self, data: XData) -> XData:
        """Create a torch geometric graph from the molecule."""
        # Compute the embeddings for each molecule
        if self.convert_molecule and data.molecule_smiles is not None:
            data.molecule_graph = self._parallel_graph(data.molecule_smiles, desc="Creating atomic graph for molecules")

        # Compute the embeddings for each block
        if self.convert_building_blocks and data.bb1_smiles is not None and data.bb2_smiles is not None and data.bb3_smiles is not None:
            data.bb1_graph = self._parallel_graph(data.bb1_smiles, desc="Creating atomic graph for bb1")
            data.bb2_graph = self._parallel_graph(data.bb2_smiles, desc="Creating atomic graph for bb2")
            data.bb3_graph = self._parallel_graph(data.bb3_smiles, desc="Creating atomic graph for bb3")

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
    atom_features = [[atom.GetAtomicNum(), atom.GetDegree()] for atom in atoms]

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

    return np.array(edge_index), np.array(edge_features)
