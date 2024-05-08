"""Compute the graph representation of the molecule."""

from concurrent.futures import ProcessPoolExecutor
from typing import Any

import torch
from rdkit import Chem
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData

NUM_FUTURES = 100
MIN_CHUNK_SIZE = 1000


class AtomGraph(VerboseTransformationBlock):
    """Create a torch geometric graph from the molecule.

    param convert_molecule: whether to convert the molecule
    param convert_bb: whether to convert the building blocks
    """

    convert_molecule: bool = True
    convert_bb: bool = True

    def atom_attribute(self, smile: str) -> torch.Tensor:
        """Extract the atom attribute from the smile

        param smile: the molecule string format
        return: tensor containing the atom feature
        """
        # Extract the atoms from the smile
        mol = Chem.MolFromSmiles(smile)
        atoms = mol.GetAtoms()

        # Extract the attributes in the atom
        atom_features = []
        for atom in atoms:
            atom_features.append([atom.GetAtomicNum(), atom.GetDegree()])

        return torch.tensor(atom_features, dtype=torch.float)

    def bond_attribute(self, smile: str) -> torch.Tensor:
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

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float)

        return edge_index, edge_features

    def torch_graph(self, smiles: list[str]) -> list[list]:
        """Create the torch graph from the smile format.

        param smile: list containing the smile format
        return: list containing the atom and bond attributes
        """
        graphs = []
        for smile in smiles:
            # Extract the atom attributes from the smile
            atom_feature = self.atom_attribute(smile)

            # Extract the edge attributes and indices
            edge_index, edge_feature = self.bond_attribute(smile)
            graphs.append([atom_feature, edge_index, edge_feature])
        return graphs

    def parallel_graph(self, smiles: list[str], desc: str) -> Any:
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
            futures = [executor.submit(self.torch_graph, chunk) for chunk in chunks]

            # Perform the multiprocessing on the chunks
            for future in tqdm(futures, total=len(futures), desc=desc):
                results.extend(future.result())

        return results

    def custom_transform(self, data: XData) -> XData:
        """Create a torch geometric graph from the molecule."""
        desc = "compute the geometric graph of the molecule"

        # Compute the embeddings for each molecule
        if self.convert_molecule and data.molecule_smiles is not None:
            data.molecule_embedding = self.parallel_graph(data.molecule_smiles, desc)

        # Compute the embeddings for each block
        if self.convert_bb and data.bb1_smiles is not None and data.bb2_smiles is not None and data.bb3_smiles is not None:
            data.bb1_embedding = self.parallel_graph(data.bb1_smiles, desc)
            data.bb2_embedding = self.parallel_graph(data.bb2_smiles, desc)
            data.bb3_embedding = self.parallel_graph(data.bb3_smiles, desc)

        return data
