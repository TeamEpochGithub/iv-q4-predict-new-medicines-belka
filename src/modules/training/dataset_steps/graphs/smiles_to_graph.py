"""Module to turn smile representation of molecule into graph representation."""
import os
from collections.abc import Callable

# from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import deepchem as dc  # type: ignore[import-not-found]
import numpy as np
import numpy.typing as npt
import torch
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import AllChem, RDConfig, rdMolDescriptors  # type: ignore[import-not-found]
from rdkit.Chem.rdchem import Mol  # type: ignore[import-not-found]
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory  # type: ignore[import-not-found]
from torch_geometric.data import Data

PHARMA_DICT = {
    "Donor": 0,
    "Acceptor": 1,
    "NegIonizable": 2,
    "PosIonizable": 3,
    "ZnBinder": 4,
    "Aromatic": 5,
    "LumpedHydrophobe": 6,
    "Hydrophobe": 7,
}
PHARMA_DICT_LEN = len(PHARMA_DICT)

# Setup AtomicNr Translation
ATOMIC_NUM_DICT = {
    5: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    14: 5,
    16: 6,
    17: 7,
    35: 8,
    53: 9,
    66: 10,
}
ATOMIC_NUM_DICT_LEN = len(ATOMIC_NUM_DICT)


@dataclass
class SmilesToGraph(TrainingBlock):
    """Turn smile representation into graph."""

    use_atom_chem_features: bool = False
    use_atom_pharmacophore_features: bool = False
    use_bond_features: bool = False
    use_atom_deep_chem_features: bool = False

    _atom_chem_features: list[Callable[[Mol], int]] = field(default_factory=list, repr=False, compare=False)
    _num_atom_chem_features: int = field(default=0, repr=False, compare=False)
    _atom_pharma_features: Callable[[Mol], npt.NDArray[np.uint8]] | None = field(default=None, repr=False, compare=False)
    _num_atom_pharma_features: int = field(default=0, repr=False, compare=False)
    _atom_deepchem_features: Callable[[Mol], npt.NDArray[np.float32]] | None = field(default=None, repr=False, compare=False)  # New field for DeepChem features
    _num_atom_features: int = field(default=0, repr=False, compare=False)

    _num_bond_features: int = field(default=0, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Post init function."""
        # Add Atom Features
        if self.use_atom_chem_features:
            self._atom_chem_features = [
                lambda x: x.GetDegree(),
                lambda x: x.GetHybridization(),
                lambda x: x.GetIsotope(),
                lambda x: ATOMIC_NUM_DICT[x.GetAtomicNum()],
                lambda x: x.GetFormalCharge(),
                lambda x: x.GetNumRadicalElectrons(),
                lambda x: x.GetIsAromatic(),
                lambda x: x.GetChiralTag(),
                lambda x: (1 if rdMolDescriptors.CalcNumLipinskiHBA(x.GetOwningMol()) > 0 else 0),
                lambda x: (1 if rdMolDescriptors.CalcNumLipinskiHBD(x.GetOwningMol()) > 0 else 0)
            ]
            self._num_atom_chem_features = len(self._atom_chem_features)
            self._num_atom_features += len(self._atom_chem_features)

        # Add Pharmacophore Features
        if self.use_atom_pharmacophore_features:
            fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
            self._atom_pharma_features = lambda x: _extract_atom_pharma_features_packed(fdef, x)  # type: ignore[misc]
            self._num_atom_pharma_features = 1
            self._num_atom_features += 1

        # Adds DeepChem Features
        if self.use_atom_deep_chem_features:
            self._atom_deepchem_features = lambda x: _extract_deepchem_features(x)
            self._num_atom_features += 10

        # Add Bond Features
        if self.use_bond_features:
            self._num_bond_features += 2

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
        graphs = []
        if y is not None:
            for i, smile in enumerate(x):
                graphs.append(self._smile_to_graph(smile, y[i]))
        else:
            graphs = [self._smile_to_graph(smile) for smile in x]

        return graphs, None

    def _smile_to_graph(self, smile: str, label: npt.NDArray[np.uint8] | None = None) -> Data:
        """Create the torch graph from the smile format.

        :param smile: list containing the smile format
        :param use_bond_attributes: Use the bond attributes in the graph
        :return: list containing the atom and bond attributes
        """
        # Convert the smile to a molecule
        mol = Chem.MolFromSmiles(smile)

        # Create the atom features
        atom_features_torch = torch.from_numpy(self._calc_atom_features(mol))

        # Create the bond features
        bond_indicies, bond_features = self._calc_bond_inidices_features(mol)
        bond_indicies_torch = torch.from_numpy(bond_indicies)
        bond_features_torch = torch.from_numpy(bond_features) if bond_features is not None else None

        # Convert the label
        if label is not None:
            label_torch = torch.from_numpy(label)

        return Data(
            x=atom_features_torch,
            edge_index=bond_indicies_torch.to(torch.long),
            edge_attr=bond_features_torch,
            y=label_torch if label is not None else None,
        )

    def _calc_atom_features(self, mol: Mol) -> npt.NDArray[np.uint8]:
        atom_features = np.empty((mol.GetNumAtoms(), self._num_atom_features), dtype=np.uint8)
        row_idx = 0

        # Add Atom Features
        if self.use_atom_chem_features:
            for idx, atom in enumerate(mol.GetAtoms()):
                chem_features = [feature(atom) for feature in self._atom_chem_features]
                atom_features[idx][row_idx : self._num_atom_chem_features] = np.array(chem_features, dtype=np.uint8)
            row_idx += self._num_atom_chem_features

        # Add Pharmacophore Features
        if self.use_atom_pharmacophore_features and self._atom_pharma_features is not None:
            atom_features[:, row_idx] = self._atom_pharma_features(mol)
            row_idx += self._num_atom_pharma_features

        return atom_features

    def _calc_bond_inidices_features(self, mol: Mol) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint8] | None]:
        bonds = mol.GetBonds()

        # Extract the attributes and the edge index
        edge_indices = np.empty((2 * len(bonds), 2), dtype=np.uint16)
        edge_features = np.empty((2 * len(bonds), 2), dtype=np.uint8)

        idx = 0
        for bond in bonds:
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices[idx] = [start, end]
            edge_indices[idx + 1] = [end, start]

            if self.use_bond_features:
                edge_features[idx] = [int(bond.GetIsConjugated()), int(bond.IsInRing())]
                edge_features[idx + 1] = [int(bond.GetIsConjugated()), int(bond.IsInRing())]

            idx += 2

        return edge_indices.T, edge_features if self.use_bond_features else None

    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return False


def _extract_atom_pharma_features_packed(fdef: MolChemicalFeatureFactory, mol: Mol) -> npt.NDArray[np.uint8]:
    mol = Chem.AddHs(mol)
    pharmafeature_list = [fdef.GetMolFeature(mol, idx) for idx in range(fdef.GetNumMolFeatures(mol))]
    mol = Chem.RemoveHs(mol)

    features = np.empty((mol.GetNumAtoms()), dtype=np.uint8)
    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()
        del_indices = []
        for i in range(len(pharmafeature_list)):
            if atom_id in pharmafeature_list[i].GetAtomIds():
                features[atom_id] += PHARMA_DICT[pharmafeature_list[i].GetFamily()]
                if len(pharmafeature_list[i].GetAtomIds()) == 1:
                    del_indices.append(i)
        for idx in del_indices:
            del pharmafeature_list[idx]
    return features


def _extract_deepchem_features(mol: Mol) -> npt.NDArray[np.float32]:
    """Extract DeepChem features.

    :param mol: The molecule to extract features from.
    :return: The extracted features.
    """
    featurizer = dc.feat.CoulombMatrixEig(max_atoms=10)
    feature_array = featurizer.featurize([mol])[0]  # Eigen-decomposition of Coulomb Matrix
    return feature_array.astype(np.float32)


def unpack_atom_features(x: torch.Tensor) -> torch.Tensor:
    """Unpack the atom features."""
    # Only Pharmacophore Features
    if x.shape[1] == 1:
        unpacked_pharma = (x[:, 4].unsqueeze(1) >> torch.arange(8, device=x.device).unsqueeze(0)) & 1
        return unpacked_pharma.squeeze(0)
    # Only Chemical Features
    if x.shape[1] == 4:
        result = torch.empty(x.shape[0], 3 + ATOMIC_NUM_DICT_LEN, device=x.device, dtype=x.dtype)
        result[:, :3] = x[:, :3]
        result[:, 3:] = torch.nn.functional.one_hot(x[:, 3].long(), ATOMIC_NUM_DICT_LEN)
        return result

    if x.shape[1] == 0:
        result = torch.empty(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        result[:, :3] = x[:, :3]
        return result

    # Only DeepChem features
    if x.shape[1] == 10:  # Dimension of Coulomb Matrix
        result = torch.empty(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        result[:, :3] = x[:, :3]
        return result

    # Both Chemical and DeepChem
    if x.shape[1] == 24:  # 10 (DeepChem) + 14 (Chem)
        result = torch.empty((x.shape[0], 3 + ATOMIC_NUM_DICT_LEN + 10), device=x.device, dtype=x.dtype)
        result[:, :3] = x[:, :3]
        result[:, 3 : 3 + ATOMIC_NUM_DICT_LEN] = torch.nn.functional.one_hot(x[:, 3].long(), ATOMIC_NUM_DICT_LEN)
        return result

    # Both Chemical and Pharmacophore Features
    result = torch.empty((x.shape[0], 3 + ATOMIC_NUM_DICT_LEN + PHARMA_DICT_LEN), device=x.device, dtype=x.dtype)
    result[:, :3] = x[:, :3]
    result[:, 3 : 3 + ATOMIC_NUM_DICT_LEN] = torch.nn.functional.one_hot(x[:, 3].long(), ATOMIC_NUM_DICT_LEN)
    unpacked_pharma = (x[:, 4].unsqueeze(1) >> torch.arange(8, device=x.device).unsqueeze(0)) & 1
    result[:, 3 + ATOMIC_NUM_DICT_LEN :] = unpacked_pharma.squeeze(0)
    return result


def unpack_edge_features(edge_attr: torch.Tensor) -> torch.Tensor:
    """Unpack the edge features."""
    return edge_attr
