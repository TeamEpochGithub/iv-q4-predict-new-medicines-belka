"""Module to standardize molecules."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from rdkit import Chem  # type: ignore[import]
from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore[import]


@dataclass
class SmilesStandardizer(TrainingBlock):
    """Encode the SMILES string into categorical data."""

    no_tautomers: bool = True

    def train(
        self,
        x: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Standardize the smiles in the smiles_array."""
        standardized_smiles = [self.standardize(smiles, no_tautomers=self.no_tautomers) for smiles in x]

        return np.array(standardized_smiles), y

    @staticmethod
    def standardize(smiles: str, *, no_tautomers: bool = True) -> str:
        """Standardize the SMILES string.

        :param smiles: The SMILES string to standardize.
        :param no_tautomers: Whether to standardize without tautomers.
        :return: The standardized SMILES string.
        """
        # Adapted from https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
        # which follows the steps in
        # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
        # as described **excellently** (by Greg) in
        # https://www.youtube.com/watch?v=eWTApNX8dJQ
        mol = Chem.MolFromSmiles(smiles)

        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol)

        # if many fragments, get the "parent" (the actual mol we are interested in)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        if no_tautomers:
            return Chem.MolToSmiles(uncharged_parent_clean_mol)

        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim to to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.

        te = rdMolStandardize.TautomerEnumerator()  # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

        return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)

    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return False
