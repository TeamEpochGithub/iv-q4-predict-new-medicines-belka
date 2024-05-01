"""Class to describe the data format."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from rdkit.DataStructs.cDataStructs import ExplicitBitVect  # type: ignore[import-not-found]


@dataclass
class XData:
    """Class to describe data format of X.

    :param building_blocks: Building blocks encoded
    :param molecule_smiles: Molecule smiles
    :param bb1: Building block 1 smiles
    :param bb2: Building block 2 smiles
    :param bb3: Buliding block 3 smiles
    """

    building_blocks: npt.NDArray[np.int16]
    retrieval: str = "SMILES"

    # SMILES
    molecule_smiles: list[str] | None = None
    bb1_smiles: list[str] | None = None
    bb2_smiles: list[str] | None = None
    bb3_smiles: list[str] | None = None

    # ECFP
    molecule_ecfp: list[ExplicitBitVect] | None = None
    bb1_ecfp: list[ExplicitBitVect] | None = None
    bb2_ecfp: list[ExplicitBitVect] | None = None
    bb3_ecfp: list[ExplicitBitVect] | None = None

    def __getitem__(self, index: int) -> npt.NDArray[Any]:
        """Get item from the data.

        :param index: Index to retrieve
        :return: Data replaced with correct building_blocks
        """
        item = self.building_blocks[index]

        if self.retrieval == "SMILES":
            if not self.molecule_smiles or not self.bb1_smiles or not self.bb2_smiles or not self.bb3_smiles:
                raise ValueError("Missing SMILE representation of building_blocks and molecule")
            mol_smile = self.molecule_smiles[index]
            return np.array([self.bb1_smiles[item[0]], self.bb2_smiles[item[1]], self.bb3_smiles[item[2]], mol_smile])
        if self.retrieval == "ECFP":
            if not self.molecule_ecfp or not self.bb1_ecfp or not self.bb2_ecfp or not self.bb3_ecfp:
                raise ValueError("Missing ECFP representation of building_blocks and molecule")
            mol_smile = self.molecule_ecfp[index]
            return np.array([self.bb1_ecfp[item[0]], self.bb2_ecfp[item[1]], self.bb3_ecfp[item[2]], mol_smile])

        return np.array([])

    def __len(self) -> int:
        """Return the length of the data.

        :return: Length of data
        """
        return len(self.building_blocks)
