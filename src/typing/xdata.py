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
    :param bb1_smiles: Building_block 1 smiles
    :param bb2_smiles: Building_block 2 smiles
    :param bb3_smiles: Building_block 3 smiles

    :param molecule_ecfp: ECFP for molecules
    :param bb1_ecfp: ECFP for building_block 1 smiles
    :param bb2_ecfp: ECFP for building_block 2 smiles
    :param bb3_ecfp: ECFP for building_block 3 smiles

    :param molecule_embedding: Embedding for molecules
    :param bb1_embedding: Embedding for building_block 1 smiles
    :param bb2_embedding: Embedding for building_block 2 smiles
    :param bb3_embedding: Embedding for building_block 3 smiles

    :param molecule_desc: Descriptors for molecules
    :param bb1_desc: Descriptors for building_block 1 smiles
    :param bb2_desc: Descriptors for building_block 2 smiles
    :param bb3_desc: Descriptors for building_block 3 smiles
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

    # Embedding
    molecule_embedding: list[npt.NDArray[np.float32]] | None = None
    bb1_embedding: list[npt.NDArray[np.float32]] | None = None
    bb2_embedding: list[npt.NDArray[np.float32]] | None = None
    bb3_embedding: list[npt.NDArray[np.float32]] | None = None

    # Descriptors
    molecule_desc: list[Any] | None = None
    bb1_desc: list[Any] | None = None
    bb2_desc: list[Any] | None = None
    bb3_desc: list[Any] | None = None

    def __getitem__(self, index: int) -> npt.NDArray[Any]:
        """Get item from the data.

        :param index: Index to retrieve
        :return: Data replaced with correct building_blocks
        """
        item = self.building_blocks[index]

        # SMILES Retrievals
        if self.retrieval == "SMILES":
            if not self.molecule_smiles or not self.bb1_smiles or not self.bb2_smiles or not self.bb3_smiles:
                raise ValueError("Missing SMILE representation of building_blocks and molecule")
            mol_smile = self.molecule_smiles[index]
            return np.array([self.bb1_smiles[item[0]], self.bb2_smiles[item[1]], self.bb3_smiles[item[2]], mol_smile])

        # ECFP Retrievals
        if self.retrieval == "ECFP":
            if not self.molecule_ecfp or not self.bb1_ecfp or not self.bb2_ecfp or not self.bb3_ecfp:
                raise ValueError("Missing ECFP representation of building_blocks and molecule")
            mol_ecfp = self.molecule_ecfp[index]
            return np.array([self.bb1_ecfp[item[0]], self.bb2_ecfp[item[1]], self.bb3_ecfp[item[2]], mol_ecfp])
        if self.retrieval == "ECFP_BB":
            if not self.bb1_ecfp or not self.bb2_ecfp or not self.bb3_ecfp:
                raise ValueError("Missing ECFP representation of building_blocks")
            return np.array([self.bb1_ecfp[item[0]], self.bb2_ecfp[item[1]], self.bb3_ecfp[item[2]]])

        # Embedding Retrievals
        if self.retrieval == "Embedding":
            return self._get_embedding(index)

        # Descriptors Retrievals
        if self.retrieval == "Descriptors":
            return self._get_descriptors(index)

        return np.array([])

    def _get_embedding(self, index: int) -> npt.NDArray[Any]:
        if not self.molecule_embedding or not self.bb1_embedding or not self.bb2_embedding or not self.bb3_embedding:
            raise ValueError("Missing embedding of building_blocks and molecule")
        return np.array([self.bb1_embedding[index], self.bb2_embedding[index], self.bb3_embedding[index], self.molecule_embedding[index]])

    def _get_descriptors(self, index: int) -> npt.NDArray[Any]:
        if not self.molecule_desc or not self.bb1_desc or not self.bb2_desc or not self.bb3_desc:
            raise ValueError("Missing descriptors of building_blocks and molecule")
        return np.array([self.bb1_desc[index], self.bb2_desc[index], self.bb3_desc[index], self.molecule_desc[index]])

    def __len(self) -> int:
        """Return the length of the data.

        :return: Length of data
        """
        return len(self.building_blocks)

    def __repr__(self) -> str:
        """Return the representation of the data."""
        return ""
