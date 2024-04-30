"""Class to describe the data format."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


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
    molecule_smiles: npt.NDArray[Any]
    bb1: list[Any]
    bb2: list[Any]
    bb3: list[Any]

    def __getitem__(self, index: int) -> npt.NDArray[Any]:
        """Get item from the data.

        :param index: Index to retrieve
        :return: Data replaced with correct building_blocks
        """
        item = self.building_blocks[index]
        mol_smile = self.molecule_smiles[index]

        return np.array([self.bb1[item[0]], self.bb2[item[1]], self.bb3[item[2]], mol_smile])

    def __len(self) -> int:
        """Return the length of the data.

        :return: Length of data
        """
        return len(self.building_blocks)
