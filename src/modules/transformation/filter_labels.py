"""Module that contains label smoothing function."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


@dataclass
class FilterLabels(VerboseTransformationBlock):
    """Class to apply a sigmoid transformation to the data."""

    protein: str = "sEH"

    def custom_transform(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply a sigmoid transformation to the data.

        :param x: XData.
        :return: XData with sigmoid transformation applied.
        """
        protein_map = {
            "BRD4": 0,
            "HSA": 1,
            "sEH": 2,
        }

        return y[:, protein_map[self.protein]]
