"""Module that contains label smoothing function."""
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock

@dataclass
class LabelSmoothing(VerboseTransformationBlock):
    """Class to apply a sigmoid transformation to the data."""

    lower_bound: float = 0.01
    upper_bound: float = 0.99

    def custom_transform(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply a sigmoid transformation to the data.

        :param x: XData.
        :return: XData with sigmoid transformation applied.
        """
        # Set 0 values to 0.01 and 1 values to 0.99
        y[y == 0] = self.lower_bound
        y[y == 1] = self.upper_bound
        return y

