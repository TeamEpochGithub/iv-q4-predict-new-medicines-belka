"""Module that shifts the predictions as transformation block."""
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock

@dataclass
class ShiftPredictions(VerboseTransformationBlock):
    """Class to apply a shift trasnformation to the data."""

    exponent: float = 0.6

    def custom_transform(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply a shift transformation to the data.

        :param x: XData.
        :return: XData with shift transformation applied.
        """
        return np.power(x, self.exponent)
