"""Module that contains Sigmoid function as transformation block."""
import numpy as np
import numpy.typing as npt

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


class Sigmoid(VerboseTransformationBlock):
    """Class to apply a sigmoid transformation to the data."""

    def custom_transform(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply a sigmoid transformation to the data.

        :param x: XData.
        :return: XData with sigmoid transformation applied.
        """
        return 1 / (1 + np.exp(-x))
