"""Base class for splitter."""
from abc import abstractmethod
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.typing.xdata import XData


class Splitter:
    """Base class for splitter."""

    @abstractmethod
    def split(
        self,
        X: XData | None,
        y: npt.NDArray[np.int8] | None,
        cache_path: Path,
    ) -> (
        list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]
        | tuple[list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]], npt.NDArray[np.int64], npt.NDArray[np.int64]]
    ):
        """Split X and y into train and test indices.

        :param X: The Xdata
        :param y: Labels
        :return: List of indices
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def includes_validation(self) -> bool:
        """Check if the splitter includes validation."""
        raise NotImplementedError
