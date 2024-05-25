"""Module to do building block reaction within dataset pipeline."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.model.training.training_block import TrainingBlock

from src.modules.chemistry_functions.bbs_to_molecule import bbs_to_molecule


@dataclass
class BBReaction(TrainingBlock):
    """Simulate chemical reaction to turn building blocks into smile strings."""

    def train(
        self,
        x: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.uint8]]:
        """Transform building blocks into the full molecule representation.

        :param x: The building block strings
        :param y: The labels (ignored)
        :return: Tuple of product smiles and labels (unchange)
        """
        result_molecules = np.array([bbs_to_molecule(bb[0], bb[1], bb[2]) for bb in x])

        return result_molecules, y

    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return False
