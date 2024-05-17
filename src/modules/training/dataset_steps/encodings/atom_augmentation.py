"""Module to augment atom encoding."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from numpy.random import default_rng


@dataclass
class AtomAugmentation(TrainingBlock):
    """Augment atom encoding class.

    :param p: Probability of augmentation
    """

    p: float = 0.5

    def train(self, X: npt.NDArray[np.int_], y: npt.NDArray[np.int_]) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Replace some atoms with similar ones.

        :param X: Input array
        :param y: Labels
        """
        replace_dict = {
            10: [11, 12, 13],  # Fluorine -> Chlorine, Bromine, Iodine
            11: [10, 12, 13],  # Chlorine -> Fluorine, Bromine, Iodine
            12: [10, 11, 13],  # Bromine -> Fluorine, Chlorine, Iodine
            13: [10, 11, 12],  # Iodine -> Fluorine, Chlorine, Bromine
            5: [7],  # Oxygen -> Sulfur
            7: [5],  # Sulfur -> Oxygen
            3: [9],  # Nitrogen -> Phosphorus
            9: [3],  # Phosphorus -> Nitrogen
        }

        # Replace some atoms with similar ones
        # X is list of encodings apply encoding_replacer to each encoding parallel
        for i in range(len(X)):
            X[i] = self.encoding_replacer(X[i], self.p, replace_dict)

        return X, y

    @staticmethod
    def encoding_replacer(encoding: npt.NDArray[np.int_], p: float, replace_dict: dict[int, list[int]]) -> npt.NDArray[np.int_]:
        """Replace the encoding with similar atoms.

        :param p: Probability of replacement
        :param replace_dict: The values that can be exchanged
        :return: Augmented encoding
        """
        rng = default_rng()
        for i in range(len(encoding)):
            if encoding[i] not in replace_dict:
                continue

            if rng.random() < p:
                encoding[i] = rng.choice(replace_dict[encoding[i]])

        return encoding

    @property
    def is_augmentation(self) -> bool:
        """Return if block is augmentation block.

        :return: is_block
        """
        return True
