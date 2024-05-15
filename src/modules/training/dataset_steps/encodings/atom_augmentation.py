from epochalyst.pipeline.model.training.training_block import TrainingBlock
import numpy.typing as npt
import numpy as np
from dataclasses import dataclass


@dataclass
class AtomAugmentation(TrainingBlock):

    p: float = 0.5

    def train(self, X: npt.NDArray[np.int_], y: npt.NDArray[np.int_]) -> tuple[npt.NDArray]:
        """Replace some atoms with similar ones"""

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

        if self.use_aug:
            # Replace some atoms with similar ones
            # X is list of encodings apply encoding_replacer to each encoding parallel
            for i in range(len(X)):
                X[i] = self.encoding_replacer(X[i], self.p, replace_dict)

            return X, y
        else:
            return X, y

    @staticmethod
    def encoding_replacer(encoding: npt.NDArray[np.int_], p: float, replace_dict: dict[int, list[int]]) -> npt.NDArray[np.int_]:
        for i in range(len(encoding)):
            if not encoding[i] in replace_dict:
                continue
            if np.random.rand() < p:
                encoding[i] = np.random.choice(replace_dict[encoding[i]])

        return encoding
