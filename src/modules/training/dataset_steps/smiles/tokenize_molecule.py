"""Module to convert the molecule smiles into a sequence of tokens."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from torchnlp.encoders.text import StaticTokenizerEncoder  # type: ignore[import-not-found]


@dataclass
class TokenizeMolecule(TrainingBlock):
    """Module to convert the molecule smiles into a sequence of tokens."""

    tokenizer_name: str = "samples_10M_window_6"
    padding_size: int = 150

    def train(
        self,
        x: npt.NDArray[np.str_],
        y: npt.NDArray[np.uint8],
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.uint8]]:
        """Convert the molecule smiles into a sequence of tokens.

        :param x: array containing the smile strings
        :param y: array containing the protein labels
        """
        # Check whether the tokenizer was trained or not
        file_path = Path(f"tm/tokenizer_{self.tokenizer_name}.pkl")
        if not file_path.exists():
            raise FileNotFoundError(f"The chosen tokenizer was not yet trained, path: {file_path}.")

        # Extract the window size and the vocab from the name
        self.window_size = int(self.tokenizer_name[-1])
        vocab = joblib.load(f"tm/tokenizer_{self.tokenizer_name}.pkl")

        # Initialize the tokenizer with the existing vocab
        encoder = StaticTokenizerEncoder("AAAA", tokenize=self.segment_molecule)
        encoder.token_to_index = {token: index for index, token in enumerate(vocab)}
        encoder.index_to_token = vocab.copy()

        # Apply the tokenizer on the molecule smiles
        return np.array([encoder.encode(smile) for smile in x]), y

    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return False

    def segment_molecule(self, smile: str) -> list[str]:
        """Transform the molecule smile for the tokenizer.

        :param smile: string representing the molecule smile
        :return: list containing the substructures
        """
        # Extract n-grams from the sequence
        length = len(smile) - self.window_size + 1
        sequence = [" ".join(smile[i : i + self.window_size]) for i in range(length)]

        # Pad the sequence with special token
        return sequence + ["PAD"] * (self.padding_size - len(sequence))
