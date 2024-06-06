"""Train a torch tokenizer on the molecule smiles."""

import pickle
from dataclasses import dataclass

import joblib
import numpy as np
import numpy.typing as npt
from torchnlp.encoders.text import StaticTokenizerEncoder  # type: ignore[import-not-found]
from tqdm import tqdm

from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.typing.xdata import XData


@dataclass
class MoleculeTokenizer(VerboseTrainingBlock):
    """Train a torch tokenizer on the blocks or molecule smiles."""

    training: str = "10M"
    window_size: int = 6

    def segment_molecule(self, smile: str) -> list[str]:
        """Transform the molecule smile for the tokenizer.

        :param smile: string representing the molecule smile
        :return: list containing the substructures"""

        # Extract n-grams from the sequence
        length = len(smile) - self.window_size + 1
        return [" ".join(smile[i: i + self.window_size]) for i in range(length)]

    def custom_train(self, x: XData, y: npt.NDArray[np.float32], **kwargs) -> tuple[XData, npt.NDArray[np.float32]]:
        """Train the torch tokenizer on the molecule smiles.

        :param x: XData containing the molecule smiles
        :param y: array containing the molecule labels
        :return: The tokenized sentences and labels
        """

        self.log_to_terminal(f"start training the tokenizer on {self.training}.")

        # Check whether the molecule smiles are present
        if x.molecule_smiles is None:
            raise ValueError("There is no SMILE information for the molecules")

        # Extract the molecule smiles as a list for torch
        tqdm_smiles = tqdm(list(x.molecule_smiles), desc="Tokenizing molecules")
        encoder = StaticTokenizerEncoder(tqdm_smiles, tokenize=self.segment_molecule)

        # Print the vocabulary size of the tokenizer
        self.log_to_terminal(f"The vocabulary size of the tokenizer {encoder.vocab_size}.")

        # Save the tokenizer as a pickle file
        with open(f"tm/tokenizer_samples={self.training}_window={str(self.window_size)}.pkl", "wb") as f:
            pickle.dump(encoder.index_to_token, f, protocol=pickle.HIGHEST_PROTOCOL)

        return x, y

    def custom_predict(self, X: XData) -> XData:
        """Predict using the model.

        :param X: XData containing the molecule smiles
        :return: the tokenized sentences
        """

        return X
