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


def identity(x: list[str]) -> list[str]:
    """Transform the molecule for the tokenizer.

    :param x: list containing the tokens
    :return: list containing the tokens
    """
    return x

@dataclass
class TokenizerAtom(VerboseTrainingBlock):
    """Train a torch tokenizer on the molecule smiles."""

    def apply_tokenizer(self, smiles: npt.NDArray[np.str_]) -> npt.NDArray[np.uint8]:
        """Apply the tokenizer on the smile molecules.

        :param smiles: the segmented smile molecules
        :param encoder: torch nlp tokenizer
        """
        return np.array([self.encoder.encode(smile) for smile in tqdm(smiles, desc="encode molecules")])

    def custom_train(self, X: XData, y: npt.NDArray[np.float32]) -> tuple[XData, npt.NDArray[np.float32]]:
        """Train the torch tokenizer on the sentences.

        :param X: XData containing the molecule smiles
        :param y: array containing the molecule labels
        :return: The tokenized sentences and labels
        """
        self.log_to_terminal("start training the tokenizer.")

        # Check whether the building blocks are present
        if X.bb1_smiles is None or X.bb2_smiles is None or X.bb3_smiles is None:
            raise ValueError("There is no SMILE information for the molecules")

        # Extract the smiles from each building block
        smiles = list(X.bb1_smiles) + list(X.bb2_smiles) + list(X.bb3_smiles)

        # Train the torch nlp tokenizer on the sequences
        tqdm_smiles = tqdm(smiles, desc="Tokenizing molecules")
        self.encoder = StaticTokenizerEncoder(tqdm_smiles, tokenize=identity)

        # Apply the torch nlp tokenizer on the building blocks
        X.bb1_ecfp = self.apply_tokenizer(X.bb1_smiles)
        X.bb2_ecfp = self.apply_tokenizer(X.bb2_smiles)
        X.bb3_ecfp = self.apply_tokenizer(X.bb3_smiles)

        # Print the vocabulary size of the tokenizer
        self.log_to_terminal(f"The vocabulary size of the tokenizer {self.encoder.vocab_size}.")

        # Save the tokenizer as a pickle file
        with open(f"tm/{self.get_hash()}.pkl", "wb") as f:
            pickle.dump(self.encoder, f, protocol=pickle.HIGHEST_PROTOCOL)

        return X, y

    def custom_predict(self, X: XData) -> XData:
        """Predict using the model.

        :param X: XData containing the molecule smiles
        :return: the tokenized sentences
        """
        # Check whether the building blocks are present
        if X.bb1_smiles is None or X.bb2_smiles is None or X.bb3_smiles is None:
            raise ValueError("There is no SMILE information for the molecules")

        # Extract the tokenizer from the pickle file
        self.encoder = joblib.load(f"tm/{self.get_hash()}.pkl")

        # Apply the torch nlp tokenizer on the building blocks
        X.bb1_ecfp = self.apply_tokenizer(X.bb1_smiles)
        X.bb2_ecfp = self.apply_tokenizer(X.bb2_smiles)
        X.bb3_ecfp = self.apply_tokenizer(X.bb3_smiles)

        return X
