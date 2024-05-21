"""Train a torch tokenizer on the molecule smiles."""

import numpy as np
import numpy.typing as npt
from rdkit import Chem
import joblib
import pickle
from tqdm import tqdm
from dataclasses import dataclass
from src.typing.xdata import XData
from torchnlp.encoders.text import StaticTokenizerEncoder
from torchnlp.encoders.text import pad_tensor
from src.modules.training.verbose_training_block import VerboseTrainingBlock

identity = lambda x: x

@dataclass
class TokenizerAtom(VerboseTrainingBlock):
    """Train a torch tokenizer on the molecule smiles."""

    def train_tokenizer(self, smiles: list[list[str]]):
        """Train the tokenizer on the smile molecules.
        :param smiles: the segmented smile molecules"""

        # Train the torch nlp tokenizer on the sequences
        tqdm_smiles = tqdm(smiles, desc="Tokenizing molecules")
        encoder = StaticTokenizerEncoder(tqdm_smiles, tokenize=identity)

        return encoder

    def apply_tokenizer(self, smiles: list[list[str]], encoder):
        """ Apply the tokenizer on the smile molecules.

        :param smiles: the segmented smile molecules
        :param encoder: torch nlp tokenizer"""

        return np.array([encoder.encode(smile) for smile in tqdm(smiles, desc="encode molecules")])

    def custom_train(self, X: XData, y: npt.NDArray[np.float32]) -> tuple[XData, npt.NDArray[np.float32]]:
        """Train the torch tokenizer on the sentences.

        :param X: XData containing the molecule smiles
        :param y: array containing the molecule labels
        :return: The tokenized sentences and labels"""

        self.log_to_terminal(f"start training the tokenizer.")

        # Extract the smiles from each building block
        smiles = list(X.bb1_smiles) + list(X.bb2_smiles) + list(X.bb3_smiles)

        # Train and apply the torch nlp tokenizer on the building blocks
        encoder = self.train_tokenizer(smiles)
        X.bb1_ecfp = self.apply_tokenizer(list(X.bb1_smiles), encoder)
        X.bb2_ecfp = self.apply_tokenizer(list(X.bb2_smiles), encoder)
        X.bb3_ecfp = self.apply_tokenizer(list(X.bb3_smiles), encoder)

        # Print the vocabulary size of the tokenizer
        self.log_to_terminal(f"the vocabulary size of the tokenizer {encoder.vocab_size}.")

        # Save the tokenizer as a torch file
        # with open(f"tm/{self.get_hash()}", 'wb') as f:
        #     f.write(encoder)

        # # Save the tokenizer as a torch file
        # joblib.dump(encoder, f"tm/{self.get_hash()}")


        return X, y

    def custom_predict(self, X: XData) -> XData:
        """Predict using the model.

        :param X: XData containing the molecule smiles
        :return: the tokenized sentences"""

        # Extract the tokenizer from the pickle file
        encoder = joblib.load(f"tm/{self.get_hash()}.pkl")

        # Apply the torch tokenizer on each building block
        X.bb1_smiles = self.apply_tokenizer(list(X.bb1_smiles), encoder)
        X.bb2_smiles = self.apply_tokenizer(list(X.bb2_smiles), encoder)
        X.bb3_smiles = self.apply_tokenizer(list(X.bb3_smiles), encoder)

        return X