"""Train a torch tokenizer on the molecule smiles."""

import numpy as np
import numpy.typing as npt
from rdkit import Chem
import pickle
import random
import torch
from tqdm import tqdm
from dataclasses import dataclass
from src.typing.xdata import XData
from torchnlp.encoders.text import StaticTokenizerEncoder
from torchnlp.encoders.text import pad_tensor
from src.modules.training.verbose_training_block import VerboseTrainingBlock

identity = lambda x: x

@dataclass
class TokenizerAtom(VerboseTrainingBlock):
    """Train a torch tokenizer on the molecule smiles.
    param window_size: the size of each word"""

    num_samples: int = 0.5

    def custom_train(self, X: XData, y: npt.NDArray[np.float32]) -> tuple[XData, npt.NDArray[np.float32]]:
        """Train the torch tokenizer on the sentences.

        :param X: XData containing the molecule smiles
        :param y: array containing the molecule labels
        :return: The tokenized sentences and labels"""

        self.log_to_terminal(f"start training the tokenizer.")

        random.seed(42)

        # extract the molecule smiles from XData
        smiles = list(X.molecule_smiles)
        sampled = random.sample(smiles, int(self.num_samples*len(smiles)))

        # train the torch nlp tokenizer on the sequences
        tqdm_smiles = tqdm(sampled, desc="Tokenizing molecules")
        encoder = StaticTokenizerEncoder(tqdm_smiles, tokenize=identity)

        # Apply the tokenizer on all the smiles
        X.molecule_ecfp = [encoder.encode(smile) for smile in tqdm(smiles, desc="encode molecules")]

        # print the vocabulary size of the tokenizer
        self.log_to_terminal(f"the vocabulary size of the tokenizer {encoder.vocab_size}.")

        # save the tokenizer as a torch file
        with open(f"tm/{self.get_hash()}.pkl", 'wb') as f:  # Use .pkl extension for clarity
            pickle.dump(encoder, f)

        return X, y

    def custom_predict(self, X: XData) -> npt.NDArray[np.float32]:
        """Predict using the model.

        :param X: XData containing the molecule smiles
        :return: the tokenized sentences"""

        # extract the smiles from the XData
        smiles = list(X.molecule_smiles)

        # extract the tokenizer from the pickle file
        with open(f"tm/{self.get_hash()}.pkl", 'rb') as f:
            encoder = pickle.load(f)

        # apply the torch tokenizer on the smiles
        X.molecule_ecfp = [encoder.encode(smile) for smile in smiles]

        return X
