"""Train a torch tokenizer on the molecule smiles."""

import numpy as np
import numpy.typing as npt
from rdkit import Chem
import pickle
from tqdm import tqdm
from dataclasses import dataclass
from src.typing.xdata import XData
from torchnlp.encoders.text import StaticTokenizerEncoder
from torchnlp.encoders.text import pad_tensor
from src.modules.training.verbose_training_block import VerboseTrainingBlock

@dataclass
class TokenizerAtom(VerboseTrainingBlock):
    """Train a torch tokenizer on the molecule smiles.
    param window_size: the size of each word"""

    window_size: int = 6

    def train(self, X: XData, y: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Train the torch tokenizer on the sentences.

        :param X: XData containing the molecule smiles
        :param y: array containing the molecule labels
        :return: The tokenized sentences and labels"""

        self.log_to_terminal(f"start training the tokenizer.")

        # extract the molecule smiles from XData
        smiles = X.molecule_smiles

        # Use tqdm to wrap the SMILES list
        tqdm_smiles = tqdm(smiles, desc="Tokenizing molecules")

        # train and apply the torch nlp tokenizer
        encoder = StaticTokenizerEncoder(tqdm_smiles, tokenize=self.segment_molecule)
        encoded = [encoder.encode(smile) for smile in smiles]
        # encoded = np.array([pad_tensor(x, length=150) for x in encoded])


        # save the tokenizer as a pickle file
        with open(f"tm/{self.get_hash()}", 'wb') as f:
            pickle.dump(encoder, f)

        # print the vocabulary size of the tokenizer
        self.log_to_terminal(f"the vocabulary size of the tokenizer {encoder.vocab_size()}.")

        return encoded, y

    def predict(self, X: XData) -> npt.NDArray[np.float32]:
        """Predict using the model.

        :param X: XData containing the molecule smiles
        :return: the tokenized sentences"""

        # extract the smiles from the XData
        smiles = X.molecule_smiles
        print(self.get_hash())

        # extract the tokenizer from the pickle file
        with open(f"tm/{self.get_hash()}", 'rb') as f:
            encoder = pickle.load(f)

        # apply the torch tokenizer on the smiles
        encoded = [encoder.encode(smile) for smile in smiles]
        return np.array([pad_tensor(x, length=150) for x in encoded])

    def segment_molecule(self, smile: list[str]) -> list[str]:
        """Transforms the molecule into a sequence of tokens.
        param smile: the smile of the molecule"""

        # # convert the smile to the molecule object
        # mol = Chem.MolFromSmiles(smile)
        #
        # # extract the atoms from the molecule
        # tokens = [atom.GetSymbol() for atom in mol.GetAtoms()]
        #
        # # extract n-grams from the sequence
        # length = len(tokens) - self.window_size + 1
        # [" ".join(tokens[i:i + self.window_size]) for i in range(length)]
        #
        return smile


