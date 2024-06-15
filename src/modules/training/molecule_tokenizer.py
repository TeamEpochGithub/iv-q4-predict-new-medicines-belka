"""Train a torch tokenizer on the molecule smiles."""

import pickle
from dataclasses import dataclass
from typing import Any

from torchnlp.encoders.text import StaticTokenizerEncoder  # type: ignore[import-not-found]
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer  # type: ignore[import-not-found]
from src.modules.objects import TrainObj, TrainPredictObj
from src.modules.training.verbose_training_block import VerboseTrainingBlock

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MTR")
@dataclass
class MoleculeTokenizer(VerboseTrainingBlock):
    """Train a torch tokenizer on the blocks or molecule smiles."""

    training: str = "10M"
    window_size: int = 6

    def segment_molecule(self, smile: str) -> list[str]:
        """Transform the molecule smile for the tokenizer.

        :param smile: string representing the molecule smile
        :return: list containing the substructures
        """
        # Extract n-grams from the sequence
        length = len(smile) - self.window_size + 1
        return [" ".join(smile[i: i + self.window_size]) for i in range(length)]


    def custom_train(self, train_predict_obj: TrainPredictObj, train_obj: TrainObj, **train_args: dict[str, Any]) -> tuple[TrainPredictObj, TrainObj]:
        """Train the torch tokenizer on the molecule smiles.

        :param x: XData containing the molecule smiles
        :param y: array containing the molecule labels
        :return: The tokenized sentences and labels
        """
        self.log_to_terminal(f"start training the tokenizer on {self.training}.")

        # Check whether the molecule smiles are present
        if train_predict_obj.x_data.molecule_smiles is None:
            raise ValueError("There is no SMILE information for the molecules")

        # Extract the molecule smiles as a list for torch
        tqdm_smiles = tqdm(list(train_predict_obj.x_data.molecule_ecfp), desc="Tokenizing molecules")
        encoder = StaticTokenizerEncoder(tqdm_smiles, tokenize=self.segment_molecule)

        # Print the vocabulary size of the tokenizer
        self.log_to_terminal(f"The vocabulary size of the tokenizer {encoder.vocab_size}.")

        # Save the tokenizer as a pickle file
        with open(f"tm/tokenizer_samples={self.training}_window={self.window_size!s}.pkl", "wb") as f:
            pickle.dump(encoder.index_to_token, f, protocol=pickle.HIGHEST_PROTOCOL)

        return train_predict_obj, train_obj

    def custom_predict(self, train_predict_obj: TrainPredictObj, **pred_args: Any) -> TrainPredictObj:
        """Predict using the model.

        :param X: XData containing the molecule smiles
        :return: the tokenized sentences
        """
        return train_predict_obj
