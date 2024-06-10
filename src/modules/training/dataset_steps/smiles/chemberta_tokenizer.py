"""Module to tokenize the molecule for chemberta.yaml model."""

from dataclasses import dataclass
from typing import Any
from transformers import AutoTokenizer
import numpy as np
import numpy.typing as npt
from epochalyst._core._caching._cacher import CacheArgs
from epochalyst.pipeline.model.training.training_block import TrainingBlock


@dataclass
class ChembertaTokenizer(TrainingBlock):
    """Module to tokenize the molecule for chemberta.yaml model.

    :param model_name: the name of the Hugging Face model
    :param padding_size: the maximum length to pad to"""

    model_name: str = "DeepChem/ChemBERTa-10M-MTR"
    padding_size: int = 150
    embedding_layer: bool = False

    def train(
            self,
            x: npt.NDArray[np.str_],
            y: npt.NDArray[np.uint8],
            _cache_args: CacheArgs | None = None,
            **train_args: Any,
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        """Convert the molecule smiles into a sequence of integers.

        :param x: array containing the molecule smiles
        :param y: array containing the protein labels"""

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        output = tokenizer(list(x), truncation=True, padding='max_length', max_length=self.padding_size)

        if self.embedding_layer:
            return np.array(output['input_ids']), y
        else:
            return output, y
    @property
    def is_augmentation(self) -> bool:
        """Check if augmentation is enabled."""
        return False

