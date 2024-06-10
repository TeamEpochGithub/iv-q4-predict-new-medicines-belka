"""Module to include the molecules in the local or public test."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import duckdb
import joblib
import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.model.training.training_block import TrainingBlock

@dataclass
class PseudoLabel(TrainingBlock):
    """Include the molecules in the local or public test."""

    path: str = 'train.csv'
    n_samples: int = 50
    def train(
            self,
            x: npt.NDArray[np.str_],
            y: npt.NDArray[np.uint8],
            **train_args: Any,
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.uint8]]:
        """Extend the numpy array with the new molecules.

        :param x: array containing the smile strings
        :param y: array containing the protein labels
        """

        # Initialize the path to csv file
        file_path = "data/raw/" + self.path

        # Define query to randomly select samples
        query = f"""
            SELECT *
            FROM read_csv_auto('{file_path}')
            USING SAMPLE {self.n_samples} ROWS
        """

        # Execute the duck db query
        samples = duckdb.query(query).to_df()
        samples = np.array(samples['molecule_smiles'])

        # Concatenate the original and new samples
        x = np.concatenate((x, samples))
        labels = np.array([[0, 0, 0] for _ in range(self.n_samples)])
        y = np.concatenate((y, labels))

        return x, y
    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return True
