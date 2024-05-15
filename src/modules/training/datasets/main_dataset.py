"""Main dataset for EEG / Spectrogram data."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from epochalyst.pipeline.model.training.training import TrainingPipeline
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from torch.utils.data import Dataset

from src.typing.xdata import DataRetrieval, XData


@dataclass
class MainDataset(Dataset):  # type: ignore[type-arg]
    """Main dataset.

    :param retrieval: The data retrieval object.
    :param steps: The steps to apply to the dataset.

    :param X: The input data.
    :param y: The labels.
    :param indices: The indices to use.
    """

    retrieval: list[str] | None = None
    steps: list[TrainingBlock] | None = None

    X: XData | None = None
    y: npt.NDArray[np.int8] | None = None
    indices: npt.NDArray[np.int32] | None = None

    def __post_init__(self) -> None:
        """Set up the dataset."""
        if self.retrieval is None:
            raise ValueError("Retrieval object must be set.")

        # Setup data retrieval
        self._retrieval_enum = getattr(DataRetrieval, self.retrieval[0])
        for retrieval in self.retrieval[1:]:
            self._retrieval_enum = self._retrieval_enum | getattr(DataRetrieval, retrieval)

        # Setup Pipeline
        self.setup_pipeline(use_augmentations=False)

    def initialize(self, X: XData, y: npt.NDArray[np.int8] | None = None, indices: list[int] | npt.NDArray[np.int32] | None = None) -> None:
        """Set up the dataset for training."""
        self.X = X
        self.y = y
        self.indices = np.array(indices, dtype=np.int32) if isinstance(indices, list) else indices

    def setup_pipeline(self, *, use_augmentations: bool) -> None:
        """Set whether to use the augmentations."""
        self._enabled_steps = []

        if self.steps is not None:
            for step in self.steps:
                if (step.is_augmentation and use_augmentations) or not step.is_augmentation:
                    self._enabled_steps.append(step)

        self._pipeline = TrainingPipeline(self._enabled_steps)
        self._pipeline.log_to_debug = lambda _: None
        self._pipeline.log_section_separator = lambda _: None

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.X is None:
            raise ValueError("Dataset not initialized.")
        if self.indices is None:
            return len(self.X)
        return len(self.indices)

    def __getitems__(self, indices: list[int]) -> tuple[Any, Any]:
        """Get items from the dataset."""
        if self.X is None:
            raise ValueError("Dataset not initialized.")

        if self.indices is not None:
            ind = self.indices[indices]

        self.X.retrieval = self._retrieval_enum
        X = self.X[ind]

        y = self.y[ind] if self.y is not None else None

        X, y = self._pipeline.train(X, y)

        return torch.Tensor(X), torch.Tensor(y)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Get an item from the dataset.

        :param idx: The index to get.
        :return: The data and the labels.
        """
        if self.X is None:
            raise ValueError("Dataset not initialized.")

        if self.indices is not None:
            idx = self.indices[idx]

        self.X.retrieval = self._retrieval_enum
        X = np.expand_dims(self.X[idx], axis=0)
        y = np.expand_dims(self.y[idx], axis=0) if self.y is not None else None

        X, y = self._pipeline.train(X, y)

        return torch.Tensor(X)[0], torch.Tensor(y)[0]
