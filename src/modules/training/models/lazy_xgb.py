"""Module for lazy xgboost trainer."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import xgboost as xgb
from epochalyst.pipeline.model.training.training_block import TrainingBlock

from src.modules.training.datasets.lazy_xgb_dataset import LazyXGBDataset
from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.typing.xdata import DataRetrieval, XData


@dataclass
class LazyXGB(VerboseTrainingBlock):
    """Lazy xgboost training class.

    :param model_name: Name of model
    :param steps: Lazy steps
    :param retrieval: Data to apply steps to from XData
    :param chunk_size: Chunk size to use
    """

    model_name: "LazyXGB"
    steps: list[TrainingBlock]
    retrieval: DataRetrieval = DataRetrieval.SMILES_MOL
    chunk_size: int = 10000

    def custom_train(self, x: XData, y: npt.NDArray[np.int8], **train_args: dict[str, Any]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
        """Train a xgboost model in batches.

        :param x: Input data
        :param y: Labels
        :return: Predictions and labels
        """
        # Set the train and test indices
        train_indices: list[int] | dict[str, Any] = train_args.get("train_indices", [])
        test_indices: list[int] | dict[str, Any] = train_args.get("test_indices", [])

        if isinstance(train_indices, dict) or isinstance(test_indices, dict):
            raise TypeError("Wrong input for train/test indices.")

        self.log_to_terminal("Extracting train and test data.")
        x.retrieval = self.retrieval

        X_train = x[train_indices]
        X_test = x[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        lazy_xgb_dataset = LazyXGBDataset(steps=self.steps, chunk_size=self.chunk_size)
        iterator = lazy_xgb_dataset.get_iterator(X_train, y_train)

        self.log_to_terminal(f"Training {self.model_name} with {self.chunk_size} chunk_size.")
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "eta": 0.1,
            "max_depth": 6,
        }
        num_boost_round = 100
        chunk_index = 0
        model = None
        for data in iterator:
            self.log_to_terminal(f"Training chunk {chunk_index}")
            model = xgb.train(params, data, num_boost_round=num_boost_round, xgb_model=model)
            chunk_index += 1
        if model is None:
            raise ValueError("XGBoost didn't train, maybe there was no data")
        self.model = model
        self.log_to_terminal("Training completed.")

        # Save the model
        trained_model_path = Path(f"tm/{self.get_hash()}")
        trained_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_model(trained_model_path)

        # Get the predictions
        test_iterator = lazy_xgb_dataset.get_iterator(X_test, y_test)
        predictions = [self.model.predict(test_data) for test_data in test_iterator]

        return np.concatenate(predictions), y_test

    def custom_predict(self, x: XData) -> npt.NDArray[np.float64]:
        """Predict using an XGBoost classifier.

        :param x: XData
        :return: Predictions
        """
        lazy_xgb_dataset = LazyXGBDataset(steps=self.steps, chunk_size=self.chunk_size)
        x.retrieval = self.retrieval
        if not hasattr(self, "model"):
            self.model = self.load_model(f"tm/{self.get_hash()}")
        iterator = lazy_xgb_dataset.get_iterator(x[:], np.zeros(len(x), dtype=np.int8))

        if self.model is None:
            raise ValueError("No model exists")

        self.log_to_terminal("Predicting.")
        predictions = []
        chunk_index = 0
        for data in iterator:
            self.log_to_terminal(f"Predicting chunk {chunk_index}")
            predictions.append(self.model.predict(data))

        return np.concatenate(predictions)

    def save_model(self, path: Path) -> None:
        """Save the model.

        :param path: Path to save model to
        """
        joblib.dump(self.model, path)

    def load_model(self, path: str) -> xgb.Booster:
        """Load the model.

        :param path: Path to load model from
        """
        self.model = joblib.load(path)

        if self.model is None:
            raise ValueError(f"Couldn't load model from {path}")
        return self.model
