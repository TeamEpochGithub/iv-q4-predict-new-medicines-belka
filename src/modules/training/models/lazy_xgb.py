"""Module for lazy xgboost trainer."""
from dataclasses import dataclass, field
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
    queue_size: int = field(default=2, init=True, repr=False, compare=False)

    # Model parameters
    eval_metric: str = "map"
    booster: str = "gbtree"
    eta: float = 0.1
    max_depth: int = 6
    objective: str = "binary:logistic"
    tree_method: str = "hist"
    max_bin: int = 256

    # Training parameters
    num_boost_round: int = 100
    device: str = "cuda"
    update: bool = False
    scale_pos_weight: int = 1

    def custom_train(self, x: XData, y: npt.NDArray[np.int8], **train_args: dict[str, Any]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
        """Train a xgboost model in batches.

        :param x: Input data
        :param y: Labels
        :return: Predictions and labels
        """
        # Set the train and validation indices
        train_indices: list[int] | dict[str, Any] = train_args.get("train_indices", [])
        validation_indices: list[int] | dict[str, Any] = train_args.get("validation_indices", [])

        # Create the model path
        trained_model_path = Path(f"tm/{self.get_hash()}")
        trained_model_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(train_indices, dict) or isinstance(validation_indices, dict):
            raise TypeError("Wrong input for train/validation indices.")

        self.log_to_terminal("Extracting train and validation data.")
        x.retrieval = self.retrieval

        X_train = x[train_indices]
        X_validation = x[validation_indices]
        y_train = y[train_indices]
        y_validation = y[validation_indices]

        lazy_xgb_dataset = LazyXGBDataset(steps=self.steps, chunk_size=self.chunk_size, max_queue_size=self.queue_size)
        iterator = lazy_xgb_dataset.get_iterator(X_train, y_train)

        self.log_to_terminal(f"Training {self.model_name} with {self.chunk_size} chunk_size.")
        params = {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "booster": self.booster,
            "eta": self.eta,
            "max_depth": self.max_depth,
            "device": self.device,
            "tree_method": self.tree_method,
            "max_bin": self.max_bin,
            "scale_pos_weight": self.scale_pos_weight,
        }
        chunk_index = 0
        model = None

        if trained_model_path.exists():
            self.log_to_terminal(f"Loading model from {trained_model_path}")
            self.model = self.load_model(trained_model_path)
        else:
            for data in iterator:
                if chunk_index == 1 and self.update:
                    params.update(
                        {
                            "process_type": "update",
                            "updater": "refresh",
                            "refresh_leaf": True,
                        },
                    )
                self.log_to_terminal(f"Training chunk {chunk_index}")
                model = xgb.train(params, data, num_boost_round=self.num_boost_round, xgb_model=model, verbose_eval=5)
                chunk_index += 1
            if model is None:
                raise ValueError("XGBoost didn't train, maybe there was no data")
            self.model = model
            self.log_to_terminal("Training completed.")

        self.save_model(trained_model_path)

        # Get the predictions
        validation_iterator = lazy_xgb_dataset.get_iterator(X_validation, y_validation)
        predictions = [self.model.predict(validation_data) for validation_data in validation_iterator]

        # Stop prefetch
        lazy_xgb_dataset.stop_prefetching()

        return np.concatenate(predictions), y_validation

    def custom_predict(self, x: XData) -> npt.NDArray[np.float64]:
        """Predict using an XGBoost classifier.

        :param x: XData
        :return: Predictions
        """
        lazy_xgb_dataset = LazyXGBDataset(steps=self.steps, chunk_size=self.chunk_size, max_queue_size=self.queue_size)
        x.retrieval = self.retrieval
        if not hasattr(self, "model"):
            self.model = self.load_model(Path(f"tm/{self.get_hash()}"))
        iterator = lazy_xgb_dataset.get_iterator(x[:], np.zeros(len(x), dtype=np.int8))

        if self.model is None:
            raise ValueError("No model exists")

        self.log_to_terminal("Predicting.")
        predictions = []
        chunk_index = 0
        for data in iterator:
            self.log_to_terminal(f"Predicting chunk {chunk_index}")
            predictions.append(self.model.predict(data))
            chunk_index += 1

        # Stop prefetch
        lazy_xgb_dataset.stop_prefetching()

        return np.concatenate(predictions)

    def save_model(self, path: Path) -> None:
        """Save the model.

        :param path: Path to save model to
        """
        joblib.dump(self.model, path)

    def load_model(self, path: Path) -> xgb.Booster:
        """Load the model.

        :param path: Path to load model from
        """
        self.model = joblib.load(path)

        if self.model is None:
            raise ValueError(f"Couldn't load model from {path}")
        return self.model
