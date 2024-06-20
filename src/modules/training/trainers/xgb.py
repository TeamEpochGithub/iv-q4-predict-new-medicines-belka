"""Module for single output lazy xgboost trainer."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import wandb
import xgboost as xgb
from epochalyst.pipeline.model.training.training_block import TrainingBlock

from src.modules.objects import TrainObj, TrainPredictObj
from src.modules.training.datasets.lazy_xgb_dataset import LazyXGBDataset
from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.typing.xdata import DataRetrieval


@dataclass
class BaseXGB(VerboseTrainingBlock):
    """Base xgboost training class.

    :param model_name: Name of model
    :param steps: Steps
    :param retrieval: Data to apply steps to from XData
    :param eval_metric: Evaluation metric
    :param booster: Booster type
    :param eta: Learning rate
    :param max_depth: Maximum depth of tree
    :param objective: Objective function
    :param tree_method: Tree method
    :param max_bin: Maximum number of bins
    :param num_boost_round: Number of boosting rounds
    :param device: Device to use
    :param update: Update model
    :param scale_pos_weight: Scale positive weight
    """

    steps: list[TrainingBlock]
    retrieval: DataRetrieval = DataRetrieval.SMILES_MOL

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

    def save_model(self, path: Path) -> None:
        """Save the model.

        :param path: Path to save model to
        """
        joblib.dump(self.model, path)
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(str(path))
            wandb.log_artifact(model_artifact)
        self.log_to_terminal(f"Model saved to {path}")

    def load_model(self, path: Path) -> xgb.Booster:
        """Load the model.

        :param path: Path to load model from
        """
        self.model = joblib.load(path)

        if self.model is None:
            raise ValueError(f"Couldn't load model from {path}")
        return self.model


@dataclass
class LazySingleXGB(BaseXGB):
    """Lazy xgboost training class.

    :param model_name: Name of model
    :param steps: Lazy steps
    :param retrieval: Data to apply steps to from XData
    :param chunk_size: Chunk size to use
    """

    model_name: str = "LazySingleXGB"
    chunk_size: int = 10000
    queue_size: int = field(default=2, init=True, repr=False, compare=False)

    def custom_train(self, train_predict_obj: TrainPredictObj, train_obj: TrainObj, **train_args: dict[str, Any]) -> tuple[TrainPredictObj, TrainObj]:
        """Train a xgboost model in batches.

        :param x: Input data
        :param y: Labels
        :return: Predictions and labels
        """
        x = train_predict_obj.x_data
        y = train_obj.y_labels_original

        # Set the train and validation indices
        train_indices: list[int] | dict[str, Any] = train_args.get("train_indices", [])
        validation_indices: list[int] | dict[str, Any] = train_args.get("validation_indices", [])

        # Create the model path
        trained_model_path_base = Path(f"tm/{self.get_hash()}")
        trained_model_path_base.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(train_indices, dict) or isinstance(validation_indices, dict):
            raise TypeError("Wrong input for train/validation indices.")

        self.log_to_terminal("Extracting train and validation data.")
        x.retrieval = self.retrieval

        predictions = []
        X_train = x[train_indices]
        X_validation = x[validation_indices]

        for protein in range(3):
            X_train = x[train_indices]
            X_validation = x[validation_indices]
            y_train = y[train_indices][:, protein]
            y_validation = y[validation_indices][:, protein]

            lazy_xgb_dataset = LazyXGBDataset(steps=self.steps, chunk_size=self.chunk_size, max_queue_size=self.queue_size)
            iterator = lazy_xgb_dataset.get_iterator(X_train, y_train)

            # Create the model path
            trained_model_path = Path(f"tm/{self.get_hash()}_{protein}")
            trained_model_path.parent.mkdir(parents=True, exist_ok=True)

            self.log_to_terminal(f"Training {self.model_name}:{protein} with {self.chunk_size} chunk_size.")
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
            self.log_to_terminal(f"Predicting on validation set using XGB:{protein}.")
            validation_iterator = lazy_xgb_dataset.get_iterator(X_validation, y_validation)
            predictions_single = np.array([self.model.predict(validation_data) for validation_data in validation_iterator]).flatten()
            predictions.append(predictions_single)

            # Stop prefetch
            lazy_xgb_dataset.stop_prefetching()

        # Return the predictions and labels
        train_predict_obj.y_predictions = np.array(predictions).T
        train_obj.y_labels_modified = y[validation_indices]

        return train_predict_obj, train_obj

    def custom_predict(self, train_predict_obj: TrainPredictObj) -> TrainPredictObj:
        """Predict using an XGBoost classifier.

        :param x: XData
        :return: Predictions
        """
        x = train_predict_obj.x_data
        x.retrieval = self.retrieval
        predictions = []

        for protein in range(3):
            lazy_xgb_dataset = LazyXGBDataset(steps=self.steps, chunk_size=self.chunk_size, max_queue_size=self.queue_size)
            if not hasattr(self, "model"):
                self.model = self.load_model(Path(f"tm/{self.get_hash()}_{protein}"))
            iterator = lazy_xgb_dataset.get_iterator(x[:], np.zeros(len(x), dtype=np.int8))

            if self.model is None:
                raise ValueError("No model exists")

            self.log_to_terminal(f"Predicting for xgb:{protein}.")
            chunk_index = 0
            predictions_single = []
            for data in iterator:
                self.log_to_terminal(f"Predicting chunk {chunk_index}")
                predictions_single.append(self.model.predict(data))
                chunk_index += 1

            predictions.append(np.array(predictions_single).flatten())

            # Stop prefetch
            lazy_xgb_dataset.stop_prefetching()

        # Return the predictions
        train_predict_obj.y_predictions = np.array(predictions).T

        return train_predict_obj


@dataclass
class LazyMultiXGB(BaseXGB):
    """Lazy xgboost training class.

    :param model_name: Name of model
    :param steps: Lazy steps
    :param retrieval: Data to apply steps to from XData
    :param chunk_size: Chunk size to use
    """

    model_name: str = "LazyXGB"
    chunk_size: int = 10000
    queue_size: int = field(default=2, init=True, repr=False, compare=False)

    def custom_train(self, train_predict_obj: TrainPredictObj, train_obj: TrainObj, **train_args: dict[str, Any]) -> tuple[TrainPredictObj, TrainObj]:
        """Train a xgboost model in batches.

        :param x: Input data
        :param y: Labels
        :return: Predictions and labels
        """
        x = train_predict_obj.x_data
        y = train_obj.y_labels_original

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
        predictions = np.array([self.model.predict(validation_data) for validation_data in validation_iterator])

        # Stop prefetch
        lazy_xgb_dataset.stop_prefetching()

        # Return the predictions and labels
        train_predict_obj.y_predictions = np.concatenate(predictions)
        train_obj.y_labels_modified = y_validation

        return train_predict_obj, train_obj

    def custom_predict(self, train_predict_obj: TrainPredictObj) -> TrainPredictObj:
        """Predict using an XGBoost classifier.

        :param x: XData
        :return: Predictions
        """
        x = train_predict_obj.x_data
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

        # Return the predictions
        train_predict_obj.y_predictions = np.concatenate(predictions)

        return train_predict_obj


@dataclass
class SingleXGB(BaseXGB):
    """Xgboost training class.

    :param model_name: Name of model
    :param steps: Steps
    :param retrieval: Data to apply steps to from XData
    :param chunk_size: Chunk size to use
    """

    model_name: str = "SingleXGB"

    def custom_train(self, train_predict_obj: TrainPredictObj, train_obj: TrainObj, **train_args: dict[str, Any]) -> tuple[TrainPredictObj, TrainObj]:
        """Train a xgboost model in batches.

        :param x: Input data
        :param y: Labels
        :return: Predictions and labels
        """
        x = train_predict_obj.x_data
        y = train_obj.y_labels_original

        # Set the train and validation indices
        train_indices: list[int] | dict[str, Any] = train_args.get("train_indices", [])
        validation_indices: list[int] | dict[str, Any] = train_args.get("validation_indices", [])

        # Create the model path
        trained_model_path_base = Path(f"tm/{self.get_hash()}")
        trained_model_path_base.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(train_indices, dict) or isinstance(validation_indices, dict):
            raise TypeError("Wrong input for train/validation indices.")

        self.log_to_terminal("Extracting train and validation data.")
        x.retrieval = self.retrieval

        predictions = []
        X_train = x[train_indices]
        X_validation = x[validation_indices]
        y_train = y[train_indices]
        y_validation = y[validation_indices]

        for step in self.steps:
            X_train, y_train = step.train(X_train, y_train)
            X_validation, y_validation = step.train(X_validation, y_validation)

        for protein in range(3):
            train_data = xgb.DMatrix(X_train, label=y_train[:, protein])
            validation_data = xgb.DMatrix(X_validation, label=y_validation[:, protein])

            # Create the model path
            trained_model_path = Path(f"tm/{self.get_hash()}_{protein}")
            trained_model_path.parent.mkdir(parents=True, exist_ok=True)

            self.log_to_terminal(f"Training {self.model_name}:{protein}")
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
                model = xgb.train(params, train_data, num_boost_round=self.num_boost_round, xgb_model=model, verbose_eval=5)
                chunk_index += 1
                if model is None:
                    raise ValueError("XGBoost didn't train, maybe there was no data")
                self.model = model
                self.log_to_terminal("Training completed.")

            self.save_model(trained_model_path)

            # Get the predictions
            self.log_to_terminal(f"Predicting on validation set using XGB:{protein}.")
            predictions_single = self.model.predict(validation_data)
            predictions.append(predictions_single)

        # Return the predictions and labels
        train_predict_obj.y_predictions = np.array(predictions).T
        train_obj.y_labels_modified = y[validation_indices]

        return train_predict_obj, train_obj

    def custom_predict(self, train_predict_obj: TrainPredictObj) -> TrainPredictObj:
        """Predict using an XGBoost classifier.

        :param x: XData
        :return: Predictions
        """
        x = train_predict_obj.x_data
        x.retrieval = self.retrieval
        predictions = []
        for step in self.steps:
            x, y = step.train(x, None)
        test_data = xgb.DMatrix(x)

        for protein in range(3):
            if not hasattr(self, "model"):
                self.model = self.load_model(Path(f"tm/{self.get_hash()}_{protein}"))

            if self.model is None:
                raise ValueError("No model exists")

            self.log_to_terminal(f"Predicting for xgb:{protein}.")
            predictions_single = self.model.predict(test_data)

            predictions.append(np.array(predictions_single))

        # Return the predictions
        train_predict_obj.y_predictions = np.array(predictions).T

        return train_predict_obj


@dataclass
class MultiXGB(BaseXGB):
    """Xgboost training class.

    :param model_name: Name of model
    :param steps: Steps
    :param retrieval: Data to apply steps to from XData
    :param chunk_size: Chunk size to use
    """

    model_name: str = "MultiXGB"

    def custom_train(self, train_predict_obj: TrainPredictObj, train_obj: TrainObj, **train_args: dict[str, Any]) -> tuple[TrainPredictObj, TrainObj]:
        """Train a xgboost model in batches.

        :param x: Input data
        :param y: Labels
        :return: Predictions and labels
        """
        x = train_predict_obj.x_data
        y = train_obj.y_labels_original

        # Set the train and validation indices
        train_indices: list[int] | dict[str, Any] = train_args.get("train_indices", [])
        validation_indices: list[int] | dict[str, Any] = train_args.get("validation_indices", [])

        # Create the model path
        trained_model_path_base = Path(f"tm/{self.get_hash()}")
        trained_model_path_base.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(train_indices, dict) or isinstance(validation_indices, dict):
            raise TypeError("Wrong input for train/validation indices.")

        self.log_to_terminal("Extracting train and validation data.")
        x.retrieval = self.retrieval

        X_train = x[train_indices]
        X_validation = x[validation_indices]
        y_train = y[train_indices]
        y_validation = y[validation_indices]

        for step in self.steps:
            X_train, y_train = step.train(X_train, y_train)
            X_validation, y_validation = step.train(X_validation, y_validation)

        train_data = xgb.DMatrix(X_train, label=y_train)
        validation_data = xgb.DMatrix(X_validation, label=y_validation)

        # Create the model path
        trained_model_path = Path(f"tm/{self.get_hash()}")
        trained_model_path.parent.mkdir(parents=True, exist_ok=True)

        self.log_to_terminal(f"Training {self.model_name}")
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
        model = None

        if trained_model_path.exists():
            self.log_to_terminal(f"Loading model from {trained_model_path}")
            self.model = self.load_model(trained_model_path)
        else:
            model = xgb.train(params, train_data, num_boost_round=self.num_boost_round, xgb_model=model, verbose_eval=5)
            if model is None:
                raise ValueError("XGBoost didn't train, maybe there was no data")
            self.model = model
            self.log_to_terminal("Training completed.")

        self.save_model(trained_model_path)

        # Get the predictions
        self.log_to_terminal("Predicting on validation set using XGB")
        predictions = self.model.predict(validation_data)

        # Return the predictions and labels
        train_predict_obj.y_predictions = np.array(predictions)
        train_obj.y_labels_modified = y[validation_indices]

        return train_predict_obj, train_obj

    def custom_predict(self, train_predict_obj: TrainPredictObj) -> TrainPredictObj:
        """Predict using an XGBoost classifier.

        :param x: XData
        :return: Predictions
        """
        x = train_predict_obj.x_data
        x.retrieval = self.retrieval
        predictions = []
        for step in self.steps:
            x, y = step.train(x, None)
        test_data = xgb.DMatrix(x)

        if not hasattr(self, "model"):
            self.model = self.load_model(Path(f"tm/{self.get_hash()}"))

        if self.model is None:
            raise ValueError("No model exists")

        self.log_to_terminal("Predicting for xgb.")
        predictions = self.model.predict(test_data)

        # Return the predictions
        train_predict_obj.y_predictions = np.array(predictions)

        return train_predict_obj
