"""Module containing Boosted Decision Tree Models."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.typing.xdata import DataRetrieval, XData


@dataclass
class DecisionTrees(VerboseTrainingBlock):
    """Class that implements (Boosted) Decision Tree Models.

    :param bdtm: Name of the model (XGBClassifier, LGBMClassifier, CatBoostClassifier, RandomForestClassifier)
    :param data: Which data to use
    :param n_estimators: Number of estimators
    :param multi_output: Predict one (false) or multiple outputs (true)
    """

    model_name: str = "XGBClassifier"
    data: list[str] = field(default_factory=lambda: ["ECFP_MOL"])
    n_estimators: int = 100
    multi_output: bool = False

    def __post_init__(self) -> None:
        """Post init method."""
        super().__post_init__()

        if len(self.data) > 1:
            raise ValueError("Only one data type is allowed.")

        if self.data[0] not in ["SMILES_MOL", "ECFP_MOL", "EMBEDDING_MOL", "DESCRIPTORS_MOL"]:
            raise ValueError(f"Invalid data type {self.data[0]}.")

        if self.model_name not in ["XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "RandomForestClassifier"]:
            raise ValueError(f"Invalid model name {self.model_name}.")

        if self.model_name in ["LGBMClassifier", "CatBoostClassifier"]:
            raise ValueError("LGBMClassifier and CatBoostClassifier are broken.")

    def custom_train(self, x: XData, y: npt.NDArray[np.int8], **train_args: dict[str, Any]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
        """Train a BDTM classifier.

        :param x: x data
        :param y: labels
        :param train_indices: Train indices
        :param test_indices: Test indices
        """
        # Set the train and test indices
        train_indices: list[int] | dict[str, Any] = train_args.get("train_indices", [])
        test_indices: list[int] | dict[str, Any] = train_args.get("test_indices", [])

        if isinstance(train_indices, dict) or isinstance(test_indices, dict):
            raise TypeError("Wrong input for train/test indices.")

        self.log_to_terminal("Extracting train and test data.")
        x.retrieval = getattr(DataRetrieval, self.data[0])

        X_train = x[train_indices]
        X_test = x[test_indices]
        y_train = y[train_indices]

        if self.data[0] == "ECFP_MOL":
            X_train = np.unpackbits(X_train, axis=1)
            X_test = np.unpackbits(X_test, axis=1)

        # Initialize the XGBoost model and fit it
        if self.model_name == "XGBClassifier":
            self.model = XGBClassifier(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)
        elif self.model_name == "LGBMClassifier":
            self.model = LGBMClassifier(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)
        elif self.model_name == "CatBoostClassifier":
            self.model = CatBoostClassifier(n_estimators=self.n_estimators, random_state=42)
        elif self.model_name == "RandomForestClassifier":
            self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)
        else:
            raise ValueError("Invalid BDTM model.")

        self.log_to_terminal(f"Training {self.model_name} with {self.n_estimators} estimators.")
        self.model.fit(X_train, y_train)
        self.log_to_terminal("Training completed.")

        # Save the model
        self.save_model(f"tm/{self.get_hash()}")

        # Get the predictions
        y_pred_proba = self.model.predict_proba(X_test)
        if self.multi_output:
            return y_pred_proba.flatten(), y[test_indices].flatten()
        return y_pred_proba[:, 1], y[test_indices]

    def custom_predict(self, x: XData) -> npt.NDArray[np.float64]:
        """Predict using an XGBoost classifier.

        :param x: XData
        :return: Predictions
        """
        x.retrieval = getattr(DataRetrieval, self.data[0])
        x_pred = x[:]

        if x_pred is None:
            raise ValueError("No data available.")

        if self.data[0] == "ECFP_MOL":
            x_pred = np.unpackbits(x_pred, axis=1)

        if not hasattr(self, "model"):
            self.model = self.load_model(f"tm/{self.get_hash()}")

        if self.multi_output:
            return self.model.predict_proba(x_pred).flatten()

        self.log_to_terminal("Predicting.")
        return self.model.predict_proba(x_pred)[:, 1]

    def save_model(self, path: str) -> None:
        """Save the model.

        :param path: Path to save model to
        """
        import joblib

        joblib.dump(self.model, path)

    def load_model(self, path: str) -> XGBClassifier | LGBMClassifier | CatBoostClassifier | RandomForestClassifier:
        """Load the model.

        :param path: Path to load model from
        """
        import joblib

        self.model = joblib.load(path)

        return self.model
