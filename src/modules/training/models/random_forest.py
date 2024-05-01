"""Module containing RandomForestClassfier class."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier

from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.typing.xdata import XData


@dataclass
class RandomForestModel(VerboseTrainingBlock):
    """Class that implements the random forest classifier."""

    n_estimators: int = 100

    def custom_train(self, x: XData, y: npt.NDArray[np.int8], **train_args: dict[str, Any]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
        """Train a random forest classifier.

        :param x: x data
        :param y: labels
        :param train_indices: Train indices
        :param test_indices: Test indices
        """
        train_indices: list[int] | dict[str, Any] = train_args.get("train_indices", [])
        test_indices: list[int] | dict[str, Any] = train_args.get("test_indices", [])

        if isinstance(train_indices, dict) or isinstance(test_indices, dict):
            raise TypeError("Wrong input for train/test indices.")

        self.rf_model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42, verbose=1, n_jobs=-1)

        X_train = np.array(x.molecule_ecfp)[train_indices]
        X_test = np.array(x.molecule_ecfp)[test_indices]
        y_train = y[train_indices]

        self.log_to_terminal(f"Training Random Forest with {self.n_estimators} estimators.")
        self.rf_model.fit(X_train, y_train)
        self.log_to_terminal("Training completed.")

        y_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]

        # Save the model
        self.save_model(f"tm/{self.get_hash()}")

        return y_pred_proba, y

    def custom_predict(self, x: XData) -> npt.NDArray[np.float64]:
        """Predict using a random forest classifier.

        :param x: XData
        :return: Predictions
        """
        x_pred = x.molecule_smiles

        if not hasattr(self, "rf_model"):
            self.rf_model = self.load_model(f"tm/{self.get_hash()}")

        return self.rf_model.predict_proba(x_pred)[:, 1]

    def save_model(self, path: str) -> None:
        """Save the model.

        :param path: Path to save model to
        """
        import joblib

        joblib.dump(self.rf_model, path)

    def load_model(self, path: str) -> RandomForestClassifier:
        """Load the model.

        :param path: Path to load model from
        """
        import joblib

        self.rf_model = joblib.load(path)

        return self.rf_model
