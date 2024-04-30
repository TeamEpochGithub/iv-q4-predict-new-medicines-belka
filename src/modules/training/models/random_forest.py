"""Module containing RandomForestClassfier class."""

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier

from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.typing.xdata import XData


class RandomForestModel(VerboseTrainingBlock):
    """Class that implements the random forest classifier."""

    def custom_train(self, x: XData, y: npt.NDArray[np.int8], **train_args: dict[str, Any]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
        """Train a random forest classifier.

        :param x: x data
        :param y: labels
        :param train_indices: Train indices
        :param test_indices: Test indices
        """
        train_indices = train_args.get("train_indices", [])
        test_indices = train_args.get("test_indices", [])
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

        X_train = x.molecule_smiles[train_indices]
        X_test = x.molecule_smiles[test_indices]
        y_train = y[train_indices]

        self.rf_model.fit(X_train, y_train)

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
