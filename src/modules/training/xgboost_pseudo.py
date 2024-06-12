"""Module containing Pseudo Boosted Decision Tree Models."""

from dataclasses import dataclass, field
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from typing import Any
import joblib
from sklearn.metrics import accuracy_score
import numpy as np
import numpy.typing as npt
from xgboost import XGBClassifier

from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.typing.xdata import DataRetrieval, XData

@dataclass
class XgboostPseudo(VerboseTrainingBlock):
    """Class that implements Pseudo Boosted Decision Tree Models.

    :param n_estimators: Number of estimators
    :param threshold: Convert the probabilities to the classes
    """

    n_estimators: int = 50
    threshold: float = 0.1
    n_samples: int = 500000

    def custom_train(self, x: XData, y: npt.NDArray[np.int8], **train_args: dict[str, Any]) -> tuple[XData, npt.NDArray[np.int8]]:
        """Train the Boosted Decision Tree and apply on the test molecules.

        :param x: XData containing the molecule fingerprints
        :param y: array containing the protein labels
        """

        # Set the train and validation indices
        train_indices: list[int] | dict[str, Any] = train_args.get("train_indices", [])
        validation_indices: list[int] | dict[str, Any] = train_args.get("validation_indices", [])

        if isinstance(train_indices, dict) or isinstance(validation_indices, dict):
            raise TypeError("Wrong input for train/validation indices.")

        self.log_to_terminal("Extracting train and validation data.")
        x.retrieval = getattr(DataRetrieval, "ECFP_MOL")

        # Extract the indices of the test and train samples
        test_indices = np.unique(np.where(y == [-1, -1, -1])[0])
        mask = np.isin(train_indices, test_indices, invert=True)
        train_indices = train_indices[mask][:self.n_samples]
        print(len(test_indices))

        # Extract the train and test molecules
        X_train = np.unpackbits(x[train_indices], axis=1)
        X_test = np.unpackbits(x[test_indices], axis=1)
        X_val = np.unpackbits(x[validation_indices], axis=1)

        y_train = y[train_indices]
        y_val = y[validation_indices]
        self.log_to_terminal(f"The number of samples in the train: {len(test_indices)}")

        # Initialize the XGBoost model and fit it
        self.model = XGBClassifier(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)

        self.log_to_terminal(f"Training XGBoost with {self.n_estimators} estimators.")
        self.model.fit(X_train, y_train)
        self.log_to_terminal("Training completed.")

        # Predict the labels of the test samples
        y_pred = self.model.predict_proba(X_test)
        y[test_indices] = (y_pred >= self.threshold).astype(int)
        print(np.mean(y[test_indices]))

        # Compute the accuracy of the model on validation
        y_pred = self.model.predict_proba(X_val)
        y_pred = (y_pred >= self.threshold).astype(int)

        accuracy = accuracy_score(y_pred, y_val)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        return x, y

    def custom_predict(self, x: XData, **train_args: dict[str, Any]) -> XData:
        """Predict using an XGBoost classifier.
        :param x: XData containing the molecule fingerprint"""

        return x
