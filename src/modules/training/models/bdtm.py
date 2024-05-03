"""Module containing Boosted Decision Tree Models."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.typing.xdata import XData


@dataclass
class BDTM(VerboseTrainingBlock):
    """Class that implements the Boosted Decision Tree Models.

    :param bdtm: Name of the model (XGBClassifier, LGBMClassifier, CatBoostClassifier)
    :param n_estimators: Number of estimators
    """

    bdtm_name: str = "XGBClassifier"
    n_estimators: int = 100
    multi_output: bool = False

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

        X_train = np.array(x.molecule_ecfp)[train_indices]
        X_test = np.array(x.molecule_ecfp)[test_indices]
        y_train = y[train_indices]

        # Initialize the XGBoost model and fit it
        if self.bdtm_name == "XGBClassifier":
            self.bdtm = XGBClassifier(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)
        elif self.bdtm_name == "LGBMClassifier":
            self.bdtm = LGBMClassifier(n_estimators=self.n_estimators, random_state=42, n_jobs=-1)
        elif self.bdtm_name == "CatBoostClassifier":
            self.bdtm = CatBoostClassifier(n_estimators=self.n_estimators, random_state=42, verbose=1)
        else:
            raise ValueError("Invalid BDTM model.")

        self.log_to_terminal(f"Training {self.bdtm_name} with {self.n_estimators} estimators.")
        self.bdtm.fit(X_train, y_train)
        self.log_to_terminal("Training completed.")

        # Save the model
        self.save_model(f"tm/{self.get_hash()}")

        # Get the predictions
        if self.multi_output:
            y_pred_proba = self.bdtm.predict_proba(X_test)
            return y_pred_proba.flatten(), y[test_indices].flatten()
        
        y_pred_proba = self.bdtm.predict_proba(X_test)[:, 1]
        return y_pred_proba, y[test_indices]

    def custom_predict(self, x: XData) -> npt.NDArray[np.float64]:
        """Predict using an XGBoost classifier.

        :param x: XData
        :return: Predictions
        """
        x_pred = x.molecule_ecfp

        if not hasattr(self, "xgb_model"):
            self.bdtm = self.load_model(f"tm/{self.get_hash()}")

        if self.multi_output:
            return self.bdtm.predict_proba(x_pred).flatten()
        
        return self.bdtm.predict_proba(x_pred)[:, 1]

    def save_model(self, path: str) -> None:
        """Save the model.

        :param path: Path to save model to
        """
        import joblib

        joblib.dump(self.bdtm, path)

    def load_model(self, path: str) -> XGBClassifier:
        """Load the model.

        :param path: Path to load model from
        """
        import joblib

        self.bdtm = joblib.load(path)

        return self.bdtm
