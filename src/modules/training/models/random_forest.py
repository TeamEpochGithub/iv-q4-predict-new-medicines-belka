from sklearn.ensemble import RandomForestClassifier
import numpy.typing as npt
import numpy as np
from src.typing.xdata import XData
from src.modules.training.verbose_training_block import VerboseTrainingBlock


class RandomForestModel(VerboseTrainingBlock):

    def custom_train(self, x: XData, y: npt.NDArray[np.int8], train_indices: list[int], test_indices: list[int]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
        """Train a random forest classifier."""

        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

        X_train = x.molecule_smiles[train_indices]
        X_test = x.molecule_smiles[test_indices]
        y_train = y[train_indices]

        self.rf_model.fit(X_train, y_train)

        y_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]

        self.t

        return y_pred_proba, y

    def custom_predict(self, x: XData):
        x_pred = x.molecule_smiles

        return self.rf_model.predict_proba(x_pred)[:, 1]
