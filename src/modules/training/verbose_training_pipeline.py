"""A verbose training pipeline that logs to the terminal and to W&B."""
from typing import Any

from epochalyst._core._caching._cacher import CacheArgs
from epochalyst.pipeline.model.training.training import TrainingPipeline

from src.modules.logging.logger import Logger
from src.modules.objects import TrainObj, TrainPredictObj


class VerboseTrainingPipeline(TrainingPipeline, Logger):
    """A verbose training pipeline that logs to the terminal and to W&B."""

    def train(self, x: Any, y: Any, cache_args: CacheArgs | None = None, **train_args: Any) -> tuple[Any, Any]:  # noqa: ANN401
        """Train the model and return the predictions."""
        train_predict_obj, train_obj = super().train(TrainPredictObj(x_data=x), TrainObj(y_labels_original=y), cache_args=cache_args, **train_args)
        return train_predict_obj.y_predictions, train_obj.y_labels_modified

    def predict(self, x: Any, cache_args: CacheArgs | None = None, **pred_args: Any) -> Any:  # noqa: ANN401
        """Predict using the model."""
        return super().predict(TrainPredictObj(x_data=x), cache_args=cache_args, **pred_args).y_predictions
