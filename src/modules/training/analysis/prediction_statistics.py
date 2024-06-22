"""Module for calculating statistics and graphs from predictions."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from matplotlib import pyplot as plt

from src.modules.objects import TrainObj, TrainPredictObj
from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.scoring.mean_average_precision_scorer import MeanAveragePrecisionScorer
from src.utils.logger import logger


@dataclass
class PredictionStatistics(VerboseTrainingBlock):
    """Create statistics from predictions."""

    def custom_train(self, train_predict_obj: TrainPredictObj, train_obj: TrainObj, **train_args: dict[str, Any]) -> tuple[TrainPredictObj, TrainObj]:
        """Generate statistics from predictions.

        :param train_predict_obj: The train_predict object.
        :param train_obj: The training object.
        :param train_args: Additional arguments.
        """
        predictions = train_predict_obj.y_predictions
        if predictions is None:
            raise ValueError("Predictions are None. Run predict first.")

        labels = train_obj.y_labels_modified if train_obj.y_labels_modified is not None else train_obj.y_labels_original

        # Generate statistics from predictions
        # Format of x is [brd4, hsa, sEH]

        scorer = MeanAveragePrecisionScorer(name="MAP")

        # BRD4 accuracy
        score_brd4 = scorer(labels[:, 0], predictions[:, 0])
        logger.info(f"brd4 val accuracy: {score_brd4}")
        # wandb.log()

        # HSA accuracy
        score_hsa = scorer(labels[:, 1], predictions[:, 1])
        logger.info(f"hsa val accuracy: {score_hsa}")

        # sEH accuracy
        score_seh = scorer(labels[:, 2], predictions[:, 2])
        logger.info(f"seh val accuracy: {score_seh}")

        if wandb.run:
            wandb.log(
                {
                    "BRD4 Val Score": score_brd4,
                    "HSA Val Score": score_hsa,
                    "sEH Val Score": score_seh,
                },
            )

        # Histogram of probabilities
        output_dir: Path | None = train_args.get("output_dir", None)  # type: ignore[assignment]
        if output_dir is not None:
            logger.info(f"Saving histogram of probabilities to {output_dir}")
            visualization_path = output_dir / "visualizations"
            visualization_path.mkdir(exist_ok=True, parents=True)

            # In the histogram apply a sigmoid to the probabilities
            # Log scale the y axis
            plt.figure()
            plt.hist(1 / (1 + np.exp(-predictions[:, 0])), bins=50, alpha=0.5, label="BRD4")
            plt.hist(1 / (1 + np.exp(-predictions[:, 1])), bins=50, alpha=0.5, label="HSA")
            plt.hist(1 / (1 + np.exp(-predictions[:, 2])), bins=50, alpha=0.5, label="sEH")
            plt.yscale("log")
            plt.legend(loc="upper right")
            plt.title("Histogram of probabilities")
            plt.xlabel("Probability")
            plt.ylabel("Frequency")
            plt.savefig(visualization_path / "histogram_validation.png")

        return train_predict_obj, train_obj

    def custom_predict(self, train_predict_obj: TrainPredictObj, **pred_args: Any) -> TrainPredictObj:
        """Predict nothing since statistics not generated for predict.

        :param x: Input
        :return: Input
        """
        return train_predict_obj
