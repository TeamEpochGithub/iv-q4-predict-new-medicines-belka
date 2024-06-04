"""Module for calculating statistics and graphs from predictions."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import wandb
from matplotlib import pyplot as plt

from src.modules.training.verbose_training_block import VerboseTrainingBlock
from src.scoring.mean_average_precision_scorer import MeanAveragePrecisionScorer
from src.utils.logger import logger


@dataclass
class PredictionStatistics(VerboseTrainingBlock):
    """Create statistics from predictions."""

    def custom_train(self, x: npt.NDArray[np.float_], y: npt.NDArray[np.int8], **train_args: Any) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
        """Generate statistics from predictions.

        :param x: Predictions
        :param y: Labels
        :return: Original x and y
        """
        # Generate statistics from predictions
        # Format of x is [brd4, hsa, sEH]

        scorer = MeanAveragePrecisionScorer(name="MAP")

        # BRD4 accuracy
        score_brd4 = scorer(y[:, 0], x[:, 0])
        logger.info(f"brd4 val accuracy: {score_brd4}")
        # wandb.log()

        # HSA accuracy
        score_hsa = scorer(y[:, 1], x[:, 1])
        logger.info(f"hsa val accuracy: {score_hsa}")

        # sEH accuracy
        score_seh = scorer(y[:, 2], x[:, 2])
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
        output_dir = train_args.get("output_dir", None)
        if output_dir is not None:
            logger.info(f"Saving histogram of probabilities to {output_dir}")
            visualization_path = output_dir / "visualizations"
            visualization_path.mkdir(exist_ok=True, parents=True)

            # In the histogram apply a sigmoid to the probabilities
            # Log scale the y axis
            plt.figure()
            plt.hist(1 / (1 + np.exp(-x[:, 0])), bins=50, alpha=0.5, label="BRD4")
            plt.hist(1 / (1 + np.exp(-x[:, 1])), bins=50, alpha=0.5, label="HSA")
            plt.hist(1 / (1 + np.exp(-x[:, 2])), bins=50, alpha=0.5, label="sEH")
            plt.yscale("log")
            plt.legend(loc="upper right")
            plt.title("Histogram of probabilities")
            plt.xlabel("Probability")
            plt.ylabel("Frequency")
            plt.savefig(visualization_path / "histogram_validation.png")

        return x, y

    def custom_predict(self, x: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Predict nothing since statistics not generated for predict.

        :param x: Input
        :return: Input
        """
        return x
