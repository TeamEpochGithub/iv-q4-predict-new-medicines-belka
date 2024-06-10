"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
import os
import warnings
from contextlib import nullcontext
from pathlib import Path

import coloredlogs
import hydra
import numpy as np
import numpy.typing as npt
import wandb
from epochalyst.logging.section_separator import print_section_separator
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from src.config.train_config import TrainConfig
from src.setup.setup_data import GetXCache, GetYCache, setup_xy
from src.setup.setup_pipeline import setup_pipeline
from src.setup.setup_runtime_args import create_cache_path, setup_cache_args, setup_train_args
from src.setup.setup_wandb import setup_wandb
from src.splitter.base import Splitter
from src.typing.xdata import XData, slice_copy
from src.utils.lock import Lock
from src.utils.logger import logger
from src.utils.set_torch_seed import set_torch_seed

# Set logging
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"  # Makes hydra give full error messages

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:
    """Train a model pipeline with a train-test split. Entry point for Hydra which loads the config file."""
    # Install coloredlogs
    coloredlogs.install()

    # Run the train config with an optional lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext

    with optional_lock():
        run_train_cfg(cfg)
        """
        #try:
        #    run_train_cfg(cfg)
        #except hydra.errors.InstantiationException as e:
        #    logger.error("Training failed to instantiate.")
        #    if wandb.run:
        #        wandb.log(
                    {
                        "Validation Score": -0.1,
                        "Test Score": -0.1,
                        "Combined Score": -0.1,
                    },
                )
            logger.error(e)
        except ValueError as e:
            logger.error(e)
            if wandb.run:
                wandb.log(
                    {
                        "Validation Score": -0.1,
                        "Test Score": -0.1,
                        "Combined Score": -0.1,
                    },
                )"""


def run_train_cfg(cfg: DictConfig) -> None:
    """Train a model pipeline with a train-test split."""
    print_section_separator("Q4 - Detect Medicine - Training")

    # Set seed
    set_torch_seed()

    # Setup Weights & Biases
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    if cfg.wandb.enabled:
        setup_wandb(cfg, "train", output_dir)

    # Preload the pipeline
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg)

    # Setup cache arguments
    cache_path = create_cache_path(cfg.cache_path, cfg.splitter, cfg.sample_size, cfg.sample_split)
    splitter_cache_path = cache_path / "splits.pkl"
    cache_args_x, cache_args_y, cache_args_train = setup_cache_args(cache_path)

    # Setup the data
    X: XData | None = None
    y: npt.NDArray[np.int_] | None = None
    train_indices: npt.NDArray[np.int64]
    validation_indices: npt.NDArray[np.int64] | None = None
    test_indices: npt.NDArray[np.int64] | None = None

    # Check if the data is cached and load if not
    if (
        not model_pipeline.get_x_cache_exists(cache_args_x)
        or not model_pipeline.get_y_cache_exists(cache_args_y)
        or (cfg.splitter is not None and not splitter_cache_path.exists())
    ):
        X, y = setup_xy(cfg)

    # Split the data into train and test if required
    if cfg.splitter is None:
        logger.info("Training on all data (full).")
        with GetXCache(model_pipeline, cache_args_x, X) as X:
            train_indices = np.arange(len(X))
        fold_idx = -1
    else:
        fold_idx = 0
        splitter: Splitter = instantiate(cfg.splitter)
        if splitter.includes_test:
            logger.info("Splitting data into train, validation and test sets.")
            splits: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]
            splits, _, test_indices = splitter.split(X=X, y=y, cache_path=splitter_cache_path)  # type: ignore[assignment, misc]
            train_indices, validation_indices = splits[fold_idx]
        else:
            logger.info("Splitting Data into train and validation sets.")
            train_indices, validation_indices = splitter.split(X=X, y=y, cache_path=splitter_cache_path)[fold_idx]  # type: ignore[assignment, misc]

    # Run the pipeline and score the results
    print_section_separator("Train model pipeline")
    train_args = setup_train_args(
        pipeline=model_pipeline,
        cache_args_x=cache_args_x,
        cache_args_y=cache_args_y,
        cache_args_train=cache_args_train,
        train_indices=train_indices,
        validation_indices=validation_indices,
        save_model=True,
        fold=fold_idx,
        output_dir=output_dir,
    )

    # Train Model and make predictions on the validation set
    validation_predictions, _ = model_pipeline.train(X, y, **train_args)

    # Make predictions on the test set if it exists
    test_predictions = None
    with GetXCache(model_pipeline, cache_args_x, X) as X:
        if test_indices is not None:
            X_sliced = slice_copy(X, test_indices)
            test_predictions = model_pipeline.predict(X_sliced)
            del X_sliced

    # Score the predictions
    with GetYCache(model_pipeline, cache_args_y, y) as y:
        scoring(
            y=y,
            validation_predictions=validation_predictions,
            validation_indices=validation_indices,
            test_predictions=test_predictions,
            test_indices=test_indices,
            cfg=cfg,
            output_dir=output_dir,
        )
    wandb.finish()


def scoring(
    y: npt.NDArray[np.int_],
    validation_predictions: npt.NDArray[np.int_] | None,
    validation_indices: npt.NDArray[np.int_] | None,
    test_predictions: npt.NDArray[np.int_] | None,
    test_indices: npt.NDArray[np.int_] | None,
    cfg: DictConfig,
    output_dir: Path | None = None,
) -> None:
    """Score the predictions and possible validation.

    :param cfg: The dictionary configuration
    :param validation_indices: the validation indices
    :param y_new: The test set labels
    :param predictions: The predictions on the validation set
    :param model_pipeline: The model pipeline
    :param x_val: XData for test set
    :param y_val: Labels for test y
    """
    print_section_separator("Scoring")

    # Set the scores to -1
    validation_score = -1.0
    test_score = -1.0
    combined_score = -1.0

    # Instantiate the scorer
    scorer = instantiate(cfg.scorer)

    # Score the validation set
    if validation_indices is not None and validation_predictions is not None:
        logger.info("Scoring on validation set")
        validation_score = scorer(y[validation_indices], validation_predictions)

    # Score the test set
    if test_indices is not None and test_predictions is not None:
        logger.info("Scoring on test set")
        test_score = scorer(y[test_indices], test_predictions)
        combined_score = 0.5 * validation_score + 0.5 * test_score

        # BRD4 accuracy
        score_brd4 = scorer(y[test_indices][:, 0], test_predictions[:, 0])
        logger.info(f"brd4 test accuracy: {score_brd4}")
        # wandb.log()

        # HSA accuracy
        score_hsa = scorer(y[test_indices][:, 1], test_predictions[:, 1])
        logger.info(f"hsa test accuracy: {score_hsa}")

        # sEH accuracy
        score_seh = scorer(y[test_indices][:, 2], test_predictions[:, 2])
        logger.info(f"seh test accuracy: {score_seh}")

        if wandb.run:
            wandb.log(
                {
                    "BRD4 Test Score": score_brd4,
                    "HSA Test Score": score_hsa,
                    "sEH Test Score": score_seh,
                },
            )

        if output_dir is not None:
            logger.info(f"Saving histogram of probabilities to {output_dir}")
            visualization_path = output_dir / "visualizations"
            visualization_path.mkdir(exist_ok=True, parents=True)

            # In the histogram apply a sigmoid to the probabilities
            # Log scale the y axis
            plt.figure()
            plt.hist(1 / (1 + np.exp(-test_predictions[:, 0])), bins=50, alpha=0.5, label="BRD4")
            plt.hist(1 / (1 + np.exp(-test_predictions[:, 1])), bins=50, alpha=0.5, label="HSA")
            plt.hist(1 / (1 + np.exp(-test_predictions[:, 2])), bins=50, alpha=0.5, label="sEH")
            plt.yscale("log")
            plt.legend(loc="upper right")
            plt.title("Histogram of probabilities")
            plt.xlabel("Probability")
            plt.ylabel("Frequency")
            plt.savefig(visualization_path / "histogram_test.png")

    # Report the scores
    logger.info(f"Validation Score: {validation_score:.6f}")
    logger.info(f"Test Score: {test_score:.6f}")
    logger.info(f"Combined Score: {combined_score:.6f}")
    if wandb.run:
        wandb.log(
            {
                "Validation Score": validation_score,
                "Test Score": test_score,
                "Combined Score": combined_score,
            },
        )


if __name__ == "__main__":
    # Run the train function
    run_train()
