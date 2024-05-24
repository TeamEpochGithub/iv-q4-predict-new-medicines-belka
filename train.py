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
from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.model import ModelPipeline
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config.train_config import TrainConfig
from src.setup.setup_data import setup_xy
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

    # Cache/Processed Path
    cache_path = create_cache_path(cfg.cache_path, cfg.splitter, cfg.sample_size, cfg.sample_split)
    splitter_cache_path = cache_path / "splits.pkl"

    # Cache arguments
    cache_args_x, cache_args_y, cache_args_train = setup_cache_args(cache_path)

    # Setup the data
    X = XData(np.array([1]))
    y = np.array([1])
    x_test, y_test = None, None

    train_indices: npt.NDArray[np.int64]
    validation_indices: npt.NDArray[np.int64]

    # Check if the data is cached
    if (
        not model_pipeline.get_x_cache_exists(cache_args_x)
        or not model_pipeline.get_y_cache_exists(cache_args_y)
        or (cfg.splitter is not None and not splitter_cache_path.exists())
    ):
        X, y = setup_xy(cfg)

    # Split the data into train and test if required
    if cfg.splitter is None:
        logger.info("Training on all data (full).")
        train_indices, validation_indices = np.arange(len(X)), np.array([])
        fold_idx = -1
    else:
        fold_idx = 0
        splitter: Splitter = instantiate(cfg.splitter)
        if splitter.includes_validation:
            logger.info("Splitting data into train, validation and test sets.")

            splits: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]
            train_validation_indices: npt.NDArray[np.int64]
            test_indices: npt.NDArray[np.int64]
            splits, train_validation_indices, test_indices = splitter.split(X=X, y=y, cache_path=splitter_cache_path)  # type: ignore[assignment]

            # Setup test data
            x_test = slice_copy(X, test_indices)
            y_test = y[test_indices].flatten()
            logger.info(f"Binds in test: {np.count_nonzero(y_test == 1) * 100 / len(y_test):.2f}%")

            # Zero-index the split
            train_indices, validation_indices = splits[fold_idx]
            X.slice_all(train_validation_indices)
            y = y[train_validation_indices]
        else:
            logger.info("Splitting Data into train and validation sets.")
            train_indices, validation_indices = splitter.split(X=X, y=y, cache_path=splitter_cache_path)[fold_idx]  # type: ignore[assignment]
        logger.info(f"Train/Test size: {len(train_indices)}/{len(validation_indices)}, Bind {np.count_nonzero(y == 1) * 100 / (len(y) * 3)}%")

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
    )
    predictions, y_new = model_pipeline.train(X, y, **train_args)
    scoring(cfg=cfg, validation_indices=validation_indices, y_new=y_new, predictions=predictions, x_test=x_test, y_test=y_test, model_pipeline=model_pipeline)

    wandb.finish()


def scoring(
    cfg: DictConfig,
    validation_indices: npt.NDArray[np.int_],
    y_new: npt.NDArray[np.int_],
    predictions: npt.NDArray[np.int_],
    model_pipeline: ModelPipeline | EnsemblePipeline,
    x_test: XData | None = None,
    y_test: npt.NDArray[np.int_] | None = None,
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
    combined_score_val_percentage = None

    # Instantiate the scorer
    scorer = instantiate(cfg.scorer)

    # Score the validation set
    if len(validation_indices) > 0:
        logger.info("Scoring on validation set")
        validation_score = scorer(y_new, predictions)

    # Score the test set
    if x_test is not None and y_test is not None:
        logger.info("Scoring on test set")
        pred_val = model_pipeline.predict(x_test)
        test_score = scorer(y_test, pred_val)
        combined_score = 0.5 * validation_score + 0.5 * test_score
        combined_score_val_percentage = 100 * (len(y_new) / (len(y_new) + len(y_test)))

    # Log the scores
    logger.info(f"Validation Score: {validation_score:.6f}")
    logger.info(f"Test Score: {test_score:.6f}")
    logger.info(f"Combined Score: {combined_score:.6f}, {str(combined_score_val_percentage) + '% of training score.' if combined_score_val_percentage is not None else ''}")
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
