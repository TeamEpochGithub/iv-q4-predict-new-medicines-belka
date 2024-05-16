"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
import os
import warnings
from contextlib import nullcontext
from pathlib import Path

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
from src.setup.setup_runtime_args import setup_train_args
from src.setup.setup_wandb import setup_wandb
from src.typing.xdata import XData, slice_copy
from src.utils.lock import Lock
from src.utils.logger import logger
from src.utils.set_torch_seed import set_torch_seed

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"
# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:
    """Train a model pipeline with a train-test split. Entry point for Hydra which loads the config file."""
    # Run the train config with an optional lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext
    with optional_lock():
        run_train_cfg(cfg)


def run_train_cfg(cfg: DictConfig) -> None:
    """Train a model pipeline with a train-test split."""
    print_section_separator("Q4 - Detect Medicine - Training")

    import coloredlogs

    coloredlogs.install()

    # Set seed
    set_torch_seed()

    # Get output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    if cfg.wandb.enabled:
        setup_wandb(cfg, "train", output_dir)

    # Preload the pipeline
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg)

    # Cache arguments for x_sys
    processed_data_path = Path(cfg.processed_path)
    processed_data_path.mkdir(parents=True, exist_ok=True)
    cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{processed_data_path}",
    }

    splitter_cache_path = Path(f"data/splits/split_{cfg.sample_size}.pkl")

    # Defaults
    X = XData(np.array([1]))
    y = np.array([1])
    val_x = None
    val_y = None

    if not model_pipeline.get_x_cache_exists(cache_args) or not model_pipeline.get_y_cache_exists(cache_args) or not splitter_cache_path.exists() or cfg.val_split:
        X, y = setup_xy(cfg)

    # Split the data into train and test if required
    if cfg.test_size == 0:
        if cfg.splitter.n_splits != 0:
            raise ValueError("Test size is 0, but n_splits is not 0. Also please set n_splits to 0 if you want to run train full.")
        logger.info("Training full.")
        train_indices, test_indices = list(range(len(X))), []  # type: ignore[arg-type]
        fold = -1
    elif cfg.val_split:
        logger.info("Splitting data into train, test, and validation sets.")
        splits, train_val_indices, val_indices = instantiate(cfg.splitter).split(X=X, y=y, cache_path=splitter_cache_path)
        train_indices, test_indices = splits[0]
        fold = 0
        val_x = slice_copy(X, val_indices)
        val_y = y[val_indices].flatten()
        logger.info(f"Bind % in validation: {np.count_nonzero(val_y == 1) * 100 / len(val_y)}")
        if len(X.building_blocks) > 1:
            X.slice_all(train_val_indices)
        if len(y) > 1:
            y = y[train_val_indices]
    else:
        logger.info("Splitting Data into train and test sets.")
        train_indices, test_indices = instantiate(cfg.splitter).split(X=X, y=y, cache_path=splitter_cache_path)[0]
        fold = 0
    logger.info(f"Bind % in train|test: {np.count_nonzero(y == 1) * 100 / (len(y) * 3)}")
    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

    # Run the model pipeline
    print_section_separator("Train model pipeline")
    train_args = setup_train_args(pipeline=model_pipeline, cache_args=cache_args, train_indices=train_indices, test_indices=test_indices, save_model=True, fold=fold)

    predictions, y_new = model_pipeline.train(X, y, **train_args)

    scoring(cfg=cfg, test_indices=test_indices, y_new=y_new, predictions=predictions, val_x=val_x, val_y=val_y, model_pipeline=model_pipeline)

    wandb.finish()


def scoring(
    cfg: DictConfig,
    test_indices: list[int],
    y_new: npt.NDArray[np.int_],
    predictions: npt.NDArray[np.int_],
    model_pipeline: ModelPipeline | EnsemblePipeline,
    val_x: XData | None = None,
    val_y: npt.NDArray[np.int_] | None = None,
) -> None:
    """Score the predictions and possible validation.

    :param cfg: The dictionary configuration
    :param test_indices: the test indices
    :param y_new: The test set labels
    :param prediction: The predictions on the test set
    :param model_pipeline: The model pipeline
    :param val_x: XData for validation set
    :param val_y: Labels for validation y
    :param val_indices: The indices for validation
    """
    print_section_separator("Scoring")
    if len(test_indices) > 0:
        scorer = instantiate(cfg.scorer)
        score = scorer(y_new, predictions)
        logger.info(f"Score: {score}")

        if wandb.run:
            wandb.log({"Score": score})
    elif wandb.run:
        wandb.log({"Score": -1})

    if val_x is not None and val_y is not None:
        scorer = instantiate(cfg.scorer)
        pred_val = model_pipeline.predict(val_x)
        val_score = scorer(val_y, pred_val)
        logger.info(f"Validation Score: {val_score}")

        combined_score = 0.66 * score + 0.33 * val_score
        logger.info(f"Percentage of training score in combined score: {len(y_new) / (len(y_new) + len(val_y))}")
        logger.info(f"Combined Score: {combined_score}")
        if wandb.run:
            wandb.log({"Validation Score": val_score})
            wandb.log({"Combined Score": combined_score})
    elif wandb.run:
        wandb.log({"Validation Score": -1})
        wandb.log({"Combined Score": -1})


if __name__ == "__main__":
    run_train()
