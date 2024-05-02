"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
import gc
import os
import warnings
from contextlib import nullcontext
from pathlib import Path

import hydra
import wandb
from epochalyst.logging.section_separator import print_section_separator
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config.train_config import TrainConfig
from src.setup.setup_data import read_train_data, sample_data, setup_train_x_data, setup_train_y_data
from src.setup.setup_pipeline import setup_pipeline
from src.setup.setup_runtime_args import setup_train_args
from src.setup.setup_wandb import setup_wandb
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

    logger.info("Finished setting up pipeline")

    # Cache arguments for x_sys
    processed_data_path = Path(cfg.processed_path)
    processed_data_path.mkdir(parents=True, exist_ok=True)
    cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{processed_data_path}",
    }

    # Read the data if required and split it in X, y
    logger.info("Reading data")
    train_data = read_train_data(Path(cfg.data_path))
    # x_cache_exists = model_pipeline.get_x_cache_exists(cache_args)
    # y_cache_exists = model_pipeline.get_y_cache_exists(cache_args)

    # Sample the data
    logger.info("Sampling data")
    train_data = sample_data(train_data, cfg.sample_size)

    # Reading X and y data
    logger.info("Reading Building Blocks and setting up X and y data")
    X, y = None, None
    # if not x_cache_exists:
    X = setup_train_x_data(Path(cfg.data_path), train_data)
    y = setup_train_y_data(train_data)
    del train_data
    gc.collect()

    # Split the data into train and test if required
    if cfg.test_size == 0:
        if cfg.splitter.n_splits != 0:
            raise ValueError("Test size is 0, but n_splits is not 0. Also please set n_splits to 0 if you want to run train full.")
        logger.info("Training full.")
        train_indices, test_indices = list(range(len(X))), []  # type: ignore[arg-type]
        fold = -1
    else:
        logger.info("Splitting Data into train and test sets.")
        train_indices, test_indices = instantiate(cfg.splitter).split(X=X, y=y)[0]
        fold = 0
    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

    # Run the model pipeline
    print_section_separator("Train model pipeline")
    train_args = setup_train_args(pipeline=model_pipeline, cache_args=cache_args, train_indices=train_indices, test_indices=test_indices, save_model=True, fold=fold)
    predictions, y_new = model_pipeline.train(X, y, **train_args)

    if y is None:
        y = y_new

    if len(test_indices) > 0:
        print_section_separator("Scoring")
        scorer = instantiate(cfg.scorer)
        score = scorer(y_new[test_indices], predictions)
        logger.info(f"Score: {score}")

        if wandb.run:
            wandb.log({"Score": score})

    wandb.finish()


if __name__ == "__main__":
    run_train()
