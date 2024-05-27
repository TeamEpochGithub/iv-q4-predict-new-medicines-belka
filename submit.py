"""Submit.py is the main script for running inference on the test set and creating a submission."""
import os
import warnings
from pathlib import Path

import coloredlogs
import hydra
import pandas as pd
import polars as pl
from epochalyst.logging.section_separator import print_section_separator
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from src.config.submit_config import SubmitConfig
from src.setup.setup_data import setup_inference_data
from src.setup.setup_pipeline import setup_pipeline
from src.setup.setup_runtime_args import setup_pred_args
from src.utils.logger import logger
from src.utils.replace_predictions import replace_predictions

# Set logging
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"  # Makes hydra give full error messages

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_submit", node=SubmitConfig)


@hydra.main(version_base=None, config_path="conf", config_name="submit")
def run_submit(cfg: DictConfig) -> None:
    """Run the main script for submitting the predictions."""
    # Set up logging
    coloredlogs.install()

    print_section_separator("Q4 - Detect Medicine - Submit")

    # Preload the pipeline
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg, is_train=False)

    # Load the test data
    logger.info("Loading inference data...")
    data_path = Path(cfg.data_path)
    inference_data = pl.read_parquet(data_path / "test.parquet")
    inference_data = inference_data.to_pandas(use_pyarrow_extension_array=True)
    X = setup_inference_data(data_path, inference_data)

    logger.info("Setting up Prediction Pipeline...")
    pred_args = setup_pred_args(model_pipeline)
    predictions = model_pipeline.predict(X, **pred_args)

    # Replace the known or unknown building block predictions to 0
    if cfg.replace_predictions != "none":
        logger.info(f"Replace {cfg.replace_predictions} predictions with 0")
        predictions = replace_predictions(data_path, X, predictions, cfg.replace_predictions)

    # Reshape predictions
    logger.info("Reshaping predictions...")
    starting_id = cfg.submission_start_id
    flattened_predictions = predictions.flatten()

    # Only keep predictions which are specified in test.parquet
    predictions_to_keep = inference_data[["is_BRD4", "is_HSA", "is_sEH"]].to_numpy().flatten().astype(bool)
    flattened_predictions = flattened_predictions[predictions_to_keep]

    # Save the submission
    logger.info("Creating submission.csv")
    submission_path = Path(cfg.submission_path)

    submission = pd.DataFrame(flattened_predictions, columns=["binds"])
    submission["id"] = range(starting_id, starting_id + len(flattened_predictions))
    submission = submission[["id", "binds"]]
    submission.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    run_submit()
