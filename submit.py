"""Submit.py is the main script for running inference on the test set and creating a submission."""
import os
import time
import warnings
from pathlib import Path

import hydra
import pandas as pd
import polars as pl
from epochalyst.logging.section_separator import print_section_separator
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from tqdm import tqdm

from src.config.submit_config import SubmitConfig
from src.setup.setup_data import setup_inference_data
from src.setup.setup_pipeline import setup_pipeline
from src.setup.setup_runtime_args import setup_pred_args
from src.utils.filter_known_predictions import filter_known_predictions
from src.utils.logger import logger

warnings.filterwarnings("ignore", category=UserWarning)

# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_submit", node=SubmitConfig)


@hydra.main(version_base=None, config_path="conf", config_name="submit")
# TODO(Epoch): Use SubmitConfig instead of DictConfig
def run_submit(cfg: DictConfig) -> None:
    """Run the main script for submitting the predictions."""
    print_section_separator("Q4 - Detect Medicine - Submit")

    # Set up logging
    import coloredlogs

    coloredlogs.install()

    # Preload the pipeline
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg, is_train=False)

    # Load the test data

    directory = Path(cfg.data_path)

    first_time = time.time()
    inference_data = pl.read_parquet(directory / "test.parquet")
    inference_data = inference_data.to_pandas(use_pyarrow_extension_array=True)

    X = setup_inference_data(directory, inference_data)
    logger.info(f"Total time to setup data: {time.time() - first_time}s")

    logger.info("Setting up Prediction Pipeline...")
    # Setup Arguments for prediction
    cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{Path(cfg.result_path)}",
    }
    pred_args = setup_pred_args(pipeline=model_pipeline, cache_args=cache_args)
    predictions = model_pipeline.predict(X, **pred_args)

    # Save the predictions
    logger.info("Reshaping predictions...")
    predictions_df = pd.DataFrame(predictions.reshape(len(predictions) // 3, 3), columns=["binds_BRD4", "binds_HSA", "binds_sEH"])
    predictions_df["is_BRD4"] = inference_data["is_BRD4"]
    predictions_df["is_HSA"] = inference_data["is_HSA"]
    predictions_df["is_sEH"] = inference_data["is_sEH"]

    if cfg.filter_pred != "none":
        logger.info(f"Filter predictions on {cfg.filter_pred}")
        predictions_df = filter_known_predictions(directory, X, predictions_df, cfg.filter_pred)

    # Map predictions to ids from test data
    original_test = pd.read_parquet("data/raw/test.parquet")

    # Predictions are (binds_BRD4, binds_HSA, binds_sEH) these should be mapped to the original ids
    # If you have id_1, id_2, id_3, id_4, id_5, id_6
    # And you have predictions (1, 0, 1), (0,1,0)
    # You should map these to id_1, id_3, id_5

    # Get the original ids
    original_ids = original_test.id

    # Create a flattened array of predictions where each row is only given if its corresponding is_ is true
    # For example if is_BRD4 is true, then the prediction is binds_BRD4 else it is skipped and shouldn't be included in the final array
    final_predictions = []
    for i in tqdm(range(predictions_df.shape[0]), desc="Flattening predictions"):
        if predictions_df.iloc[i].is_BRD4:
            final_predictions.append(predictions_df.iloc[i].binds_BRD4)
        if predictions_df.iloc[i].is_HSA:
            final_predictions.append(predictions_df.iloc[i].binds_HSA)
        if predictions_df.iloc[i].is_sEH:
            final_predictions.append(predictions_df.iloc[i].binds_sEH)

    logger.info("Saving submission...")
    final_predictions_df = pd.DataFrame({"id": original_ids, "binds": final_predictions})
    final_predictions_df.to_csv(directory / "submission.csv", index=False)

    logger.info(f"Submission saved to {directory / 'submission.csv'}")


if __name__ == "__main__":
    run_submit()
