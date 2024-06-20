"""The main script for Cross Validation. Takes in the raw data, does CV and logs the results."""
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import coloredlogs
import hydra
import numpy as np
import numpy.typing as npt
import randomname
import wandb
from epochalyst.logging.section_separator import print_section_separator
from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.model import ModelPipeline
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config.cv_config import CVConfig
from src.setup.setup_data import GetXCache, GetYCache, create_pseudo_labels, setup_xy
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
cs.store(name="base_cv", node=CVConfig)


@hydra.main(version_base=None, config_path="conf", config_name="cv")
def run_cv(cfg: DictConfig) -> None:
    """Do cv on a model pipeline with K fold split. Entry point for Hydra which loads the config file."""
    # Install coloredlogs
    coloredlogs.install()

    # Run the cv config with an optional lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext
    with optional_lock():
        run_cv_cfg(cfg)


def run_cv_cfg(cfg: DictConfig) -> None:
    """Do cv on a model pipeline with K fold split."""
    print_section_separator("Q4 - Detect Medicine - CV")

    # Set seed
    set_torch_seed()

    # Set up Weights & Biases
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    if cfg.wandb.enabled:
        wandb_group_name = randomname.get_name()
        setup_wandb(cfg, "cv", output_dir, name=wandb_group_name, group=wandb_group_name)

    # Preload the pipeline
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg)

    # Setup cache arguments
    cache_path = create_cache_path(cfg.cache_path, cfg.splitter, cfg.sample_size, cfg.sample_split, pseudo_label=cfg.pseudo_label)
    splitter_cache_path = cache_path / "splits.pkl"
    cache_args_x, cache_args_y, cache_args_train = setup_cache_args(cache_path)

    # Setup the data
    X: XData | None = None
    y: npt.NDArray[np.int_] | None = None
    splits: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]
    test_indices: npt.NDArray[np.int64] | None = None
    data_cached: bool = True

    # Check if the data is cached and load if not
    if (
        not model_pipeline.get_x_cache_exists(cache_args_x)
        or not model_pipeline.get_y_cache_exists(cache_args_y)
        or (cfg.splitter is not None and not splitter_cache_path.exists())
    ):
        X, y = setup_xy(cfg)
        data_cached = False

    # Split the data into train and, optionally, test
    if cfg.splitter is None:
        raise ValueError("Splitter is required for cross validation.")
    splitter: Splitter = instantiate(cfg.splitter)
    if splitter.includes_test:
        logger.info("Splitting data into train, validation and test sets.")
        splits, train_validation_indices, test_indices = splitter.split(X=X, y=y, cache_path=splitter_cache_path)  # type: ignore[assignment]
    else:
        logger.info("Splitting Data into train and validation sets.")
        splits = splitter.split(X=X, y=y, cache_path=splitter_cache_path)  # type: ignore[assignment]

    # Create Score arrays
    with GetYCache(model_pipeline, cache_args_y, y) as y:
        validation_scores = []
        test_scores = []
        combined_scores = []
        oof_predictions = np.zeros(y.shape, dtype=np.float64)

    # Run Folds
    for fold_no, (train_indices, validation_indices) in enumerate(splits):
        validation_score, test_score, predictions = run_fold(
            fold_no,
            X,
            y,
            train_indices,
            validation_indices,
            test_indices,
            cfg,
            output_dir,
            cache_args_x,
            cache_args_y,
            cache_args_train,
            data_cached=data_cached,
        )
        validation_scores.append(validation_score)
        test_scores.append(test_score)
        combined_scores.append((test_score + validation_score) / 2)
        oof_predictions[validation_indices] = predictions

    scoring(cfg, model_pipeline, cache_args_y, y, validation_scores, test_scores, combined_scores, test_indices, train_validation_indices, oof_predictions)

    wandb.finish()


def scoring(
    cfg: DictConfig,
    model_pipeline: ModelPipeline | EnsemblePipeline,
    cache_args_y: dict[str, Any],
    y: npt.NDArray[np.int_],
    validation_scores: list[float],
    test_scores: list[float],
    combined_scores: list[float],
    test_indices: npt.NDArray[np.int64] | None,
    train_validation_indices: npt.NDArray[np.int64] | None | tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]],
    oof_predictions: npt.NDArray[np.float64],
) -> None:
    """Calculate final scores.

    :param cfg: Configuration
    :param model_pipeline: The model pipeline
    :param cache_args_y: Cache arguments for y.
    :param y: Y values
    """
    # Average Scores
    with GetYCache(model_pipeline, cache_args_y, y) as y:
        avg_val_score = np.average(np.array(validation_scores))
        avg_test_score = np.average(np.array(test_scores))
        avg_combined_score = np.average(np.array(combined_scores))
        if test_indices is not None:
            oof_score = instantiate(cfg.scorer)(y[train_validation_indices], oof_predictions[train_validation_indices])
        else:
            oof_score = instantiate(cfg.scorer)(y, oof_predictions)

    # Report Scores
    print_section_separator("CV - Results")
    logger.info(f"Avg Val Score: {avg_val_score}")
    logger.info(f"Avg Test Score: {avg_test_score}")
    logger.info(f"Avg Combined Score: {avg_combined_score}")
    logger.info(f"OOF Score: {oof_score}")
    if wandb.run:
        wandb.log(
            {
                "Validation Score": avg_val_score,
                "Test Score": avg_test_score,
                "Combined Score": avg_combined_score,
                "OOF Score": oof_score,
            },
        )


def run_fold(
    fold_no: int,
    X: Any,  # noqa: ANN401
    y: Any,  # noqa: ANN401
    train_indices: npt.NDArray[np.int_],
    validation_indices: npt.NDArray[np.int_],
    test_indices: npt.NDArray[np.int_] | None,
    cfg: DictConfig,
    _output_dir: Path,
    cache_args_x: dict[str, Any],
    cache_args_y: dict[str, Any],
    cache_args_train: dict[str, Any],
    *,
    data_cached: bool = False,
) -> tuple[float, float, npt.NDArray[np.int_]]:
    """Run a single fold of the cross validation.

    :param i: The fold number.
    :param X: The input data.
    :param y: The labels.
    :param train_indices: The indices of the training data.
    :param test_indices: The indices of the test data.
    :param cfg: The config file.
    :param scorer: The scorer to use.
    :param output_dir: The output directory for the prediction plots.
    :param processed_y: The processed labels.
    :return: The score of the fold and the predictions.
    """
    print_section_separator(f"CV - Fold {fold_no}")

    X, y, train_indices, test_indices = create_pseudo_labels(X=X, y=y, train_indices=train_indices, test_indices=test_indices, cfg=cfg, data_cached=data_cached)

    logger.info(f"Train/Validation size: {len(train_indices)}/{len(validation_indices)}")
    logger.info("Creating clean pipeline for this fold")
    model_pipeline = setup_pipeline(cfg)
    train_args = setup_train_args(
        pipeline=model_pipeline,
        cache_args_x=cache_args_x,
        cache_args_y=cache_args_y,
        cache_args_train=cache_args_train,
        train_indices=train_indices,
        validation_indices=validation_indices,
        fold=fold_no,
        save_model=cfg.save_folds,
        output_dir=_output_dir,
    )
    validation_predictions, _ = model_pipeline.train(X, y, **train_args)

    # Get Validation Score
    with GetYCache(model_pipeline, cache_args_y, y) as y:
        validation_score = instantiate(cfg.scorer)(y[validation_indices], validation_predictions)

    # Get the test score
    test_score = -1.0
    combined_score = -1.0
    if test_indices is not None:
        with GetXCache(model_pipeline, cache_args_x, X) as X, GetYCache(model_pipeline, cache_args_y, y) as y:
            X_sliced = slice_copy(X, test_indices)
            prediction_args = {"train_sys": {"MainTrainer": {"use_single_model": True}}}
            test_predictions = model_pipeline.predict(X_sliced, **prediction_args)
            test_score = instantiate(cfg.scorer)(y[test_indices], test_predictions)
            combined_score = 0.5 * validation_score + 0.5 * test_score
            del X_sliced

    # Setup Fold Folder
    # fold_dir = output_dir / str(fold_no)  # Files specific to a run can be saved here
    # logger.debug(f"Output Directory: {fold_dir}")

    # Report Scores
    logger.info(f"Score, fold {fold_no} - Val: {validation_score:.4f}, Test: {test_score:.4f}")
    if wandb.run:
        wandb.log(
            {
                f"Validation Score - {fold_no}": validation_score,
                f"Test Score - {fold_no}": test_score,
                f"Combined Score - {fold_no}": combined_score,
            },
        )
    return validation_score, test_score, validation_predictions


if __name__ == "__main__":
    run_cv()
