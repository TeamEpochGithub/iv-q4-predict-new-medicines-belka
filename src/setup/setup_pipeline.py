"""File containing functions related to setting up the pipeline."""
import select
import sys
import termios
from enum import Enum
from pathlib import Path
from typing import Any

from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.model import ModelPipeline
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.utils.logger import logger


def find_trained_model_files(model_pipeline: ModelPipeline) -> dict[int | str, Path]:
    """Find all the trained model files."""
    trained_model_files: list[Path] = []
    for step in reversed(model_pipeline.train_sys.steps):
        if isinstance(step, TorchTrainer):
            tm_model_file = step.get_model_path().with_suffix("").stem
            tm_path = step.get_model_path().parent
            trained_model_files = [file for file in tm_path.glob(f"{tm_model_file}*") if file.is_file()]

    saved_model_files: dict[int | str, Path] = {}
    for file in trained_model_files:
        if "checkpoint" in file.stem:
            epoch = int(file.stem.split("_")[-1])
            saved_model_files.update({epoch: file})
        elif saved_model_files.get("full", None) is None:
            saved_model_files["full"] = file

    return saved_model_files


def check_model_trained(model_pipeline: ModelPipeline) -> None:
    """Check if the model has already been trained and ask the user if they want to retrain it."""
    saved_model_files = find_trained_model_files(model_pipeline)

    if len(saved_model_files) > 0:
        logger.info("Model already trained.")
        while True:
            logger.info(f"Available checkpoints: {list(saved_model_files.keys())}")
            logger.info("Resume from latest checkpoint? [Enter epoch to resume training or 'y' to resume from latest checkpoint or 'n' to retrain from scratch]:")
            i, o, e = select.select([sys.stdin], [], [], 10)

            if not i:
                logger.info("Timeout. Exiting. Resuming with default option ('y')")
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
                break

            input_str = sys.stdin.readline().strip()

            if input_str.lower() in ["y", "yes", "full"]:
                logger.info("Resuming from latest checkpoint")
                break

            if input_str.lower() in ["n", "no"]:
                logger.info("Deleting all checkpoints and retraining from scratch")
                for file in saved_model_files.values():
                    file.unlink()
                break

            if input_str.isnumeric() and int(input_str) in saved_model_files:
                logger.info(f"Resuming from epoch {input_str} and deleting all checkpoints after it")
                for epoch, file in saved_model_files.items():
                    if isinstance(epoch, int) and epoch > int(input_str) or epoch == "full":
                        file.unlink()
                break

            logger.info("Invalid input")


def setup_pipeline(cfg: DictConfig, *, is_train: bool = True) -> ModelPipeline | EnsemblePipeline:
    """Instantiate the pipeline.

    :param pipeline_cfg: The model pipeline config. Root node should be a ModelPipeline
    :param is_train: Whether the pipeline is used for training
    """
    logger.info("Instantiating the pipeline")

    if is_train:
        test_size = cfg.get("splitter", {}).get("n_splits", -1.0)

    if "model" in cfg:
        model_cfg = cfg.model
        model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
        if isinstance(model_cfg_dict, dict) and is_train:
            model_cfg_dict = update_model_cfg_test_size(model_cfg_dict, test_size)
        pipeline_cfg = OmegaConf.create(model_cfg_dict)
    elif "ensemble" in cfg:
        ensemble_cfg = cfg.ensemble
        ensemble_cfg_dict = OmegaConf.to_container(ensemble_cfg, resolve=True)
        ensemble_cfg_dict = update_ensemble_cfg_dict(ensemble_cfg_dict, test_size, is_train=is_train)
        pipeline_cfg = OmegaConf.create(ensemble_cfg_dict)
    else:
        raise ValueError("Neither model nor ensemble specified in config.")

    model_pipeline = instantiate(pipeline_cfg)
    logger.debug(f"Pipeline: \n{model_pipeline}")

    return model_pipeline


def update_ensemble_cfg_dict(
    ensemble_cfg_dict: dict[str | bytes | int | Enum | float | bool, Any] | list[Any] | str | None,
    test_size: float,
    *,
    is_train: bool,
) -> dict[str | bytes | int | Enum | float | bool, Any]:
    """Update the ensemble_cfg_dict.

    :param ensemble_cfg_dict: The original ensemble_cfg_dict
    :param test_size: Test size to add to the models
    :param is_train: Boolean whether models are being trained
    """
    if isinstance(ensemble_cfg_dict, dict):
        ensemble_cfg_dict["steps"] = list(ensemble_cfg_dict["steps"].values())
        if is_train:
            for model in ensemble_cfg_dict["steps"]:
                update_model_cfg_test_size(model, test_size)

        return ensemble_cfg_dict

    return {}


def update_model_cfg_test_size(
    cfg: dict[str | bytes | int | Enum | float | bool, Any] | list[Any] | str | None,
    test_size: float = -1.0,
) -> dict[str | bytes | int | Enum | float | bool, Any] | list[Any] | str | None:
    """Update the test size in the model config.

    :param cfg: The model config.
    :param test_size: The test size.

    :return: The updated model config.
    """
    if cfg is None:
        raise ValueError("cfg should not be None")
    if isinstance(cfg, dict):
        for model in cfg["train_sys"]["steps"]:
            if model["_target_"] == "src.modules.training.main_trainer.MainTrainer":
                model["n_folds"] = test_size
    return cfg
