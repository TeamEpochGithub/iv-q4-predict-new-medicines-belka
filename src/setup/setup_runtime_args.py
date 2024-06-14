"""File containing functions related to setting up runtime arguments for pipelines."""
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.model import ModelPipeline
from omegaconf import DictConfig


def create_cache_path(root_cache_path: str, splitter_cfg: DictConfig, sample_size: int, sample_split: float, pseudo_label: str) -> Path:
    """Create cache path for processed data.

    :param splitter_cfg: Splitter configuration
    :param sample_size: Sample size
    :param sample_split: Sample split
    :return: Path to the cache
    """
    sample_size_pretty = f"{sample_size:_}"
    sample_split_pretty = f"_{sample_split!s}" if sample_split != -1 else ""
    splitter_cfg_hash = str(hashlib.sha256(str(splitter_cfg).encode("utf-8")).hexdigest())[:5]

    cache_path = Path(root_cache_path) / f"{sample_size_pretty}{sample_split_pretty}{'_' + splitter_cfg_hash if splitter_cfg_hash else ''}"

    if pseudo_label == "local":
        cache_path = Path(str(cache_path) + "_pl")
    if pseudo_label == "public":
        cache_path = Path(str(cache_path) + "_pb")

    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def setup_cache_args(processed_data_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Set cache arguments for pipeline.

    :return: Tuple containing cache arguments for x, y, and train
    """
    cache_args_x = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{processed_data_path / 'x'}",
    }
    cache_args_y = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{processed_data_path / 'y'}",
    }
    cache_args_train = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{processed_data_path / 'train'}",
    }

    return cache_args_x, cache_args_y, cache_args_train


def setup_train_args(
    pipeline: ModelPipeline | EnsemblePipeline,
    cache_args_x: dict[str, Any],
    cache_args_y: dict[str, Any],
    cache_args_train: dict[str, Any],
    train_indices: npt.NDArray[np.int_],
    validation_indices: npt.NDArray[np.int_] | None,
    output_dir: Path,
    fold: int = -1,
    *,
    save_model: bool = False,
    save_model_preds: bool = False,
) -> dict[str, Any]:
    """Set train arguments for pipeline.

    :param pipeline: Pipeline to receive arguments
    :param cache_args: Caching arguments
    :param train_indices: Train indices
    :param validation_indices: Validation indices
    :param fold: Fold number if it exists
    :param save_model: Whether to save the model to File
    :param save_model_preds: Whether to save the model predictions
    :return: Dictionary containing arguments
    """
    if validation_indices is None:
        validation_indices = np.array([])

    x_sys = {
        "cache_args": cache_args_x,
    }
    Path(cache_args_x["storage_path"]).mkdir(parents=True, exist_ok=True)

    y_sys = {
        "cache_args": cache_args_y,
    }
    Path(cache_args_y["storage_path"]).mkdir(parents=True, exist_ok=True)

    main_trainer = {
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "save_model": save_model,
    }

    if fold > -1:
        main_trainer["fold"] = fold

    train_sys = {
        "MainTrainer": main_trainer,
        "DecisionTrees": main_trainer,
        "GraphTrainer": main_trainer,
        "XgboostPseudo": main_trainer,
        "PredictionStatistics": {
            "output_dir": output_dir,
        },
        "ImageTrainer": main_trainer,
        "LazyXGB": main_trainer,
        "TwoHeadedTrainer": main_trainer,
    }

    if save_model_preds:
        train_sys["cache_args"] = cache_args_train
        Path(cache_args_train["storage_path"]).mkdir(parents=True, exist_ok=True)

    pred_sys: dict[str, Any] = {}

    train_args = {
        "x_sys": x_sys,
        "y_sys": y_sys,
        "train_sys": train_sys,
        "pred_sys": pred_sys,
    }

    if isinstance(pipeline, EnsemblePipeline):
        train_args = {
            "ModelPipeline": train_args,
        }

    return train_args


def setup_pred_args(pipeline: ModelPipeline | EnsemblePipeline, cache_args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Set train arguments for pipeline.

    :param pipeline: Pipeline to receive arguments
    :return: Dictionary containing arguments
    """
    # pred_args = {
    #     "train_sys": {
    #         "MainTrainer": {
    #             # "batch_size": 16,
    #             # "model_folds": cfg.model_folds,
    #         },
    #     },
    # }
    pred_args: dict[str, Any] = {}

    if cache_args is not None:
        pred_args["x_sys"] = {
            "cache_args": cache_args,
        }

    if isinstance(pipeline, EnsemblePipeline):
        pred_args = {
            "ModelPipeline": pred_args,
        }

    return pred_args
