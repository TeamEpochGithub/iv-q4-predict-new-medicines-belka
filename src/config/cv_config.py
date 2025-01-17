"""Schema for the cross validation configuration."""
from dataclasses import dataclass
from typing import Any

from src.config.wandb_config import WandBConfig


@dataclass
class CVConfig:
    """Schema for the cross validation configuration.

    :param model: Model pipeline.
    :param ensemble: Ensemble pipeline.
    :param raw_data_path: Path to the raw data.
    :param raw_target_path: Path to the raw target.
    :param processed_path: Path to put processed data.
    :param scorer: Scorer object to be instantiated.
    :param wandb: Whether to log to Weights & Biases and other settings.
    :param splitter: Cross validation splitter.
    :param allow_multiple_instances: Whether to allow multiple instances of training at the same time.
    :param save_folds: Whether to save the fold models
    """

    model: Any
    ensemble: Any
    data_path: str
    cache_path: str
    train_file_name: str

    scorer: Any
    wandb: WandBConfig
    splitter: Any
    sample_size: int = 10000
    sample_split: float = 0.5
    allow_multiple_instances: bool = False

    # Data additions
    pseudo_label: str = "none"
    submission_path: str | None = None
    pseudo_binding_ratio: float = 0.05
    pseudo_confidence_threshold: float = 0.6
    seh_binding_dataset: bool = False

    save_folds: bool = True
