"""Schema for the train configuration."""
from dataclasses import dataclass
from typing import Any

from src.config.wandb_config import WandBConfig


@dataclass
class TrainConfig:
    """Schema for the train configuration.

    :param model: The model pipeline.
    :param ensemble: The ensemble pipeline.
    :param raw_data_path: Path to the raw data.
    :param raw_target_path: Path to the raw target.
    :param processed_path: Path to put processed data.
    :param scorer: Scorer object to be instantiated.
    :param wandb: Whether to log to Weights & Biases and other settings.
    :param splitter: Cross validation splitter.
    :param test_size: Size of the test set.
    :param allow_multiple_instances: Whether to allow multiple instances of training at the same time.
    :param pseudo_label: Include the local or test samples into train
    """

    model: Any
    ensemble: Any
    data_path: str
    cache_path: str
    scorer: Any
    wandb: WandBConfig
    splitter: Any
    sample_size: int = 10000
    sample_split: float = 0.5
    allow_multiple_instances: bool = False

    # Data additions
    pseudo_label: str = "none"
    submission_path: str | None = None
    pseudo_binding_threshold: float = 0.5
    model_sampling: bool = False
