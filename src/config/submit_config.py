"""Schema for the submit configuration."""
from dataclasses import dataclass
from typing import Any


@dataclass
class SubmitConfig:
    """Schema for the submit configuration.

    :param model: Model pipeline.
    :param ensemble: Ensemble pipeline.
    :param raw_data_path: Path to the raw data.
    :param raw_target_path: Path to the raw target.
    :param result_path: Path to the result.
    :param filter_pred: Type of filter
    """

    model: Any
    ensemble: Any
    post_ensemble: Any
    submission_start_id: int
    data_path: str
    submission_path: str
    replace_predictions: str
