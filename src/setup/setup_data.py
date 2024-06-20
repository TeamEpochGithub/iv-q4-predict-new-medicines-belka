"""File containing all methods related to processing and formatting data before it is inserted into a pipeline.

- Since these methods are very competition specific none have been implemented here yet.
- Usually you'll have one data setup for training and another for making submissions.
"""
import gc
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from epochalyst.pipeline.model.model import ModelPipeline
from omegaconf import DictConfig

from src.typing.xdata import XData
from src.utils.logger import logger


def sample_data(train_data: pd.DataFrame, sample_size: int, sample_split: float) -> pd.DataFrame:
    """Sample the data.

    :param train_data: Training data
    :param sample_size: Size of the sample
    :return: Sampled data
    """
    if sample_split < 0:
        if sample_size > len(train_data):
            logger.info("Sample size is larger than the data, returning the data as is.")
            return train_data
        return train_data.sample(sample_size, random_state=42)  # type: ignore[call-arg]
    binds_df = train_data[(train_data["binds_BRD4"] == 1) | (train_data["binds_HSA"] == 1) | (train_data["binds_sEH"] == 1)]
    no_binds_df = train_data[(train_data["binds_BRD4"] == 0) & (train_data["binds_HSA"] == 0) & (train_data["binds_sEH"] == 0)]
    return pd.concat(
        [
            binds_df.sample(min(round(sample_size * (sample_split)), len(binds_df)), random_state=42),
            no_binds_df.sample(min(round(sample_size * (1 - sample_split)), len(no_binds_df)), random_state=42),
        ],
    )  # type: ignore[call-arg]


def read_train_data(directory: Path) -> pd.DataFrame:
    """Read the training data.

    :param path: Usually raw path is a parameter
    :return: Training data
    """
    train_data = pl.read_parquet(directory / "train.parquet")
    return train_data.to_pandas(use_pyarrow_extension_array=True)


def setup_train_x_data(directory: Path, train_data: pd.DataFrame) -> Any:  # noqa: ANN401
    """Create train x data for pipeline.

    :param path: Usually raw path is a parameter
    :return: x data
    """
    with open(directory / "train_dicts/BBs_dict_reverse_1.p", "br") as f1:
        BBs_dict_reverse_1 = pickle.load(f1)  # noqa: S301 (Security issue)
    with open(directory / "train_dicts/BBs_dict_reverse_2.p", "br") as f2:
        BBs_dict_reverse_2 = pickle.load(f2)  # noqa: S301 (Security issue)
    with open(directory / "train_dicts/BBs_dict_reverse_3.p", "br") as f3:
        BBs_dict_reverse_3 = pickle.load(f3)  # noqa: S301 (Security issue)

    # Turn to numpy array
    bb1 = np.array(list(BBs_dict_reverse_1.values()), dtype=f"U{max([len(i) for i in BBs_dict_reverse_1.values()])}")
    del BBs_dict_reverse_1
    bb2 = np.array(list(BBs_dict_reverse_2.values()), dtype=f"U{max([len(i) for i in BBs_dict_reverse_2.values()])}")
    del BBs_dict_reverse_2
    bb3 = np.array(list(BBs_dict_reverse_3.values()), dtype=f"U{max([len(i) for i in BBs_dict_reverse_3.values()])}")
    del BBs_dict_reverse_3

    building_blocks = train_data[["buildingblock1_smiles", "buildingblock2_smiles", "buildingblock3_smiles"]].to_numpy(dtype=np.int16)
    molecule_smiles = train_data["molecule_smiles"].to_numpy(dtype=f'U{train_data["molecule_smiles"].str.len().max()}')

    return XData(
        encoded_rows=building_blocks,
        molecule_smiles=molecule_smiles,
        bb1_smiles=bb1,
        bb2_smiles=bb2,
        bb3_smiles=bb3,
    )


def setup_train_y_data(train_data: pd.DataFrame) -> Any:  # noqa: ANN401
    """Create train y data for pipeline.

    :param path: Usually raw path is a parameter
    :return: y data
    """
    return train_data[["binds_BRD4", "binds_HSA", "binds_sEH"]].to_numpy(dtype=np.int8)


def setup_xy(cfg: DictConfig) -> tuple[XData, npt.NDArray[np.int8]]:
    """Set up x and y data with sampling.

    :param cfg: The configuration to setup
    :return: Tuple of x and y
    """
    # Read the data if required and split it in X, y
    logger.info("Reading data")
    train_data = read_train_data(Path(cfg.data_path))

    # Sample the data
    if cfg.sample_size is not None and cfg.sample_size > 0:
        logger.info(f"Sampling data: {cfg.sample_size:,} samples")
        train_data = sample_data(train_data, cfg.sample_size, cfg.sample_split)

    # Reading X and y data
    logger.info("Reading Building Blocks and setting up X and y data")

    # if not x_cache_exists:
    X = setup_train_x_data(Path(cfg.data_path), train_data)
    y = setup_train_y_data(train_data)
    del train_data
    gc.collect()
    return X, y


class GetXCache:
    """Context manager to get X cache."""

    def __init__(self, model_pipeline: ModelPipeline, cache_args_x: dict[str, Any], X: XData | None = None) -> None:
        """Initialize the context manager."""
        self.model_pipeline = model_pipeline
        self.cache_args_x = cache_args_x
        self.X = X

    def __enter__(self) -> XData:
        """Get X cache."""
        if self.X is not None:
            return self.X

        self.X = self.model_pipeline.x_sys._get_cache(self.model_pipeline.x_sys.get_hash(), self.cache_args_x)  # noqa: SLF001
        return self.X

    def __exit__(self, *args: object) -> None:
        """Delete X cache."""
        del self.X


class GetYCache:
    """Context manager to get X cache."""

    def __init__(self, model_pipeline: ModelPipeline, cache_args_y: dict[str, Any], y: npt.NDArray[np.int_] | None = None) -> None:
        """Initialize the context manager."""
        self.model_pipeline = model_pipeline
        self.cache_args_y = cache_args_y
        self.y = y

    def __enter__(self) -> npt.NDArray[np.int_]:
        """Get y cache."""
        if self.y is not None:
            return self.y

        self.y = self.model_pipeline.y_sys._get_cache(self.model_pipeline.y_sys.get_hash(), self.cache_args_y)  # noqa: SLF001
        return self.y

    def __exit__(self, *args: object) -> None:
        """Delete y cache."""
        del self.y


def setup_inference_data(directory: Path, inference_data: pd.DataFrame) -> Any:  # noqa: ANN401
    """Create data for inference with pipeline.

    :param path: Usually raw path is a parameter
    :return: Inference data
    """
    with open(directory / "test_dicts/BBs_dict_reverse_1_test.p", "br") as f1:
        BBs_dict_reverse_1 = pickle.load(f1)  # noqa: S301 (Security issue)
    with open(directory / "test_dicts/BBs_dict_reverse_2_test.p", "br") as f2:
        BBs_dict_reverse_2 = pickle.load(f2)  # noqa: S301 (Security issue)
    with open(directory / "test_dicts/BBs_dict_reverse_3_test.p", "br") as f3:
        BBs_dict_reverse_3 = pickle.load(f3)  # noqa: S301 (Security issue)

    # Turn to numpy array
    bb1 = np.array(list(BBs_dict_reverse_1.values()))
    del BBs_dict_reverse_1
    bb2 = np.array(list(BBs_dict_reverse_2.values()))
    del BBs_dict_reverse_2
    bb3 = np.array(list(BBs_dict_reverse_3.values()))
    del BBs_dict_reverse_3

    smile_encoding = inference_data[["buildingblock1_smiles", "buildingblock2_smiles", "buildingblock3_smiles"]].to_numpy(dtype=np.int16)
    molecule_smiles = inference_data["molecule_smiles"].to_numpy()

    return XData(
        smile_encoding,
        molecule_smiles=molecule_smiles,
        bb1_smiles=bb1,
        bb2_smiles=bb2,
        bb3_smiles=bb3,
    )


def setup_submission_pseudo_label_data(submission_path: Path, shrunken_test_data: pd.DataFrame, pseudo_binding_threshold: float = 0.5) -> npt.NDArray[np.int_]:
    """Create pseudo label data from submission.

    :param submission_path: Path
    :param shrunken_test_data: The test molecules
    :param pseudo_binding_threshold: The threshold to make bind go to 1
    :return: Pseudo labels
    """
    # Load the data
    submission = pd.read_csv(submission_path)
    raw_data = pd.read_parquet("data/raw/test.parquet")

    # Merge submission data with raw data on 'id'
    raw_data = raw_data.merge(submission[["id", "binds"]], on="id", how="left")

    # Round the binds column to 0 or 1
    raw_data["binds"] = (raw_data["binds"] >= pseudo_binding_threshold).astype(int)

    # Create a pivot table to spread the binds values across the protein types
    pivot_df = raw_data.pivot_table(index="molecule_smiles", columns="protein_name", values="binds", fill_value=0).reset_index()

    # Ensure the pivot table has the necessary columns
    pivot_df = pivot_df.rename(
        columns={
            "BRD4": "binds_BRD4",
            "HSA": "binds_HSA",
            "sEH": "binds_sEH",
        },
    )

    # Initialize columns in shrunken_test_data to ensure they exist before assignment
    shrunken_test_data["binds_BRD4"] = 0
    shrunken_test_data["binds_HSA"] = 0
    shrunken_test_data["binds_sEH"] = 0

    # Merge the pivot table with shrunken_test_data on 'molecule_smiles'
    shrunken_test_data = shrunken_test_data.merge(pivot_df, on="molecule_smiles", how="left", suffixes=("", "_new"))

    # Update the columns with the new values and drop the temporary columns
    shrunken_test_data["binds_BRD4"] = shrunken_test_data["binds_BRD4_new"].fillna(0).astype(int)
    shrunken_test_data["binds_HSA"] = shrunken_test_data["binds_HSA_new"].fillna(0).astype(int)
    shrunken_test_data["binds_sEH"] = shrunken_test_data["binds_sEH_new"].fillna(0).astype(int)

    # Drop the temporary '_new' columns
    shrunken_test_data = shrunken_test_data.drop(columns=["binds_BRD4_new", "binds_HSA_new", "binds_sEH_new"])

    # Return the pseudo labels
    return shrunken_test_data[["binds_BRD4", "binds_HSA", "binds_sEH"]].to_numpy(dtype=np.int8)
