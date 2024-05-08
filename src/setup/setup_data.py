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
            binds_df.sample(min(round(sample_size * (1 - sample_split)), len(binds_df)), random_state=42),
            no_binds_df.sample(min(round(sample_size * sample_split), len(no_binds_df)), random_state=42),
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

    # Turn to list
    BBs_dict_reverse_1 = list(BBs_dict_reverse_1.values())
    BBs_dict_reverse_2 = list(BBs_dict_reverse_2.values())
    BBs_dict_reverse_3 = list(BBs_dict_reverse_3.values())

    smile_encoding = train_data[["buildingblock1_smiles", "buildingblock2_smiles", "buildingblock3_smiles"]].to_numpy(dtype=np.int16)

    molecule_smiles = train_data["molecule_smiles"].to_numpy()

    return XData(
        smile_encoding,
        molecule_smiles=molecule_smiles,
        bb1_smiles=BBs_dict_reverse_1,
        bb2_smiles=BBs_dict_reverse_2,
        bb3_smiles=BBs_dict_reverse_3,
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
    logger.info("Sampling data")
    train_data = sample_data(train_data, cfg.sample_size, cfg.sample_split)

    # Reading X and y data
    logger.info("Reading Building Blocks and setting up X and y data")

    # if not x_cache_exists:
    X = setup_train_x_data(Path(cfg.data_path), train_data)
    y = setup_train_y_data(train_data)
    del train_data
    gc.collect()
    return X, y


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

    # Turn to list
    BBs_dict_reverse_1 = list(BBs_dict_reverse_1.values())
    BBs_dict_reverse_2 = list(BBs_dict_reverse_2.values())
    BBs_dict_reverse_3 = list(BBs_dict_reverse_3.values())

    smile_encoding = inference_data[["buildingblock1_smiles", "buildingblock2_smiles", "buildingblock3_smiles"]].to_numpy(dtype=np.int16)

    molecule_smiles = inference_data["molecule_smiles"].to_numpy()

    return XData(
        smile_encoding,
        molecule_smiles=molecule_smiles,
        bb1_smiles=BBs_dict_reverse_1,
        bb2_smiles=BBs_dict_reverse_2,
        bb3_smiles=BBs_dict_reverse_3,
    )


def setup_splitter_data() -> Any:  # noqa: ANN401
    """Create data for splitter.

    :return: Splitter data
    """
    raise NotImplementedError("Setup splitter data is competition specific, implement within competition repository")
