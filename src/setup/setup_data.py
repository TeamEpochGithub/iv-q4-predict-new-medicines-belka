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
import wandb
from epochalyst.pipeline.model.model import ModelPipeline
from omegaconf import DictConfig

from src.typing.xdata import XData
from src.utils.logger import logger

FULL_DATA_SIZE = 98415610
KAGGLE_DATA_SIZE = 878022


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


def read_train_data(directory: Path, file_name: str) -> pd.DataFrame:
    """Read the training data.

    :param path: Usually raw path is a parameter
    :return: Training data
    """
    train_data = pl.read_parquet(directory / file_name)
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
    train_data = read_train_data(Path(cfg.data_path), cfg.train_file_name)

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


def setup_submission_pseudo_label_data(
    submission_path: Path,
    shrunken_test_data: pd.DataFrame,
    pseudo_binding_ratio: float = 0.05,
    pseudo_confidence_threshold: float = 0.5,
) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.int_]]:
    """Create pseudo label data from submission.

    :param submission_path: Path
    :param shrunken_test_data: The test molecules
    :param pseudo_binding_ratio: The threshold to make bind go to 1
    :return: Pseudo labels
    """
    # Load the data
    submission = pd.read_csv(submission_path)
    raw_data = pd.read_parquet("data/raw/test.parquet")

    # Merge submission data with raw data on 'id'
    raw_data = raw_data.merge(submission[["id", "binds"]], on="id", how="left")

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
    shrunken_test_data["binds_BRD4"] = shrunken_test_data["binds_BRD4_new"].fillna(0.0)
    shrunken_test_data["binds_HSA"] = shrunken_test_data["binds_HSA_new"].fillna(0.0)
    shrunken_test_data["binds_sEH"] = shrunken_test_data["binds_sEH_new"].fillna(0.0)

    # Drop the temporary '_new' columns
    shrunken_test_data = shrunken_test_data.drop(columns=["binds_BRD4_new", "binds_HSA_new", "binds_sEH_new"])

    # Select top and bottom confidence thresholds if pseudo_confidence_threshold is less than 0.5
    top_confidence_threshold = {
        "binds_BRD4": 0.5,
        "binds_HSA": 0.5,
        "binds_sEH": 0.5,
    }
    bottom_confidence_threshold = {
        "binds_BRD4": 0.5,
        "binds_HSA": 0.5,
        "binds_sEH": 0.5,
    }

    # Find binding thresholds
    top_confidence_threshold = shrunken_test_data[["binds_BRD4", "binds_HSA", "binds_sEH"]].quantile(1 - pseudo_confidence_threshold * pseudo_binding_ratio)
    bottom_confidence_threshold = shrunken_test_data[["binds_BRD4", "binds_HSA", "binds_sEH"]].quantile(pseudo_confidence_threshold)

    # Set the binds columns to NaN if the confidence is between the thresholds
    shrunken_test_data.loc[
        (shrunken_test_data["binds_BRD4"] > bottom_confidence_threshold["binds_BRD4"]) & (shrunken_test_data["binds_BRD4"] < top_confidence_threshold["binds_BRD4"]),
        "binds_BRD4",
    ] = np.nan

    shrunken_test_data.loc[
        (shrunken_test_data["binds_HSA"] > bottom_confidence_threshold["binds_HSA"]) & (shrunken_test_data["binds_HSA"] < top_confidence_threshold["binds_HSA"]),
        "binds_HSA",
    ] = np.nan

    shrunken_test_data.loc[
        (shrunken_test_data["binds_sEH"] > bottom_confidence_threshold["binds_sEH"]) & (shrunken_test_data["binds_sEH"] < top_confidence_threshold["binds_sEH"]),
        "binds_sEH",
    ] = np.nan

    # Drop the rows with NaN values in the 'binds' column
    shrunken_test_data = shrunken_test_data.dropna(subset=["binds_BRD4", "binds_HSA", "binds_sEH"])

    # If greater than pseudo_binding_threshold set binds to 1 else 0
    shrunken_test_data.loc[shrunken_test_data["binds_BRD4"] >= top_confidence_threshold["binds_BRD4"], "binds_BRD4"] = 1
    shrunken_test_data.loc[shrunken_test_data["binds_HSA"] >= top_confidence_threshold["binds_HSA"], "binds_HSA"] = 1
    shrunken_test_data.loc[shrunken_test_data["binds_sEH"] >= top_confidence_threshold["binds_sEH"], "binds_sEH"] = 1
    shrunken_test_data.loc[shrunken_test_data["binds_BRD4"] <= bottom_confidence_threshold["binds_BRD4"], "binds_BRD4"] = 0
    shrunken_test_data.loc[shrunken_test_data["binds_HSA"] <= bottom_confidence_threshold["binds_HSA"], "binds_HSA"] = 0
    shrunken_test_data.loc[shrunken_test_data["binds_sEH"] <= bottom_confidence_threshold["binds_sEH"], "binds_sEH"] = 0

    molecule_smiles = shrunken_test_data["molecule_smiles"].to_numpy()

    # Log confidences
    if wandb.run:
        wandb.log(
            {
                "BRD4 Confidence Top": top_confidence_threshold["binds_BRD4"],
                "HSA Confidence Top": top_confidence_threshold["binds_HSA"],
                "sEH Confidence Top": top_confidence_threshold["binds_sEH"],
                "BRD4 Confidence Bottom": bottom_confidence_threshold["binds_BRD4"],
                "HSA Confidence Bottom": bottom_confidence_threshold["binds_HSA"],
                "sEH Confidence Bottom": bottom_confidence_threshold["binds_sEH"],
            },
        )

    # Return the pseudo labels
    return molecule_smiles, shrunken_test_data[["binds_BRD4", "binds_HSA", "binds_sEH"]].to_numpy(dtype=np.int8)


def create_pseudo_labels(
    X: XData | None,
    y: npt.NDArray[np.int_] | None,
    train_indices: npt.NDArray[np.int_],
    test_indices: npt.NDArray[np.int_] | None,
    cfg: DictConfig,
    *,
    data_cached: bool,
) -> tuple[XData | None, npt.NDArray[np.int_] | None, npt.NDArray[np.int_], npt.NDArray[np.int_] | None]:
    """Include the test molecule smiles into the training test.

    :param X: XData containing the molecule smiles
    :param y: array containing the protein labels
    """
    if data_cached:
        # Check whether the indices are not empty
        if cfg.pseudo_label == "local":
            if test_indices is None:
                raise ValueError("The test indices are empty.")
            test_size = test_indices.shape[0]
        return X, y, train_indices, test_indices

    if X is None or y is None or X.molecule_smiles is None or y.shape[0] == 0 or X.molecule_smiles.shape[0] == 0:
        raise ValueError("The features or the labels are empty.")

    test_size = 0
    original_size = X.molecule_smiles.shape[0]

    if cfg.seh_binding_dataset:
        # Load in sEH binding dataset data
        seh_binding_dataset = pd.read_csv("data/shrunken/seh_binding.csv")
        smiles = seh_binding_dataset["molecule_smiles"].to_numpy().astype(np.str_)
        labels = seh_binding_dataset[["binds_BRD4", "binds_HSA", "binds_sEH"]].to_numpy(dtype=np.int_)
        X.molecule_smiles = np.concatenate((X.molecule_smiles, smiles))
        y = np.concatenate((y, labels), dtype=np.int_)
        test_size += labels.shape[0]

    if cfg.pseudo_label == "submission":
        if cfg.submission_path is None:
            raise ValueError("Submission path needs to be specified if you want to pseudo label with submission data")
        submission_path = Path(cfg.submission_path)
        smiles, labels = setup_submission_pseudo_label_data(submission_path, pd.read_csv("data/shrunken/test.csv"), cfg.pseudo_binding_ratio, cfg.pseudo_confidence_threshold)
        test_size += labels.shape[0]
        X.molecule_smiles = np.concatenate((X.molecule_smiles, smiles))
        y = np.concatenate((y, labels), dtype=np.int_)
    elif cfg.pseudo_label == "local":
        if test_indices is None or X.molecule_smiles is None:
            raise ValueError("The test indices are empty, need test_indices if using local pseudo labels.")
        smiles = X.molecule_smiles[test_indices]
        X.molecule_smiles = np.concatenate((X.molecule_smiles, smiles))
        y = np.concatenate((y, np.zeros((smiles.shape[0], 3), dtype=np.int_)), dtype=np.int_)
        test_size += smiles.shape[0]

    new_indices = np.array([original_size + idx for idx in range(test_size)], dtype=np.int_)
    train_indices = np.concatenate((train_indices, new_indices)).astype(np.int_)

    return X, y, train_indices, test_indices
