"""File containing all methods related to processing and formatting data before it is inserted into a pipeline.

- Since these methods are very competition specific none have been implemented here yet.
- Usually you'll have one data setup for training and another for making submissions.
"""
import pickle
from pathlib import Path
from typing import Any

import polars as pl

from src.typing.xdata import XData


def setup_train_x_data(raw_path: str) -> Any:  # noqa: ANN401
    """Create train x data for pipeline.

    :param path: Usually raw path is a parameter
    :return: x data
    """
    directory = Path(raw_path)

    train_data = pl.read_parquet(directory / "train.parquet")
    train_data = train_data.to_pandas(use_pyarrow_extension_array=True)

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

    return XData(
        train_data[["buildingblock1_smiles", "buildingblock2_smiles", "buildingblock3_smiles"]].to_numpy(),
        list(train_data["molecule_smiles"]),
        bb1=BBs_dict_reverse_1,
        bb2=BBs_dict_reverse_2,
        bb3=BBs_dict_reverse_3,
    )


def setup_train_y_data(raw_path: str) -> Any:  # noqa: ANN401
    """Create train y data for pipeline.

    :param path: Usually raw path is a parameter
    :return: y data
    """
    raise NotImplementedError(f"Setup train data y is competition specific, raw_path:{raw_path}, implement within competition repository")


def setup_inference_data() -> Any:  # noqa: ANN401
    """Create data for inference with pipeline.

    :param path: Usually raw path is a parameter
    :return: Inference data
    """
    raise NotImplementedError("Setup inference data is competition specific, implement within competition repository, it might be the same as setup_train_x")


def setup_splitter_data() -> Any:  # noqa: ANN401
    """Create data for splitter.

    :return: Splitter data
    """
    raise NotImplementedError("Setup splitter data is competition specific, implement within competition repository")
