"""Replace the known or unknown building block predictions to 0."""
import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.typing.xdata import DataRetrieval, XData


def replace_predictions(directory: Path, x: XData, y: npt.NDArray[np.int_], bb_type: str) -> pd.DataFrame:
    """Replace the known or unknown building block predictions to 0.

    param directory: raw path the unique training blocks
    param bb_type: [known, unknown] whether the known or unknown predictions are replaced
    """
    # Extract the first building blocks in train
    with open(directory / "train_dicts/BBs_dict_reverse_1.p", "br") as f1:
        blocks = list(pickle.load(f1).values())  # noqa: S301 (Security issue)

    # Initialize the retrieval to first building block
    x.retrieval = DataRetrieval.SMILES_BB1

    # Filter the unknown molecules
    has_known_building_blocks = np.empty(len(x), dtype=np.int8)

    for idx in range(len(x)):
        if x[idx] in blocks:
            has_known_building_blocks[idx] = 1
        else:
            has_known_building_blocks[idx] = 0

    # Check whether we have to replace known or unknown
    if bb_type == "known":
        y = y * has_known_building_blocks[:, None]
    elif bb_type == "unknown":
        y = y * (1 - has_known_building_blocks)[:, None]

    return y
