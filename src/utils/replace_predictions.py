"""Replace the known or unknown building block predictions to 0."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.typing.xdata import DataRetrieval, XData


def replace_predictions(directory: Path, x: XData, y: pd.DataFrame, bb_type: str) -> pd.DataFrame:
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
    accepted = []

    for idx in range(len(x)):
        if x[idx] in blocks:
            accepted.append(1)
        else:
            accepted.append(0)

    # Check whether we have to replace known or unknown
    accept = np.array(accepted)
    if bb_type == "unknown":
        accept = 1 - accept

    # Replace the predictions with the value 0
    y.loc[(accept == 0), ["binds_BRD4", "binds_HSA", "binds_sEH"]] = 0

    return y
