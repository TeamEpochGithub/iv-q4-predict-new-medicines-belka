"""Module to encode SMILES into categorical data."""
import joblib
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData


class NotebookEncoding(VerboseTransformationBlock):
    """Class that replicates the encoding of the highest public notebook."""

    def custom_transform(self, x: XData) -> XData:
        """Encode the SMILE strings into categorical data.

        :param x: XData.
        :return: XData with ecfp replaced by encodings.
        """
        enc = {
            "l": 0,
            "y": 1,
            "@": 2,
            "3": 3,
            "H": 4,
            "S": 5,
            "F": 6,
            "C": 7,
            "r": 8,
            "s": 9,
            "/": 10,
            "c": 11,
            "o": 12,
            "+": 13,
            "I": 14,
            "5": 15,
            "(": 16,
            "2": 17,
            ")": 18,
            "9": 19,
            "i": 20,
            "#": 21,
            "6": 22,
            "8": 23,
            "4": 24,
            "=": 25,
            "1": 26,
            "O": 27,
            "[": 28,
            "D": 29,
            "B": 30,
            "]": 31,
            "N": 32,
            "7": 33,
            "n": 34,
            "-": 35,
        }

        smiles = x.molecule_smiles

        def encode_smile(smile: str) -> npt.NDArray[np.uint8]:
            """Encode a char based on enc dictionary.

            :param smile: The smile string
            :return: Encoded array
            """
            tmp = [enc[i] for i in smile]
            tmp = tmp + [0] * (142 - len(tmp))
            return np.array(tmp).astype(np.uint8)

        smiles_enc = joblib.Parallel(n_jobs=-1)(joblib.delayed(encode_smile)(smile) for smile in tqdm(smiles))
        smiles_enc = np.stack(smiles_enc)

        x.molecule_ecfp = smiles_enc

        return x
