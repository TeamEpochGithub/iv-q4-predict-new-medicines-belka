"""Module to encode SMILES into categorical data."""
import joblib
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData


class SimpleMoleculeEncoding(VerboseTransformationBlock):
    """Class that replicates the encoding of the highest public notebook."""

    def custom_transform(self, x: XData) -> XData:
        """Encode the SMILE strings into categorical data.

        :param x: XData.
        :return: XData with ecfp replaced by encodings.
        """
        enc = {
            "l": 1,
            "y": 2,
            "@": 3,
            "3": 4,
            "H": 5,
            "S": 6,
            "F": 7,
            "C": 8,
            "r": 9,
            "s": 10,
            "/": 11,
            "c": 12,
            "o": 13,
            "+": 14,
            "I": 15,
            "5": 16,
            "(": 17,
            "2": 18,
            ")": 19,
            "9": 20,
            "i": 21,
            "#": 22,
            "6": 23,
            "8": 24,
            "4": 25,
            "=": 26,
            "1": 27,
            "O": 28,
            "[": 29,
            "D": 30,
            "B": 31,
            "]": 32,
            "N": 33,
            "7": 34,
            "n": 35,
            "-": 36,
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

        smiles_enc = joblib.Parallel(n_jobs=-1)(joblib.delayed(encode_smile)(smile) for smile in tqdm(smiles, desc="Encoding SMILES"))
        smiles_enc = np.stack(smiles_enc)

        x.molecule_ecfp = smiles_enc

        return x
