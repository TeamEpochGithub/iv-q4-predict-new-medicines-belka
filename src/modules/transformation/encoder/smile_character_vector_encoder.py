"""Custom encoding."""
import joblib
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData


class SmileCharacterVectorEncoder(VerboseTransformationBlock):
    """Class that encodes the molecule smiles into vectors ."""

    def custom_transform(self, x: XData) -> XData:
        """Encode the SMILE strings.

        :param x: XData.
        :return: XData with ecfp replaced by encodings.
        """
        # [Encoding, Atomic Number, Typical Valence, Electronegativity]
        enc = {
            "l": [1, 0.0, 0.0, 0.0],
            "y": [2, 0.0, 0.0, 0.0],
            "@": [3, 0.0, 0.0, 0.0],
            "3": [4, 0.0, 0.0, 0.0],
            "H": [5, 1, 1, 2.20],
            "S": [6, 16, 2, 2.58],
            "F": [7, 9, 1, 3.98],
            "C": [8, 6, 4, 2.55],
            "r": [9, 0.0, 0.0, 0.0],
            "s": [10, 0.0, 0.0, 0.0],
            "/": [11, 0.0, 0.0, 0.0],
            "c": [12, 6, 4, 2.55],
            "o": [13, 8, 2, 3.44],
            "+": [14, 0.0, 0.0, 0.0],
            "I": [15, 53, 1, 2.66],
            "5": [16, 0.0, 0.0, 0.0],
            "(": [17, 0.0, 0.0, 0.0],
            "2": [18, 0.0, 0.0, 0.0],
            ")": [19, 0.0, 0.0, 0.0],
            "9": [20, 0.0, 0.0, 0.0],
            "i": [21, 0.0, 0.0, 0.0],
            "#": [22, 0.0, 0.0, 0.0],
            "6": [23, 0.0, 0.0, 0.0],
            "8": [24, 0.0, 0.0, 0.0],
            "4": [25, 0.0, 0.0, 0.0],
            "=": [26, 0.0, 0.0, 0.0],
            "1": [27, 0.0, 0.0, 0.0],
            "O": [28, 8, 2, 3.44],
            "[": [29, 0.0, 0.0, 0.0],
            "D": [30, 0.0, 0.0, 0.0],
            "B": [31, 5, 3, 2.04],  # Assuming Boron
            "]": [32, 0.0, 0.0, 0.0],
            "N": [33, 7, 3, 3.04],
            "7": [34, 0.0, 0.0, 0.0],
            "n": [35, 7, 3, 3.04],
            "-": [36, 0.0, 0.0, 0.0],
        }

        smiles = x.molecule_smiles

        def encode_smile(smile: str) -> npt.NDArray[np.float32]:
            """Encode a char based on enc dictionary.

            :param smile: The smile string
            :return: Encoded array
            """
            tmp = [enc[i] for i in smile]
            tmp = tmp + [[0, 0, 0, 0]] * (142 - len(tmp))
            return np.array(tmp).astype(np.float32)

        smiles_enc = joblib.Parallel(n_jobs=-1)(joblib.delayed(encode_smile)(smile) for smile in tqdm(smiles, desc="Encoding SMILES"))
        smiles_enc = np.stack(smiles_enc)

        x.molecule_ecfp = smiles_enc

        return x
