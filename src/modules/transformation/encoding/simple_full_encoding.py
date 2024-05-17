"""Module to encode SMILES into categorical data."""
import joblib
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData


class SimpleFullEncoding(VerboseTransformationBlock):
    """Class that replicates the encoding of the highest public notebook."""

    def custom_transform(self, x: XData) -> XData:
        """Encode the SMILE strings into categorical data.

        :param x: XData.
        :return: XData with ecfp replaced by encodings.
        """
        enc = {
            # Atoms
            "C": 1,
            "c": 2,
            "N": 3,
            "n": 4,
            "O": 5,
            "o": 6,
            "S": 7,
            "s": 8,
            "P": 9,
            "F": 10,
            "Cl": 11,
            "Br": 12,
            "I": 13,
            "i": 14,
            "B": 15,
            "H": 16,
            # Bonds
            "-": 17,
            "=": 18,
            "#": 19,
            # Bond configuration
            "/": 20,
            # Branches
            "(": 21,
            ")": 22,
            # Brackets
            "[": 23,
            "]": 24,
            # Numbers
            "1": 25,
            "2": 26,
            "3": 27,
            "4": 28,
            "5": 29,
            "6": 30,
            "7": 31,
            "8": 32,
            "9": 33,
            # Stereochem
            "@": 34,
            # DNA
            "Dy": 35,
            # Charges
            "+": 36,
        }

        smiles = x.molecule_smiles

        def encode_smile(smile: str) -> npt.NDArray[np.uint8]:
            """Encode a char based on enc dictionary.

            :param smile: The smile string
            :return: Encoded array
            """
            x = 0
            tmp = []
            while x < len(smile):
                # Check two first
                two_chars = smile[x : x + 2]
                if two_chars in enc:
                    tmp.append(enc[two_chars])
                    x += 2
                else:
                    tmp.append(enc[smile[x]])
                    x += 1
            tmp = tmp + [0] * (142 - len(tmp))
            return np.array(tmp).astype(np.uint8)

        smiles_enc = joblib.Parallel(n_jobs=-1)(joblib.delayed(encode_smile)(smile) for smile in tqdm(smiles, desc="Encoding Atomwise"))
        smiles_enc = np.stack(smiles_enc)

        x.molecule_ecfp = smiles_enc

        return x
