"""Module to encode the individual building blocks."""
import joblib
import numpy as np
import numpy.typing as npt
from rdkit import Chem  # type: ignore[import-not-found]
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData


class BBEncoder(VerboseTransformationBlock):
    """Class that transforms smiles of building blocks to encoding."""

    def custom_transform(self, x: XData) -> XData:
        """Encode the SMILE building block strings.

        :param x: XData
        :return: XData with ecfp replaced by encodings
        """
        fmoc = Chem.MolFromSmiles("O=COCC1c2ccccc2-c2ccccc21")  # Biphenyl core
        boc = Chem.MolFromSmiles("CC(C)(C)OC(=O)")  # Tert butyl acetate

        MAX_ENC_SIZE_BB1 = 42
        MAX_ENC_SIZE_BB2 = 54
        MAX_ENC_SIZE_BB3 = 43

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
            ".": 37,
            "\\": 38,
            "%": 39,
            "0": 40,
        }

        def remove_substructures(block: str) -> str:
            """Remove substructures from block.

            :param block: SMILE of block
            :return: SMILE of modified block
            """
            block_mol = Chem.MolFromSmiles(block)
            if block_mol.HasSubstructMatch(fmoc):
                # Delete fmoc
                block_mol = Chem.rdmolops.DeleteSubstructs(block_mol, fmoc, onlyFrags=False)
            elif block_mol.HasSubstructMatch(boc):
                # Delete boc
                block_mol = Chem.rdmolops.DeleteSubstructs(block_mol, boc, onlyFrags=False)
            else:
                raise ValueError("Can't find substructure")

            return Chem.MolToSmiles(block_mol)

        def encode_block(block: str, bb: int) -> npt.NDArray[np.uint8]:
            """Encode a char based on enc dictionary.

            :param smile: The smile string
            :return: Encoded array
            """
            x = 0
            tmp = []
            while x < len(block):
                # Check two first
                two_chars = block[x : x + 2]
                if two_chars in enc:
                    tmp.append(enc[two_chars])
                    x += 2
                else:
                    tmp.append(enc[block[x]])
                    x += 1
            tmp = tmp + [0] * (bb - len(tmp))
            return np.array(tmp).astype(np.uint8)

        def bb1_encoder(block: str) -> npt.NDArray[np.uint8]:
            return encode_block(remove_substructures(block), MAX_ENC_SIZE_BB1)

        bb1_encoding = joblib.Parallel(n_jobs=-1)(joblib.delayed(bb1_encoder)(smile) for smile in tqdm(x.bb1_smiles, desc="Encoding block 1"))
        x.bb1_ecfp = bb1_encoding

        bb2_encoding = joblib.Parallel(n_jobs=-1)(joblib.delayed(encode_block)(smile, MAX_ENC_SIZE_BB2) for smile in tqdm(x.bb2_smiles, desc="Encoding block 2"))
        x.bb2_ecfp = bb2_encoding

        bb3_encoding = joblib.Parallel(n_jobs=-1)(joblib.delayed(encode_block)(smile, MAX_ENC_SIZE_BB3) for smile in tqdm(x.bb3_smiles, desc="Encoding block 3"))
        x.bb3_ecfp = bb3_encoding

        return x
