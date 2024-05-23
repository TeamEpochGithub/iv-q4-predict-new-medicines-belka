"""Module to encode SMILES into categorical data."""
from dataclasses import dataclass

import joblib
import numpy as np
import numpy.typing as npt
from rdkit import Chem  # type: ignore[import-not-found]
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData
from src.utils.logger import logger

MAX_ENC_SIZE_MOLECULE = 142
MAX_ENC_SIZE_BB1 = 42
MAX_ENC_SIZE_BB2 = 54
MAX_ENC_SIZE_BB3 = 43

ENCODING = {
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


@dataclass
class SmileAtomEncoder(VerboseTransformationBlock):
    """Class that encodes the molecules atomwise, instead of characterwise as in SmileCharacterEncoder."""

    convert_building_blocks: bool = False
    convert_molecules: bool = False

    def custom_transform(self, x: XData) -> XData:
        """Encode the SMILE strings into categorical data.

        :param x: XData.
        :return: XData with ecfp replaced by encodings.
        """
        smiles = x.molecule_smiles

        if self.convert_molecules:
            smiles_enc = joblib.Parallel(n_jobs=-1)(joblib.delayed(encode_smiles)(smile, MAX_ENC_SIZE_MOLECULE) for smile in tqdm(smiles, desc="Encoding Molcules Atomise"))
            x.molecule_ecfp = np.stack(smiles_enc)

        if self.convert_building_blocks:
            bb1_encoding = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(encode_smiles_bb_1)(smile, MAX_ENC_SIZE_BB1) for smile in tqdm(x.bb1_smiles, desc="Encoding block 1 Atomwise")
            )
            x.bb1_ecfp = bb1_encoding

            bb2_encoding = joblib.Parallel(n_jobs=-1)(joblib.delayed(encode_smiles)(smile, MAX_ENC_SIZE_BB2) for smile in tqdm(x.bb2_smiles, desc="Encoding block 2 Atomwise"))
            x.bb2_ecfp = bb2_encoding

            bb3_encoding = joblib.Parallel(n_jobs=-1)(joblib.delayed(encode_smiles)(smile, MAX_ENC_SIZE_BB3) for smile in tqdm(x.bb3_smiles, desc="Encoding block 3 Atomwise"))
            x.bb3_ecfp = bb3_encoding

        return x


def encode_smiles(smiles: str, padding: int) -> npt.NDArray[np.uint8]:
    """Encode an atom based on enc dictionary.

    :param smile: The smile string
    :return: Encoded array
    """
    x = 0
    tmp = []
    while x < len(smiles):
        # Check two first
        two_chars = smiles[x : x + 2]
        if two_chars in ENCODING:
            tmp.append(ENCODING[two_chars])
            x += 2
        elif smiles[x] in ENCODING:
            tmp.append(ENCODING[smiles[x]])
            x += 1
        else:
            logger.info("Can't find " + smiles[x] + " nor " + smiles[x : x + 2] + ". Skipping...")
            x += 1
    tmp = tmp + [0] * (padding - len(tmp))
    return np.array(tmp).astype(np.uint8)


def encode_smiles_bb_1(block: str, padding: int) -> npt.NDArray[np.uint8]:
    """Encode an atom based on enc dictionary."""
    return encode_smiles(remove_substructures(block), padding)


def remove_substructures(block: str) -> str:
    """Remove substructures from block.

    :param block: SMILE of block
    :return: SMILE of modified block
    """
    fmoc = Chem.MolFromSmiles("O=COCC1c2ccccc2-c2ccccc21")  # Biphenyl core
    boc = Chem.MolFromSmiles("CC(C)(C)OC(=O)")  # Tert butyl acetate

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
