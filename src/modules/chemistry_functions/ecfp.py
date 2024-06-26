"""Module to convert array of smiles into ecfp fingerprints in parallel."""
from concurrent.futures import ProcessPoolExecutor
from enum import Enum

import numpy as np
import numpy.typing as npt
from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import AllChem  # type: ignore[import-not-found]
from rdkit.DataStructs.cDataStructs import ExplicitBitVect  # type: ignore[import-not-found]
from tqdm import tqdm

NUM_FUTURES = 100
MIN_CHUNK_SIZE = 1000


class EcfpReturnType(Enum):
    """Enum to store the return type of the ECFP block."""

    RDKIT = 0
    NP_UNPACKED = 1
    NP_PACKED = 2


def convert_smiles_array(
    smiles_array: npt.NDArray[np.bytes_],
    radius: int,
    bits: int,
    *,
    return_type: EcfpReturnType = EcfpReturnType.NP_PACKED,
    use_features: bool = False,
    progressbar_desc: str | None = None,
) -> npt.NDArray[np.uint8] | list[ExplicitBitVect]:
    """Worker function to process a batch of SMILES strings into a packed ECFP array.

    :param smiles: A list of SMILES strings.
    :param radius: The radius of the ECFP.
    :param bits: The number of bits in the ECFP.
    :param use_features: Whether to use features in the ECFP.
    :param progressbar_desc: Description for the progress bar.
    :return: A numpy array of fingerprints.
    """
    iterator = tqdm(enumerate(smiles_array), desc=progressbar_desc) if progressbar_desc is not None else enumerate(smiles_array)
    result: list[ExplicitBitVect] | npt.NDArray[np.uint8]
    fingerprint: ExplicitBitVect

    match return_type:
        case EcfpReturnType.RDKIT:
            result = []
            for _, smile in iterator:
                mol = Chem.MolFromSmiles(smile)
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits, useFeatures=use_features)
                result.append(fingerprint)

        case EcfpReturnType.NP_UNPACKED:
            result = np.empty((len(smiles_array), bits), dtype=np.uint8)
            for idx, smile in iterator:
                mol = Chem.MolFromSmiles(smile)
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits, useFeatures=use_features)
                result[idx] = np.array(fingerprint)

        case EcfpReturnType.NP_PACKED:
            result = np.empty((len(smiles_array), bits // 8), dtype=np.uint8)
            for idx, smile in iterator:
                mol = Chem.MolFromSmiles(smile)
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits, useFeatures=use_features)
                result[idx] = np.packbits(np.array(fingerprint))

    return result


def convert_smile_array_parallel(smiles_array: npt.NDArray[np.bytes_], radius: int, bits: int, *, use_features: bool, desc: str) -> npt.NDArray[np.uint8]:
    """Convert a list of SMILES strings into their ECFP fingerprints using multiprocessing.

    :param smile_array: A list of SMILES strings.
    :param desc: Description for logging purposes.
    :return: A numpy array of fingerprints.
    """
    chunk_size = len(smiles_array) // NUM_FUTURES
    chunk_size = max(chunk_size, MIN_CHUNK_SIZE)
    chunks = [smiles_array[i : i + chunk_size] for i in range(0, len(smiles_array), chunk_size)]

    result = np.empty((len(smiles_array), bits // 8), dtype=np.uint8)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(convert_smiles_array, smiles_array=chunk, radius=radius, bits=bits, use_features=use_features) for chunk in chunks]

        last_idx = 0
        for future in tqdm(futures, total=len(futures), desc=desc):
            partial_result = future.result()
            result[last_idx : last_idx + len(partial_result)] = partial_result
            last_idx += len(partial_result)

    return result
