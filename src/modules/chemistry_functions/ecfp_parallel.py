"""Module to convert array of smiles into ecfp fingerprints in parallel."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import numpy.typing as npt
from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import AllChem  # type: ignore[import-not-found]
from tqdm import tqdm

NUM_FUTURES = 100
MIN_CHUNK_SIZE = 1000


def convert_smile_batch(
    smiles: npt.NDArray[np.bytes_],
    radius: int,
    bits: int,
    *,
    use_features: bool = False,
    progressbar_desc: str | None = None,
) -> npt.NDArray[np.uint8]:
    """Worker function to process a batch of SMILES strings.

    :param smiles: A list of SMILES strings.
    :param radius: The radius of the ECFP.
    :param bits: The number of bits in the ECFP.
    :param use_features: Whether to use features in the ECFP.
    :param progressbar_desc: Description for the progress bar.
    :return: A numpy array of fingerprints.
    """
    result = np.empty((len(smiles), bits // 8), dtype=np.uint8)
    iterator = tqdm(enumerate(smiles), desc=progressbar_desc) if progressbar_desc is not None else enumerate(smiles)
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
        futures = [executor.submit(convert_smile_batch, smiles=chunk, radius=radius, bits=bits, use_features=use_features) for chunk in chunks]

        last_idx = 0
        for future in tqdm(futures, total=len(futures), desc=desc):
            partial_result = future.result()
            result[last_idx : last_idx + len(partial_result)] = partial_result
            last_idx += len(partial_result)

    return result
