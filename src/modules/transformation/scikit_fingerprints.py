"""Converts the molcule building blocks into Fingerprints using scikit-fingerprints."""
import gc
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit.DataStructs.cDataStructs import ExplicitBitVect  # type: ignore[import-not-found]
from skfp.fingerprints import ECFPFingerprint  # type: ignore[import-not-found]
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData

NUM_FUTURES = 100
MIN_CHUNK_SIZE = 1000


@dataclass
class ScikitFingerprints(VerboseTransformationBlock):
    """Converts the molecule building blocks into fingerprints using RDKit.

    :param convert_building_blocks: Whether to convert the building blocks
    :param convert_molecules: Whether to convert the molecules
    :param replace_array: Whether to replace the array with the ECFP fingerprints

    :param bits: The number of bits in the ECFP
    :param radius: The radius of the ECFP
    :param useFeatures: Whether to use features in the ECFP
    """

    convert_building_blocks: bool = False
    convert_molecules: bool = False
    replace_array: bool = False

    fingerprint: Any = None

    @staticmethod
    def _convert_smile(smiles: list[str], fingerprint: Any) -> list[ExplicitBitVect]:  # noqa: ANN401
        """Worker function to process a single SMILES string."""
        f1 = ECFPFingerprint(fp_size=1024)
        return np.concatenate((fingerprint.fit_transform(X=smiles), f1.fit_transform(X=smiles)), axis=1)
        # return fingerprint.fit_transform(X=smiles) + f1.fit_transform(X=smiles)

    def _convert_smile_array(self, smile_array: list[str], desc: str) -> list[ExplicitBitVect] | list[dict[str, ExplicitBitVect]]:
        if not isinstance(smile_array[0], str):
            self.log_to_warning("Not a SMILE (string) array. Skipping conversion.")

        fingerprint = [self._convert_smile([smile], self.fingerprint)[0] for smile in tqdm(smile_array, desc=desc)]
        if self.replace_array:
            return fingerprint
        return [{"smile": smile, "fingerprint": fingerprint} for smile, fingerprint in zip(smile_array, fingerprint, strict=False)]

    def _convert_smile_array_parallel(self, smile_array: list[str], desc: str) -> list[ExplicitBitVect]:
        """Convert a list of SMILES strings into the fingerprints using multiprocessing.

        :param smile_array: A list of SMILES strings.
        :param desc: Description for logging purposes.
        :return: A list of fingerprints or a list of dictionaries containing SMILES and fingerprint pairs.
        """
        if not isinstance(smile_array[0], str):
            self.log_to_warning("Not a SMILE (string) array. Skipping conversion.")
            return []

        chunk_size = len(smile_array) // NUM_FUTURES
        chunk_size = max(chunk_size, MIN_CHUNK_SIZE)
        chunks = [smile_array[i : i + chunk_size] for i in range(0, len(smile_array), chunk_size)]

        results = []
        with ProcessPoolExecutor() as executor:
            self.log_to_terminal("Creating futures for fingerprint conversion.")
            futures = [executor.submit(self._convert_smile, chunk, self.fingerprint) for chunk in chunks]
            for future in tqdm(futures, total=len(futures), desc=desc):
                results.extend(future.result())

        return results

    def custom_transform(self, data: XData) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        if self.convert_building_blocks:
            if data.bb1_smiles is None or data.bb2_smiles is None or data.bb3_smiles is None:
                raise ValueError("There is no SMILE information for at least on building block. Can't convert to fingerprint")
            data.bb1_ecfp = self._convert_smile_array(data.bb1_smiles, desc="Creating fingerprint for bb1")
            data.bb2_ecfp = self._convert_smile_array(data.bb2_smiles, desc="Creating fingerprint for bb2")
            data.bb3_ecfp = self._convert_smile_array(data.bb3_smiles, desc="Creating fingerprint for bb3")

        if self.convert_molecules:
            if data.molecule_smiles is None:
                raise ValueError("There is no SMILE information for the molecules, can't convert to fingerprint")
            data.molecule_ecfp = self._convert_smile_array_parallel(data.molecule_smiles, desc="Creating fingerprint for molecules")
            if self.replace_array:
                data.molecule_smiles = None

        gc.collect()
        return data
