"""Converts the molcule building blocks into Extended Connectivity Fingerprints (ECFP) using RDKit."""
import gc
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import AllChem  # type: ignore[import-not-found]
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData

NUM_FUTURES = 100
MIN_CHUNK_SIZE = 1000


@dataclass
class ECFP(VerboseTransformationBlock):
    """Converts the molcule building blocks into Extended Connectivity Fingerprints (ECFP) using RDKit.

    :param convert_building_blocks: Whether to convert the building blocks
    :param convert_molecules: Whether to convert the molecules
    :param delete_smiles: Whether to delete the SMILES after conversion

    :param bits: The number of bits in the ECFP
    :param radius: The radius of the ECFP
    :param use_features: Whether to use features in the ECFP
    """

    convert_building_blocks: bool = False
    convert_molecules: bool = False
    delete_smiles: bool = False

    bits: int = 128
    radius: int = 2
    use_features: bool = False

    @staticmethod
    def _convert_smile_batch(
        smiles: npt.NDArray[np.str_],
        radius: int,
        bits: int,
        *,
        use_features: bool = False,
        progressbar_desc: str | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Worker function to process a single SMILES string.

        :param smiles: A list of SMILES strings.
        :param radius: The radius of the ECFP.
        :param bits: The number of bits in the ECFP.
        :param use_features: Whether to use features in the ECFP.
        :param progressbar_desc: Description for the progress bar.
        :return: A list of ECFP fingerprints or a list of dictionaries containing SMILES and ECFP pairs.
        """
        result = np.empty((len(smiles), bits // 8), dtype=np.uint8)
        iterator = tqdm(enumerate(smiles), desc=progressbar_desc) if progressbar_desc is not None else enumerate(smiles)
        for idx, smile in iterator:
            mol = Chem.MolFromSmiles(smile)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits, useFeatures=use_features)
            result[idx] = np.packbits(np.array(fingerprint))
        return result

    def _convert_smile_array_parallel(self, smile_array: npt.NDArray[np.str_], desc: str) -> npt.NDArray[np.uint8]:
        """Convert a list of SMILES strings into their ECFP fingerprints using multiprocessing.

        :param smile_array: A list of SMILES strings.
        :param desc: Description for logging purposes.
        :return: A list of ECFP fingerprints or a list of dictionaries containing SMILES and ECFP pairs.
        """
        chunk_size = len(smile_array) // NUM_FUTURES
        chunk_size = max(chunk_size, MIN_CHUNK_SIZE)
        chunks = [smile_array[i : i + chunk_size] for i in range(0, len(smile_array), chunk_size)]

        result = np.empty((len(smile_array), self.bits // 8), dtype=np.uint8)
        with ProcessPoolExecutor() as executor:
            self.log_to_terminal("Creating futures for ECFP conversion.")
            futures = [executor.submit(self._convert_smile_batch, smiles=chunk, radius=self.radius, bits=self.bits, use_features=self.use_features) for chunk in chunks]

            last_idx = 0
            for future in tqdm(futures, total=len(futures), desc=desc):
                partial_result = future.result()
                result[last_idx : last_idx + len(partial_result)] = partial_result
                last_idx += len(partial_result)

        return result

    def custom_transform(self, data: XData) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        if self.convert_building_blocks:
            if data.bb1_smiles is None or data.bb2_smiles is None or data.bb3_smiles is None:
                raise ValueError("There is no SMILE information for at least on building block. Can't convert to ECFP")
            data.bb1_ecfp = self._convert_smile_batch(
                data.bb1_smiles,
                radius=self.radius,
                bits=self.bits,
                use_features=self.use_features,
                progressbar_desc="Creating ECFP for bb1",
            )
            data.bb2_ecfp = self._convert_smile_batch(
                data.bb2_smiles,
                radius=self.radius,
                bits=self.bits,
                use_features=self.use_features,
                progressbar_desc="Creating ECFP for bb2",
            )
            data.bb3_ecfp = self._convert_smile_batch(
                data.bb3_smiles,
                radius=self.radius,
                bits=self.bits,
                use_features=self.use_features,
                progressbar_desc="Creating ECFP for bb3",
            )

        if self.convert_molecules:
            if data.molecule_smiles is None:
                raise ValueError("There is no SMILE information for the molecules, can't convert to ECFP")
            data.molecule_ecfp = self._convert_smile_array_parallel(data.molecule_smiles, desc="Creating ECFP for molecules")
            if self.delete_smiles:
                data.molecule_smiles = None

        gc.collect()
        return data
