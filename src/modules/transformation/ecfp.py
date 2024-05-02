"""Converts the molcule building blocks into Extended Connectivity Fingerprints (ECFP) using RDKit."""
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import AllChem  # type: ignore[import-not-found]
from rdkit.DataStructs.cDataStructs import ExplicitBitVect  # type: ignore[import-not-found]
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
    :param replace_array: Whether to replace the array with the ECFP fingerprints

    :param bits: The number of bits in the ECFP
    :param radius: The radius of the ECFP
    :param useFeatures: Whether to use features in the ECFP
    """

    convert_building_blocks: bool = False
    convert_molecules: bool = False
    replace_array: bool = False

    bits: int = 128
    radius: int = 2
    use_features: bool = False

    @staticmethod
    def _convert_smile(smiles: list[str], radius: int, bits: int, *, use_features: bool = False) -> list[ExplicitBitVect]:
        """Worker function to process a single SMILES string."""
        result = []
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            result.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bits, useFeatures=use_features))
        return result

    def _convert_smile_array(self, smile_array: list[str], desc: str) -> list[ExplicitBitVect] | list[dict[str, ExplicitBitVect]]:
        if not isinstance(smile_array[0], str):
            self.log_to_warning("Not a SMILE (string) array. Skipping conversion.")

        ecfp = [self._convert_smile([smile], radius=self.radius, bits=self.bits, use_features=self.use_features)[0] for smile in tqdm(smile_array, desc=desc)]
        if self.replace_array:
            return ecfp
        return [{"smile": smile, "ecfp": ecfp} for smile, ecfp in zip(smile_array, ecfp, strict=False)]

    def _convert_smile_array_parallel(self, smile_array: list[str], desc: str) -> list[ExplicitBitVect]:
        """Convert a list of SMILES strings into their ECFP fingerprints using multiprocessing.

        :param smile_array: A list of SMILES strings.
        :param desc: Description for logging purposes.
        :return: A list of ECFP fingerprints or a list of dictionaries containing SMILES and ECFP pairs.
        """
        if not isinstance(smile_array[0], str):
            self.log_to_warning("Not a SMILE (string) array. Skipping conversion.")
            return []

        chunk_size = len(smile_array) // NUM_FUTURES
        chunk_size = max(chunk_size, MIN_CHUNK_SIZE)
        chunks = [smile_array[i : i + chunk_size] for i in range(0, len(smile_array), chunk_size)]

        results = []
        with ProcessPoolExecutor() as executor:
            self.log_to_terminal("Creating futures for ECFP conversion.")
            futures = [executor.submit(self._convert_smile, chunk, radius=self.radius, bits=self.bits, use_features=self.use_features) for chunk in chunks]
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
                raise ValueError("There is no SMILE information for at least on building block. Can't convert to ECFP")
            data.bb1_ecfp = self._convert_smile_array(data.bb1_smiles, desc="Converting bb1")
            data.bb2_ecfp = self._convert_smile_array(data.bb2_smiles, desc="Converting bb2")
            data.bb3_ecfp = self._convert_smile_array(data.bb3_smiles, desc="Converting bb3")

        if self.convert_molecules:
            if data.molecule_smiles is None:
                raise ValueError("There is no SMILE information for the molecules, can't convert to ECFP")
            data.molecule_ecfp = self._convert_smile_array_parallel(data.molecule_smiles, desc="Converting molecules")
            if self.replace_array:
                data.molecule_smiles = None

        gc.collect()
        return data
