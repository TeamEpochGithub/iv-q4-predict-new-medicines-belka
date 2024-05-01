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


@dataclass
class ECFP(VerboseTransformationBlock):
    """Converts the molcule building blocks into Extended Connectivity Fingerprints (ECFP) using RDKit.

    :param convert_building_blocks: Whether to convert the building blocks
    :param convert_molecules: Whether to convert the molecules
    :param replace_array: Whether to replace the array with the ECFP fingerprints
    :param chunk_size: The size of the chunks to process in parallel

    :param bits: The number of bits in the ECFP
    :param radius: The radius of the ECFP
    :param useFeatures: Whether to use features in the ECFP
    """

    convert_building_blocks: bool = False
    convert_molecules: bool = False
    replace_array: bool = False
    chunk_size: int = 100000

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

    def _convert_smile_array_parallel(self, smile_array: list[str], desc: str) -> list[ExplicitBitVect] | list[dict[str, ExplicitBitVect]]:
        """Convert a list of SMILES strings into their ECFP fingerprints using multiprocessing.

        :param smile_array: A list of SMILES strings.
        :param desc: Description for logging purposes.
        :return: A list of ECFP fingerprints or a list of dictionaries containing SMILES and ECFP pairs.
        """
        if not isinstance(smile_array[0], str):
            self.log_to_warning("Not a SMILE (string) array. Skipping conversion.")
            return []

        chunks = [smile_array[i : i + self.chunk_size] for i in range(0, len(smile_array), self.chunk_size)]

        results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._convert_smile, chunk, radius=self.radius, bits=self.bits, use_features=self.use_features) for chunk in chunks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                results.extend(future.result())

        if self.replace_array:
            return results

        flat_smiles = [smile for chunk in chunks for smile in chunk]
        return [{"smile": smile, "ecfp": ecfp} for smile, ecfp in zip(flat_smiles, results, strict=False)]

    def custom_transform(self, data: XData) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        if self.convert_building_blocks:
            data.bb1 = self._convert_smile_array(data.bb1, desc="Converting bb1")
            data.bb2 = self._convert_smile_array(data.bb2, desc="Converting bb2")
            data.bb3 = self._convert_smile_array(data.bb3, desc="Converting bb3")

        if self.convert_molecules:
            data.molecule_smiles = self._convert_smile_array_parallel(data.molecule_smiles, desc="Converting molecules")

        gc.collect()
        return data
