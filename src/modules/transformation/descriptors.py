"""Transformation block for descriptors."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from dask import compute, delayed
from rdkit import Chem  # type: ignore[import]
from rdkit.Chem import Descriptors as rdkitDesc  # type: ignore[import]
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData

NUM_FUTURES = 100
MIN_CHUNK_SIZE = 1000


@dataclass
class Descriptors(VerboseTransformationBlock):
    """Calculates the descriptors of molecules from SMILES.

    :param: descriptor_names: List of descriptor names.
    :param convert_molecules: Whether to convert the molecules.
    :param convert_bbs: Whether to convert the building blocks.
    """

    descriptor_names: ClassVar[list[str]] = [
        "MaxAbsEStateIndex",
        "MaxEStateIndex",
        "MinAbsEStateIndex",
        "MinEStateIndex",
        "qed",
        "SPS",
        "MolWt",
        "HeavyAtomMolWt",
        "ExactMolWt",
        "NumValenceElectrons",
        "NumRadicalElectrons",
        "MaxPartialCharge",
        "MinPartialCharge",
        "MaxAbsPartialCharge",
        "MinAbsPartialCharge",
        "FpDensityMorgan1",
        "FpDensityMorgan2",
        "FpDensityMorgan3",
        "BCUT2D_MWHI",
        "BCUT2D_MWLOW",
        "BCUT2D_CHGHI",
        "BCUT2D_CHGLO",
        "BCUT2D_LOGPHI",
        "BCUT2D_LOGPLOW",
        "BCUT2D_MRHI",
        "BCUT2D_MRLOW",
        "AvgIpc",
        "BalabanJ",
        "BertzCT",
        "Chi0",
        "Chi0n",
        "Chi0v",
        "HallKierAlpha",
        "Ipc",
        "LabuteASA",
        "PEOE_VSA1",
        "SMR_VSA1",
        "SlogP_VSA1",
        "TPSA",
        "EState_VSA1",
        "VSA_EState1",
        "FractionCSP3",
        "MolLogP",
        "MolMR",
    ]
    convert_molecules: bool = False
    convert_building_blocks: bool = False

    def custom_transform(self, data: XData) -> XData:
        """Apply a custom transformation to the molecule smiles.

        Converts each SMILES in data to molecular descriptors and compiles results into a DataFrame.

        :param data: The data to transform
        :return: XData object containing the descriptors for each molecule.
        """
        if self.convert_molecules:
            data.molecule_desc = self.transform_molecule(data)
        if self.convert_building_blocks:
            data.bb1_desc, data.bb2_desc, data.bb3_desc = self.transform_bb(data).values()
        return data

    def transform_molecule(self, data: XData) -> npt.NDArray[np.float32]:
        """Compute descriptors for the whole molecule.

        :param data: The data to transform
        :return: A XData object containing the descriptors for each molecule.
        """
        if data.molecule_smiles is None:
            raise ValueError("Molecule smiles are not provided.")

        chunk_size = len(data.molecule_smiles) // NUM_FUTURES
        chunk_size = max(chunk_size, MIN_CHUNK_SIZE)
        chunks = [data.molecule_smiles[i : i + chunk_size] for i in range(0, len(data.molecule_smiles), chunk_size)]

        result = np.empty([len(data), len(self.descriptor_names)], dtype=np.float32)
        last_idx = 0
        with ProcessPoolExecutor() as executor:
            self.log_to_terminal("Creating futures for molecule descriptor calculation.")
            futures = [executor.submit(self._calculate_descriptors, chunk, names=self.descriptor_names) for chunk in chunks]

            for future in tqdm(futures, desc="Creating Descriptors for molecules", total=len(futures)):
                partial_result = future.result()
                result[last_idx : last_idx + len(partial_result)] = partial_result
                last_idx += len(partial_result)

        np.nan_to_num(result, copy=False)
        return result

    def transform_bb(self, data: XData) -> dict[str, npt.NDArray[np.float32]]:
        """Compute descriptors for each building block.

        :param data: The data to transform
        :return: A XData object containing the descriptors for each building block.
        """
        if data.bb1_smiles is None or data.bb2_smiles is None or data.bb3_smiles is None:
            raise ValueError("Building block smiles are not provided.")

        # Delayed tasks
        futures_bb1 = [self._delayed_descriptors(smiles) for smiles in data.bb1_smiles]
        futures_bb2 = [self._delayed_descriptors(smiles) for smiles in data.bb2_smiles]
        futures_bb3 = [self._delayed_descriptors(smiles) for smiles in data.bb3_smiles]

        with tqdm(total=len(futures_bb1) + len(futures_bb2) + len(futures_bb3), desc="Creating Descriptors for BBs") as pbar:
            bb1_desc = compute(*futures_bb1)
            pbar.update(len(futures_bb1))
            bb2_desc = compute(*futures_bb2)
            pbar.update(len(futures_bb2))
            bb3_desc = compute(*futures_bb3)
            pbar.update(len(futures_bb3))

        return {
            "bb1": bb1_desc,
            "bb2": bb2_desc,
            "bb3": bb3_desc,
        }

    @delayed
    def _delayed_descriptors(self, smiles: str) -> npt.NDArray[np.float32]:
        """Delays the computation of descriptors."""
        return self._calculate_descriptors(np.array([smiles]), self.descriptor_names)[0]

    @staticmethod
    def _calculate_descriptors(smiles: npt.NDArray[np.str_], names: list[str]) -> npt.NDArray[np.float32]:
        """Calculate the descriptors for the molecule using the predefined descriptor functions."""
        functions = [getattr(rdkitDesc, desc_name) for desc_name in names]
        result = np.empty([len(smiles), len(names)], dtype=np.float32)
        for idx, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            for sub_idx, desc_fn in enumerate(functions):
                result[idx][sub_idx] = desc_fn(mol)

        return result
