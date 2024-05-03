"""Transformation block for descriptors."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, ClassVar

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
        "MaxAbsEStateIndex",
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
    convert_bbs: bool = False

    def custom_transform(self, data: XData) -> XData:
        """Apply a custom transformation to the molecule smiles.

        Converts each SMILES in data to molecular descriptors and compiles results into a DataFrame.

        :param data: The data to transform
        :return: A DataFrame where each row contains the descriptors of a molecule.
        """
        self.descriptor_functions = {desc_name: getattr(rdkitDesc, desc_name) for desc_name in self.descriptor_names}
        if self.convert_molecules and self.convert_bbs:
            transformed_X = self.transform_molecule(data)
            return self.transform_bb(transformed_X)
        if self.convert_molecules:
            return self.transform_molecule(data)
        if self.convert_bbs:
            return self.transform_bb(data)
        return data

    def transform_molecule(self, data: XData) -> XData:
        """Compute descriptors for the whole molecule.

        :param data: The data to transform
        :return: A XData object containing the descriptors for each molecule.
        """
        molecule_smiles = data.molecule_smiles if data.molecule_smiles is not None else []

        chunk_size = len(molecule_smiles) // NUM_FUTURES
        chunk_size = max(chunk_size, MIN_CHUNK_SIZE)
        chunks = [molecule_smiles[i : i + chunk_size] for i in range(0, len(molecule_smiles), chunk_size)]

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.calculate_descriptors, chunk, names=self.descriptor_names) for chunk in chunks]

            mol_desc = []
            for future in tqdm(futures, desc="Creating Descriptors for molecules", total=len(futures)):
                mol_desc.extend(list(future.result()))

        data.molecule_desc = mol_desc
        return data

    def transform_bb(self, data: XData) -> XData:
        """Compute descriptors for each building block.

        :param data: The data to transform
        :return: A XData object containing the descriptors for each building block.
        """
        # Delayed tasks
        bb1_smiles = data.bb1_smiles if data.bb1_smiles is not None else []
        bb2_smiles = data.bb2_smiles if data.bb2_smiles is not None else []
        bb3_smiles = data.bb3_smiles if data.bb3_smiles is not None else []
        futures_bb1 = [self.delayed_descriptors(smiles) for smiles in bb1_smiles]
        futures_bb2 = [self.delayed_descriptors(smiles) for smiles in bb2_smiles]
        futures_bb3 = [self.delayed_descriptors(smiles) for smiles in bb3_smiles]

        with tqdm(total=len(futures_bb1) + len(futures_bb2) + len(futures_bb3), desc="Creating Descriptors for BBs") as pbar:
            bb1_desc = compute(*futures_bb1)
            pbar.update(len(futures_bb1))
            bb2_desc = compute(*futures_bb2)
            pbar.update(len(futures_bb2))
            bb3_desc = compute(*futures_bb3)
            pbar.update(len(futures_bb3))

        # Updating the XData instance
        data.bb1_desc = bb1_desc if bb1_desc is not None else []
        data.bb2_desc = bb2_desc if bb2_desc is not None else []
        data.bb3_desc = bb3_desc if bb3_desc is not None else []

        return data

    @delayed
    def delayed_descriptors(self, smiles: str) -> list[Any]:
        """Delays the computation of descriptors."""
        return self.calculate_descriptors([smiles], self.descriptor_names)[0]

    @staticmethod
    def calculate_descriptors(smiles: list[str], names: list[str]) -> list[Any]:
        """Calculate the descriptors for the molecule using the predefined descriptor functions."""
        functions = {desc_name: getattr(rdkitDesc, desc_name) for desc_name in names}
        results = []
        if smiles is not None:
            for smile in smiles:
                mol = Chem.MolFromSmiles(smile)
                results.append([functions[desc_name](mol) for desc_name in names])
        else:
            return [0 in range(len(names))]

        return results
