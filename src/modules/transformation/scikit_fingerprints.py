"""Converts the molcule building blocks into the specified fingerprint using SKFP."""
import gc
from dataclasses import dataclass
from typing import Any

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData

NUM_FUTURES = 100
MIN_CHUNK_SIZE = 1000


@dataclass
class ScikitFingerprints(VerboseTransformationBlock):
    """Converts the molecule building blocks into fingerprints using RDKit.

    :param convert_building_blocks: Whether to convert the building blocks
    :param convert_molecules: Whether to convert the molecules
    :param delete_smiles: Whether to delete the SMILES after conversion

    :param fingerprint: The fingerprint to use
    """

    convert_building_blocks: bool = False
    convert_molecules: bool = False
    delete_smiles: bool = False

    fingerprint: Any = None

    def custom_transform(self, data: XData) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        if self.convert_building_blocks:
            if data.bb1_smiles is None or data.bb2_smiles is None or data.bb3_smiles is None:
                raise ValueError("There is no SMILE information for at least on building block. Can't convert to Fingerprint")

            self.log_to_terminal("Creating Fingerprint for building block 1")
            data.bb1_ecfp = self.fingerprint.fit_transform(X=data.bb1_smiles)
            self.log_to_terminal("Creating Fingerprint for building block 2")
            data.bb2_ecfp = self.fingerprint.fit_transform(X=data.bb2_smiles)
            self.log_to_terminal("Creating Fingerprint for building block 3")
            data.bb3_ecfp = self.fingerprint.fit_transform(X=data.bb3_smiles)

        if self.convert_molecules:
            if data.molecule_smiles is None:
                raise ValueError("There is no SMILE information for the molecules, can't convert to ECFP")
            self.log_to_terminal("Creating Fingerprint for molecules")
            chunk_size = len(data) // NUM_FUTURES
            chunk_size = max(chunk_size, MIN_CHUNK_SIZE)
            data.molecule_ecfp = self.fingerprint.fit_transform(X=data.molecule_smiles, batch_size=chunk_size, n_jobs=-1)

        if self.delete_smiles:
            del data.bb1_smiles
            del data.bb2_smiles
            del data.bb3_smiles
            del data.molecule_smiles

        gc.collect()
        return data
