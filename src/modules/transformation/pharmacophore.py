"""Converts the molcule building blocks into Extended Connectivity Fingerprints (ECFP) using RDKit."""
import gc
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from skfp.fingerprints import PharmacophoreFingerprint # type: ignore[import-not-found]

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData


@dataclass
class Pharmacophore(VerboseTransformationBlock):
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

    def custom_transform(self, data: XData) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        pf = PharmacophoreFingerprint(fp_size=bits, n_jobs=-1)
        if self.convert_building_blocks:
            if data.bb1_smiles is None or data.bb2_smiles is None or data.bb3_smiles is None:
                raise ValueError("There is no SMILE information for at least on building block. Can't convert to ECFP")
            data.bb1_ecfp = pf.transform(data.bb1_smiles)
            data.bb2_ecfp = pf.transform(data.bb2_smiles)
            data.bb3_ecfp = pf.transform(data.bb3_smiles)

        if self.convert_molecules:
            if data.molecule_smiles is None:
                raise ValueError("There is no SMILE information for the molecules, can't convert to ECFP")
            data.molecule_ecfp = pf.transform(data.molecule_smiles)
            if self.replace_array:
                data.molecule_smiles = None

        gc.collect()
        return data
