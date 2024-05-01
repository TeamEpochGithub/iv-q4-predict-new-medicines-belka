"""Inefficient block for morgan fingerprints."""
import numpy as np
import numpy.typing as npt
import pandas as pd
from rdkit import Chem  # type: ignore[import-not-found]
from rdkit.Chem import AllChem  # type: ignore[import-not-found]
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData


class Morgan(VerboseTransformationBlock):
    """Class for creating Morgan fingerprints."""

    def custom_transform(self, x: XData) -> XData:
        """Transform XData into molecules with fingerprints.

        :param x: XData
        :return: Transformed XData
        """
        smile_df = pd.DataFrame({"smiles": x.molecule_smiles})

        smile_df["smiles"] = smile_df["smiles"].apply(Chem.MolFromSmiles)

        # Generate ECFPs
        def generate_ecfp(molecule: str, radius: int = 2, bits: int = 1024) -> npt.NDArray[np.int8]:
            if molecule is None:
                return None
            return np.array(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

        tqdm.pandas()
        smile_df["smiles"] = smile_df["smiles"].progress_apply(generate_ecfp)
        x.molecule_smiles = smile_df["smiles"].to_numpy()

        return x
