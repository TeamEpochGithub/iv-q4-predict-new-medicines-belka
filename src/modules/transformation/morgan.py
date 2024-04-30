from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm


class Morgan(VerboseTransformationBlock):

    def custom_transform(self, x: XData) -> XData:
        df = pd.DataFrame({'smiles': x.molecule_smiles})

        df['smiles'] = df['smiles'].apply(Chem.MolFromSmiles)

        # Generate ECFPs
        def generate_ecfp(molecule, radius=2, bits=1024):
            if molecule is None:
                return None
            return np.array(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

        tqdm.pandas()
        df['smiles'] = df['smiles'].progress_apply(generate_ecfp)
        x.molecule_smiles = df['smiles'].to_numpy()

        return x
