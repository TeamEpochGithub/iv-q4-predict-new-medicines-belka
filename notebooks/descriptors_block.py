from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd

dtypes = {'buildingblock1_smiles': np.int16, 'buildingblock2_smiles': np.int16, 'buildingblock3_smiles': np.int16,
          'binds_BRD4':np.byte, 'binds_HSA':np.byte, 'binds_sEH':np.byte}

train = pd.read_csv('data/shrunken/train.csv', dtype = dtypes,  nrows = 15000)
print(train.head())
# %%
import os
print("Current Working Directory:", os.getcwd())
# %%
from src.typing.xdata import XData

x_data = XData(
    building_blocks=np.array(train[['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles']], dtype=np.int16),
    molecule_smiles=train['molecule_smiles'].tolist(),
    bb1=train['buildingblock1_smiles'].tolist(),
    bb2=train['buildingblock2_smiles'].tolist(),
    bb3=train['buildingblock3_smiles'].tolist()
)

# %%
# Apply Transformation block
from src.modules.transformation.descriptors_transformation_block import Descriptors
descriptors_transformation_block = Descriptors()
descriptors_transformation_block.custom_transform(x_data)


