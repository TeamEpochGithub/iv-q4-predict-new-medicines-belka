# %%
from pathlib import Path

from rdkit.Chem import Descriptors
from rdkit import Chem
import numpy as np
import pandas as pd


from src.modules.transformation.descriptors_transformation_block import Descriptors
from src.setup.setup_data import setup_train_x_data
from src.typing.xdata import XData

dtypes = {'buildingblock1_smiles': np.int16, 'buildingblock2_smiles': np.int16, 'buildingblock3_smiles': np.int16,
          'binds_BRD4':np.byte, 'binds_HSA':np.byte, 'binds_sEH':np.byte}

train = pd.read_csv('data/shrunken/train.csv', dtype = dtypes, nrows=1000000)
train_samples = train.sample(frac = 0.1)

# %%
object_data = setup_train_x_data(Path("/home/daniel-de-dios/PycharmProjects/q4-detect-medicine/data/shrunken"), train_samples)
# %%
block = Descriptors(convert_molecules= True, convert_bbs=True)
transformed_data = block.custom_transform(object_data)