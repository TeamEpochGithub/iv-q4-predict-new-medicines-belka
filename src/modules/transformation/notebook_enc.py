from src.typing.xdata import XData
import joblib
from tqdm import tqdm
from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
import numpy as np


class NotebookEncoding(VerboseTransformationBlock):

    def custom_transform(self, x: XData) -> XData:
        enc = {'l': 1, 'y': 2, '@': 3, '3': 4, 'H': 5, 'S': 6, 'F': 7, 'C': 8, 'r': 9, 's': 10, '/': 11, 'c': 12, 'o': 13,
               '+': 14, 'I': 15, '5': 16, '(': 17, '2': 18, ')': 19, '9': 20, 'i': 21, '#': 22, '6': 23, '8': 24, '4': 25, '=': 26,
               '1': 27, 'O': 28, '[': 29, 'D': 30, 'B': 31, ']': 32, 'N': 33, '7': 34, 'n': 35, '-': 36}

        smiles = x.molecule_smiles

        def encode_smile(smile):
            tmp = [enc[i] for i in smile]
            tmp = tmp + [0]*(142-len(tmp))
            return np.array(tmp).astype(np.uint8)

        smiles_enc = joblib.Parallel(n_jobs=-1)(joblib.delayed(encode_smile)(smile) for smile in tqdm(smiles))
        smiles_enc = np.stack(smiles_enc)

        x.molecule_ecfp = smiles_enc

        return x
