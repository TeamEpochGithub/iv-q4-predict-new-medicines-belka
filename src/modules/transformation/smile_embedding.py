"""Create the embeddings of the molecules using smiles2vec"""
import numpy.typing as npt
import numpy as np
from rdkit import Chem
from src.typing.xdata import XData
from gensim.models import Word2Vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock

class SmileEmbedding(VerboseTransformationBlock):
    """ Create the embeddings of the building blocks and the molecule.

    param model_path: the path of the pre-trained mol2vec
    param unseen: the token of the unseen fingerprints
    param molecule: """

    model_path: str = 'https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl'
    unseen: str = "UNK"
    molecule: bool = True
    building_block: bool = True

    def embeddings(self, smiles:list[str])-> list:
        """Compute the embeddings of the molecules or blocks.

        param smile: list containing the molecules as strings
        return: list containing the embeddings of the atoms"""

        # extract the embedding of the unseen token
        unseen_vec = self.model.get_vector(self.unseen)
        keys = set(self.model.key_to_index)

        features = []
        for smile in tqdm(smiles):
            # create the molecule from the smile format
            molecule = Chem.MolFromSmiles(smile)

            # create a sentence containing the substructures
            sentence = MolSentence(mol2alt_sentence(molecule, 1))

            # compute the embeddings of each structure
            embeddings = []
            for structure in sentence:

                # check whether the structure exists
                if structure in set(sentence) & keys:
                    embeddings.append(self.model.get_vector(structure))
                else:
                    embeddings.append(unseen_vec)

            features.append(np.array(embeddings))
        print('hello world')
        return features


    def custom_transform(self, data: XData) -> XData:

        # load the pre-trained model from gensim
        self.model = Word2Vec.load(self.model_path).wv

        # compute the embeddings for each molecule
        if self.molecule:
            data.molecule_smiles = self.embeddings(data.molecule_smiles)

        # compute the embeddings for each block
        if self.building_block:
            data.bb1 = self.embeddings(data.bb1)
            data.bb2 = self.embeddings(data.bb2)
            data.bb3 = self.embeddings(data.bb3)

        return data












