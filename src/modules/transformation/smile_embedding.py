
"""Create the embeddings of the molecules using smiles2vec"""
import numpy.typing as npt

import numpy as np
from rdkit import Chem
from src.typing.xdata import XData
from gensim.models import Word2Vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from dask.distributed import Client

import dask
import dask.bag as db


class SmileEmbedding(VerboseTransformationBlock):
    """ Create the embeddings of the building blocks and the molecule.

    param model_path: the path of the pre-trained mol2vec
    param unseen: the token of the unseen fingerprints
    param molecule: """

    model_path: str = 'https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl'
    unseen: str = "UNK"
    molecule: bool = True
    building_block: bool = True

    # def embeddings(self, smiles:list[str])-> list:
    #     """Compute the embeddings of the molecules or blocks.
    #
    #     param smile: list containing the molecules as strings
    #     return: list containing the embeddings of the atoms"""
    #
    #     # extract the embedding of the unseen token
    #     unseen_vec = self.model.get_vector(self.unseen)
    #     keys = set(self.model.key_to_index)
    #
    #
    #     for smile in tqdm(smiles):
    #         # create the molecule from the smile format
    #         molecule = Chem.MolFromSmiles(smile)
    #
    #         # create a sentence containing the substructures
    #         sentence = MolSentence(mol2alt_sentence(molecule, 1))
    #
    #         # compute the embeddings of each structure
    #         embeddings = []
    #         for structure in sentence:
    #
    #             # check whether the structure exists
    #             if structure in set(sentence) & keys:
    #                 embeddings.append(self.model.get_vector(structure))
    #             else:
    #                 embeddings.append(unseen_vec)
    #
    #         features.append(np.array(embeddings))
    #
    #     return features

    def embeddings(self, smile:str) -> npt.NDArray[np.float32]:
        """Compute the embeddings of a molecule or block

        param smile: the molecule as the smile format
        return: a numpy array containing the embeddings"""

        # create the molecule from the smile format
        molecule = Chem.MolFromSmiles(smile)

        # create a sentence containing the substructures
        sentence = MolSentence(mol2alt_sentence(molecule, 1))

        # compute the embeddings of each structure
        embeddings = []
        for structure in sentence:
            # check whether the structure exists
            if structure in set(sentence) & self.keys:
                embeddings.append(self.model.get_vector(structure))
            else:
                embeddings.append(self.unseen_vec)

        return np.array(embeddings)

    def multi_process(self, smiles:list[str]) -> list:
        # Initialize a Dask client
        client = Client()

        b = db.from_sequence(smiles)
        embedding = b.map(self.embeddings)
        embedding = embedding.compute()

        # Close the client if not needed anymore
        client.close()

        return embedding



    def custom_transform(self, data: XData) -> XData:


        # load the pre-trained model from gensim
        self.model = Word2Vec.load(self.model_path).wv

        # extract the embedding of the unseen token
        self.unseen_vec = self.model.get_vector(self.unseen)
        self.keys = set(self.model.key_to_index)

        # compute the embeddings for each molecule
        if self.molecule:
            data.molecule_smiles = self.multi_process(data.molecule_smiles)

        # compute the embeddings for each block
        if self.building_block:
            data.bb1 = self.multi_process(data.bb1)
            data.bb2 = self.multi_process(data.bb2)
            data.bb3 = self.multi_process(data.bb3)

        return data
