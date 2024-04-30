"""Create the embeddings of the molecules using smiles2vec"""
import numpy.typing as npt
import numpy as np
from rdkit import Chem
from src.typing.xdata import XData
from gensim.models import Word2Vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock

class SmileEmbedding(VerboseTransformationBlock):
    """ Create the embeddings of the building blocks and the molecule.

    param model_path: the path of the pre-trained mol2vec
    param unseen: the token of the unseen fingerprints
    param molecule: """

    model_path: str = 'https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl'
    unseen: str = "UNK"
    _molecule: bool = True
    _building_block: bool = True

    def single_embedding(self, smile:str)-> npt.NDArray[np.float32]:
        """Compute the embedding of a molecule or a building block.

        param smile: the smile of the building block
        return: embeddings of the atoms in the molecule"""

        # create the molecule from the smile format
        molecule = Chem.MolFromSmiles(smile)

        # create a sentence containing the substructures
        sentence = MolSentence(mol2alt_sentence(molecule, 1))

        # extract the embedding of the unseen token
        unseen_vec = self.model.wv.get_vector(self.unseen)

        # compute the embedding of each substructure
        embeddings = []

        for structure in sentence:
            if structure in set(sentence) & self.keys:
                embeddings.append(self.model.wv.get_vector(structure))
            else:
                embeddings.append(unseen_vec)

        return np.array(embeddings)

    def multi_embeddings(self, smiles: list[str]) -> list:
        """Compute the embeddings of the molecules or blocks.

        param smiles: a list containing the smiles
        return: a list containing the embeddings"""

        return [self.single_embedding(smile) for smile in smiles]

    def custom_transform(self, data: XData) -> XData:

        # load the pre-trained model from gensim
        self.model = Word2Vec.load(self.model_path)

        # extract the tokens from the model
        self.keys = set(self.model.wv.key_to_index)

        # compute the embeddings for each molecule
        if self._molecule:
            smiles = data.molecule_smiles
            data.molecules_smiles = self.multi_embeddings(smiles)


        # compute the embeddings for each building block
        if self._building_block:

            data.bb1 = self.multi_embeddings(data.bb1)
            data.bb2 = self.multi_embeddings(data.bb2)
            data.bb3 = self.multi_embeddings(data.bb3)

        return data












