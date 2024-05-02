"""Create the embeddings of the molecules using smiles2vec."""
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import numpy.typing as npt
from gensim.models import Word2Vec
from mol2vec.features import MolSentence, mol2alt_sentence
from rdkit import Chem  # type: ignore[import]
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import XData

gensim_logger = logging.getLogger("gensim")
gensim_logger.setLevel(logging.WARNING)


class SmileEmbedding(VerboseTransformationBlock):
    """Create the embeddings of the building blocks and the molecule.

    param model_path: the path of the pre-trained mol2vec
    param unseen: the token of the unseen fingerprints
    param molecule:
    param building
    """

    model_path: str = "https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl"
    unseen: str = "UNK"
    molecule: bool = False
    building: bool = True
    chunk_size: int = 10000

    def embeddings(self, smiles: list[str]) -> list[npt.NDArray[np.float32]]:
        """Compute the embeddings of the molecules or blocks.

        param smile: list containing the molecules as strings
        return: list containing the embeddings of the atoms
        """
        # Extract the embedding of the unseen token
        unseen_vec = self.model.get_vector(self.unseen)
        keys = set(self.model.key_to_index)

        features = []
        for smile in smiles:
            # Create the molecule from the smile format
            molecule = Chem.MolFromSmiles(smile)

            # Create a sentence containing the substructures
            sentence = MolSentence(mol2alt_sentence(molecule, 1))

            # Compute the embeddings of each structure
            embeddings = []
            for structure in sentence:
                # check whether the structure exists
                if structure in set(sentence) & keys:
                    embeddings.append(self.model.get_vector(structure))
                else:
                    embeddings.append(unseen_vec)

            features.append(np.array(embeddings))

        return features

    def parallel_embeddings(self, smiles: list[str], desc: str) -> list[npt.NDArray[np.float32]]:
        """Compute the embeddings of the molecules using multiprocessing.

        param smiles: list containing the smiles of the molecules
        param desc: message to be shown during process
        """
        # Divide the smiles molecules into chunks
        chunks = [smiles[i : i + self.chunk_size] for i in range(0, len(smiles), self.chunk_size)]

        # Initialize the multiprocessing with the chunks
        results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.embeddings, chunk) for chunk in chunks]

            # Perform the multiprocessing on the chunks
            for future in tqdm(futures, total=len(futures), desc=desc):
                results.extend(future.result())

        return results

    def custom_transform(self, data: XData) -> XData:
        """Compute the embeddings of the molecules in training.

        param data: the training or test set
        """
        # Load the pre-trained model from gensim
        self.model = Word2Vec.load(self.model_path).wv

        # Extract the embedding of the unseen token
        self.unseen_vec = self.model.get_vector(self.unseen)
        self.keys = set(self.model.key_to_index)

        desc = "compute the embeddings of the molecule"

        #
        # Compute the embeddings for each molecule
        if self.molecule and data.molecule_smiles is not None:
            data.molecule_embedding = self.parallel_embeddings(data.molecule_smiles, desc)

        # Compute the embeddings for each block
        if self.building and data.bb1_smiles is not None and data.bb2_smiles is not None and data.bb3_smiles is not None:
            data.bb1_embedding = self.parallel_embeddings(data.bb1_smiles, desc)
            data.bb2_embedding = self.parallel_embeddings(data.bb2_smiles, desc)
            data.bb3_embedding = self.parallel_embeddings(data.bb3_smiles, desc)

        return data
