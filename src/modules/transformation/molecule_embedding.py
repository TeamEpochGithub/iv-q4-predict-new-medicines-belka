""" Transforms the sequence of embeddings into a single embedding."""
from src.typing.xdata import XData
import numpy.typing as npt
import numpy as np
from dataclasses import dataclass
from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
@dataclass
class MoleculeEmbedding(VerboseTransformationBlock):
    """ Transforms the sequence of embeddings into a single embedding.
    param transform: the type of transformation (concat or mean)
    param n_concat: the number of arrays to be concatenated"""

    n_concat: int = 5
    name_transform: str = "concat"

    def concat_embedding(self, blocks: list[npt.NDArray[np.float32]]) -> list[npt.NDArray[np.float32]]:
        """Transforms concatenate the embeddings in each building block.

        param blocks: list containing the embeddings of each block
        return: list containing a single embedding for each block"""

        embeddings = []
        for block in blocks:
            # compute the index of the sections in the sequence
            split = [i * len(block) // self.n_concat for i in range(1, self.n_concat)]

            # divide the sequence into sections and compute the mean
            sections = np.split(block, split)
            arrays = [part.mean(axis=0) for part in sections]

            # concatenate the embeddings into one single embedding
            embeddings.append(np.concatenate(arrays))

        return embeddings

    def custom_transform(self, data: XData) -> XData:

        if data.bb1_embedding is None or data.bb2_embedding is None or data.bb3_embedding is None:
            raise ValueError("Missing embedding representation of the building block")

        if self.name_transform == "concat":
            data.bb1_embedding = self.concat_embedding(data.bb1_embedding)
            data.bb2_embedding = self.concat_embedding(data.bb2_embedding)
            data.bb3_embedding = self.concat_embedding(data.bb3_embedding)

            data.retrieval = 'Embedding'
            data.molecule_embedding = [data[i].flatten() for i in range(len(data.building_blocks))]

            aa = 1


        return data

