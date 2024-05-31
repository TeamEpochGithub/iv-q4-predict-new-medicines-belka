"""Transforms the sequence of embeddings into a single embedding."""
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.xdata import DataRetrieval, XData


@dataclass
class MoleculeEmbedding(VerboseTransformationBlock):
    """Transform the sequence of embeddings into a single embedding.

    param transform: the type of transformation (concat or mean)
    param n_concat: the number of arrays to be concatenated
    """

    n_concat: int = 5
    name_transform: str = "concat"

    def concat_embedding(self, blocks: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Transform concatenate the embeddings in each building block.

        param blocks: list containing the embeddings of each block
        return: list containing a single embedding for each block
        """
        embeddings = []
        for block in blocks:
            # compute the index of the sections in the sequence
            split = [i * len(block) // self.n_concat for i in range(1, self.n_concat)]

            # divide the sequence into sections and compute the mean
            sections = np.split(block, split)
            arrays = [part.mean(axis=0) for part in sections]

            # concatenate the embeddings into one single embedding
            embeddings.append(np.concatenate(arrays))

        return np.array(embeddings)

    def custom_transform(self, X: XData) -> XData:
        """Transform the sequence of embeddings into a single embedding."""
        if X.bb1_embedding is None or X.bb2_embedding is None or X.bb3_embedding is None:
            raise ValueError("Missing embedding representation of the building block")

        if self.name_transform == "concat":
            X.bb1_embedding = self.concat_embedding(X.bb1_embedding)
            X.bb2_embedding = self.concat_embedding(X.bb2_embedding)
            X.bb3_embedding = self.concat_embedding(X.bb3_embedding)

            X.retrieval = DataRetrieval.EMBEDDING
            X.molecule_embedding = np.array([X[i].flatten() for i in range(len(X))])  # type: ignore[union-attr]

        return X
