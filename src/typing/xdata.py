"""Class to describe the data format."""
import gc
from dataclasses import dataclass
from enum import IntFlag
from typing import Any
from typing import Callable
import numpy as np
import numpy.typing as npt

# from rdkit.DataStructs.cDataStructs import ExplicitBitVect  # type: ignore[import-not-found]


class DataRetrieval(IntFlag):
    """Class to select which data to retrieve in XData.

    Possible to select multiple datapoints at once by 'or'ing them together with the '|' operator.
    """

    SMILES_MOL = 2**0
    SMILES_BB1 = 2**1
    SMILES_BB2 = 2**2
    SMILES_BB3 = 2**3
    SMILES = SMILES_MOL | SMILES_BB1 | SMILES_BB2 | SMILES_BB3
    SMILES_BB = SMILES_BB1 | SMILES_BB2 | SMILES_BB3
    ECFP_MOL = 2**4
    ECFP_BB1 = 2**5
    ECFP_BB2 = 2**6
    ECFP_BB3 = 2**7
    ECFP = ECFP_MOL | ECFP_BB1 | ECFP_BB2 | ECFP_BB3
    ECFP_BB = ECFP_BB1 | ECFP_BB2 | ECFP_BB3
    EMBEDDING_MOL = 2**8
    EMBEDDING_BB1 = 2**9
    EMBEDDING_BB2 = 2**10
    EMBEDDING_BB3 = 2**11
    EMBEDDING = EMBEDDING_MOL | EMBEDDING_BB1 | EMBEDDING_BB2 | EMBEDDING_BB3
    EMBEDDING_BB = EMBEDDING_BB1 | EMBEDDING_BB2 | EMBEDDING_BB3
    DESCRIPTORS_MOL = 2**12
    DESCRIPTORS_BB1 = 2**13
    DESCRIPTORS_BB2 = 2**14
    DESCRIPTORS_BB3 = 2**15
    DESCRIPTORS = DESCRIPTORS_MOL | DESCRIPTORS_BB1 | DESCRIPTORS_BB2 | DESCRIPTORS_BB3
    DESCRIPTORS_BB = DESCRIPTORS_BB1 | DESCRIPTORS_BB2 | DESCRIPTORS_BB3
    GRAPHS_MOL = 2**16
    GRAPHS_BB1 = 2**17
    GRAPHS_BB2 = 2**18
    GRAPHS_BB3 = 2**19
    GRAPHS = GRAPHS_MOL | GRAPHS_BB1 | GRAPHS_BB2 | GRAPHS_BB3
    GRAPHS_BB = GRAPHS_BB1 | GRAPHS_BB2 | GRAPHS_BB3


@dataclass
class XData:
    """Class to describe data format of X.

    :param building_blocks: Building blocks encoded

    :param molecule_smiles: Molecule smiles
    :param bb1_smiles: Building_block 1 smiles
    :param bb2_smiles: Building_block 2 smiles
    :param bb3_smiles: Building_block 3 smiles

    :param molecule_ecfp: ECFP for molecules
    :param bb1_ecfp: ECFP for building_block 1 smiles
    :param bb2_ecfp: ECFP for building_block 2 smiles
    :param bb3_ecfp: ECFP for building_block 3 smiles

    :param molecule_embedding: Embedding for molecules
    :param bb1_embedding: Embedding for building_block 1 smiles
    :param bb2_embedding: Embedding for building_block 2 smiles
    :param bb3_embedding: Embedding for building_block 3 smiles

    :param molecule_desc: Descriptors for molecules
    :param bb1_desc: Descriptors for building_block 1 smiles
    :param bb2_desc: Descriptors for building_block 2 smiles
    :param bb3_desc: Descriptors for building_block 3 smiles
    """

    building_blocks: npt.NDArray[np.int16]
    retrieval: DataRetrieval = DataRetrieval.SMILES

    # SMILES
    molecule_smiles: npt.NDArray[np.str_] | None = None
    bb1_smiles: npt.NDArray[np.str_] | None = None
    bb2_smiles: npt.NDArray[np.str_] | None = None
    bb3_smiles: npt.NDArray[np.str_] | None = None

    # ECFP
    molecule_ecfp: npt.NDArray[np.uint8] | None = None
    bb1_ecfp: npt.NDArray[np.uint8] | None = None
    bb2_ecfp: npt.NDArray[np.uint8] | None = None
    bb3_ecfp: npt.NDArray[np.uint8] | None = None

    # Embedding
    molecule_embedding: npt.NDArray[np.float32] | None = None
    bb1_embedding: npt.NDArray[np.float32] | None = None
    bb2_embedding: npt.NDArray[np.float32] | None = None
    bb3_embedding: npt.NDArray[np.float32] | None = None

    # Descriptors
    molecule_desc: npt.NDArray[np.float32] | None = None
    bb1_desc: npt.NDArray[np.float32] | None = None
    bb2_desc: npt.NDArray[np.float32] | None = None
    bb3_desc: npt.NDArray[np.float32] | None = None

    # Graph
    molecule_graph: list[Any] | None = None
    bb1_graph: list[Any] | None = None
    bb2_graph: list[Any] | None = None
    bb3_graph: list[Any] | None = None

    # tokenizer
    tokenizer: Callable[[str], npt.NDArray[np.float32]] | None = None

    def slice_all(self, slice_array: npt.NDArray[np.int_]) -> None:
        """Slice all existing arrays in x data by numpy array.

        :param slice_array: Array to slice by
        """
        if self.building_blocks is not None:
            self.building_blocks = self.building_blocks[slice_array]
        if self.molecule_smiles is not None:
            self.molecule_smiles = self.molecule_smiles[slice_array]
        if self.molecule_ecfp is not None:
            self.molecule_ecfp = self.molecule_ecfp[slice_array]
        if self.molecule_embedding is not None:
            self.molecule_embedding = self.molecule_embedding[slice_array]
        if self.molecule_desc is not None:
            self.molecule_desc = self.molecule_desc[slice_array]

        gc.collect()

    def __getitem__(self, idx: int | npt.NDArray[np.int_] | list[int] | slice) -> npt.NDArray[Any] | list[Any]:  # noqa: C901 PLR0912 PLR0915
        """Get item from the data.

        :param index: Index to retrieve
        :return: Data replaced with correct building_blocks
        """
        if not isinstance(idx, np.integer):
            return self._getitems(idx)  # type: ignore[arg-type]

        result = []
        item = self.building_blocks[idx]

        # SMILES
        if self.retrieval & DataRetrieval.SMILES_MOL:
            if self.molecule_smiles is None:
                raise ValueError("No SMILES data available.")
            result.append(self.molecule_smiles[idx])
        if self.retrieval & DataRetrieval.SMILES_BB1:
            if self.bb1_smiles is None:
                raise ValueError("No SMILES data available.")
            result.append(self.bb1_smiles[item[0]])
        if self.retrieval & DataRetrieval.SMILES_BB2:
            if self.bb2_smiles is None:
                raise ValueError("No SMILES data available.")
            result.append(self.bb2_smiles[item[1]])
        if self.retrieval & DataRetrieval.SMILES_BB3:
            if self.bb3_smiles is None:
                raise ValueError("No SMILES data available.")
            result.append(self.bb3_smiles[item[2]])

        # ECFP
        if self.retrieval & DataRetrieval.ECFP_MOL:
            if self.molecule_ecfp is None:
                raise ValueError("No ECFP data available.")
            result.append(self.molecule_ecfp[idx])
        if self.retrieval & DataRetrieval.ECFP_BB1:
            if self.bb1_ecfp is None:
                raise ValueError("No ECFP data available.")
            result.append(self.bb1_ecfp[item[0]])
        if self.retrieval & DataRetrieval.ECFP_BB2:
            if self.bb2_ecfp is None:
                raise ValueError("No ECFP data available.")
            result.append(self.bb2_ecfp[item[1]])
        if self.retrieval & DataRetrieval.ECFP_BB3:
            if self.bb3_ecfp is None:
                raise ValueError("No ECFP data available.")
            result.append(self.bb3_ecfp[item[2]])

        # EMBEDDINGS
        if self.retrieval & DataRetrieval.EMBEDDING_MOL:
            if self.molecule_embedding is None:
                raise ValueError("No embedding data available.")
            result.append(self.molecule_embedding[idx])
        if self.retrieval & DataRetrieval.EMBEDDING_BB1:
            if self.bb1_embedding is None:
                raise ValueError("No embedding data available.")
            result.append(self.bb1_embedding[item[0]])
        if self.retrieval & DataRetrieval.EMBEDDING_BB2:
            if self.bb2_embedding is None:
                raise ValueError("No embedding data available.")
            result.append(self.bb2_embedding[item[1]])
        if self.retrieval & DataRetrieval.EMBEDDING_BB3:
            if self.bb3_embedding is None:
                raise ValueError("No embedding data available.")
            result.append(self.bb3_embedding[item[2]])

        # DESCRIPTORS
        if self.retrieval & DataRetrieval.DESCRIPTORS_MOL:
            if self.molecule_desc is None:
                raise ValueError("No descriptor data available.")
            result.append(self.molecule_desc[idx])
        if self.retrieval & DataRetrieval.DESCRIPTORS_BB1:
            if self.bb1_desc is None:
                raise ValueError("No descriptor data available.")
            result.append(self.bb1_desc[item[0]])
        if self.retrieval & DataRetrieval.DESCRIPTORS_BB2:
            if self.bb2_desc is None:
                raise ValueError("No descriptor data available.")
            result.append(self.bb2_desc[item[1]])
        if self.retrieval & DataRetrieval.DESCRIPTORS_BB3:
            if self.bb3_desc is None:
                raise ValueError("No descriptor data available.")
            result.append(self.bb3_desc[item[2]])

        # GRAPHS
        if self.retrieval & DataRetrieval.GRAPHS_MOL:
            if self.molecule_graph is None:
                raise ValueError("No graph data available.")
            result.append(self.molecule_graph[idx])
        if self.retrieval & DataRetrieval.GRAPHS_BB1:
            if self.bb1_graph is None:
                raise ValueError("No graph data available.")
            result.append(self.bb1_graph[item[0]])
        if self.retrieval & DataRetrieval.GRAPHS_BB2:
            if self.bb2_graph is None:
                raise ValueError("No graph data available.")
            result.append(self.bb2_graph[item[1]])
        if self.retrieval & DataRetrieval.GRAPHS_BB3:
            if self.bb3_graph is None:
                raise ValueError("No graph data available.")
            result.append(self.bb3_graph[item[2]])

        if len(result) == 1:
            return result[0]
        return result

    def _getitems(self, indices: npt.NDArray[np.int_] | list[int] | slice) -> npt.NDArray[Any]:
        """Retrieve items for all indices based on the specified retrieval flags.

        :param indices: List of indices to retrieve
        """
        if self.retrieval == DataRetrieval.SMILES_MOL:
            if self.molecule_smiles is None:
                raise ValueError("No SMILES data available.")
            return self.molecule_smiles[indices]

        if self.retrieval == DataRetrieval.ECFP_MOL:
            if self.molecule_ecfp is None:
                raise ValueError("No ECFP data available.")
            return self.molecule_ecfp[indices]

        if self.retrieval == DataRetrieval.EMBEDDING_MOL:
            if self.molecule_embedding is None:
                raise ValueError("No embedding data available.")
            return self.molecule_embedding[indices]

        if self.retrieval == DataRetrieval.DESCRIPTORS_MOL:
            if self.molecule_desc is None:
                raise ValueError("No descriptor data available.")
            return self.molecule_desc[indices]

        if isinstance(indices, slice):
            indices_new = range(
                indices.start if indices.start is not None else 0,
                indices.stop if indices.stop is not None else len(self),
                indices.step if indices.step is not None else 1,
            )
            return np.array([self[i] for i in indices_new])
        return np.array([self[i] for i in indices])

    def __len__(self) -> int:
        """Return the length of the data.

        :return: Length of data
        """
        return len(self.building_blocks)

    def __repr__(self) -> str:
        """Return a string representation of the data.

        :return: String representation of the data
        """
        return f"XData with {len(self.building_blocks)} entries"


def slice_copy(xdata: XData, slice_array: npt.NDArray[np.int_]) -> XData:
    """Make a copy of xdata with sliced versions of the lists.

    :param slice_array: The array with indices to retrieve
    :param return: The sliced xdata class
    """
    return XData(
        building_blocks=xdata.building_blocks[slice_array],
        retrieval=xdata.retrieval,
        # SMILES
        molecule_smiles=xdata.molecule_smiles[slice_array] if xdata.molecule_smiles is not None else None,
        bb1_smiles=xdata.bb1_smiles,
        bb2_smiles=xdata.bb2_smiles,
        bb3_smiles=xdata.bb3_smiles,
        # ECFP
        molecule_ecfp=xdata.molecule_ecfp[slice_array] if xdata.molecule_ecfp is not None else None,
        bb1_ecfp=xdata.bb1_ecfp,
        bb2_ecfp=xdata.bb2_ecfp,
        bb3_ecfp=xdata.bb3_ecfp,
        # Embedding
        molecule_embedding=xdata.molecule_embedding[slice_array] if xdata.molecule_embedding is not None else None,
        bb1_embedding=xdata.bb1_embedding,
        bb2_embedding=xdata.bb2_embedding,
        bb3_embedding=xdata.bb3_embedding,
        # Descriptors
        molecule_desc=xdata.molecule_desc[slice_array] if xdata.molecule_desc is not None else None,
        bb1_desc=xdata.bb1_desc,
        bb2_desc=xdata.bb2_desc,
        bb3_desc=xdata.bb3_desc,
    )
