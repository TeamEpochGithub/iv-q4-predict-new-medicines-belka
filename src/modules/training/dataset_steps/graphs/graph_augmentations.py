""" Graph Augmentations Class. """
from dataclasses import dataclass
from random import random
from typing import Tuple, List, Any

import numpy as np
from epochalyst.pipeline.model.training.training_block import TrainingBlock
import numpy.typing as npt
from rdkit import Chem
from rdkit.Chem import rdchem
import torch
from torch_geometric.data import Data


@dataclass
class GraphAugmentationBlock(TrainingBlock):
    """ Graph Augmentations Block. """

    p_node_drop: float = 0.1
    p_edge_drop: float = 0.1
    p_edge_mask: float = 0.1
    p_node_mask: float = 0.1

    def augment_graph(self, data: Data) -> Data:
        """Apply augmentations to the graph."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Node dropping
        num_nodes = x.size(0)
        node_mask = torch.rand(num_nodes) > self.p_node_drop
        kept_nodes = node_mask.nonzero(as_tuple=False).view(-1)
        x = x[kept_nodes]

        # Mapping from old nodes to new nodes
        node_map = -torch.ones(num_nodes, dtype=torch.long)
        node_map[kept_nodes] = torch.arange(kept_nodes.size(0))

        # Edge dropping
        edge_mask = torch.rand(edge_index.size(1)) > self.p_edge_drop
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

        # Remove edges connected to dropped nodes
        valid_edge_mask = (node_map[edge_index[0]] >= 0) & (node_map[edge_index[1]] >= 0)
        edge_index = edge_index[:, valid_edge_mask]
        edge_attr = edge_attr[valid_edge_mask] if edge_attr is not None else None

        # Remap edge indices
        edge_index = node_map[edge_index]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = data.y)

    def apply_mask(self, data: Data) -> Data:
        """ Applies mask to the nodes and edge attributes"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Node attribute masking
        if self.p_node_mask > 0:
            mask = torch.rand(x.size(0), x.size(1)) < self.p_node_mask
            x = x * ~mask

        # Edge attribute masking
        if edge_attr is not None and self.p_edge_mask > 0:
            mask = torch.rand(edge_attr.size(0), edge_attr.size(1)) < self.p_edge_mask
            edge_attr = edge_attr * ~mask

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=data.y)

    def train(
        self,
        x: List[Data],
        y: npt.NDArray[np.uint8] | None = None,
        **train_args: Any,
    ) -> Tuple[List[Data], None]:
        """Apply augmentations and return augmented Data objects.

        :param x: The x molecule data in Data format
        :param y: The binding data
        :return: List of augmented molecule graphs and labels
        """
        augmented_graphs = [self.augment_graph(data) for data in x]
        masked_graphs = [self.apply_mask(data) for data in augmented_graphs]
        return masked_graphs, y

    @property
    def is_augmentation(self) -> bool:
        """Return whether the block is an augmentation."""
        return True