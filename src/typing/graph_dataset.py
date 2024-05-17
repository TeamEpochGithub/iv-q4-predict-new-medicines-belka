"""Dataset for graph representations."""
from typing import Optional, Generic, Tuple, TypeVar

import torch
from torch_geometric.data import Data, Dataset

T = TypeVar('T', bound=tuple[torch.Tensor, ...])

class GraphDataset(Dataset, Generic[T]):
    """Dataset class for graph representations."""

    def __init__(self, graphs: list, labels: Optional[torch.Tensor] = None, device: torch.device = None):
        super(GraphDataset, self).__init__()
        self.device = device
        self.graphs = [self.convert_to_data_object(graph) for graph in graphs]
        self.labels = labels.to(device) if labels is not None else None

    def convert_to_data_object(self, graph) -> Data:
        """ Converts a graph representation to a data object.

        @param: graph: graph representation
        @return: data object
        """
        node_features = graph[0]
        edge_index = graph[1]
        edge_features = graph[2]

        atom_features = torch.from_numpy(node_features).float().to(self.device)
        edge_index = torch.from_numpy(edge_index).long().to(self.device)
        edge_features = torch.from_numpy(edge_features).float().to(self.device)

        data = Data(x=atom_features, edge_index=edge_index.t().contiguous(), edge_attr=edge_features)
        return data

    def __len__(self) -> int:
        """ Returns the length of the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> T:
        """ Returns the item at the given index.

        @param: idx: index
        """
        graph = self.graphs[idx]
        if self.labels is not None:
            graph.y = self.labels[idx]
        return graph