"""Dataset for graph representations."""
from typing import Optional, Generic, Tuple, TypeVar

import torch
from torch_geometric.data import Data, Dataset

T = TypeVar('T', bound=tuple[torch.Tensor, ...])

class GraphDataset(Dataset, Generic[T]):
    """Dataset class for graph representations."""

    def __init__(self, graphs: list, labels: Optional[torch.Tensor] = None):
        super(GraphDataset, self).__init__()
        self.graphs = [self.convert_to_data_object(graph) for graph in graphs]
        self.labels = labels

    def convert_to_data_object(self, graph) -> Data:
        node_features = graph[0]
        edge_index = graph[1]
        edge_features = graph[2]

        atom_features = torch.from_numpy(node_features).float()
        edge_index = torch.from_numpy(edge_index).long()
        edge_features = torch.from_numpy(edge_features).float()

        data = Data(x=atom_features, edge_index=edge_index.t().contiguous(), edge_attr=edge_features)
        return data

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> T:
        graph = self.graphs[idx]
        if self.labels is not None:
            graph.y = self.labels[idx]
        return graph