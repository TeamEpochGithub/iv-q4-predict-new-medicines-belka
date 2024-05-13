""" Dataset for graph representations."""

from torch_geometric.data import Dataset, Data
import torch

class GraphDataset(Dataset):
    """Dataset class for graph representations."""
    def __init__(self, graphs, labels):
        """ Initializes the Dataset.

        @param: graphs: List of graphs
        @param: labels: List of labels
        """
        super(GraphDataset, self).__init__()
        self.graphs = [self.convert_to_data_object(graph) for graph in graphs]
        self.labels = labels

    def convert_to_data_object(self, graph):
        """ Converts each graph to a Data object.

        @param: graph: Graph array representation.
        """
        node_features = graph[0]
        edge_index = graph[1]
        edge_features = graph[2]

        atom_features = torch.from_numpy(node_features).float()
        edge_index = torch.from_numpy(edge_index).long()
        edge_features = torch.from_numpy(edge_features).float()

        data = Data(x=atom_features,
                    edge_index=edge_index.t().contiguous(),
                    edge_attr=edge_features)
        return data

    def len(self):
        """ Returns the size of the dataset. """
        return len(self.graphs)

    def get(self, idx):
        """ Returns the data at the given index. """
        graph = self.graphs[idx]
        graph.y = self.labels[idx]
        return graph
