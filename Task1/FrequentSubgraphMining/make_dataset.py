from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

def make_dataset():
    dataset_name = 'IMDB-MULTI'
    dataset = TUDataset(root=f'{dataset_name}', name=dataset_name)
    graphs = [to_networkx(data=graph, to_undirected=True) for graph in dataset]
    return [list(graph.edges) for graph in graphs]