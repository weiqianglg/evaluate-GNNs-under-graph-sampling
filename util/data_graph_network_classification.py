import os.path as osp
import random

from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset
from torch_geometric.datasets import GNNBenchmarkDataset
from util.data_graph_common import DgDataset, subgraph_wrapper, SingleGraphData
from train.metrics import accuracy_MNIST_CIFAR


class NetworkClassificationSingleData(DgDataset):
    def __init__(self, dataset, already_double_edge=False, pre_manipulate_type="ogb"):
        super().__init__()
        self.dataset = dataset
        self.data = dataset

        self.double_edge = not already_double_edge
        if self.double_edge:
            edge_index = to_undirected(self.data.edge_index, num_nodes=self.data.num_nodes)
            self.data.edge_index = edge_index

        self.pre_manipulate(pre_manipulate_type)

    def pre_manipulate(self, pre_manipulate_type):
        if pre_manipulate_type == 'ogb':
            raise NotImplementedError

    def subgraph(self, observed_node, observed_edge):
        num_nodes = self.data.x.shape[0]
        x = self.data.x[observed_node]
        y = self.data.y
        edge_index = subgraph_wrapper(observed_node, observed_edge=observed_edge,
                                      edge_index=self.data.edge_index, num_nodes=num_nodes)
        self.data = Data(x=x, edge_index=edge_index, y=y)


N, V, E, F = 0, 0, 0, 0


class NetworkClassificationSingleGraphData(SingleGraphData):
    def __init__(self, dataset, params: dict):
        super().__init__(dataset, params)
        global V, E, N
        N += 1
        V += self.graph.number_of_nodes()
        E += self.graph.number_of_edges()

    def get_observed_graph(self, params):
        super().get_observed_graph(params)
        global F
        F = self.dataset.num_features


class NetworkClassificationDataset(DgDataset):
    def __init__(self, dataset, already_double_edge=False, evaluator=None, pre_manipulate_type="ogb"):
        super().__init__()
        self.dataset = dataset
        self.datas = []

        self.already_double_edge = already_double_edge

        self.pre_manipulate_type = pre_manipulate_type
        self.evaluator = evaluator

    def subgraph(self, observed_node, observed_edge):
        raise NotImplementedError(f"{self.__class__.__name__} is a set of graphs, subgraph is not supported.")

    def get_observed_graph(self, params, keep_=0.001):
        for data in tqdm(self.dataset):
            if random.random() > keep_:
                continue
            data = NetworkClassificationSingleData(data, self.already_double_edge, self.pre_manipulate_type)
            dg = NetworkClassificationSingleGraphData(data, params)
            dg.get_observed_graph(params)
            self.datas.append(dg.dataset.data)

    @property
    def num_features(self):
        return self.dataset.num_features

    @property
    def n_classes(self):
        return self.dataset.num_classes


def get_data(name, split='train'):
    evaluator = accuracy_MNIST_CIFAR
    pre_manipulate_dataset = ""
    if name in ['MNIST', 'CIFAR10']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
        dataset = GNNBenchmarkDataset(
            root=path,
            name=name,
            split=split,
            pre_transform=T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
        )
        return NetworkClassificationDataset(dataset, already_double_edge=True, evaluator=evaluator,
                                            pre_manipulate_type=pre_manipulate_dataset)
    else:
        raise ValueError(f"{name} dataset is not supported.")


def load_data(params):
    dgs_train = get_data(params["dataset"], split='train')
    dgs_train.get_observed_graph(params)
    dgs_val = get_data(params["dataset"], split='val')
    dgs_val.get_observed_graph(params)
    dgs_test = get_data(params["dataset"], split='test')
    dgs_test.get_observed_graph(params)
    return dgs_train, dgs_val, dgs_test


if __name__ == '__main__':
    import json

    fpath = "../config/superpixels_graph_classification_GCN_MNIST_100k.json"
    with open(fpath) as f:
        config = json.load(f)
    print(N, V, E, F)
    load_data(config)
    print(N, V, E, F)
