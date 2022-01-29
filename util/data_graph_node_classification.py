import logging
import os.path as osp
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WikiCS
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import Actor
from torch_geometric.datasets import GNNBenchmarkDataset
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch
import numpy as np
from scipy.stats import entropy
from train.metrics import weighted_f1_score
from util.data_graph_common import DgDataset, subgraph_wrapper, SingleGraphData


class NodeClassificationSingleData(DgDataset):
    """wrapper node classification dataset"""

    def __init__(self, dataset, already_double_edge=False, pre_manipulate_type="ogb", n_classes=-1):
        super().__init__()
        self.num_classes = n_classes
        self.dataset = dataset
        self.data = dataset

        self.double_edge = not already_double_edge
        if self.double_edge:
            edge_index = to_undirected(self.data.edge_index, num_nodes=self.data.num_nodes)
            self.data.edge_index = edge_index

        self.has_mask = hasattr(self.data, 'train_mask')
        self.get_randn_mask()

    def get_randn_mask(self):
        data = self.data
        if self.has_mask and data.train_mask.dim() > 1:  # like Actor, there are 10 splits, we pick only one
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.test_mask = data.test_mask[:, 0]

    def subgraph(self, observed_node, observed_edge):
        num_nodes = self.data.x.shape[0]

        x, y = self.data.x[observed_node], self.data.y[observed_node]
        if self.has_mask:
            train_mask = self.data.train_mask[observed_node]
            val_mask = self.data.val_mask[observed_node]
            test_mask = self.data.test_mask[observed_node]

        edge_index = subgraph_wrapper(observed_node, observed_edge=observed_edge,
                                      edge_index=self.data.edge_index, num_nodes=num_nodes)

        n_idy = torch.zeros(self.n_classes, dtype=y.dtype)
        observed_y = y.unique()
        n_idy[observed_y] = torch.arange(observed_y.size(0))
        y = n_idy[y]

        if self.has_mask:
            self.data = Data(x=x, y=y, edge_index=edge_index,
                             train_mask=train_mask,
                             val_mask=val_mask,
                             test_mask=test_mask)
        else:
            self.data = Data(x=x, y=y, edge_index=edge_index)

    @property
    def n_classes(self):
        if self.num_classes > 0:
            return self.num_classes
        else:
            return torch.unique(self.data.y).size(0)

    @property
    def num_features(self):
        return self.dataset.num_features

    def train_test_kl(self):
        if not self.has_mask:
            return -1
        value = torch.unique(self.data.y)
        value = value.cpu().detach().numpy()
        n_classes = value.shape[0]
        y1, y2 = self.data.y[self.data.train_mask], self.data.y[self.data.test_mask]
        y1 = y1.cpu().detach().numpy()
        y2 = y2.cpu().detach().numpy()
        value1, counts1 = np.unique(y1, return_counts=True)
        value2, counts2 = np.unique(y2, return_counts=True)
        y1, y2 = np.zeros(n_classes), np.zeros(n_classes)
        for i, v in enumerate(value1):
            y1[value == v] = counts1[i]
        for i, v in enumerate(value2):
            y2[value == v] = counts2[i]
        ent = entropy(y1, y2)
        return ent


N, V, E, F = 0, 0, 0, 0


class NodeClassificationGraphData(SingleGraphData):
    def __init__(self, dataset, params: dict):
        super().__init__(dataset, params)
        self.train_test_distance = None
        global V, E, N
        N += 1
        V += self.graph.number_of_nodes()
        E += self.graph.number_of_edges()

    def get_observed_graph(self, params):
        super().get_observed_graph(params)
        self.train_test_distance = self.dataset.train_test_kl()
        global F
        F = self.dataset.num_features


class NodeClassificationDataset(DgDataset):
    def __init__(self, dataset, already_double_edge=False, evaluator=None, pre_manipulate_type="ogb"):
        super().__init__()
        self.dataset = dataset
        self.data = None
        self.datas = []

        self.already_double_edge = already_double_edge

        self.pre_manipulate_type = pre_manipulate_type

        self.evaluator = evaluator

    def pre_manipulate(self, pre_manipulate_type):
        if pre_manipulate_type == 'ogb':
            self.data.y = self.data.y.flatten()
            split_idx = self.dataset.get_idx_split()
            mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            mask[split_idx['train']] = 1
            self.data.train_mask = mask
            mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            mask[split_idx['valid']] = 1
            self.data.val_mask = mask
            mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            mask[split_idx['test']] = 1
            self.data.test_mask = mask

    def subgraph(self, observed_node, observed_edge):
        raise NotImplementedError(f"{self.__class__.__name__} is a set of graphs, subgraph is not supported.")

    def get_observed_graph(self, params, keep_=1):
        for data in tqdm(self.dataset):
            if random.random() > keep_:
                continue
            self.data = data
            self.pre_manipulate(self.pre_manipulate_type)
            data = NodeClassificationSingleData(self.data, self.already_double_edge, self.pre_manipulate_type,
                                                self.n_classes)
            dg = NodeClassificationGraphData(data, params)
            dg.get_observed_graph(params)
            self.datas.append(dg.dataset.data)

    @property
    def num_features(self):
        return self.dataset.num_features

    @property
    def n_classes(self):
        return self.dataset.num_classes


class OgbEvaluator:
    def __init__(self, name):
        self.evaluator = Evaluator(name=name)

    def __call__(self, scores, targets):
        scores = scores.argmax(dim=-1, keepdim=True)
        targets = targets.unsqueeze(dim=1)
        return self.evaluator.eval({
            "y_true": targets,
            "y_pred": scores
        })['acc']


def get_data(name, split=None):
    doubled_edge = False
    evaluator = weighted_f1_score
    pre_manipulate_type = None
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
        dataset = Planetoid(path, name, pre_transform=T.NormalizeFeatures())
    elif name in ['WikiCS']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data', 'WikiCS')
        dataset = WikiCS(path, pre_transform=T.NormalizeFeatures())
    elif name in ['cornell', 'texas', 'wisconsin']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
        dataset = WebKB(path, name, pre_transform=T.NormalizeFeatures())
    elif name in ['Actor']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data', 'film')
        dataset = Actor(path, pre_transform=T.NormalizeFeatures())
    elif name in ['CLUSTER']:
        doubled_edge = True
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
        dataset = GNNBenchmarkDataset(
            root=path,
            name=name,
            split=split,
            pre_transform=T.NormalizeFeatures()
        )
    elif name in ['ogbn-arxiv']:
        doubled_edge = True
        evaluator = OgbEvaluator(name='ogbn-arxiv')
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
        dataset = PygNodePropPredDataset(name, path, pre_transform=T.NormalizeFeatures())
        pre_manipulate_type = "ogb"
    else:
        raise ValueError(f"{name} dataset is not supported.")
    return NodeClassificationDataset(dataset, doubled_edge, evaluator, pre_manipulate_type)


def load_data(params):
    DATASET_NAME = params['dataset']
    if DATASET_NAME in ['CLUSTER']:
        single_graph = False
    else:
        single_graph = True
    if single_graph:
        dataset = get_data(params["dataset"])
        dataset.get_observed_graph(params)
        return dataset,
    else:
        dgs_train = get_data(params["dataset"], split='train')
        dgs_train.get_observed_graph(params)
        dgs_val = get_data(params["dataset"], split='val')
        dgs_val.get_observed_graph(params)
        dgs_test = get_data(params["dataset"], split='test')
        dgs_test.get_observed_graph(params)
        return dgs_train, dgs_val, dgs_test



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, datefmt="%m-%dT%H:%M:%S", format="%(asctime)s %(message)s")
    import json

    fpath = "../config/node_classification_gcn.json"
    with open(fpath) as f:
        config = json.load(f)
    dg = load_data(config)
    if len(dg) == 1:
        print(dg[0].datas[0].train_mask.sum().item() / V)
        print(dg[0].datas[0].val_mask.sum().item() / V)
        print(dg[0].datas[0].test_mask.sum().item() / V)
    else:
        print(len(dg[0].datas))
        print(len(dg[1].datas))
        print(len(dg[2].datas))

    print(N, V, E, F)
