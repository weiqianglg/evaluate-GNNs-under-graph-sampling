import logging
import os.path as osp
import random

import networkx as nx
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Actor
from train.metrics import link_auc_score

from torch_geometric.transforms import RandomLinkSplit
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from util.data_graph_common import DgDataset, subgraph_wrapper, SingleGraphData


class EdgeClassificationDataset(DgDataset):
    def __init__(self, dataset, already_double_edge=False, evaluator=None, pre_manipulate_type="ogb"):
        super().__init__()
        self.dataset = dataset
        self.data = self.dataset[0]  # single graph

        self.double_edge = not already_double_edge

        if self.double_edge:
            edge_index = to_undirected(self.data.edge_index, num_nodes=self.data.num_nodes)
            self.data.edge_index = edge_index

        self.train_edges = None
        self.val_edges = None
        self.val_edges_neg = None
        self.test_edges = None
        self.test_edges_neg = None
        self.pre_manipulate(pre_manipulate_type)
        self.evaluator = evaluator

    def pre_manipulate(self, pre_manipulate_type):
        # if pre_manipulate_type == 'ogb':
        #     split_edge = self.dataset.get_edge_split()
        #     self.train_edges = split_edge['train']['edge'].t()  # positive train edges
        #     self.val_edges = split_edge['valid']['edge'].t()  # positive val edges
        #     self.val_edges_neg = split_edge['valid']['edge_neg'].t()  # negative val edges
        #     self.test_edges = split_edge['test']['edge'].t()  # positive test edges
        #     self.test_edges_neg = split_edge['test']['edge_neg'].t()  # negative test edges
        split_edges = RandomLinkSplit(is_undirected=True, split_labels=True)
        train_data, val_data, test_data = split_edges(self.data)
        self.train_edges = train_data.pos_edge_label_index
        self.val_edges = val_data.pos_edge_label_index
        self.val_edges_neg = val_data.neg_edge_label_index
        self.test_edges = test_data.pos_edge_label_index
        self.test_edges_neg = test_data.neg_edge_label_index

    def subgraph(self, observed_node, observed_edge):
        num_nodes = self.data.x.shape[0]
        x = self.data.x[observed_node]
        edge_index = subgraph_wrapper(observed_node, observed_edge=observed_edge,
                                      edge_index=self.data.edge_index, num_nodes=num_nodes)

        self.train_edges = subgraph_wrapper(observed_node, observed_edge=observed_edge,
                                            edge_index=self.train_edges, num_nodes=num_nodes,
                                            observed_edge_in_graph=False)
        self.val_edges = subgraph_wrapper(observed_node, observed_edge=observed_edge,
                                          edge_index=self.val_edges, num_nodes=num_nodes,
                                          observed_edge_in_graph=False)
        self.val_edges_neg = subgraph_wrapper(observed_node, observed_edge=None,
                                              edge_index=self.val_edges_neg, num_nodes=num_nodes,
                                              observed_edge_in_graph=False)
        self.test_edges = subgraph_wrapper(observed_node, observed_edge=observed_edge,
                                           edge_index=self.test_edges, num_nodes=num_nodes,
                                           observed_edge_in_graph=False)
        self.test_edges_neg = subgraph_wrapper(observed_node, observed_edge=None,
                                               edge_index=self.test_edges_neg, num_nodes=num_nodes,
                                               observed_edge_in_graph=False)

        self.data = Data(x=x, edge_index=edge_index)

    def t(self):
        """change train_edges from 2xn to nx2, if neg edges is less than pos edges, randomly add some neg edges"""
        logging.debug(f"subgraph {self.data}, train edges {self.train_edges.size(1)}, "
                      f"val edges {self.val_edges.size(1)}, val neg edges {self.val_edges_neg.size(1)},"
                      f"test edges {self.test_edges.size(1)}, test neg edges {self.test_edges_neg.size(1)}.")

        num_nodes = self.data.x.shape[0]
        if self.val_edges_neg.size(1) < self.val_edges.size(1):
            extra_edges_neg = negative_sampling(self.val_edges, num_nodes=num_nodes)
            self.val_edges_neg = torch.cat([self.val_edges_neg, extra_edges_neg], dim=1)
        if self.test_edges_neg.size(1) < self.test_edges.size(1):
            extra_edges_neg = negative_sampling(self.test_edges, num_nodes=num_nodes)
            self.test_edges_neg = torch.cat([self.test_edges_neg, extra_edges_neg], dim=1)

        self.train_edges.t_()
        self.val_edges.t_()
        self.val_edges_neg.t_()
        self.test_edges.t_()
        self.test_edges_neg.t_()

        logging.debug(f"subgraph added extra val and test neg edges, "
                      f"val edges {len(self.val_edges)}, val neg edges {len(self.val_edges_neg)},"
                      f"test edges {len(self.test_edges)}, test neg edges {len(self.test_edges_neg)}.")

    @property
    def num_features(self):
        return self.dataset.data.num_features

    @property
    def n_classes(self):
        return -1


class OgbEvaluator:
    def __init__(self, name):
        self.evaluator = Evaluator(name=name)

    def __call__(self, scores, targets):
        K = self.evaluator.K
        return self.evaluator.eval({
            'y_pred_pos': scores,
            'y_pred_neg': targets,
        })[f'hits@{K}']


def get_data(name):
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
        dataset = Planetoid(path, name, pre_transform=T.NormalizeFeatures())
        return EdgeClassificationDataset(dataset, already_double_edge=False, evaluator=link_auc_score)
    elif name in ['Actor']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data', 'film')
        dataset = Actor(path, pre_transform=T.NormalizeFeatures())
        return EdgeClassificationDataset(dataset, already_double_edge=False, evaluator=link_auc_score)
    elif name in ['ogbl-collab', 'OGBL-COLLAB']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
        dataset = PygLinkPropPredDataset('ogbl-collab', path, pre_transform=T.NormalizeFeatures())
        return EdgeClassificationDataset(dataset, already_double_edge=True, evaluator=OgbEvaluator(name='ogbl-collab'))
    elif name in ['ogbl-ppa', 'OGBL-PPA']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data')
        dataset = PygLinkPropPredDataset('ogbl-ppa', path, pre_transform=T.ToUndirected())
        dataset.data.x = dataset.data.x.float()
        return EdgeClassificationDataset(dataset, already_double_edge=True, evaluator=OgbEvaluator(name='ogbl-ppa'))
    else:
        raise ValueError(f"{name} dataset is not supported.")


class EdgeClassificationSingleGraphData(SingleGraphData):
    def __init__(self, dataset, params: dict):
        super().__init__(dataset, params)

    def get_observed_graph(self, params):
        super().get_observed_graph(params)
        self.dataset.t()


def load_data(params):
    dataset = get_data(params["dataset"])
    dg = EdgeClassificationSingleGraphData(dataset, params)
    dg.get_observed_graph(params)
    return dg


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, datefmt="%m-%dT%H:%M:%S",
                        format="%(asctime)s %(message)s")
    import json

    fpath = "../config/COLLAB_edge_classification_GAT_40k.json"
    with open(fpath) as f:
        config = json.load(f)
    load_data(config)
