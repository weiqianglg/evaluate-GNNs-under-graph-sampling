from abc import ABC
import logging
import random
import os.path as osp
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_networkx
import pandas as pd
from itertools import product
import networkx as nx
import littleballoffur


def subgraph_wrapper(observed_node, *, observed_edge, edge_index, num_nodes, observed_edge_in_graph=True):
    """get observed_edge from edge_index and relabel them.
    if observed_edge is None, get induced subgraph from observed_node.
    edge_index is the whole graph or the part graph (for example, train_edges in link prediction task).
    edge_index SHOULD be undirected, that is if (u,v) in edge_index, (v,u) must in too.
    num_nodes is the whole graph's nodes number.
    observed_edge_in_graph indicates that edge_index IS the whole graph."""
    if observed_edge is None and len(observed_node) == num_nodes:  # return the raw edge_index
        return edge_index
    elif observed_edge is None:
        n_mask = torch.zeros(num_nodes, dtype=torch.bool)
        n_mask[observed_node] = 1
        mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
        observed_edge = edge_index[:, mask]
    elif observed_edge_in_graph:
        observed_edge = torch.tensor(observed_edge, dtype=torch.long).t_()
    else:
        # the following code is slow
        # e_mask = torch.zeros(edge_index.shape[-1], dtype=torch.bool)
        # for u, v in observed_edge:
        #     e_mask = e_mask | ((edge_index[0] == u) & (edge_index[1] == v))
        # observed_edge = edge_index[:, e_mask]
        edge_index_set = set(map(tuple, edge_index.t().tolist()))
        observed_edge = set(observed_edge) & edge_index_set
        observed_edge = torch.tensor(list(observed_edge), dtype=torch.long).t_()

    observed_edge_index = to_undirected(observed_edge, num_nodes=len(observed_node))

    n_idx = torch.zeros(num_nodes, dtype=torch.long)
    n_idx[observed_node] = torch.arange(len(observed_node))

    observed_edge_index = n_idx[observed_edge_index]
    return observed_edge_index


class DgDataset(Dataset, ABC):
    def __init__(self):
        self.data = None

    def subgraph(self, observed_node, observed_edge):
        pass

    @property
    def n_classes(self):
        return -1

    @property
    def num_features(self):
        return -1


def get_graph_topology(dataset, only_largest_cc, store_graphml, graph_name, cache=True):
    pkl_file = osp.join(osp.dirname(osp.realpath(__file__)), f"{graph_name}_{only_largest_cc}.pkl")
    if cache and osp.exists(pkl_file):
        with open(pkl_file, "rb") as f:
            _nodes, _g = pickle.load(f)
        logging.debug(f"load from {pkl_file}.")
        if _nodes and dataset:
            dataset.subgraph(_nodes, observed_edge=None)
        return _g
    else:
        _nodes = None
        g = to_networkx(dataset.data, to_undirected=True)
        logging.debug(
            f"G({g.number_of_nodes()}, {g.number_of_edges()}, "
            f"{dataset.num_features:d}) from {graph_name}.")
        if only_largest_cc:
            largest_cc = list(max(nx.connected_components(g), key=len))
            dataset.subgraph(largest_cc, observed_edge=None)  # littleballoffur need consecutive numeric index
            g = to_networkx(dataset.data, to_undirected=True)
            logging.debug(
                f"G'({g.number_of_nodes()}, {g.number_of_edges()}, "
                f"{dataset.num_features:d}) from the largest connected component.")
            _nodes = largest_cc
        if store_graphml:
            nx.write_graphml(g, osp.join(store_graphml, f'{graph_name}.graphml'))
        if cache:
            with open(pkl_file, "wb") as f:
                pickle.dump((_nodes, g), f)
            logging.debug(f"dump to {pkl_file}.")
    return g


def get_sampled_graph(graph, sample_method, percent_of_nodes, seed, subgraph):
    nodes_number = graph.number_of_nodes()
    if percent_of_nodes == 1.0:  # the whole graph, no need to sample
        return graph.copy()
    number_of_nodes = nodes_number * percent_of_nodes
    sample_method = getattr(littleballoffur, sample_method)
    sample_method = sample_method(number_of_nodes, seed=seed, subgraph=subgraph)  # keep the rest default
    start_node = random.choice(list(graph.nodes))
    logging.debug(f"start sampling from {start_node}.")
    return sample_method.sample(graph, start_node)


class SingleGraphData(object):
    def __init__(self, dataset, params: dict):
        self.dataset = dataset
        self.graph = get_graph_topology(self.dataset, params["largest_cc"], params["store_graphml"], params["dataset"],
                                        params["cache"])
        self.nodes_number = self.graph.number_of_nodes()
        self.observed_graph = None
        self.train_test_distance = None

    def get_observed_graph(self, params):
        self.observed_graph = get_sampled_graph(self.graph,
                                                params['sample']["sample_method"], params['sample']['percent_of_nodes'],
                                                params['sample']["seed"], params['sample']["subgraph"])

        self.dataset.subgraph(list(self.observed_graph.nodes),
                              list(self.observed_graph.edges))
        logging.debug(
            f"G''({self.dataset.data.x.shape[0]}, {self.dataset.data.edge_index.shape[1]//2}, "
            f"{self.dataset.num_features:d}, {self.dataset.n_classes:d}) from {params['sample']}.")
        self.graph = None


def induced_edge_compare(graph_name="ogbn-arxiv"):
    graph = get_graph_topology(None, True, False, graph_name, True)
    sample = ['BreadthFirstSearchSampler', 'RandomWalkSampler', 'ForestFireSampler',
              'MetropolisHastingsRandomWalkSampler']
    ratio = [0.1, 0.3, 0.5]
    induced = [False, True]
    seed = range(40, 50)
    df = pd.DataFrame()
    for s, r, i, seed_ in product(sample, ratio, induced, seed):
        print("starting sample", s, r, i, seed_)
        observed_graph = get_sampled_graph(graph, s, r, seed_, i)
        v = observed_graph.number_of_nodes()
        e = observed_graph.number_of_edges()
        df = df.append({
            "sample": s,
            "ratio": r,
            "induced": '1' if i else '0',
            "N": v,
            "E": e
        }, ignore_index=True)
    df.to_excel('../result/induced_edge_compare.xlsx')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, datefmt="%m-%dT%H:%M:%S", format="%(asctime)s %(message)s")

    induced_edge_compare()
    exit(0)
    import sys

    sys.path.append(".")
    G = nx.complete_graph(5)


    class Empty:
        def subgraph(self, observed_node, observed_edge):
            edge_index = subgraph_wrapper(observed_node, observed_edge=observed_edge,
                                          edge_index=self.data.edge_index, num_nodes=5)
            return to_networkx(self.data, to_undirected=True)


    dataset = Empty()
    from torch_geometric.data import Data

    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t_()
    edge_index = to_undirected(edge_index)
    dataset.data = Data(edge_index=edge_index, x=torch.ones((5, 1)))
    dataset.num_features = 0
    g = get_graph_topology(dataset, only_largest_cc=True, store_graphml="", graph_name="")
    assert G.number_of_nodes() == g.number_of_nodes(), G.number_of_edges() == g.number_of_edges()
    g1 = get_sampled_graph(g, "MetropolisHastingsRandomWalkSampler", 0.5, 42, False)
    assert g1.number_of_edges() < g1.number_of_nodes() * (g1.number_of_nodes() - 1) // 2
    e1 = subgraph_wrapper(list(g1.nodes), observed_edge=list(g1.edges), edge_index=edge_index, num_nodes=5)
    assert e1.shape == (2, 2 * g1.number_of_edges())
    g1 = get_sampled_graph(g, "MetropolisHastingsRandomWalkSampler", 0.5, 42, True)
    assert g1.number_of_edges() == g1.number_of_nodes() * (g1.number_of_nodes() - 1) // 2
    e1 = subgraph_wrapper(list(g1.nodes), observed_edge=None, edge_index=edge_index, num_nodes=5)
    assert e1.shape == (2, 2 * g1.number_of_edges())
    from torch_geometric.utils.num_nodes import maybe_num_nodes

    assert maybe_num_nodes(e1) == g1.number_of_nodes()

    e1 = subgraph_wrapper([1, 2, 3], observed_edge=[(1, 2), (2, 3)], edge_index=edge_index, num_nodes=5,
                          observed_edge_in_graph=False)
    assert e1.shape == (2, 2 * 2)
    print(e1)
