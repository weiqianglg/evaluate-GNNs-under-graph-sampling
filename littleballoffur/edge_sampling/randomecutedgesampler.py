import random
from collections import Counter
from itertools import combinations
import networkx as nx
import networkit as nk
from typing import Union, List
from littleballoffur.sampler import Sampler

NKGraph = type(nk.graph.Graph())
NXGraph = nx.classes.graph.Graph


class CutEdgeSampler(Sampler):
    r"""cut communities into 2 disjoint parts, and only keep edges between them.
    """
    def __init__(self, community: str = 'label', keep_edge_in_commnity="no", number_of_nodes: int = 100,
                 seed: int = 42, subgraph: bool = False):
        self.community_label = community
        self.keep_in_edge = keep_edge_in_commnity
        self.seed = seed
        self._set_seed()
        self.random_left_p = number_of_nodes # this p will be recalculated in _random_split, p=number_of_nodes/|V|

    def _split_two_disjoint_set(self, graph, citerion):
        left, right = set(), set()
        for n in self.backend.get_node_iterator(graph):
            if citerion(graph, n):
                left.add(n)
            else:
                right.add(n)
        return left, right

    def _random_split(self, graph):
        self.random_left_p /= graph.number_of_nodes()
        def _random_left(graph, n):
            return random.random() < self.random_left_p
        return self._split_two_disjoint_set(graph, _random_left)

    def _random_community_split(self, graph):
        labels = [self.backend.get_node_attr(graph, n)[self.community_label] for n in
                  self.backend.get_node_iterator(graph)]
        labels = Counter(labels)
        label_set = set(labels.keys())
        r = []
        for i in range(1, len(label_set) // 2 + 1):
            for left_label in combinations(label_set, i):
                r.append(
                    left_label
                )
        left_label = random.choice(r)
        left_count = sum([labels[l] for l in left_label])
        right_count = sum([labels[l] for l in label_set-set(left_label)])

        def _in_left_community(graph, n):
            return self.backend.get_node_attr(graph, n)[self.community_label] in left_label
        return self._split_two_disjoint_set(graph, _in_left_community)

    def _largest_community_split(self, graph):
        """keep the largest community left, and the others right"""
        labels = [self.backend.get_node_attr(graph, n)[self.community_label] for n in
                  self.backend.get_node_iterator(graph)]
        labels = Counter(labels)
        largest = sorted(labels.items(), key=lambda x: x[1], reverse=True)[0][0]
        def _in_left_community(graph, n):
            return self.backend.get_node_attr(graph, n)[self.community_label] == largest
        return self._split_two_disjoint_set(graph, _in_left_community)

    def _select_edges(self, graph, left, right):
        self._sampled_edges = []
        if self.keep_in_edge == 'left':
            keep_edge_community = left
        elif self.keep_in_edge == 'right':
            keep_edge_community = right
        elif self.keep_in_edge == 'random':
            keep_edge_community = random.choice([left, right])
        else:
            keep_edge_community = None
        for u, v in self.backend.get_edge_iterator(graph):
            if (u in left and v in right) or (u in right and v in left) or (
                    keep_edge_community and u in keep_edge_community and v in keep_edge_community):
                self._sampled_edges.append((u, v))

    def sample(self, graph: Union[NXGraph, NKGraph], start_node: int = 0) -> Union[NXGraph, NKGraph]:
        self._deploy_backend(graph)

        left, right = self._random_split(graph)
        self._select_edges(graph, left, right)
        new_graph = self.backend.graph_from_edgelist(self._sampled_edges)
        return new_graph
