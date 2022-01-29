import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""


class GraphSageLayer(nn.Module):

    def __init__(self, in_feats, out_feats, activation, dropout,
                 aggregator_type="mean", batch_norm=True, residual=False,
                 bias=True):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        self.activation = activation
        self.aggregator_type = aggregator_type
        self.batch_norm = batch_norm
        self.residual = residual

        if in_feats != out_feats:
            self.residual = False

        self.dropout = nn.Dropout(p=dropout)

        self.sageconv = SAGEConv(in_feats, out_feats, bias=bias, aggr=aggregator_type)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)

    def forward(self, feature, edge_index):
        h_in = feature  # to be used for residual connection

        h = self.dropout(feature)

        h = self.sageconv(h, edge_index)

        h = self.activation(h)

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.residual:
            h = h_in + h  # residual connection

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, aggregator={}, residual={})'.format(self.__class__.__name__,
                                                                                        self.in_channels,
                                                                                        self.out_channels,
                                                                                        self.aggregator_type,
                                                                                        self.residual)
