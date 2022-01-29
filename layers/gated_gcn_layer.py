import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ResGatedGraphConv, GatedGraphConv

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""


class GatedGCNLayer(nn.Module):
    """
        Param: []
    """

    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv = ResGatedGraphConv(input_dim, output_dim, root_weight=residual)

    def forward(self, feature, edge_index):
        h = self.conv(feature, edge_index)

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization

        h = F.relu(h)  # non-linear activation
        h = self.dropout(h)

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)

