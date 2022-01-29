import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GMMConv

"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""


class GMMLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of GMMConv

    Parameters
    ----------
    in_dim : 
        Number of input features.
    out_dim : 
        Number of output features.
    dim : 
        Dimensionality of pseudo-coordinte.
    kernel : 
        Number of kernels :math:`K`.
    aggr_type : 
        Aggregator type (``sum``, ``mean``, ``max``).
    dropout :
        Required for dropout of output features.
    batch_norm :
        boolean flag for batch_norm layer.
    residual : 
        If True, use residual connection inside this layer. Default: ``False``.
    bias : 
        If True, adds a learnable bias to the output. Default: ``True``.
    
    """

    def __init__(self, in_dim, out_dim, dim, kernel, aggr_type, dropout,
                 batch_norm, residual=False, bias=True):
        super().__init__()

        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout

        self.gmmconv = GMMConv(in_dim, out_dim, dim, kernel, aggr=aggr_type, bias=bias)
        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(out_dim)

        if in_dim != out_dim:
            self.residual = False

    def forward(self, feature, edge_index, edge_attr):
        h_in = feature  # to be used for residual connection

        h = self.gmmconv(feature, edge_index, edge_attr)

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization

        h = F.relu(h)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection

        h = F.dropout(h, self.dropout, training=self.training)

        return h
