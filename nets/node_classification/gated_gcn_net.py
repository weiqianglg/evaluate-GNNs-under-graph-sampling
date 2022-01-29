import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout


class GatedGCNNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer
        self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                   self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        if not self.readout:
            out_dim = n_classes
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        if self.readout:
            self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, feature, edge_index):
        # input embedding
        h = self.embedding_h(feature)
        for conv in self.layers:
            h = conv(h, edge_index)
        if self.readout:
            h = self.MLP_layer(h)
        return h

    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss
