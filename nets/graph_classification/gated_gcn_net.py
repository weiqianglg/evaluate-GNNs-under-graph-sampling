import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

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
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']

        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                   self.batch_norm, self.residual) for _ in range(n_layers - 1)])
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, feature, edge_index, batch):
        h = self.embedding_h(feature)
        for conv in self.layers:
            h = conv(h, edge_index)

        if self.readout == "sum":
            hg = global_add_pool(h, batch)
        elif self.readout == "max":
            hg = global_max_pool(h, batch)
        elif self.readout == "mean":
            hg = global_mean_pool(h, batch)
        else:
            hg = global_mean_pool(h, batch)  # default readout is mean nodes

        return self.MLP_layer(hg)

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
