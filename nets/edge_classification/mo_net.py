import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import degree
from torch_geometric.utils.num_nodes import maybe_num_nodes

import numpy as np

"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""

from layers.gmm_layer import GMMLayer
from layers.mlp_readout_layer import MLPReadout

class MoNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.name = 'MoNet'
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        kernel = net_params['kernel']                       # for MoNet
        dim = net_params['pseudo_dim_MoNet']                # for MoNet
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']                          
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']  
        self.device = net_params['device']
        
        aggr_type = "add"                               # default for MoNet
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Hidden layer
        for _ in range(n_layers-1):
            self.layers.append(GMMLayer(hidden_dim, hidden_dim, dim, kernel, aggr_type,
                                        dropout, batch_norm, residual))
            self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
            
        # Output layer
        self.layers.append(GMMLayer(hidden_dim, out_dim, dim, kernel, aggr_type,
                                    dropout, batch_norm, residual))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        
        self.MLP_layer = MLPReadout(2*out_dim, 1)

    def forward(self, feature, edge_index):
        h = self.embedding_h(feature)

        row, col = edge_index
        nodes_num = maybe_num_nodes(edge_index)
        srcs = 1 / torch.sqrt(degree(row, nodes_num)[row] + 1)
        dsts = 1 / torch.sqrt(degree(col, nodes_num)[col] + 1)
        pseudo = torch.cat((srcs.unsqueeze(-1), dsts.unsqueeze(-1)), dim=1)

        for i in range(len(self.layers)):
            h = self.layers[i](h, edge_index, self.pseudo_proj[i](pseudo))
        return h
    
    def edge_predictor(self, h_i, h_j):
        x = torch.cat([h_i, h_j], dim=1)
        x = self.MLP_layer(x)
        
        return torch.sigmoid(x)
    
    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss
        
        return loss
