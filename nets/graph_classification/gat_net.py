import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer
from layers.mlp_readout_layer import MLPReadout

class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim * num_heads)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
                                              dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        
    def forward(self, feature, edge_index, batch):
        h = self.embedding_h(feature)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(h, edge_index)
        
        if self.readout == "sum":
            hg = global_add_pool(h, batch)
        elif self.readout == "max":
            hg = global_max_pool(h, batch)
        elif self.readout == "mean":
            hg = global_mean_pool(h, batch)
        else:
            hg = global_mean_pool(h, batch) # default readout is mean nodes
            
        return self.MLP_layer(hg)
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss