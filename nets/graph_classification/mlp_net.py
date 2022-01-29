import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from layers.mlp_readout_layer import MLPReadout

class MLPNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.gated = net_params['gated']
        self.readout = net_params['readout']
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        feat_mlp_modules = [
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers-1):
            feat_mlp_modules.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            feat_mlp_modules.append(nn.ReLU())
            feat_mlp_modules.append(nn.Dropout(dropout))
        self.feat_mlp = nn.Sequential(*feat_mlp_modules)
        
        if self.gated:
            self.gates = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        self.readout_mlp = MLPReadout(hidden_dim, n_classes)

    def forward(self, feature, edge_index, batch):
        h = self.in_feat_dropout(feature)
        h = self.feat_mlp(h)
        if self.gated:
            h = torch.sigmoid(self.gates(h)) * h

        if self.readout == "sum":
            hg = global_add_pool(h, batch)
        elif self.readout == "max":
            hg = global_max_pool(h, batch)
        elif self.readout == "mean":
            hg = global_mean_pool(h, batch)
        else:
            hg = global_mean_pool(h, batch)  # default readout is mean nodes
        
        return self.readout_mlp(hg)
        
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss