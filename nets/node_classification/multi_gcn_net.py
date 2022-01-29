import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout


class MultiGCNNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']

        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList()
        self.layers.extend([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                     self.batch_norm, self.residual) for _ in range(n_layers)])

        self.jump = JumpingKnowledge(mode='cat')
        self.MLP_layer = MLPReadout(hidden_dim * n_layers, n_classes, 0)


    def forward(self, feature, edge_index):
        # input embedding

        h = self.embedding_h(feature)
        h = self.in_feat_dropout(h)
        xs = []
        # GCNs
        for conv in self.layers:
            h = conv(h, edge_index)
            xs += [h]
        h = self.jump(xs)
        h = self.MLP_layer(h)

        return h


    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero(as_tuple=False)].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss
