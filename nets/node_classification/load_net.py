"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.node_classification.gcn_net import GCNNet
from nets.node_classification.gat_net import GATNet
from nets.node_classification.graphsage_net import GraphSageNet
from nets.node_classification.mlp_net import MLPNet
from nets.node_classification.gin_net import GINNet
from nets.node_classification.mo_net import MoNet as MoNet_
from nets.node_classification.multi_gcn_net import MultiGCNNet
from nets.node_classification.gated_gcn_net import GatedGCNNet


def GCN(net_params):
    return GCNNet(net_params)


def GAT(net_params):
    return GATNet(net_params)


def GraphSage(net_params):
    return GraphSageNet(net_params)


def MLP(net_params):
    return MLPNet(net_params)


def GIN(net_params):
    return GINNet(net_params)


def MoNet(net_params):
    return MoNet_(net_params)


def MultiGCN(net_params):
    return MultiGCNNet(net_params)


def GatedGCN(net_params):
    return GatedGCNNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'MLP': MLP,
        'GIN': GIN,
        'MoNet': MoNet,
        'MultiGCN': MultiGCN,
        'GatedGCN': GatedGCN,
    }

    return models[MODEL_NAME](net_params)
