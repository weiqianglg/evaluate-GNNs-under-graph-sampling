"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)

from tqdm import tqdm
import torch.nn.functional as F


def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(pos_edge_index.device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train_epoch_sparse(model, optimizer, device, data, train_edges, batch_size, epoch):
    model.train()
    edge_index = data.edge_index
    x = data.x

    total_loss = total_examples = 0
    for perm in tqdm(DataLoader(range(train_edges.size(0)), batch_size, shuffle=True)):
        optimizer.zero_grad()

        # Compute node embeddings
        h = model(x, edge_index)

        # Positive samples
        edge = train_edges[perm].t()
        pos_out = model.edge_predictor(h[edge[0]], h[edge[1]])

        # Just do some trivial random sampling
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=x.device)
        neg_out = model.edge_predictor(h[edge[0]], h[edge[1]])

        loss = model.loss(pos_out, neg_out)

        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.detach().item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples, optimizer


def evaluate_network_sparse(model, device, data, pos_train_edges,
                            pos_valid_edges, neg_valid_edges,
                            pos_test_edges, neg_test_edges,
                            evaluator, batch_size, epoch):
    edge_index = data.edge_index
    x = data.x
    model.eval()

    with torch.no_grad():

        h = model(x, edge_index)

        pos_train_preds = []
        for perm in DataLoader(range(pos_train_edges.size(0)), batch_size):
            edge = pos_train_edges[perm].t()
            pos_train_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edges.size(0)), batch_size):
            edge = pos_valid_edges[perm].t()
            pos_valid_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(range(pos_valid_edges.size(0)), batch_size):
            edge = neg_valid_edges[perm].t()
            neg_valid_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        val_loss = model.loss(pos_valid_pred, neg_valid_pred)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edges.size(0)), batch_size):
            edge = pos_test_edges[perm].t()
            pos_test_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(pos_test_edges.size(0)), batch_size):
            edge = neg_test_edges[perm].t()
            neg_test_preds += [model.edge_predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

    train_hits = evaluator(pos_train_pred, neg_valid_pred)  # negative samples for valid == training

    valid_hits = evaluator(pos_valid_pred, neg_valid_pred)

    test_hits = evaluator(pos_test_pred, neg_test_pred)

    return train_hits, valid_hits, test_hits, val_loss
