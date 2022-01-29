import logging
import time
import torch
import numpy as np
from tqdm import tqdm

from train.metrics import weighted_f1_score


def train_epoch_sparse(model, optimizer, device, data_loader):
    model.train()
    epoch_loss = 0
    for iter, batch_graphs in enumerate(data_loader):
        batch_x = batch_graphs.x.to(device)  # num x feat
        batch_edge_index = batch_graphs.edge_index.to(device)
        batch_labels = batch_graphs.y.to(device)
        batch = batch_graphs.batch.to(device)
        optimizer.zero_grad()
        batch_scores = model.forward(batch_x, batch_edge_index, batch)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

    epoch_loss /= (iter + 1)

    return epoch_loss, optimizer


def evaluate_network_sparse(model, device, data_loader, evaluator):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    with torch.no_grad():
        for iter, batch_graphs in enumerate(data_loader):
            batch_x = batch_graphs.x.to(device)  # num x feat
            batch_edge_index = batch_graphs.edge_index.to(device)
            batch_labels = batch_graphs.y.to(device)
            batch = batch_graphs.batch.to(device)
            batch_scores = model.forward(batch_x, batch_edge_index, batch)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += evaluator(batch_scores, batch_labels)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)

    return epoch_test_loss, epoch_test_acc
