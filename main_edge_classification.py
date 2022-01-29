import numpy as np
import os
import socket
import time
import logging
import glob
import argparse, json

import torch

import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from util.run import gpu_setup
from util.run import view_model_param
from util.run import set_seed
from util.run import multi_dataset_sample
from util.run import parse_args
from util.data_graph_edge_classification import load_data

from nets.edge_classification.load_net import gnn_model  # import all GNNS


def train_val_pipeline(MODEL_NAME, dataset, params, net_params):
    t0 = time.time()
    per_epoch_time = []

    graph = dataset.data

    evaluator = dataset.evaluator

    train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg = dataset.train_edges, dataset.val_edges, dataset.val_edges_neg, dataset.test_edges, dataset.test_edges_neg

    device = net_params['device']
    set_seed(device, params['seed'])

    model = gnn_model(MODEL_NAME, net_params)

    model = model.to(device)
    graph.edge_index = graph.edge_index.to(device)
    graph.x = graph.x.to(device)
    train_edges = train_edges.to(device)
    val_edges = val_edges.to(device)
    val_edges_neg = val_edges_neg.to(device)
    test_edges = test_edges.to(device)
    test_edges_neg = test_edges_neg.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    epoch_train_losses = []
    epoch_train_hits, epoch_val_hits = [], []

    from train.train_edge_classification import train_epoch_sparse as train_epoch
    from train.train_edge_classification import evaluate_network_sparse as evaluate_network

    t = tqdm(range(params['epochs']))
    patience = params['num_epochs_patience']
    vlss_mn = np.inf
    vacc_mx = 0.0
    train_acc_early_model = None
    vacc_early_model = None
    tacc_early_model = None
    curr_step = 0
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in t:

            t.set_description('Epoch %d' % epoch)

            start = time.time()

            epoch_train_loss, optimizer = train_epoch(model, optimizer, device, graph, train_edges,
                                                      params['batch_size'], epoch)
            epoch_train_hits, epoch_val_hits, epoch_test_hits = 0, 0, 0
            epoch_train_hits, epoch_val_hits, epoch_test_hits, epoch_val_loss = evaluate_network(
                model, device, graph, train_edges, val_edges, val_edges_neg, test_edges, test_edges_neg, evaluator,
                params['batch_size'], epoch)

            t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                          train_loss=epoch_train_loss, train_hits=epoch_train_hits,
                          val_hits=epoch_val_hits, test_hits=epoch_test_hits)

            per_epoch_time.append(time.time() - start)

            scheduler.step(epoch_val_hits)

            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

            # Stop training after params['max_time'] hours
            if time.time() - t0 > params['max_time'] * 3600:
                print('-' * 89)
                print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                break

            # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
            if epoch_val_hits >= vacc_mx or epoch_val_loss <= vlss_mn:
                if epoch_val_hits >= vacc_mx and epoch_val_loss <= vlss_mn:
                    train_acc_early_model = epoch_train_hits
                    vacc_early_model = epoch_val_hits
                    tacc_early_model = epoch_test_hits
                vacc_mx = np.max((epoch_val_hits, vacc_mx))
                vlss_mn = np.min((epoch_val_loss.item(), vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    logging.info(f"!BREAK! val acc or loss can not be improved in {patience} epoch.")
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    train_hits, val_hits, test_hits = train_acc_early_model, vacc_early_model, tacc_early_model
    logging.info(f"Hit@K train {train_hits * 100:.4f}%, val {val_hits * 100:.4f}%, test {test_hits * 100:.4f}%")
    logging.info("Convergence Time (Epochs): {:.4f}".format(epoch))
    logging.info("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    logging.info("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    return test_hits


def edge_classification_run(config):
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    MODEL_NAME = config['model']
    DATASET_NAME = config['dataset']

    dg = load_data(config)
    dataset = dg.dataset

    params = config['params']
    net_params = config['net_params']
    net_params['in_dim'] = dataset.data.x.shape[-1]
    net_params['n_classes'] = 1  # binary prediction
    net_params['device'] = device
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, type="edge-model")
    metric = train_val_pipeline(MODEL_NAME, dataset, params, net_params)
    return {"dataset": DATASET_NAME, "model": MODEL_NAME,
            "test_acc": metric,
            "left_p": config['sample']["percent_of_nodes"]}


if __name__ == '__main__':
    import sys

    sys.path.append(".")
    logging.basicConfig(level=logging.DEBUG, datefmt="%m-%dT%H:%M:%S", format="%(asctime)s %(message)s")
    config, datasets, sample_paras, ratio = parse_args()
    multi_dataset_sample(edge_classification_run, config, datasets, sample_paras, ratio)

    # fpath = "./config/node_classification_sage.json"
    # with open(fpath) as f:
    #     config = json.load(f)
    # edge_classification_run(config)
