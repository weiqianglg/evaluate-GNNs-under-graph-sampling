import logging
import os.path as osp
import numpy as np
import time
import argparse
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from util.data_graph_network_classification import load_data
from nets.graph_classification.load_net import gnn_model  # import GNNs
from util.run import gpu_setup
from util.run import view_model_param
from util.run import set_seed
from util.run import multi_dataset_sample
from util.run import parse_args


def train_val_pipeline(MODEL_NAME, dataset, params, net_params):
    start0 = time.time()
    device = net_params['device']

    set_seed(device, params['seed'])

    trainset, valset, testset = dataset
    train_loader = DataLoader(trainset.datas, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(valset.datas, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(testset.datas, batch_size=params['batch_size'], shuffle=False)
    logging.info(f"Training Graphs: {len(trainset.datas)}")
    logging.info(f"Validation Graphs: {len(valset.datas)}")
    logging.info(f"Test Graphs: {len(testset.datas)}")
    evaluator = trainset.evaluator

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    from train.train_network_classification import train_epoch_sparse as train_epoch
    from train.train_network_classification import evaluate_network_sparse as evaluate

    t = tqdm(range(params['epochs']))
    patience = params['num_epochs_patience']
    vlss_mn = np.inf
    vacc_mx = 0.0
    vacc_early_model = None
    tacc_early_model = None
    curr_step = 0
    try:
        for epoch in t:
            t.set_description('Epoch %d' % epoch)
            start = time.time()

            epoch_train_loss, optimizer = train_epoch(model, optimizer, device, train_loader)
            epoch_val_loss, epoch_val_acc = evaluate(model, device, val_loader, evaluator)

            t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                          train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                          val_acc=epoch_val_acc)

            scheduler.step(epoch_val_loss)

            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                logging.info("!BREAK! lr smaller or equal to min lr threshold.")
                break
            if time.time() - start0 > params['max_time'] * 3600:
                logging.info("!BREAK! max_time for training elapsed {:.2f} hours.".format(params['max_time']))
                break

            # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
            if epoch_val_acc >= vacc_mx or epoch_val_loss <= vlss_mn:
                if epoch_val_acc >= vacc_mx and epoch_val_loss <= vlss_mn:
                    vacc_early_model = epoch_val_acc
                    _, tacc_early_model = evaluate(model, device, test_loader, evaluator)
                vacc_mx = np.max((epoch_val_acc, vacc_mx))
                vlss_mn = np.min((epoch_val_loss, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    logging.info(f"!BREAK! val acc or loss can not be improved in {patience} epoch.")
                    break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    logging.info(f"Best Test Accuracy: {tacc_early_model:.4f}")
    logging.info(f"Best Val Accuracy: {vacc_early_model:.4f}")
    return tacc_early_model


def network_classification_run(config):
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])

    MODEL_NAME = config['model']
    DATASET_NAME = config['dataset']
    dataset = load_data(config)

    # parameters
    params = config['params']
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']

    net_params['in_dim'] = dataset[0].datas[0].x.shape[-1]
    net_params['n_classes'] = dataset[0].n_classes

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, "graph-model")
    test_acc = train_val_pipeline(MODEL_NAME, dataset, params, net_params)
    return {"dataset": DATASET_NAME, "model": MODEL_NAME,
            "test_acc": test_acc,
            "left_p": config['sample']["percent_of_nodes"]}


if __name__ == '__main__':
    import sys

    sys.path.append(".")
    logging.basicConfig(level=logging.INFO, datefmt="%m-%dT%H:%M:%S", format="%(asctime)s %(message)s")
    config, datasets, sample_paras, ratio = parse_args()
    multi_dataset_sample(network_classification_run, config, datasets, sample_paras, ratio)
