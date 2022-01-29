"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import logging
import time
import torch
import numpy as np
from tqdm import tqdm


# def train_network_sparse(model, optimizer, device, data, params, scheduler, evaluator):
#     start0 = time.time()
#     model.train()
#     x = data.x.to(device)
#     edge_index = data.edge_index.to(device)
#     y = data.y.to(device)
#     train_mask = data.train_mask.to(device)
#     val_mask = data.val_mask.to(device)
#     t = tqdm(range(params['epochs']))
#     patience = params['num_epochs_patience']
#     vlss_mn = np.inf
#     vacc_mx = 0.0
#     vacc_early_model = None
#     tacc_early_model = None
#     state_dict_early_model = None
#     curr_step = 0
#     for epoch in t:
#         optimizer.zero_grad()
#         scores = model.forward(x, edge_index)
#         loss = model.loss(scores[train_mask], y[train_mask])
#         loss.backward()
#         optimizer.step()
#
#         with torch.no_grad():
#             val_loss = model.loss(scores[val_mask], y[val_mask])
#             val_acc = evaluator(scores[val_mask], y[val_mask])
#             if epoch % params['print_epoch_interval'] == 0:
#                 logging.debug(f"epoch {epoch}, loss {loss:.4f}, val_loss {val_loss:.4f}, val_acc {val_acc:.4f}")
#                 t.display(f"train loss {loss:.4f}, val loss {val_loss:.4f}.")
#         scheduler.step(val_loss)
#
#         if optimizer.param_groups[0]['lr'] < params['min_lr']:
#             logging.info("!BREAK! lr smaller or equal to min lr threshold.")
#             break
#
#         # Stop training after params['max_time'] hours
#         if time.time() - start0 > params['max_time'] * 3600:
#             logging.info("!BREAK! max_time for training elapsed {:.2f} hours.".format(params['max_time']))
#             break
#
#         # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
#         if val_acc >= vacc_mx or val_loss <= vlss_mn:
#             if val_acc >= vacc_mx and val_loss <= vlss_mn:
#                 vacc_early_model = val_acc
#                 tacc_early_model = evaluator(scores[train_mask], y[train_mask])
#                 state_dict_early_model = model.state_dict()
#             vacc_mx = np.max((val_acc, vacc_mx))
#             vlss_mn = np.min((val_loss.item(), vlss_mn))
#             curr_step = 0
#         else:
#             curr_step += 1
#             if curr_step >= patience:
#                 logging.info(f"!BREAK! val acc or loss can not be improved in {patience} epoch.")
#                 break
#
#     return tacc_early_model, vacc_early_model, state_dict_early_model
#
#
# def evaluate_network_sparse(model, device, data, mask, evaluator, state_dict_early_model=None):
#     if state_dict_early_model:
#         model.load_state_dict(state_dict_early_model)
#     model.eval()
#     x = data.x.to(device)
#     edge_index = data.edge_index.to(device)
#     y = data.y.to(device)
#     mask = mask.to(device)
#     with torch.no_grad():
#         scores = model.forward(x, edge_index)
#         test_loss = model.loss(scores[mask], y[mask])
#         test_acc = evaluator(scores[mask], y[mask])
#     return test_loss, test_acc


def train_epoch_sparse(model, optimizer, device, data_loader, single_graph=True):
    model.train()
    epoch_loss = 0
    for iter, batch_graphs in enumerate(data_loader):
        batch_x = batch_graphs.x.to(device)  # num x feat
        batch_edge_index = batch_graphs.edge_index.to(device)
        batch_labels = batch_graphs.y.to(device)
        if single_graph:
            train_mask = batch_graphs.train_mask.to(device)
        optimizer.zero_grad()
        batch_scores = model.forward(batch_x, batch_edge_index)
        if single_graph:
            loss = model.loss(batch_scores[train_mask], batch_labels[train_mask])
        else:
            loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

    epoch_loss /= (iter + 1)

    return epoch_loss, optimizer


def evaluate_network_sparse(model, device, data_loader, evaluator, single_graph=True, eva_type='val'):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    with torch.no_grad():
        for iter, batch_graphs in enumerate(data_loader):
            batch_x = batch_graphs.x.to(device)  # num x feat
            batch_edge_index = batch_graphs.edge_index.to(device)
            batch_labels = batch_graphs.y.to(device)
            if single_graph:
                if eva_type == 'val':
                    mask = batch_graphs.val_mask.to(device)
                elif eva_type == 'test':
                    mask = batch_graphs.test_mask.to(device)
                else:
                    raise TypeError("evaluation type for single graph must be val or test.")
            batch_scores = model.forward(batch_x, batch_edge_index)
            if single_graph:
                loss = model.loss(batch_scores[mask], batch_labels[mask])
                acc = evaluator(batch_scores[mask], batch_labels[mask])
            else:
                loss = model.loss(batch_scores, batch_labels)
                acc = evaluator(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += acc
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= (iter + 1)

    return epoch_test_loss, epoch_test_acc
