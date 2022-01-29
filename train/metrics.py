import torch
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np


def accuracy_MNIST_CIFAR(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores == targets).float().sum().item()
    return acc / len(targets)


def accuracy_CITATION_GRAPH(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores == targets).float().sum().item()
    acc = acc / len(targets)
    return acc


def weighted_f1_score(scores, targets):
    scores = scores.cpu().detach().argmax(dim=1).numpy()
    targets = targets.cpu().detach().numpy()
    acc = f1_score(scores, targets, average='weighted')
    return acc


def binary_f1_score(scores, targets):
    """Computes the F1 score using scikit-learn for binary class labels. 
    
    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().numpy()
    y_pred = scores.argmax(dim=1).cpu().numpy()
    return f1_score(y_true, y_pred, average='binary')


def link_auc_score(pos_pred, neg_pred):
    pos_len, neg_len = len(pos_pred), len(neg_pred)
    link_labels = np.zeros(pos_len + neg_len, dtype=np.uint8)
    link_labels[:pos_len] = 1
    link_probs = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
    return roc_auc_score(link_labels, link_probs)
