import logging
import os.path as osp
from itertools import product
import numpy as np
import os
import random
import argparse
import json
import pandas as pd
import torch

from nets.node_classification.load_net import gnn_model as node_gnn_model  # import GNNs
from nets.edge_classification.load_net import gnn_model as edge_gnn_model  # import GNNs
from nets.graph_classification.load_net import gnn_model as graph_gnn_model  # import GNNs


def gpu_setup(use_gpu, gpu_id):
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        torch.cuda.set_device(gpu_id)
        logging.info(f'cuda available with GPU:{torch.cuda.get_device_name()}:{gpu_id}')
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        logging.info('cuda not available')
        device = torch.device("cpu")
    return device


def view_model_param(MODEL_NAME, net_params, type="node-model"):
    if "node" in type:
        model = node_gnn_model(MODEL_NAME, net_params)
    elif "edge" in type:
        model = edge_gnn_model(MODEL_NAME, net_params)
    elif "graph" in type:
        model = graph_gnn_model(MODEL_NAME, net_params)
    else:
        model = None
    total_param = 0
    logging.info(f"MODEL DETAILS:\n{model}")
    # for name, p in model.named_parameters():
    #     print(name, p.shape)
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    logging.info(f'{MODEL_NAME} Total parameters: {total_param}')
    return total_param


def set_seed(device, seed_):
    # setting seeds
    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed_)


def try_another_seed_if_failed(model_func, config, times=5):
    result = None
    result = model_func(config)
    return result
    for t in range(times):
        try:
            result = model_func(config)
            break
        except Exception as e:
            logging.error(f"try {t} time:")
            logging.error(e)
            config['sample']['seed'] = config['sample']['seed'] * 10 + t
            continue
    return result


def dataframe_to_excel(metric, config):
    outpath = config["out_dir"]
    if not osp.exists(outpath):
        os.makedirs(outpath)
    result_path = osp.join(outpath, f"{config['dataset']}-"
                                    f"{config['model']}-"
                                    f"{config['sample']['sample_method']}-"
                                    f"{config['sample']['subgraph']}.xlsx")
    logging.info(f"\n{metric} \nout to {result_path}.")
    metric.to_excel(result_path)


def sample_(model_func, config, percent_of_nodes, round_per_sample=4):
    metric = pd.DataFrame()
    for sample_percent_of_nodes in percent_of_nodes:
        config['sample']["percent_of_nodes"] = sample_percent_of_nodes
        sample_init_seed = config['sample']['seed']
        para_init_seed = config['params']['seed']
        for seed in range(1, 1 + round_per_sample):
            config['sample']['seed'] = sample_init_seed + seed
            config['params']['seed'] = para_init_seed + seed
            result = try_another_seed_if_failed(model_func, config)
            metric = metric.append(result, ignore_index=True)
            dataframe_to_excel(metric, config)


def multi_dataset_sample(model_func, model_config_path, ds, sample_paras, percent_of_nodes):
    with open(model_config_path) as f:
        config = json.load(f)

    for dataset, (sample_method, subgraph) in product(ds, sample_paras):
        config["dataset"] = dataset
        config["sample"]["sample_method"] = sample_method
        config["sample"]["subgraph"] = subgraph
        sample_(model_func, config, percent_of_nodes)  # 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--sample', action="extend", nargs='*', type=str, help="give sample method(s)")
    parser.add_argument('--dataset', action="extend", nargs='*', type=str, help="give dataset(s)")
    parser.add_argument('--ratio', action="extend", nargs='*', type=float, help="give sample ratio(s)")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    sample_method_arg = {
        "rw": 'RandomWalkSampler',
        "bfs": 'BreadthFirstSearchSampler',
        "ffs": 'ForestFireSampler',
        "mhrw": 'MetropolisHastingsRandomWalkSampler',
        "mh": 'MetropolisHastingsRandomWalkSampler',
    }
    induced_arg = {
        "1": True,
        "true": True,
        "0": False,
        'false': False,
    }

    if args.sample:  # rw 1 ffs 0
        sample_paras = []
        for i in range(0, len(args.sample), 2):
            sample_paras.append((
                sample_method_arg[args.sample[i].lower()],
                induced_arg[args.sample[i + 1].lower()]
            ))
    else:
        sample_paras = [
            # ('CutEdgeSampler', False),
            ('RandomWalkSampler', False),
            ('RandomWalkSampler', True),
            ('BreadthFirstSearchSampler', False),
            ('BreadthFirstSearchSampler', True),
            ('ForestFireSampler', False),
            ('ForestFireSampler', True),
            ('MetropolisHastingsRandomWalkSampler', False),
            ('MetropolisHastingsRandomWalkSampler', True)
        ]
    if args.dataset:
        datasets = args.dataset
    else:
        datasets = [config['dataset']]
    if args.ratio:
        ratio = np.array(args.ratio)
    else:
        step_percent, stop_precent = 0.1, 0.5
        ratio = np.linspace(step_percent, stop_precent, int(stop_precent / step_percent))
        ratio = np.hstack([ratio, 1.0]) # add 100%
    logging.info(f"config file {args.config}")
    logging.info(f"datasets {datasets}")
    logging.info(f"sample paras {sample_paras}")
    logging.info(f"ratio paras {ratio}")
    return args.config, datasets, sample_paras, ratio


if __name__ == '__main__':
    import sys

    sys.path.append(".")
    logging.basicConfig(level=logging.DEBUG, datefmt="%m-%dT%H:%M:%S",
                        format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    args = parser.parse_args()
    datasets = ['Citeseer']  ##,['Actor', 'Citeseer', 'Cora', 'Pubmed']
    sample_paras = [
        ('CutEdgeSampler', False),
        # ('RandomWalkSampler', False),
        # # ('RandomWalkSampler', True),
        # ('BreadthFirstSearchSampler', False),
        # # ('BreadthFirstSearchSampler', True),
        # ('ForestFireSampler', False),
        # # ('ForestFireSampler', True),
        # ('MetropolisHastingsRandomWalkSampler', False),
        # ('MetropolisHastingsRandomWalkSampler', True)
    ]
    multi_dataset_sample(args.config, datasets, sample_paras, step_percent=1, add_all=True)
