#!/bin/bash
export LD_LIBRARY_PATH="/home/weiq/anaconda3/lib"
export CUDA_VISIBLE_DEVICES="0"

#python main_network_classification.py --config ./config/superpixels_graph_classification_GatedGCN_MNIST_100k.json
python main_network_classification.py --config ./config/superpixels_graph_classification_GatedGCN_CIFAR10_100k.json --sample mhrw 0 mhrw 1
