#!/bin/bash
export LD_LIBRARY_PATH="/home/weiq/anaconda3/lib"
export CUDA_VISIBLE_DEVICES="0"


#python main_network_classification.py --config ./config/superpixels_graph_classification_GIN_CIFAR10_100k.json --sample rw 1 --ratio 0.5
#
#python main_network_classification.py --config ./config/superpixels_graph_classification_GatedGCN_MNIST_100k.json --sample bfs 0 --ratio 0.3
#
#python main_network_classification.py --config ./config/superpixels_graph_classification_GatedGCN_MNIST_100k.json --sample bfs 1 --ratio 0.1

python main_network_classification.py --config ./config/superpixels_graph_classification_GCN_MNIST_100k.json --sample mhrw 1 --ratio 0.4 0.5

#python main_network_classification.py --config ./config/superpixels_graph_classification_GIN_MNIST_100k.json --sample ffs 1 --ratio 0.5