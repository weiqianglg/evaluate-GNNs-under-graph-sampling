#!/bin/bash
export LD_LIBRARY_PATH="/home/weiq/anaconda3/lib"
export CUDA_VISIBLE_DEVICES="0"


#python main_network_classification.py --config ./config/superpixels_graph_classification_GIN_MNIST_100k.json --sample rw 1 bfs 0 bfs 1 ffs 0 ffs 1 mhrw 0 mhrw 1

python main_network_classification.py --config ./config/superpixels_graph_classification_GIN_CIFAR10_100k.json --sample ffs 0 ffs 1 mhrw 0 mhrw 1
