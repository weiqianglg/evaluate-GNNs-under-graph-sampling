#!/bin/bash
export LD_LIBRARY_PATH="/home/weiq/anaconda3/lib"
export CUDA_VISIBLE_DEVICES="1"


python main_network_classification.py --config ./config/superpixels_graph_classification_GatedGCN_MNIST_100k.json --sample bfs 0 --ratio 0.4 0.5

python main_network_classification.py --config ./config/superpixels_graph_classification_GatedGCN_MNIST_100k.json --sample ffs 0 ffs 1 --ratio 0.1 0.2 0.3 0.4 0.5