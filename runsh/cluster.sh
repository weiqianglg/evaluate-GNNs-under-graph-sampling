#!/bin/bash
export LD_LIBRARY_PATH="/home/weiq/anaconda3/lib"
export CUDA_VISIBLE_DEVICES="0"
python main_node_classification.py --config ./config/SBMs_node_clustering_GAT_CLUSTER_100k.json --sample bfs 0 bfs 1 ffs 0 ffs 1 mhrw 0 mhrw 1
python main_node_classification.py --config ./config/SBMs_node_clustering_GatedGCN_CLUSTER_100k.json
#python main_node_classification.py --config ./config/SBMs_node_clustering_GCN_CLUSTER_100k.json
#python main_node_classification.py --config ./config/SBMs_node_clustering_GIN_CLUSTER_100k.json
#python main_node_classification.py --config ./config/SBMs_node_clustering_GraphSage_CLUSTER_100k.json
#python main_node_classification.py --config ./config/SBMs_node_clustering_MLP_CLUSTER_100k.json
#python main_node_classification.py --config ./config/SBMs_node_clustering_MoNet_CLUSTER_100k.json