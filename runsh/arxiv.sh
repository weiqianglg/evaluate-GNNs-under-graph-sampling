#!/bin/bash
export LD_LIBRARY_PATH="/home/weiq/anaconda3/lib"
export CUDA_VISIBLE_DEVICES="0"

#python main_node_classification.py --config ./config/arxiv_node_classification_GAT_100K.json
#python main_node_classification.py --config ./config/arxiv_node_classification_GatedGCN_100K.json
python main_node_classification.py --config ./config/arxiv_node_classification_GCN_100K.json
python main_node_classification.py --config ./config/arxiv_node_classification_GIN_100K.json
python main_node_classification.py --config ./config/arxiv_node_classification_GraphSage_100K.json
#python main_node_classification.py --config ./config/arxiv_node_classification_MLP_100K.json
python main_node_classification.py --config ./config/arxiv_node_classification_MoNet_100K.json
