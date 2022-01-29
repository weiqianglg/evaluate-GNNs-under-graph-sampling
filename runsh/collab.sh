#!/bin/bash
export LD_LIBRARY_PATH="/home/weiq/anaconda3/lib"
export CUDA_VISIBLE_DEVICES="1"

#python main_edge_classification.py --config ./config/COLLAB_edge_classification_GAT_40k.json
#python main_edge_classification.py --config ./config/COLLAB_edge_classification_GatedGCN_40k.json
python main_edge_classification.py --config ./config/COLLAB_edge_classification_GCN_40k.json --sample bfs 0 --ratio 0.3

#python main_edge_classification.py --config ./config/COLLAB_edge_classification_GIN_40k.json
#python main_edge_classification.py --config ./config/COLLAB_edge_classification_GraphSage_40k.json
#python main_edge_classification.py --config ./config/COLLAB_edge_classification_MLP_40k.json --sample bfs 0 --ratio 0.3
#python main_edge_classification.py --config ./config/COLLAB_edge_classification_MoNet_40k.json
