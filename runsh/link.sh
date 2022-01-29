#!/bin/bash
export LD_LIBRARY_PATH="/home/weiq/anaconda3/lib"
export CUDA_VISIBLE_DEVICES="1"


python main_edge_classification.py --config ./config/COLLAB_edge_classification_GAT_40k.json --sample ffs 0 --ratio 0.3
