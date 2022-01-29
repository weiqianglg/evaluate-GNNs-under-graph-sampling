#!/bin/bash
export LD_LIBRARY_PATH="/home/weiq/anaconda3/lib"
export CUDA_VISIBLE_DEVICES="0"
python main_node_classification.py --config ./config/node_classification_gated.json --ratio 0.5 --sample bfs 0 --dataset Actor

