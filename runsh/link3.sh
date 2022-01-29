#!/bin/bash
export LD_LIBRARY_PATH="/home/weiq/anaconda3/lib"
export CUDA_VISIBLE_DEVICES="1"


python main_edge_classification.py --config ./config/node_classification_sage.json --dataset Actor --sample ffs 0 --ratio 0.3
python main_edge_classification.py --config ./config/node_classification_sage.json --dataset Cora --sample rw 0 --ratio 0.3