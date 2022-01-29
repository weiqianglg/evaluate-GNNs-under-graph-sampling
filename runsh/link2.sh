#!/bin/bash
export LD_LIBRARY_PATH="/home/weiq/anaconda3/lib"
export CUDA_VISIBLE_DEVICES="0"


python main_edge_classification.py --config ./config/edge_classification_sage.json --dataset Pubmed --ratio 0.3 --sample bfs 0 ffs 0 mhrw 0 rw 0
python main_edge_classification.py --config ./config/edge_classification_gat.json --dataset Pubmed --ratio 0.3 --sample bfs 0 ffs 0 mhrw 0 rw 0
python main_edge_classification.py --config ./config/edge_classification_mlp.json --dataset Pubmed --ratio 0.3 --sample bfs 0 ffs 0 mhrw 0 rw 0

