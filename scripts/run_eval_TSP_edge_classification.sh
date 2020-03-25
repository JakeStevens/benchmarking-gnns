#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_mol_opt
# tmux detach
# pkill python

# bash run_eval_TSP_edge_classification.sh




############
# GNNs
############

#GatedGCN
#GCN
#GraphSage
#MLP
#GIN
#MoNet
#GAT
#DiffPool





############
# TSP
############

seed0=41
code=eval_TSP_edge_classification.py 
tmux new -s benchmark_TSP_edge_classification -d
tmux send-keys "conda activate benchmark_gnn" C-m
dataset=TSP
nets=(GIN GCN GraphSage GatedGCN GAT DiffPool MoNet)
for net in ${nets[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_${net}.json' &
wait" C-m
    done
tmux send-keys "tmux kill-session -t benchmark_TSP_edge_classification" C-m
