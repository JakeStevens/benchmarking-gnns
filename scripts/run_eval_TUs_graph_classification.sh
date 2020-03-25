#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_mol_opt
# tmux detach
# pkill python

# bash run_eval_TUs_graph_classification.sh




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
# TUs
############

seed0=41
code=eval_TUs_graph_classification.py 
tmux new -s benchmark_TUs_graph_classification -d
tmux send-keys "conda activate benchmark_gnn" C-m
datasets=(ENZYMES DD PROTEINS_full)
nets=(GIN GCN GraphSage GatedGCN GAT DiffPool MoNet)
for net in ${nets[@]}; do
    for dataset in ${datasets[@]}; do
      tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TUs_graph_classification_${net}_${dataset}.json' &
wait" C-m
    done
done
tmux send-keys "tmux kill-session -t benchmark_TUs_graph_classification" C-m
