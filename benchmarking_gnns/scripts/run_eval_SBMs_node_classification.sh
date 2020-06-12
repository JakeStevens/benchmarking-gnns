#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_mol_opt
# tmux detach
# pkill python

# bash run_eval_SBMs_node_classification_CLUSTER.sh




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
# SBM
############

seed0=41
code=eval_SBMs_node_classification.py 
tmux new -s benchmark_SBMs_node_classification -d
tmux send-keys "conda activate benchmark_gnn" C-m
datasets=(SBM_CLUSTER SBM_PATTERN)
nets=(GIN GCN GraphSage GatedGCN GAT DiffPool MoNet)
for net in ${nets[@]}; do
    for dataset in ${datasets[@]}; do
      short=${dataset:4}
      tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_${net}_${short}.json' &
wait" C-m
    done
done
tmux send-keys "tmux kill-session -t benchmark_SBMs_node_classification" C-m

