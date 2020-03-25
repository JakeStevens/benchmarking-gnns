#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_mol_opt
# tmux detach
# pkill python

# bash run_eval_molecules_graph_regression_ZINC.sh


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
# ZINC
############

seed0=41
code=eval_molecules_graph_regression.py 
dataset=ZINC
tmux new -s benchmark_molecules_graph_regression -d

tmux send-keys "conda activate benchmark_gnn" C-m

nets=(GIN GCN GraphSage GatedGCN GAT DiffPool MoNet)
for net in ${nets[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_${net}_ZINC.json' &
wait" C-m
done

tmux send-keys "tmux kill-session -t benchmark_molecules_graph_regression" C-m
