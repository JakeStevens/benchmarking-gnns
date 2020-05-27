#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_cit_graph
# tmux detach
# pkill python

# bash run_all_eval.sh


############
# GNNs
############

#GCN
#GraphSage
#GAT


###############
# Start Session
###############
tmux new -s benchmark_all_eval -d
tmux send-keys "conda activate gnns" C-m


###############
# Citation Graphs
#################
code=eval_CitationGraphs_node_classification.py 
datasets=(CORA CITESEER PUBMED)
nets=(GCN GraphSage GAT )
for dataset in ${datasets[@]}; do
    for net in ${nets[@]}; do
        tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --config 'configs/CitationGraphs_node_classification_$net.json' --epochs 100 --batch_norm False --graph_norm False &
wait" C-m
    done
done


############
# ZINC
############

seed0=41
code=eval_molecules_graph_regression.py 
dataset=ZINC

nets=(GIN GCN GraphSage GatedGCN GAT DiffPool) # MoNet)
for net in ${nets[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/molecules_graph_regression_${net}_ZINC.json' --epochs 100 --batch_norm False --graph_norm False &
wait" C-m
done


############
# SBM
############

seed0=41
code=eval_SBMs_node_classification.py 
datasets=(SBM_CLUSTER SBM_PATTERN)
nets=(GIN GCN GraphSage GatedGCN GAT DiffPool) # MoNet)
for net in ${nets[@]}; do
    for dataset in ${datasets[@]}; do
      short=${dataset:4}
      tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_${net}_${short}.json' --epochs 10 --batch_norm False --graph_norm False &
wait" C-m
    done
done


############
# superpixel  
############

seed0=41
code=eval_superpixels_graph_classification.py 
datasets=(CIFAR10 MNIST)
nets=(GIN GCN GraphSage GatedGCN GAT DiffPool) # MoNet)
for net in ${nets[@]}; do
    for dataset in ${datasets[@]}; do
      tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/superpixels_graph_classification_${net}_${dataset}.json' --epochs 10 --batch_norm False --graph_norm False &
wait" C-m
    done
done


############
# TSP
############

seed0=41
code=eval_TSP_edge_classification.py 
dataset=TSP
nets=(GIN GCN GraphSage GatedGCN GAT DiffPool) # MoNet)
for net in ${nets[@]}; do
    tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TSP_edge_classification_${net}.json' --epochs 10 --batch_norm False --graph_norm False &
wait" C-m
    done



############
# TUs
############


seed0=41
code=eval_TUs_graph_classification.py 
datasets=(ENZYMES DD PROTEINS_full)
nets=(GIN GCN GraphSage GatedGCN GAT DiffPool) # MoNet)
for net in ${nets[@]}; do
    for dataset in ${datasets[@]}; do
      tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/TUs_graph_classification_${net}_${dataset}.json' --epochs 100 --batch_norm False --graph_norm False &
wait" C-m
    done
done



##############
# End Session
##############
tmux send-keys "tmux kill-session -t benchmark_all_eval" C-m
