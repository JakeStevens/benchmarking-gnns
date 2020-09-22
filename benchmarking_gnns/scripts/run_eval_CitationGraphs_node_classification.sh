#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_cit_graph
# tmux detach
# pkill python

# bash run_eval_CitationGraphs_node_classification.sh


############
# GNNs
############

#GCN
#GraphSage
#GAT

code=eval_CitationGraphs_node_classification.py 
tmux new -s benchmark_CitationGraphs_node_classification -d
tmux send-keys "conda activate benchmark_gnn" C-m

datasets=(CORA CITESEER PUBMED)
#nets=(GCN GraphSage)
layers=(1 2 3)
hiddens=(16 128 1024)
for layer in ${layers[@]}; do
  for hidden in ${hiddens[@]}; do
    for dataset in ${datasets[@]}; do
      tmux send-keys "
nvidia-smi --query-gpu=power.draw --format=csv --loop-ms=10 > gcn-$dataset-$hidden-$layer.power &
smi=\$!
python $code --dataset $dataset --gpu_id 0 --L $layer --hidden_dim $hidden --config 'configs/CitationGraphs_node_classification_GCN.json' --epochs 2000
kill \$smi
wait" C-m

      tmux send-keys "      
nvidia-smi --query-gpu=power.draw --format=csv --loop-ms=10 > graphsage-$dataset-$hidden-$layer.power &
smi=\$!
python $code --dataset $dataset --gpu_id 0 --L $layer --hidden_dim $hidden --config 'configs/CitationGraphs_node_classification_GraphSage.json' --epochs 2000
kill \$smi
wait" C-m

      tmux send-keys "
nvidia-smi --query-gpu=power.draw --format=csv --loop-ms=10 > graphsage-max-$dataset-$hidden-$layer.power &
smi=\$!
python $code --dataset $dataset --gpu_id 0 --L $layer --hidden_dim $hidden --sage_aggregator pool --config 'configs/CitationGraphs_node_classification_GraphSage.json' --epochs 2000
kill \$smi
wait" C-m
    done
  done
done
tmux send-keys "tmux kill-session -t benchmark_CitationGraphs_node_classification" C-m
