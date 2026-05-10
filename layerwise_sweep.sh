#!/bin/bash
# layerwise_sweep.sh
# ==================
# Full (grad_type × layer) sweep for one model + dataset.
#
# $1  DATASET      cola | sst2 | rotten_tomatoes
# $2  MODEL_NAME   HF repo-id or local path
# $3  DEVICE       cuda:0 | cpu
# $4+ extra flags
#
# Examples:
#   ./layerwise_sweep.sh cola  openai-community/gpt2-large cuda:0
#   ./layerwise_sweep.sh sst2  bert-base-uncased            cuda:0 --aux_dataset piqa
#   ./layerwise_sweep.sh cola  meta-llama/Llama-3.2-1B      cuda:1

array=( "$@" )
len=${#array[@]}
last_args="${array[@]:3:$len}"

python3 layerwise_sweep.py \
    --dataset      "$1" \
    --split        test \
    --n_inputs     10 \
    -b             1 \
    --model_name   "$2" \
    --device       "$3" \
    --sip_epochs   2 \
    --sip_batch_size 4 \
    --aux_dataset  wikitext \
    --save_dir     "./sweep_results/${1}_$(echo $2 | tr '/' '_')" \
    $last_args
