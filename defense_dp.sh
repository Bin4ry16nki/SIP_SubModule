#!/bin/bash
# defense_dp.sh
# =============
# Same 7-positional-arg interface as PGL's defense_dp.sh.
#
# $1  DATASET       $2  BATCH_SIZE  $3  MODEL_NAME
# $4  GRAD_TYPE     $5  ATTACK_LAYER
# $6  NOISE         noise_multiplier (e.g. 0.01 to 0.5)
# $7  CLIPPING      clipping_bound   (default 1.0)
# $8+ extra flags
#
# Examples:
#   ./defense_dp.sh sst2 1 openai-community/gpt2-large all_layers all 0.01 1.0
#   ./defense_dp.sh cola 1 bert-base-uncased            layer_encoder 0 0.1 1.0

array=( "$@" )
len=${#array[@]}
last_args="${array[@]:7:$len}"

python3 dpsgd_defense.py \
    --dataset          "$1" \
    --split            test \
    --n_inputs         10 \
    -b                 "$2" \
    --model_name       "$3" \
    --grad_type        "$4" \
    --attack_layer     "$5" \
    --noise_multiplier "$6" \
    --clipping_bound   "$7" \
    --sip_epochs       2 \
    --sip_batch_size   4 \
    --aux_dataset      wikitext \
    --save_dir         "./results/dp_${1}_${4}_layer${5}_noise${6}" \
    $last_args
