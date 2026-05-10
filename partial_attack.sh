#!/bin/bash
# partial_attack.sh
# =================
# Same 5-positional-arg interface as PGL's partial_attack.sh.
#
# $1  DATASET       cola | sst2 | rotten_tomatoes
# $2  BATCH_SIZE    e.g. 1
# $3  MODEL_NAME    HF repo-id or local path, e.g.:
#                     openai-community/gpt2-large
#                     bert-base-uncased
#                     /abs/path/to/model
#                     gpt/gpt2-large   (resolved via DUALGUARD_MODEL_PATH)
# $4  GRAD_TYPE     layer_encoder | attn_query | attn_key | attn_value |
#                   attn_qkv | attn_output | ffn_fc | ffn_output |
#                   word_emb | encoder | all_layers
# $5  ATTACK_LAYER  0 | 0,1,2 | all
# $6+ extra flags   e.g. --aux_dataset piqa --sip_epochs 3
#
# Examples:
#   ./partial_attack.sh cola 1 openai-community/gpt2-large all_layers all
#   ./partial_attack.sh sst2 1 bert-base-uncased            layer_encoder 0
#   ./partial_attack.sh sst2 1 bert-base-uncased            attn_qkv 0,1,2,3 --aux_dataset piqa
#   ./partial_attack.sh cola 1 meta-llama/Llama-3.2-1B      ffn_output all

array=( "$@" )
len=${#array[@]}
last_args="${array[@]:5:$len}"

python3 partial_attack.py \
    --dataset      "$1" \
    --split        test \
    --n_inputs     10 \
    -b             "$2" \
    --model_name   "$3" \
    --grad_type    "$4" \
    --attack_layer "$5" \
    --sip_epochs   2 \
    --sip_batch_size 4 \
    --aux_dataset  wikitext \
    --save_dir     "./results/${1}_${4}_layer${5}" \
    $last_args
