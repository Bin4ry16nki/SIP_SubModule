# Layerwise SIP Privacy Analysis

Replaces gradient inversion ([partial-gradients-leakage](https://arxiv.org/abs/2406.00999))
with the SIP forward-inversion attack ([DualGuard](https://github.com/ZJUWalker/DualGuard)).

---

## Setup

```bash
conda create -n sip_layerwise python=3.12
conda activate sip_layerwise
pip install -e .
pip install rouge datasets nltk evaluate
python -c "import nltk; nltk.download('wordnet')"
```

No model files need to be manually downloaded.  Models are pulled from HuggingFace
Hub automatically.  To use local weights instead, set the environment variable:

```bash
export DUALGUARD_MODEL_PATH=/path/to/your/models
```

When set, `--model_name gpt/gpt2-large` resolves to `$DUALGUARD_MODEL_PATH/gpt/gpt2-large`.
When not set, `--model_name openai-community/gpt2-large` downloads from HF Hub directly.

---

## Quick start

```bash
# GPT-2  (HF Hub, no local path needed)
./partial_attack.sh cola 1 openai-community/gpt2-large layer_encoder 0

# BERT
./partial_attack.sh sst2 1 bert-base-uncased layer_encoder 0

# BERT with PIQA as the GRU training dataset
./partial_attack.sh sst2 1 bert-base-uncased attn_qkv 0,1,2,3 --aux_dataset piqa

# LLaMA
./partial_attack.sh cola 1 meta-llama/Llama-3.2-1B ffn_output all

# All layers, all grad_types (full sweep)
./layerwise_sweep.sh cola openai-community/gpt2-large cuda:0

# DP defence
./defense_dp.sh sst2 1 bert-base-uncased all_layers all 0.1 1.0
```

---

## Model support

| Model family | Example --model_name |
|---|---|
| GPT-2 | `openai-community/gpt2-large` |
| LLaMA | `meta-llama/Llama-3.2-1B` |
| Qwen-2 | `Qwen/Qwen2-1.5B` |
| BERT base | `bert-base-uncased` |
| BERT fine-tuned | `textattack/bert-base-uncased-CoLA` |
| TinyBERT | `huawei-noah/TinyBERT_General_4L_312D` |

---

## Aux training datasets for the GRU inverter

| --aux_dataset | Description |
|---|---|
| `wikitext` | WikiText-2 plain text (default) |
| `piqa` | PIQA question+solution pairs |

---

## grad_type vocabulary

| grad_type | Submodule captured | BERT parameter equivalent |
|---|---|---|
| `layer_encoder` | Full block output | `encoder.layer.N.*` |
| `attn_query` | Q-projection | `attention.self.query` |
| `attn_key` | K-projection | `attention.self.key` |
| `attn_value` | V-projection | `attention.self.value` |
| `attn_qkv` | Combined Q+K+V | all three above |
| `attn_output` | Attention output proj | `attention.output.dense` |
| `ffn_fc` | FFN first linear | `intermediate.dense` |
| `ffn_output` | FFN second linear | `output.dense` |
| `word_emb` | Embedding layer | `embeddings.word_embeddings` |
| `encoder` | All blocks (layer_encoder × N) | all encoder layers |
| `all_layers` | encoder + word_emb | all parameters |

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DUALGUARD_MODEL_PATH` | `""` (HF Hub) | Local model weights root |
| `DUALGUARD_DATA_PATH` | `""` (HF cache) | Local dataset root |
| `DUALGUARD_NLTK_PATH` | `~/nltk_data` | NLTK wordnet location |
| `DUALGUARD_PROJECT_ROOT` | auto-detected | Log and checkpoint dirs |

---

## Output

```
results/{dataset}_{grad_type}_layer{N}/
    results.json            # { "layer_encoder__layer0": {rouge_l_f1, meteor, token_acc, loss} }

sweep_results/{dataset}_{model}/
    {grad_type}/results.json
    all_results.json
    summary.csv
    summary_table.txt
```
