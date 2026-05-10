import os
import random
from typing import Tuple
import numpy as np
import torch
from rouge import Rouge
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BertForSequenceClassification, BertTokenizer,
                          PreTrainedModel, PretrainedConfig)
from trl import DPOTrainer
import nltk
from nltk.translate import meteor_score
from dualguard.utils import env

# ── NLTK wordnet path ──────────────────────────────────────────────────────
nltk.data.path.append(env.nltk_path)

# ── model path root — empty string means "treat name as HF repo id" ────────
MODEL_PATH = env.model_path   # may be ""


def set_random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def get_embed_size(target_config: PretrainedConfig):
    for attr in ('n_embd', 'hidden_size', 'd_model'):
        if hasattr(target_config, attr):
            return getattr(target_config, attr)
    return 0


# ── model loading ──────────────────────────────────────────────────────────

def _resolve_path(model_name: str) -> str:
    """
    Return a path or HF repo-id suitable for from_pretrained().

    Rules (in order):
      1. If model_name is already an absolute path that exists → use as-is.
      2. If MODEL_PATH is set → join MODEL_PATH/model_name (local weights).
      3. Otherwise → treat model_name as a HuggingFace repo id
         (e.g. "openai-community/gpt2-large", "bert-base-uncased").
    """
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name
    if MODEL_PATH:
        candidate = os.path.join(MODEL_PATH, model_name)
        if os.path.exists(candidate):
            return candidate
        # If the joined path does not exist but MODEL_PATH was explicitly set,
        # still try it (from_pretrained will give a clear error).
        return candidate
    # No MODEL_PATH → HF Hub repo id
    return model_name


def load_model_and_tokenizer(model_name: str = "openai-community/gpt2-large",
                              ) -> Tuple[PreTrainedModel, AutoTokenizer]:
    path = _resolve_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager")
    return model, tokenizer


def load_bert_model_and_tokenizer(model_name: str = "bert-base-uncased",
                                  num_labels: int = 2,
                                  ) -> Tuple[PreTrainedModel, BertTokenizer]:
    """
    Load a BERT model for sequence classification + its tokenizer.
    model_name may be a HF repo id or a local path.
    """
    path = _resolve_path(model_name)
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    return model, tokenizer


def load_model(model_name: str = "openai-community/gpt2-large") -> PreTrainedModel:
    path = _resolve_path(model_name)
    return AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager")


def load_tokenizer(model_name: str = "openai-community/gpt2-large") -> AutoTokenizer:
    path = _resolve_path(model_name)
    tok = AutoTokenizer.from_pretrained(path)
    tok.pad_token = tok.eos_token
    return tok


def list_models(keyword: str = None):
    if not MODEL_PATH or not os.path.isdir(MODEL_PATH):
        return []
    model_paths = []
    for prefix in os.listdir(MODEL_PATH):
        p = os.path.join(MODEL_PATH, prefix)
        if os.path.isdir(p):
            for name in os.listdir(p):
                model_paths.append(os.path.join(prefix, name))
    if keyword:
        return [x for x in model_paths if keyword.lower() in x.lower()]
    return model_paths


# ── text decoding ──────────────────────────────────────────────────────────

def new_mask(input_tensor: torch.Tensor) -> torch.Tensor:
    result = input_tensor.clone()
    first_one_indices = (input_tensor == 1).float().argmax(dim=1)
    for i, idx in enumerate(first_one_indices):
        if (input_tensor[i] == 1).any():
            result[i, :idx] = 1
    return result


def decode_with_extra_space(tok, sent, mask):
    strs = []
    for idx, tid in enumerate(sent):
        if mask[idx] == 0:
            return ' '.join(strs)
        strs.append(tok.decode(tid, skip_special_tokens=True))
    return ' '.join(strs)


def decode_with_extra_space_evaluate(tok, sent, mask):
    strs = []
    for idx, tid in enumerate(sent):
        if mask[idx]:
            strs.append(tok.decode(tid, skip_special_tokens=True))
    return ' '.join(strs)


# ── evaluation metrics ─────────────────────────────────────────────────────

def evaluate_attacker_rouge(tok, attacker_logits, batch, **kwargs):
    device = kwargs.get('device', 'cuda')
    _msk  = batch['attention_mask'].to(device)
    mask  = new_mask(_msk).bool().to(device)
    _input = batch['input'].to(device)

    if 'input_santi_mask' in batch:
        mask_s = batch['input_santi_mask']
        zero_indexes = torch.where(mask_s == 0)
        masked_atk_ids = attacker_logits.argmax(dim=-1)
        masked_atk_ids[zero_indexes] = tok.unk_token_id
        masked_gt_ids = batch['input_ids'].clone()
        masked_gt_ids[zero_indexes] = tok.unk_token_id
        atk_txts = [decode_with_extra_space(tok, s, torch.ones(s.shape)) for s in masked_atk_ids]
        gt_txts  = [decode_with_extra_space(tok, s, torch.ones(s.shape)) for s in masked_gt_ids]
    else:
        atk_txts = [decode_with_extra_space(tok, s, m)
                    for s, m in zip(attacker_logits.argmax(dim=-1), mask)]
        gt_txts  = [decode_with_extra_space(tok, s, m)
                    for s, m in zip(_input, mask)]

    rouge      = calculate_rouge_text(atk_txts, gt_txts, print_comparison=False)
    meteor     = calculate_meteor(atk_txts, gt_txts)
    token_acc  = calculate_token_acc(tok, atk_txts, gt_txts)
    return rouge, meteor, token_acc


def calculate_token_acc(tok, texts, labels):
    f1_avg = 0
    for g, r in zip(texts, labels):
        gen_tok = set(tok.tokenize(g))
        ref_tok = set(tok.tokenize(r))
        overlap  = len(gen_tok & ref_tok)
        prec = overlap / len(gen_tok) if gen_tok else 0
        rec  = overlap / len(ref_tok) if ref_tok else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        f1_avg += f1
    return f1_avg / len(texts) if texts else 0


def calculate_meteor(texts, labels):
    total = 0
    for text, label in zip(texts, labels):
        total += meteor_score.meteor_score([label.split()], text.split())
    return total / len(texts) if texts else 0


def calculate_rouge_text(texts, labels, print_comparison=False):
    my_rouge = Rouge()
    hyps, refs = zip(*zip(texts, labels))
    if print_comparison:
        for h, r in zip(hyps, refs):
            print(f'{r}==>{h}')
    try:
        hyps   = ['<EMP>' if len(h) == 0 else h for h in hyps]
        result = my_rouge.get_scores(hyps, refs, avg=True, ignore_empty=True)
    except Exception as e:
        print(f'rouge error {e}')
        result = {k: {'f': 0.0, 'p': 0.0, 'r': 0.0}
                  for k in ('rouge-1', 'rouge-2', 'rouge-l')}
    return result


def calculate_rouge(tok, logits, labels, print_comparison=False, is_tokens=False):
    if not is_tokens:
        output_texts = [tok.decode(logits.argmax(dim=-1)[i], skip_special_tokens=True)
                        for i in range(len(logits))]
    else:
        output_texts = [tok.decode(s, skip_special_tokens=True) for s in logits]
    return calculate_rouge_text(output_texts, labels, print_comparison=print_comparison)


# ── loss functions ─────────────────────────────────────────────────────────

def calc_unshift_loss(lm_logits, labels):
    labels = labels.to(lm_logits.device)
    loss_fct = CrossEntropyLoss()
    return loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))


def calc_shifted_loss(lm_logits, labels):
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss()
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def calc_unshift_loss_logits(lm_logits, labels):
    labels = labels.to(lm_logits.device)
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    return loss_fct(lm_logits.view(-1, lm_logits.size(-1)),
                    labels.view(-1, labels.size(-1)))


# ── misc ───────────────────────────────────────────────────────────────────

def evaluate_attacker_mse(attacker_outputs, batch):
    return torch.nn.functional.mse_loss(attacker_outputs, batch)


def dist_corr(x, y):
    x = x.float(); y = y.float()
    n = x.size(0)
    a = torch.cdist(x, x); b = torch.cdist(y, y)
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov2 = (A * B).sum() / n**2
    dvar  = (A**2).sum() / n**2
    dvar2 = (B**2).sum() / n**2
    return dcov2 / (torch.sqrt(dvar) * torch.sqrt(dvar2))


def get_embedding_layer(llm):
    for path in [('model', 'embed_tokens'), ('transformer', 'wte'),
                 ('transformer', 'embedding'), ('transformer', 'word_embeddings'),
                 ('bert', 'embeddings', 'word_embeddings')]:
        obj = llm
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    for attr in ('embed_tokens', 'word_embeddings', 'embedding'):
        if hasattr(llm, attr):
            return getattr(llm, attr)
    return None


def get_vocab_size(config: PretrainedConfig):
    if hasattr(config, 'vocab_size'):
        return config.vocab_size
