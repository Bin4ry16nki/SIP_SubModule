"""
layerwise_sip.py  —  dualguard/attack/layerwise_sip.py
=======================================================
Replaces gradient-inversion (PGL) with SIP forward-inversion (DualGuard).

Supported model families
------------------------
  GPT-2   (GPT2LMHeadModel)
  LLaMA   (LlamaForCausalLM)
  Qwen-2  (Qwen2ForCausalLM)
  BERT    (BertModel / BertForSequenceClassification / BertForMaskedLM)

grad_type vocabulary  (identical to PGL)
-----------------------------------------
  layer_encoder  – full transformer block output
  attn_query     – Q-projection output
  attn_key       – K-projection output
  attn_value     – V-projection output
  attn_qkv       – combined Q+K+V
  attn_output    – attention output projection
  ffn_fc         – FFN first linear
  ffn_output     – FFN second linear / down_proj
  word_emb       – embedding layer output  (maps to PGL word_emb)
  encoder        – layer_encoder for every layer
  all_layers     – encoder + word_emb
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertModel,
    GPT2LMHeadModel,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)

from dualguard.attack.sip import GRUDRInverter
from dualguard.utils.model import calc_unshift_loss, evaluate_attacker_rouge, get_embed_size

log = logging.getLogger(__name__)

GRAD_TYPES = [
    "layer_encoder", "attn_query", "attn_key", "attn_value",
    "attn_qkv", "attn_output", "ffn_fc", "ffn_output",
    "word_emb", "encoder", "all_layers",
]


# ── model-family detection ─────────────────────────────────────────────────

def _is_gpt2(m):       return isinstance(m, GPT2LMHeadModel)
def _is_llama_qwen(m): return isinstance(m, (LlamaForCausalLM, Qwen2ForCausalLM))
def _is_bert(m):       return isinstance(m, (BertModel,
                                              BertForSequenceClassification,
                                              BertForMaskedLM))

def _num_layers(model: nn.Module) -> int:
    if _is_gpt2(model):       return model.config.n_layer
    if _is_bert(model):       return model.config.num_hidden_layers
    return model.config.num_hidden_layers  # LLaMA / Qwen

def _hidden_size(model: nn.Module) -> int:
    return get_embed_size(model.config)

def _layer_list(model: nn.Module) -> nn.ModuleList:
    if _is_gpt2(model):                    return model.transformer.h
    if _is_llama_qwen(model):              return model.model.layers
    # BERT: BertForSequenceClassification wraps BertModel
    bert = getattr(model, 'bert', model)   # unwrap if BertForSequenceClassification
    return bert.encoder.layer


# ── hook helpers ────────────────────────────────────────────────────────────

def _basic_hook(store: dict, key: str):
    def fn(module, inp, output):
        val = output[0] if isinstance(output, (tuple, list)) else output
        store[key] = val.detach().float()
    return fn


class _GPT2QKVSlice:
    """GPT-2 c_attn → (B,S,3H): slice out q/k/v third."""
    def __init__(self, store, key, which, H):
        self.store, self.key, self.which, self.H = store, key, which, H
    def __call__(self, module, inp, output):
        out = (output[0] if isinstance(output, (tuple,list)) else output).detach().float()
        H   = self.H
        {"q": lambda: out[..., :H],
         "k": lambda: out[..., H:2*H],
         "v": lambda: out[..., 2*H:]}
        self.store[self.key] = {"q": out[..., :H],
                                "k": out[..., H:2*H],
                                "v": out[..., 2*H:]}[self.which]


class _MeanQKV:
    """LLaMA/Qwen: accumulate q,k,v and store mean once all three fire."""
    def __init__(self, store, key, part, acc):
        self.store, self.key, self.part, self.acc = store, key, part, acc
    def __call__(self, module, inp, output):
        val = (output[0] if isinstance(output,(tuple,list)) else output).detach().float()
        self.acc[self.part] = val
        if all(p in self.acc for p in ("q","k","v")):
            q, k, v = self.acc["q"], self.acc["k"], self.acc["v"]
            D = q.shape[-1]
            if k.shape[-1] != D: k = k.repeat_interleave(D//k.shape[-1], dim=-1)
            if v.shape[-1] != D: v = v.repeat_interleave(D//v.shape[-1], dim=-1)
            self.store[self.key] = (q+k+v)/3.0
            self.acc.clear()


# ── register one (grad_type, layer_idx) pair ───────────────────────────────

def _register_one(model: nn.Module, grad_type: str,
                  layer_idx: int, store: dict) -> List:
    key      = f"{grad_type}__layer{layer_idx}"
    handles  = []
    layers   = _layer_list(model)
    block    = layers[layer_idx]
    H        = _hidden_size(model)
    is_gpt2  = _is_gpt2(model)
    is_bert  = _is_bert(model)

    # ── full block output ──────────────────────────────────────────────
    if grad_type == "layer_encoder":
        handles.append(block.register_forward_hook(_basic_hook(store, key)))
        return handles

    # ── embedding ──────────────────────────────────────────────────────
    if grad_type == "word_emb":
        if is_gpt2:
            handles.append(model.transformer.drop.register_forward_hook(
                _basic_hook(store, key)))
        elif is_bert:
            bert = getattr(model, 'bert', model)
            handles.append(bert.embeddings.register_forward_hook(
                _basic_hook(store, key)))
        else:
            handles.append(model.model.embed_tokens.register_forward_hook(
                _basic_hook(store, key)))
        return handles

    # ── GPT-2 specific ─────────────────────────────────────────────────
    if is_gpt2:
        if grad_type in ("attn_query", "attn_key", "attn_value"):
            which = {"attn_query":"q","attn_key":"k","attn_value":"v"}[grad_type]
            handles.append(block.attn.c_attn.register_forward_hook(
                _GPT2QKVSlice(store, key, which, H)))
        elif grad_type == "attn_qkv":
            handles.append(block.attn.c_attn.register_forward_hook(
                _basic_hook(store, key)))          # full 3H tensor
        elif grad_type == "attn_output":
            handles.append(block.attn.c_proj.register_forward_hook(
                _basic_hook(store, key)))
        elif grad_type == "ffn_fc":
            handles.append(block.mlp.c_fc.register_forward_hook(
                _basic_hook(store, key)))
        elif grad_type == "ffn_output":
            handles.append(block.mlp.c_proj.register_forward_hook(
                _basic_hook(store, key)))
        return handles

    # ── BERT specific ──────────────────────────────────────────────────
    # BERT block = BertLayer:
    #   attention.self.{query, key, value}  (Linear)
    #   attention.output.dense              (Linear)
    #   intermediate.dense                  (Linear)  ← ffn_fc
    #   output.dense                        (Linear)  ← ffn_output
    if is_bert:
        attn = block.attention
        if grad_type == "attn_query":
            handles.append(attn.self.query.register_forward_hook(
                _basic_hook(store, key)))
        elif grad_type == "attn_key":
            handles.append(attn.self.key.register_forward_hook(
                _basic_hook(store, key)))
        elif grad_type == "attn_value":
            handles.append(attn.self.value.register_forward_hook(
                _basic_hook(store, key)))
        elif grad_type == "attn_qkv":
            acc: Dict = {}
            for part, proj in [("q", attn.self.query),
                                ("k", attn.self.key),
                                ("v", attn.self.value)]:
                handles.append(proj.register_forward_hook(
                    _MeanQKV(store, key, part, acc)))
        elif grad_type == "attn_output":
            handles.append(attn.output.dense.register_forward_hook(
                _basic_hook(store, key)))
        elif grad_type == "ffn_fc":
            handles.append(block.intermediate.dense.register_forward_hook(
                _basic_hook(store, key)))
        elif grad_type == "ffn_output":
            handles.append(block.output.dense.register_forward_hook(
                _basic_hook(store, key)))
        return handles

    # ── LLaMA / Qwen ───────────────────────────────────────────────────
    if grad_type in ("attn_query", "attn_key", "attn_value"):
        which = {"attn_query":"q","attn_key":"k","attn_value":"v"}[grad_type]
        proj  = {"q": block.self_attn.q_proj,
                 "k": block.self_attn.k_proj,
                 "v": block.self_attn.v_proj}[which]
        handles.append(proj.register_forward_hook(_basic_hook(store, key)))
    elif grad_type == "attn_qkv":
        acc: Dict = {}
        for part, proj in [("q", block.self_attn.q_proj),
                            ("k", block.self_attn.k_proj),
                            ("v", block.self_attn.v_proj)]:
            handles.append(proj.register_forward_hook(
                _MeanQKV(store, key, part, acc)))
    elif grad_type == "attn_output":
        handles.append(block.self_attn.o_proj.register_forward_hook(
            _basic_hook(store, key)))
    elif grad_type == "ffn_fc":
        handles.append(block.mlp.gate_proj.register_forward_hook(
            _basic_hook(store, key)))
    elif grad_type == "ffn_output":
        handles.append(block.mlp.down_proj.register_forward_hook(
            _basic_hook(store, key)))
    return handles


def register_hooks(model: nn.Module, grad_type: str,
                   layer_idxs: List[int], store: dict) -> List:
    handles = []
    if grad_type in ("encoder", "all_layers"):
        for idx in layer_idxs:
            handles += _register_one(model, "layer_encoder", idx, store)
        if grad_type == "all_layers":
            handles += _register_one(model, "word_emb", 0, store)
    else:
        for idx in layer_idxs:
            handles += _register_one(model, grad_type, idx, store)
    return handles


def remove_hooks(handles: List) -> None:
    for h in handles:
        h.remove()


# ── activation size at a given capture point ───────────────────────────────

def _act_size(model: nn.Module, grad_type: str) -> int:
    H = _hidden_size(model)
    # GPT-2 c_attn fuses QKV → 3H
    if _is_gpt2(model) and grad_type == "attn_qkv":
        return 3 * H
    # BERT intermediate is 4H by default
    if _is_bert(model) and grad_type == "ffn_fc":
        return getattr(model.config, 'intermediate_size', 4 * H)
    return H


# ── forward pass helper: works for both causal LMs and BERT ────────────────

def _forward(model: nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> None:
    """Run a forward pass without gradients, populating hook stores."""
    with torch.no_grad():
        if _is_bert(model):
            bert = getattr(model, 'bert', model)
            bert(input_ids=ids, attention_mask=mask)
        else:
            model(input_ids=ids, attention_mask=mask)


# ── train one GRUDRInverter ─────────────────────────────────────────────────

def train_inverter(model, tokenizer, grad_type, layer_idx,
                   store, train_loader, epochs, gru_hidden, gru_bidir,
                   device) -> GRUDRInverter:
    key       = f"{grad_type}__layer{layer_idx}"
    act_size  = _act_size(model, grad_type)
    vocab_size = model.config.vocab_size

    inverter = GRUDRInverter(n_embed=act_size, vocab_size=vocab_size,
                              hidden_size=gru_hidden,
                              bidirectional=gru_bidir).to(device)
    opt = torch.optim.AdamW(inverter.parameters(), lr=1e-3, weight_decay=1e-5)
    model.eval(); inverter.train()

    total = epochs * len(train_loader)
    with tqdm(total=total, desc=f"  train [{key}]", leave=False) as pbar:
        for _ in range(epochs):
            for batch in train_loader:
                opt.zero_grad()
                ids  = batch["input"].to(device)
                mask = batch["attention_mask"].to(device)
                _forward(model, ids, mask)
                if key not in store:
                    pbar.update(1); continue
                logits = inverter(store[key])
                loss   = calc_unshift_loss(logits, ids)
                loss.backward()
                opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)
    return inverter


# ── evaluate one GRUDRInverter ──────────────────────────────────────────────

def eval_inverter(model, tokenizer, inverter, grad_type, layer_idx,
                  store, eval_loader, device, sample_rate=1.0) -> dict:
    key = f"{grad_type}__layer{layer_idx}"
    model.eval(); inverter.eval()

    rouge_s = meteor_s = tacc_s = loss_s = 0.0
    n_steps = max(1, int(len(eval_loader) * sample_rate))
    done    = 0

    with tqdm(total=n_steps, desc=f"  eval  [{key}]", leave=False) as pbar:
        for batch in eval_loader:
            if done >= n_steps: break
            ids  = batch["input"].to(device)
            mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                _forward(model, ids, mask)
                if key not in store:
                    done += 1; pbar.update(1); continue
                logits = inverter(store[key])

            loss_s  += calc_unshift_loss(logits, ids).item()
            res, meteor, tacc = evaluate_attacker_rouge(
                tokenizer, logits, batch, device=device)
            rouge_s  += res["rouge-l"]["f"]
            meteor_s += meteor
            tacc_s   += tacc
            done += 1
            pbar.set_postfix(rouge=f"{rouge_s/done:.3f}")
            pbar.update(1)

    d = max(done, 1)
    return {"rouge_l_f1": rouge_s/d, "meteor": meteor_s/d,
            "token_acc": tacc_s/d,   "loss":   loss_s/d}


# ── activation store with optional defence ─────────────────────────────────

def _make_store(noise: Optional[float], pct_mask: Optional[float]) -> dict:
    if noise is None and pct_mask is None:
        return {}

    class _DefStore(dict):
        def __setitem__(self, k, v):
            if isinstance(v, torch.Tensor):
                if pct_mask is not None:
                    v = v * (torch.rand_like(v) > pct_mask).float()
                if noise is not None:
                    v = v + torch.randn_like(v) * noise
            super().__setitem__(k, v)

    return _DefStore()


# ── top-level sweep ─────────────────────────────────────────────────────────

def run_layerwise_sip(
    model:            nn.Module,
    tokenizer:        AutoTokenizer,
    train_loader:     DataLoader,
    eval_loader:      DataLoader,
    grad_type:        str,
    layer_idxs:       Optional[List[int]],
    epochs:           int,
    gru_hidden:       int,
    gru_bidir:        bool,
    device:           str,
    save_dir:         str,
    sample_rate:      float = 1.0,
    save_inverters:   bool  = False,
    defense_noise:    Optional[float] = None,
    defense_pct_mask: Optional[float] = None,
    run_log:          Optional[logging.Logger] = None,
) -> Dict[str, dict]:
    if run_log is None: run_log = log

    model.to(device); model.eval()
    os.makedirs(save_dir, exist_ok=True)

    n = _num_layers(model)
    if layer_idxs is None:
        layer_idxs = list(range(n))
    else:
        layer_idxs = [i for i in layer_idxs if 0 <= i < n]

    # Expand meta-types
    if grad_type in ("encoder", "all_layers"):
        pairs = [("layer_encoder", idx) for idx in layer_idxs]
        if grad_type == "all_layers":
            pairs = [("word_emb", 0)] + pairs
    else:
        pairs = [(grad_type, idx) for idx in layer_idxs]

    store   = _make_store(defense_noise, defense_pct_mask)
    handles = []
    for gt, idx in pairs:
        handles += _register_one(model, gt, idx, store)

    results: dict = {}
    for gt, idx in tqdm(pairs, desc=f"SIP sweep [{grad_type}]"):
        key = f"{gt}__layer{idx}"
        run_log.info(f"[SIP] Training inverter: {key}")

        inverter = train_inverter(model, tokenizer, gt, idx, store,
                                  train_loader, epochs, gru_hidden, gru_bidir, device)
        if save_inverters:
            torch.save(inverter.state_dict(),
                       os.path.join(save_dir, f"inverter_{key}.pt"))

        metrics = eval_inverter(model, tokenizer, inverter, gt, idx,
                                store, eval_loader, device, sample_rate)
        results[key] = metrics
        run_log.info(
            f"[SIP] {key:42s}  rouge-L={metrics['rouge_l_f1']:.4f}"
            f"  meteor={metrics['meteor']:.4f}"
            f"  token_acc={metrics['token_acc']:.4f}")

        _write_json(results, os.path.join(save_dir, "results.json"))
        del inverter
        torch.cuda.empty_cache()

    remove_hooks(handles)
    return results


# ── I/O ─────────────────────────────────────────────────────────────────────

def _write_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def print_results_table(results: dict) -> None:
    hdr = f"{'Capture point':<42} {'Rouge-L':>9} {'METEOR':>9} {'Token-Acc':>10} {'Loss':>9}"
    sep = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for k in sorted(results):
        m = results[k]
        print(f"{k:<42} {m['rouge_l_f1']*100:>8.2f}%"
              f" {m['meteor']*100:>8.2f}%"
              f" {m['token_acc']*100:>9.2f}%"
              f" {m['loss']:>9.4f}")
    print(sep + "\n")
