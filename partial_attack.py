"""
partial_attack.py  —  sip_layerwise/partial_attack.py
======================================================
Main entry point.  max_seq_len is derived automatically from the loaded
tokenizer so no manual flag is needed.
"""

import datetime, json, os, sys, time
import numpy as np
import torch

from args_factory import get_args
from data_utils import get_eval_loader, get_train_loader
from dualguard.attack.layerwise_sip import run_layerwise_sip, print_results_table
from dualguard.utils.model import (
    load_model_and_tokenizer, load_bert_model_and_tokenizer, set_random_seed)


def _is_bert_name(name: str) -> bool:
    return "bert" in name.lower() or "tinybert" in name.lower()


def print_metrics(results: dict, label: str = "Aggregate") -> None:
    print(f"\n[{label} metrics]", flush=True)
    for k in sorted(results):
        m = results[k]
        print(f"  {k:<42} | "
              f"rouge-L={m['rouge_l_f1']*100:6.2f}% | "
              f"meteor={m['meteor']*100:6.2f}% | "
              f"token_acc={m['token_acc']*100:6.2f}% | "
              f"loss={m['loss']:.4f}", flush=True)


def main():
    print("\n\n\nCommand:", " ".join(sys.argv), "\n\n\n", flush=True)

    args = get_args()
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    set_random_seed(args.rng_seed)

    # ── load model + tokenizer ─────────────────────────────────────────────
    print(f"Loading model: {args.model_name}", flush=True)
    if _is_bert_name(args.model_name):
        model, tokenizer = load_bert_model_and_tokenizer(
            args.model_name, num_labels=args.bert_num_labels)
    else:
        model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.eval()

    # max_seq_len is derived once from the tokenizer and reused everywhere
    from data_utils import _model_max_len
    max_seq_len = _model_max_len(tokenizer)
    print(f"Model max sequence length: {max_seq_len}", flush=True)

    # ── data loaders ──────────────────────────────────────────────────────
    print(f"Building eval loader  : {args.dataset}/{args.split}", flush=True)
    eval_loader = get_eval_loader(
        tokenizer, args.dataset, args.split,
        args.n_inputs, args.batch_size,
        max_seq_len=max_seq_len, rng_seed=args.rng_seed)

    print(f"Building train loader : {args.aux_dataset}", flush=True)
    train_loader = get_train_loader(
        args.aux_dataset, tokenizer, args.sip_batch_size,
        max_seq_len=max_seq_len)

    # ── run sweep ─────────────────────────────────────────────────────────
    print("\n\nAttacking..\n", flush=True)
    t0 = time.time()

    results = run_layerwise_sip(
        model            = model,
        tokenizer        = tokenizer,
        train_loader     = train_loader,
        eval_loader      = eval_loader,
        grad_type        = args.grad_type,
        layer_idxs       = args.attack_layer,
        epochs           = args.sip_epochs,
        gru_hidden       = args.gru_hidden,
        gru_bidir        = args.gru_bidirectional,
        device           = args.device,
        save_dir         = args.save_dir,
        sample_rate      = args.sample_rate,
        save_inverters   = args.save_inverters,
        defense_noise    = args.defense_noise,
        defense_pct_mask = args.defense_pct_mask,
    )

    elapsed = str(datetime.timedelta(seconds=int(time.time()-t0)))
    print(f"\nDone.  Total time: {elapsed}", flush=True)
    print_results_table(results)
    print_metrics(results)
    print(f"Results written to: {os.path.join(args.save_dir, 'layerwise_sip_results.json')}",
          flush=True)


if __name__ == "__main__":
    main()
