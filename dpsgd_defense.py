"""
dpsgd_defense.py  —  sip_layerwise/dpsgd_defense.py
====================================================
DP defence variant.  Applies L2-clip + Gaussian noise to captured
activations before the GRU inverter sees them.
"""

import datetime, json, os, sys, time
import numpy as np
import torch

from args_factory import get_args
from data_utils import get_eval_loader, get_train_loader, _model_max_len
from dualguard.attack.layerwise_sip import run_layerwise_sip, print_results_table
from dualguard.utils.model import (
    load_model_and_tokenizer, load_bert_model_and_tokenizer, set_random_seed)


def _is_bert_name(name: str) -> bool:
    return "bert" in name.lower() or "tinybert" in name.lower()


def print_metrics(results: dict, label: str) -> None:
    print(f"\n[{label}]", flush=True)
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

    noise_std = args.noise_multiplier * args.clipping_bound
    print(f"DP defence  noise_multiplier={args.noise_multiplier}"
          f"  clipping_bound={args.clipping_bound}"
          f"  => activation noise std={noise_std:.4f}", flush=True)

    if _is_bert_name(args.model_name):
        model, tokenizer = load_bert_model_and_tokenizer(
            args.model_name, num_labels=args.bert_num_labels)
    else:
        model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.eval()

    max_seq_len = _model_max_len(tokenizer)
    print(f"Model max sequence length: {max_seq_len}", flush=True)

    eval_loader  = get_eval_loader(
        tokenizer, args.dataset, args.split,
        args.n_inputs, args.batch_size,
        max_seq_len=max_seq_len, rng_seed=args.rng_seed)
    train_loader = get_train_loader(
        args.aux_dataset, tokenizer, args.sip_batch_size,
        max_seq_len=max_seq_len)

    save_dir = os.path.join(
        args.save_dir,
        f"dp_{args.dataset}_{args.grad_type}_layer{args.attack_layer}_noise{args.noise_multiplier}")
    os.makedirs(save_dir, exist_ok=True)

    print("\n\nAttacking with DP defence..\n", flush=True)
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
        save_dir         = save_dir,
        sample_rate      = args.sample_rate,
        save_inverters   = False,
        defense_noise    = noise_std,
        defense_pct_mask = args.defense_pct_mask,
    )

    elapsed = str(datetime.timedelta(seconds=int(time.time()-t0)))
    print(f"\nDone. Total time: {elapsed}", flush=True)
    print_results_table(results)
    print_metrics(results, label=f"DP Results (sigma={noise_std:.4f})")

    out = os.path.join(save_dir, "dpsgd_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {out}", flush=True)


if __name__ == "__main__":
    main()
