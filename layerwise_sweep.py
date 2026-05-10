"""
layerwise_sweep.py  —  sip_layerwise/layerwise_sweep.py
========================================================
Full automated (grad_type × layer) sweep.
"""

import argparse, csv, datetime, io, json, os, sys, time
import numpy as np
import torch

from data_utils import get_eval_loader, get_train_loader, _model_max_len
from dualguard.attack.layerwise_sip import (
    GRAD_TYPES, run_layerwise_sip, print_results_table)
from dualguard.utils.model import (
    load_model_and_tokenizer, load_bert_model_and_tokenizer, set_random_seed)

SWEEP_TYPES = [
    "layer_encoder", "attn_query", "attn_key", "attn_value",
    "attn_qkv", "attn_output", "ffn_fc", "ffn_output",
]


def _is_bert_name(name: str) -> bool:
    return "bert" in name.lower() or "tinybert" in name.lower()


def get_args():
    p = argparse.ArgumentParser(description="Full layerwise SIP sweep")
    p.add_argument("--dataset",    required=True,
                   choices=["cola","sst2","rotten_tomatoes"])
    p.add_argument("--split",      default="test", choices=["val","test"])
    p.add_argument("--n_inputs",   type=int, default=10)
    p.add_argument("-b","--batch_size", type=int, default=1)
    p.add_argument("--model_name", default="openai-community/gpt2-large")
    p.add_argument("--bert_num_labels", type=int, default=2)
    p.add_argument("--aux_dataset",default="wikitext")
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--rng_seed",   type=int, default=101)
    p.add_argument("--sip_epochs", type=int, default=2)
    p.add_argument("--sip_batch_size", type=int, default=4)
    p.add_argument("--gru_hidden", type=int, default=256)
    p.add_argument("--gru_bidirectional",
                   type=lambda x: x.lower() not in ("false","0","no"),
                   default=True, metavar="BOOL")
    p.add_argument("--save_dir",   default="./sweep_results")
    p.add_argument("--save_inverters", action="store_true")
    p.add_argument("--sample_rate",type=float, default=1.0)
    p.add_argument("--grad_types", default=",".join(SWEEP_TYPES))
    p.add_argument("--attack_layers", default="all")
    p.add_argument("--defense_noise",    type=float, default=None)
    p.add_argument("--defense_pct_mask", type=float, default=None)
    return p.parse_args()


def main():
    args = get_args()
    np.random.seed(args.rng_seed)
    torch.manual_seed(args.rng_seed)
    set_random_seed(args.rng_seed)
    os.makedirs(args.save_dir, exist_ok=True)

    grad_types = [g.strip() for g in args.grad_types.split(",")]
    layer_idxs = (None if args.attack_layers.strip().lower() == "all"
                  else [int(x) for x in args.attack_layers.split(",")])

    print(f"Loading model: {args.model_name}", flush=True)
    if _is_bert_name(args.model_name):
        model, tokenizer = load_bert_model_and_tokenizer(
            args.model_name, num_labels=args.bert_num_labels)
    else:
        model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.eval()

    max_seq_len = _model_max_len(tokenizer)
    print(f"Model max sequence length: {max_seq_len}", flush=True)

    eval_loader  = get_eval_loader(
        tokenizer, args.dataset, args.split, args.n_inputs, args.batch_size,
        max_seq_len=max_seq_len, rng_seed=args.rng_seed)
    train_loader = get_train_loader(
        args.aux_dataset, tokenizer, args.sip_batch_size,
        max_seq_len=max_seq_len)

    all_results: dict = {}
    t_total = time.time()

    for gt in grad_types:
        print(f"\n{'='*60}\nSweeping grad_type = {gt}\n{'='*60}", flush=True)
        gt_dir = os.path.join(args.save_dir, gt)
        os.makedirs(gt_dir, exist_ok=True)

        t_gt = time.time()
        res  = run_layerwise_sip(
            model            = model,
            tokenizer        = tokenizer,
            train_loader     = train_loader,
            eval_loader      = eval_loader,
            grad_type        = gt,
            layer_idxs       = layer_idxs,
            epochs           = args.sip_epochs,
            gru_hidden       = args.gru_hidden,
            gru_bidir        = args.gru_bidirectional,
            device           = args.device,
            save_dir         = gt_dir,
            sample_rate      = args.sample_rate,
            save_inverters   = args.save_inverters,
            defense_noise    = args.defense_noise,
            defense_pct_mask = args.defense_pct_mask,
        )
        elapsed = str(datetime.timedelta(seconds=int(time.time()-t_gt)))
        print(f"grad_type={gt} done in {elapsed}", flush=True)

        with open(os.path.join(gt_dir, f"{gt}_results.json"), "w") as f:
            json.dump(res, f, indent=2)
        all_results.update(res)

    with open(os.path.join(args.save_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    csv_path = os.path.join(args.save_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "capture_key","grad_type","layer_idx",
            "rouge_l_f1","meteor","token_acc","loss"])
        w.writeheader()
        for key, m in sorted(all_results.items()):
            parts = key.split("__layer")
            w.writerow({"capture_key": key,
                        "grad_type":   parts[0],
                        "layer_idx":   parts[1] if len(parts)>1 else "N/A",
                        **m})

    buf = io.StringIO()
    _old = sys.stdout; sys.stdout = buf
    print_results_table(all_results)
    sys.stdout = _old
    with open(os.path.join(args.save_dir, "summary_table.txt"), "w") as f:
        f.write(buf.getvalue())

    total = str(datetime.timedelta(seconds=int(time.time()-t_total)))
    print(f"\nFull sweep complete in {total}.", flush=True)
    print(f"Results: {args.save_dir}", flush=True)
    print_results_table(all_results)


if __name__ == "__main__":
    main()
