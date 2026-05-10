"""
args_factory.py
===============
CLI argument parser for the layerwise SIP privacy analysis.

Keeps every PGL argument name intact.  Gradient-inversion-only parameters
(loss, n_steps, lr*, opt_alg, coeff_*, use_swaps*, init*, neptune, baseline)
are removed and replaced with SIP-specific ones.

Model loading
-------------
  --model_name  accepts:
    • A HuggingFace repo id: "openai-community/gpt2-large", "bert-base-uncased"
    • A local absolute path: "/data/models/gpt2-large"
    • A relative name resolved against DUALGUARD_MODEL_PATH env var

  --bert_num_labels  (only used when model is BERT-based)

Aux training datasets
---------------------
  --aux_dataset  wikitext | piqa  (or any dataset registered in datasets.py)
"""

import argparse
from dualguard.attack.layerwise_sip import GRAD_TYPES


def _layer_list(s: str):
    if s.strip().lower() == "all":
        return None
    try:
        return [int(x.strip()) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Cannot parse '{s}' — expected comma-separated ints or 'all'.")


def get_args():
    p = argparse.ArgumentParser(
        description="Layerwise SIP forward-inversion privacy analysis.")

    # ── sweep axes ──────────────────────────────────────────────────────
    p.add_argument("--grad_type", default="layer_encoder", choices=GRAD_TYPES)
    p.add_argument("--attack_layer", type=_layer_list, default=[0],
                   metavar="LAYERS",
                   help="Layer indices: int, comma-separated, or 'all'.")

    # ── evaluation dataset (same as PGL) ─────────────────────────────────
    p.add_argument("--dataset", required=True,
                   choices=["cola", "sst2", "rotten_tomatoes"])
    p.add_argument("--split", required=True, choices=["val", "test"])
    p.add_argument("--n_inputs", type=int, required=True)
    p.add_argument("-b", "--batch_size", type=int, default=1)
    p.add_argument("--rng_seed", type=int, default=101)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--print_every", type=int, default=50)

    # ── defence (same names as PGL) ──────────────────────────────────────
    p.add_argument("--defense_noise", type=float, default=None)
    p.add_argument("--defense_pct_mask", type=float, default=None)
    p.add_argument("--noise_multiplier", type=float, default=0.01)
    p.add_argument("--clipping_bound",   type=float, default=1.0)

    # ── model ────────────────────────────────────────────────────────────
    p.add_argument("--model_name", type=str, default="openai-community/gpt2-large",
                   help=(
                       "HF repo id OR local path.  Examples:\n"
                       "  openai-community/gpt2-large\n"
                       "  bert-base-uncased\n"
                       "  /data/models/bert-base-uncased\n"
                       "  gpt/gpt2-large  (resolved via DUALGUARD_MODEL_PATH)"))
    p.add_argument("--bert_num_labels", type=int, default=2,
                   help="Number of classification labels for BERT models.")

    # ── SIP / GRU training ───────────────────────────────────────────────
    p.add_argument("--aux_dataset", type=str, default="wikitext",
                   help="Auxiliary dataset for GRU training: wikitext | piqa")
    p.add_argument("--sip_epochs",      type=int,   default=2)
    p.add_argument("--sip_batch_size",  type=int,   default=4)
    p.add_argument("--gru_hidden",      type=int,   default=256)
    p.add_argument("--gru_bidirectional",
                   type=lambda x: x.lower() not in ("false","0","no"),
                   default=True, metavar="BOOL")

    # ── output ───────────────────────────────────────────────────────────
    p.add_argument("--save_dir", type=str, default="./layerwise_sip_results")
    p.add_argument("--save_inverters", action="store_true")
    p.add_argument("--sample_rate", type=float, default=1.0)

    return p.parse_args()
