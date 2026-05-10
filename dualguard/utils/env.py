"""
env.py  —  dualguard/utils/env.py
==================================
Runtime-configurable paths.  All values can be overridden by environment
variables so the codebase works on any machine without editing this file.

Priority order for every path:
  1. Corresponding environment variable  (e.g. DUALGUARD_MODEL_PATH)
  2. Value set below as default

Quickest setup — just export two variables before running:
    export DUALGUARD_MODEL_PATH=/path/to/your/models
    export DUALGUARD_DATA_PATH=/path/to/your/datasets   # optional; HF cache used if absent
"""

import os

# ── model weights root ──────────────────────────────────────────────────────
# If DUALGUARD_MODEL_PATH is not set we fall back to '' which makes
# load_model_and_tokenizer() treat model_name as a plain HuggingFace repo id
# (e.g. "openai-community/gpt2-large") — no local path needed.
model_path = os.environ.get("DUALGUARD_MODEL_PATH", "")

# ── dataset cache root ──────────────────────────────────────────────────────
# When DUALGUARD_DATA_PATH is set the dataset classes load from local disk.
# When it is '' the HuggingFace datasets library downloads & caches normally.
data_root           = os.environ.get("DUALGUARD_DATA_PATH", "")
dataset_cache_dir   = os.path.join(data_root, "") if data_root else ""

# ── derived paths (used by dataset classes) ─────────────────────────────────
model_download_dir  = model_path
model_cache_dir     = os.path.join(data_root, "cache")  if data_root else ""
attacker_path       = os.path.join(data_root, "models", "attacker") if data_root else ""
mapper_path         = os.path.join(data_root, "models", "mapper")   if data_root else ""
reducer_path        = os.path.join(data_root, "models", "reducer")  if data_root else ""
lora_path           = os.path.join(data_root, "models", "lora")     if data_root else ""

# ── NLTK wordnet path ────────────────────────────────────────────────────────
# Used by calculate_meteor() in model.py.
# Set DUALGUARD_NLTK_PATH or place wordnet under ~/nltk_data (default NLTK location).
nltk_path = os.environ.get("DUALGUARD_NLTK_PATH",
                            os.path.expanduser("~/nltk_data"))

# ── project-internal save dirs ──────────────────────────────────────────────
project_root    = os.environ.get("DUALGUARD_PROJECT_ROOT",
                                  os.path.dirname(os.path.dirname(
                                      os.path.dirname(os.path.abspath(__file__)))))

sip_model_dir   = os.path.join(project_root, "dualguard", "static", "sip")
warmup_model_dir= os.path.join(project_root, "dualguard", "static", "warmup")
usl_model_dir   = os.path.join(project_root, "dualguard", "static", "usl")

log_dir         = os.path.join(project_root, "dualguard", "log")
sip_log_dir     = os.path.join(log_dir, "sip")
usl_log_dir     = os.path.join(log_dir, "usl")
warmup_log_dir  = os.path.join(log_dir, "warmup")
eval_log_dir    = os.path.join(log_dir, "eval")

# ── create log dirs on import so loggers never fail ─────────────────────────
for _d in (log_dir, sip_log_dir, usl_log_dir, warmup_log_dir, eval_log_dir):
    os.makedirs(_d, exist_ok=True)
