"""
data_utils.py  —  sip_layerwise/data_utils.py
==============================================
Dataset utilities.  max_seq_len is always derived from the tokenizer's
model_max_length so BERT (512) and GPT-2 (1024) are handled automatically
without any manual flag.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def _model_max_len(tokenizer: AutoTokenizer) -> int:
    """
    Return the model's hard sequence-length limit.
    Guards against HuggingFace's default sentinel (1_000_000_000+).
    """
    raw = getattr(tokenizer, 'model_max_length', 512)
    return raw if raw <= 10_000 else 512


class EvalDataset(Dataset):
    """
    Loads cola / sst2 / rotten_tomatoes with PGL's exact sampling logic.
    Tokenises with the target model's tokenizer and respects its max length.
    Each __getitem__ returns one pre-batched dict:
        input          : LongTensor (batch_size, seq_len)
        attention_mask : LongTensor (batch_size, seq_len)
    """

    def __init__(self, tokenizer: AutoTokenizer, dataset_name: str,
                 split: str, n_inputs: int, batch_size: int,
                 max_seq_len: Optional[int] = None, rng_seed: int = 101):
        np.random.seed(rng_seed)

        # Derive max_seq_len from tokenizer if not supplied
        if max_seq_len is None or max_seq_len <= 0:
            max_seq_len = _model_max_len(tokenizer)

        seq_key = {"cola": "sentence", "sst2": "sentence",
                   "rotten_tomatoes": "text"}[dataset_name]

        if dataset_name in ("cola", "sst2"):
            full = load_dataset("glue", dataset_name)["train"]
        else:
            full = load_dataset(dataset_name)["train"]

        idxs = list(range(len(full)))
        np.random.shuffle(idxs)

        n_samples = n_inputs * batch_size
        if split == "test":
            assert n_samples <= 1000, "max 1000 test samples"
            idxs = idxs[:n_samples]
        else:   # val
            idxs = idxs[1000:]
            assert len(idxs) >= n_samples
            zipped = sorted([(i, len(full[i][seq_key])) for i in idxs],
                            key=lambda x: x[1])
            chunk  = len(zipped) // n_samples
            idxs   = []
            for i in range(n_samples):
                idxs.append(zipped[chunk*i + np.random.randint(chunk)][0])
            np.random.shuffle(idxs)

        self.items: List[dict] = []
        for i in range(n_inputs):
            texts = [full[idxs[i*batch_size + j]][seq_key] for j in range(batch_size)]
            enc   = tokenizer(texts,
                              padding="max_length",
                              truncation=True,
                              max_length=max_seq_len,
                              return_tensors="pt")
            self.items.append({
                "input":          enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            })

    def __len__(self):        return len(self.items)
    def __getitem__(self, i): return self.items[i]


def _identity_collate(batch):
    assert len(batch) == 1
    return batch[0]


def get_eval_loader(tokenizer: AutoTokenizer, dataset_name: str,
                    split: str, n_inputs: int, batch_size: int,
                    max_seq_len: Optional[int] = None,
                    rng_seed: int = 101) -> DataLoader:
    ds = EvalDataset(tokenizer, dataset_name, split, n_inputs,
                     batch_size, max_seq_len, rng_seed)
    return DataLoader(ds, batch_size=1, shuffle=False,
                      collate_fn=_identity_collate)


def get_train_loader(aux_dataset: str, tokenizer: AutoTokenizer,
                     batch_size: int,
                     max_seq_len: Optional[int] = None) -> DataLoader:
    """
    Build a training DataLoader for the GRU inverter.

    max_seq_len is passed down to get_dataloader_unsliced() so that every
    dataset class (WikiText, PIQA, …) correctly caps its sequences at the
    model's limit.  When None the tokenizer's model_max_length is used.
    """
    from dualguard.utils.exp import get_dataset
    if max_seq_len is None or max_seq_len <= 0:
        max_seq_len = _model_max_len(tokenizer)
    ds = get_dataset(aux_dataset, tokenizer=tokenizer, client_ids=[])
    return ds.get_dataloader_unsliced(batch_size, "train",
                                      max_seq_len=max_seq_len)
