"""
datasets.py  —  dualguard/utils/datasets/datasets.py
=====================================================
Dataset loading uses _load() which tries local disk first, then HF Hub.
"""

import ast
from functools import partial

import torch
from datasets import load_dataset, Dataset
from trl import DataCollatorForCompletionOnlyLM

from dualguard.utils.datasets.base import FedDataset
from dualguard.utils.exp import register_dataset
from dualguard.utils import env as env_config


# ── path-aware loader ──────────────────────────────────────────────────────

def _load(name: str, *args, **kwargs):
    cache = env_config.dataset_cache_dir
    if cache:
        import os
        local = os.path.join(cache, name)
        if os.path.exists(local):
            return load_dataset(local, *args, **kwargs)
    return load_dataset(name, *args, **kwargs)


# ── PIQA ───────────────────────────────────────────────────────────────────

@register_dataset('piqa')
class PIQAFedDataset(FedDataset):
    def __init__(self, tokenizer, client_ids, shrink_frac=0.3, **kwargs):
        super().__init__(tokenizer, client_ids,
                         dataset=_load('piqa'),
                         types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac, **kwargs)
        self.q_temp = "### Question:\n"
        self.a_temp = "### Solution:\n"
        if self.completion_only:
            response_template_ids = tokenizer.encode(
                '\n' + self.a_temp, add_special_tokens=False)[2:]
            self.co_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template_ids, tokenizer=tokenizer)

    def _format(self, example):
        q = self.q_temp + example["goal"]
        a = self.a_temp + (example["sol1"] if example['label'] == 0 else example["sol2"])
        return {'q': q, 'a': a, 'input': q + '\n' + a}

    def get_answer_mask(self, input_ids, a_ids, tokenizer):
        answer_mask = torch.zeros_like(input_ids, dtype=torch.int)
        for i, (input_seq, a_seq) in enumerate(zip(input_ids, a_ids)):
            inp = input_seq.tolist()
            a   = [t for t in a_seq.tolist()
                   if t not in (tokenizer.pad_token_id, tokenizer.bos_token_id)]
            for j in range(len(inp) - len(a) + 1):
                if inp[j:j+len(a)] == a:
                    answer_mask[i, j:j+len(a)] = 1
                    break
        return answer_mask

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        qs    = [b['q']     for b in batch]
        as_   = [b['a']     for b in batch]
        cap   = max_seq_len if max_seq_len > 0 else 512
        enc   = self.tokenizer(texts, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=cap)
        input_ids = enc['input_ids']
        attn_mask = enc['attention_mask']
        if self.completion_only:
            a_enc = self.tokenizer(as_, return_tensors='pt', padding='max_length',
                                   truncation=True, max_length=cap)
            answer_mask = self.get_answer_mask(input_ids, a_enc['input_ids'], self.tokenizer)
        ret = {'input': input_ids, 'input_ids': input_ids,
               'attention_mask': attn_mask, 'input_text': texts}
        if extra_info:
            ret['input_text_q'] = qs
        if self.completion_only:
            labels = input_ids.clone()
            labels[answer_mask == 0] = -100
            ret['labels'] = labels
        return ret

    def _pre_process(self, ds, batch_size):
        return ds.map(self._format)


@register_dataset('piqa-mini')
class PIQAMiniFedDataset(PIQAFedDataset):
    def __init__(self, tokenizer, client_ids, shrink_frac=0.05, **kwargs):
        super().__init__(tokenizer, client_ids, shrink_frac=shrink_frac, **kwargs)


# ── generic GLUE helper ────────────────────────────────────────────────────

class _GLUEDataset(FedDataset):
    """Base for single-sentence GLUE tasks."""
    _glue_name  = None   # e.g. 'stsb'
    _text_field = None   # e.g. 'sentence1'

    def __init__(self, tokenizer, client_ids, shrink_frac=1.0, **kwargs):
        super().__init__(tokenizer, client_ids,
                         dataset=_load('glue', self._glue_name),
                         types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac, **kwargs)

    def _format(self, example):
        return {'input': example[self._text_field]}

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        cap   = max_seq_len if max_seq_len > 0 else 512
        enc   = self.tokenizer(texts, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=cap)
        return {'input': enc['input_ids'], 'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'], 'input_text': texts}

    def _pre_process(self, ds, batch_size):
        return ds.map(self._format)


class _GLUEPairDataset(_GLUEDataset):
    _text_field2 = None

    def _format(self, example):
        return {'input': example[self._text_field] + ' [SEP] ' + example[self._text_field2]}


@register_dataset('stsb')
class STSBFedDataset(_GLUEPairDataset):
    _glue_name='stsb'; _text_field='sentence1'; _text_field2='sentence2'

@register_dataset('qnli')
class QNLIFedDataset(_GLUEPairDataset):
    _glue_name='qnli'; _text_field='question'; _text_field2='sentence'

@register_dataset('mrpc')
class MRPCFedDataset(_GLUEPairDataset):
    _glue_name='mrpc'; _text_field='sentence1'; _text_field2='sentence2'

@register_dataset('rte')
class RTEFedDataset(_GLUEPairDataset):
    _glue_name='rte'; _text_field='sentence1'; _text_field2='sentence2'

@register_dataset('cola', dra_train_label='test')
class CoLAFedDataset(_GLUEDataset):
    _glue_name='cola'; _text_field='sentence'


# ── GSM8K ──────────────────────────────────────────────────────────────────

@register_dataset('gsm8k', dra_train_label='test')
class GSM8KFedDataset(FedDataset):
    def __init__(self, tokenizer, client_ids, shrink_frac=1.0, **kwargs):
        super().__init__(tokenizer, client_ids,
                         dataset=_load('gsm8k', 'main'),
                         types=['train', 'test'],
                         shrink_frac=shrink_frac, **kwargs)

    def _format(self, example):
        return {'input': example['question'] + '\n' + example['answer']}

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        cap   = max_seq_len if max_seq_len > 0 else 512
        enc   = self.tokenizer(texts, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=cap)
        return {'input': enc['input_ids'], 'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'], 'input_text': texts}

    def _pre_process(self, ds, batch_size):
        return ds.map(self._format)


# ── DialogSum ──────────────────────────────────────────────────────────────

@register_dataset('dialogsum')
class DialogSumFedDataset(FedDataset):
    def __init__(self, tokenizer, client_ids, shrink_frac=1.0, **kwargs):
        super().__init__(tokenizer, client_ids,
                         dataset=_load('knkarthick/dialogsum'),
                         types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac, **kwargs)

    def _format(self, example):
        return {'input': example['dialogue']}

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        cap   = max_seq_len if max_seq_len > 0 else 512
        enc   = self.tokenizer(texts, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=cap)
        return {'input': enc['input_ids'], 'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'], 'input_text': texts}

    def _pre_process(self, ds, batch_size):
        return ds.map(self._format)


# ── CodeAlpaca ─────────────────────────────────────────────────────────────

@register_dataset('codealpaca', dra_train_label='test')
class CodeAlpacaFedDataset(FedDataset):
    def __init__(self, tokenizer, client_ids, shrink_frac=1.0, **kwargs):
        super().__init__(tokenizer, client_ids,
                         dataset=_load('HuggingFaceH4/CodeAlpaca_20K'),
                         types=['train', 'test'],
                         shrink_frac=shrink_frac, **kwargs)

    def _format(self, example):
        return {'input': example.get('prompt','') + '\n' + example.get('completion','')}

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        cap   = max_seq_len if max_seq_len > 0 else 512
        enc   = self.tokenizer(texts, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=cap)
        return {'input': enc['input_ids'], 'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'], 'input_text': texts}

    def _pre_process(self, ds, batch_size):
        return ds.map(self._format)


# ── IMDB ───────────────────────────────────────────────────────────────────

@register_dataset('imdb', dra_train_label='unsupervised')
class IMDBFedDataset(FedDataset):
    def __init__(self, tokenizer, client_ids, shrink_frac=1.0, **kwargs):
        super().__init__(tokenizer, client_ids,
                         dataset=_load('imdb'),
                         types=['train', 'test', 'unsupervised'],
                         shrink_frac=shrink_frac, **kwargs)

    def _format(self, example):
        return {'input': example['text']}

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        cap   = max_seq_len if max_seq_len > 0 else 512
        enc   = self.tokenizer(texts, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=cap)
        return {'input': enc['input_ids'], 'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'], 'input_text': texts}

    def _pre_process(self, ds, batch_size):
        return ds.map(self._format)


# ── WikiText-2 ─────────────────────────────────────────────────────────────

@register_dataset('wikitext')
class WikiTextFedDataset(FedDataset):
    def __init__(self, tokenizer, client_ids, shrink_frac=0.3, **kwargs):
        dataset = _load('wikitext', 'wikitext-2-v1')
        super().__init__(tokenizer, client_ids, dataset,
                         types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac, **kwargs)

    def _format(self, example):
        pass

    def _pre_process(self, ds, batch_size):
        tokenizer  = self.tokenizer

        # Cap block_size at the model's maximum position embeddings so BERT
        # (512) and GPT-2 (1024) both work correctly without warnings.
        model_max  = getattr(tokenizer, 'model_max_length', 512)
        # Guard against the HF default sentinel value (very large int)
        if model_max > 10_000:
            model_max = 512
        raw_block  = self.uni_length if self.uni_length > 0 else 128
        block_size = min(raw_block, model_max)

        def tokenize_fn(examples):
            # truncation=True silences the "sequence longer than max" warnings
            return tokenizer(examples["text"],
                             truncation=True,
                             max_length=model_max)

        def group_texts(examples):
            cat   = {k: sum(examples[k], []) for k in examples.keys()}
            total = (len(cat[list(cat.keys())[0]]) // block_size) * block_size
            result = {k: [t[i:i+block_size] for i in range(0, total, block_size)]
                      for k, t in cat.items()}
            result["labels"]     = result["input_ids"].copy()
            result["input"]      = result["input_ids"].copy()
            result["input_text"] = [tokenizer.decode(ii) for ii in result["input_ids"]]
            return result

        tok_ds = ds.map(tokenize_fn, batched=True, num_proc=4,
                        remove_columns=["text"])
        return tok_ds.map(group_texts, batched=True,
                          batch_size=batch_size, num_proc=4)

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        input_ids = torch.tensor([b['input'] for b in batch])
        attn_mask = torch.tensor([b['attention_mask'] for b in batch])
        return {'input':          input_ids,
                'input_ids':      input_ids,
                'attention_mask': attn_mask,
                'input_text':     [b.get('input_text', '') for b in batch]}


# ── WikiText-103 ───────────────────────────────────────────────────────────

@register_dataset('wikitext-103')
class WikiText103FedDataset(WikiTextFedDataset):
    def __init__(self, tokenizer, client_ids, shrink_frac=0.1, **kwargs):
        from dualguard.utils.datasets.base import FedDataset as _Base
        dataset = _load('wikitext', 'wikitext-103-v1')
        _Base.__init__(self, tokenizer, client_ids, dataset,
                       types=['train', 'test', 'validation'],
                       shrink_frac=shrink_frac, **kwargs)


# ── E2E ────────────────────────────────────────────────────────────────────

@register_dataset('e2e')
class E2EFedDataset(FedDataset):
    def __init__(self, tokenizer, client_ids, shrink_frac=1.0, **kwargs):
        super().__init__(tokenizer, client_ids,
                         dataset=_load('e2e_nlg'),
                         types=['train', 'test', 'validation'],
                         shrink_frac=shrink_frac, **kwargs)

    def _format(self, example):
        return {'input': example['meaning_representation'] + ' | ' + example['human_reference']}

    def _col_fun(self, batch, max_seq_len=-1, extra_info=True):
        texts = [b['input'] for b in batch]
        cap   = max_seq_len if max_seq_len > 0 else 512
        enc   = self.tokenizer(texts, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=cap)
        return {'input': enc['input_ids'], 'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'], 'input_text': texts}

    def _pre_process(self, ds, batch_size):
        return ds.map(self._format)
