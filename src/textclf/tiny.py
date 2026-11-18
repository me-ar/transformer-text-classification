"""Tiny, randomly-initialised stand-ins so tests and smoke runs need no model downloads."""
from __future__ import annotations

import hashlib
import re

import numpy as np


def tiny_tokenizer(vocab_size: int = 2000):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    words = [f"h{i}" for i in range(vocab_size - len(vocab))]
    vocab.update({w: i + 4 for i, w in enumerate(words)})
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]")

    class Hashed:
        """Maps any word to a hashed vocab entry, so real text produces varied ids."""

        def __call__(self, texts, **kw):
            mapped = [" ".join(f"h{int(hashlib.md5(w.lower().encode()).hexdigest(), 16) % len(words)}"
                               for w in re.findall(r"\w+", t)) or "[UNK]" for t in texts]
            kw.pop("return_token_type_ids", None)
            enc = fast(mapped, **kw)
            enc.pop("token_type_ids", None)
            return enc

    return Hashed()


def tiny_distilbert(n_labels: int):
    from transformers import DistilBertConfig, DistilBertForSequenceClassification
    cfg = DistilBertConfig(vocab_size=2000, dim=32, n_layers=2, n_heads=2, hidden_dim=64,
                           max_position_embeddings=128, num_labels=n_labels)
    return DistilBertForSequenceClassification(cfg), tiny_tokenizer()


def hashing_encoder(dim: int = 512):
    def encode(texts):
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)
    return encode
