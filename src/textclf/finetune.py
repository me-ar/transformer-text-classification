"""Fine-tune a transformer encoder for classification — full fine-tuning or LoRA.

A plain PyTorch loop (no Trainer) so every step is visible: dynamic padding, linear warm-up/decay,
gradient clipping and early stopping on validation macro-F1.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


@dataclass
class FTConfig:
    epochs: int = 4
    lr: float = 5e-5
    batch_size: int = 32
    max_len: int = 64
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    patience: int = 2
    seed: int = 42
    lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_targets: tuple[str, ...] = ("q_lin", "v_lin")


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def apply_lora(model, cfg: FTConfig):
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=list(cfg.lora_targets), task_type="SEQ_CLS",
                      modules_to_save=["pre_classifier", "classifier"])  # new heads are trained fully
    return get_peft_model(model, lcfg)


def _loader(tokenizer, texts, labels, cfg, shuffle):
    items = list(zip(texts, labels, strict=True))

    def collate(batch):
        enc = tokenizer([t for t, _ in batch], padding=True, truncation=True, max_length=cfg.max_len,
                        return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in batch])
        return enc

    g = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(items, batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate, generator=g)


@torch.no_grad()
def predict_proba(model, tokenizer, texts, cfg: FTConfig) -> np.ndarray:
    model.eval()
    out = []
    for batch in _loader(tokenizer, list(texts), [0] * len(texts), cfg, shuffle=False):
        batch.pop("labels")
        out.append(torch.softmax(model(**batch).logits.float(), -1))
    return torch.cat(out).numpy()


def train(model, tokenizer, train_texts, train_y, val_texts, val_y, cfg: FTConfig, log=print):
    torch.manual_seed(cfg.seed)
    if cfg.lora:
        model = apply_lora(model, cfg)
    trainable, total = count_params(model)
    log(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    dl = _loader(tokenizer, list(train_texts), list(train_y), cfg, shuffle=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(dl)
    warm = int(cfg.warmup_ratio * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    best, best_state, wait, history = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in dl:
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total_loss += loss.item()
        val_pred = predict_proba(model, tokenizer, val_texts, cfg).argmax(1)
        f1 = f1_score(val_y, val_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": round(total_loss / len(dl), 4), "val_macro_f1": round(f1, 4),
                        "elapsed_s": round(time.time() - t0, 1)})
        log(f"  epoch {epoch}: loss={total_loss / len(dl):.4f} val macro-F1={f1:.4f} ({time.time() - t0:.0f}s)")
        if f1 > best:
            best, wait = f1, 0
            best_state = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, {"history": history, "trainable_params": trainable, "total_params": total,
                   "train_seconds": round(time.time() - t0, 1)}


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def temperature_scale(val_logits_proba: np.ndarray, val_y: np.ndarray) -> float:
    """Fit a single temperature on validation (Guo et al., 2017) to fix over-confidence."""
    logp = np.log(np.clip(val_logits_proba, 1e-12, 1))
    best_t, best_nll = 1.0, math.inf
    for t in np.linspace(0.5, 5.0, 91):
        z = logp / t
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        nll = -np.log(np.clip(p[np.arange(len(val_y)), val_y], 1e-12, 1)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def apply_temperature(proba: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(proba, 1e-12, 1)) / t
    z = z - z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)
