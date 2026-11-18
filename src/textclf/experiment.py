"""Benchmark: TF-IDF vs frozen embeddings vs full fine-tuning vs LoRA on Banking77.

Usage: python -m textclf.experiment            # full run (downloads models)
       python -m textclf.experiment --smoke    # tiny offline run for testing the pipeline
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import data
from .classic import EmbeddingClassifier, TfidfClassifier, sentence_transformer_encoder
from .finetune import (
    FTConfig,
    apply_temperature,
    expected_calibration_error,
    predict_proba,
    temperature_scale,
    train,
)

BASE_MODEL = "distilbert-base-uncased"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SHOTS = [5, 10, 20, 50, 1000]


def load_transformer(name: str, n_labels: int, smoke: bool):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not smoke:
        tok = AutoTokenizer.from_pretrained(name)
        return AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_labels), tok
    from .tiny import tiny_distilbert
    return tiny_distilbert(n_labels)


def metrics(y, proba) -> dict:
    pred = proba.argmax(1)
    return {"accuracy": round(accuracy_score(y, pred), 4), "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "ece": round(expected_calibration_error(proba, y), 4)}


def timed_proba(fn, texts, n_rep: int = 1) -> tuple[np.ndarray, float]:
    t = time.perf_counter()
    for _ in range(n_rep):
        p = fn(texts)
    return p, 1000 * (time.perf_counter() - t) / (n_rep * len(texts))


def main(smoke: bool = False, out: Path | None = None) -> dict:
    out = out or data.ROOT / "reports"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    sp = data.load()
    if smoke:
        sp.train, sp.val, sp.test = (data.subsample_per_class(sp.train, 4), sp.val.iloc[:200], sp.test.iloc[:300])
    yv, yt = sp.val["label"].to_numpy(), sp.test["label"].to_numpy()
    print(f"train={len(sp.train)} val={len(sp.val)} test={len(sp.test)} classes={sp.n_classes}", flush=True)
    rows, probas, extra = [], {}, {}

    def record(name, proba, latency, train_s, trainable, **kw):
        m = metrics(yt, proba)
        rows.append({"model": name, **m, **kw, "latency_ms_per_example": round(latency, 3),
                     "train_seconds": round(train_s, 1), "trainable_params": trainable})
        probas[name] = proba
        print(f"{name:32s} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} ECE={m['ece']:.4f}", flush=True)

    # 1) TF-IDF
    t = time.perf_counter()
    tfidf = TfidfClassifier().fit(sp.train.text, sp.train.label, sp.val.text, yv)
    tr = time.perf_counter() - t
    p, lat = timed_proba(tfidf.predict_proba, sp.test.text)
    record("TF-IDF + LogReg", p, lat, tr, int(tfidf.clf.coef_.size))

    # 2) Frozen embeddings
    if smoke:
        from .tiny import hashing_encoder
        encode = hashing_encoder()
    else:
        encode = sentence_transformer_encoder(EMB_MODEL)
    t = time.perf_counter()
    emb = EmbeddingClassifier(encode).fit(sp.train.text, sp.train.label)
    tr = time.perf_counter() - t
    p, lat = timed_proba(emb.predict_proba, sp.test.text)
    record("MiniLM embeddings + LogReg", p, lat, tr, int(emb.clf.coef_.size))

    # 3) + 4) Fine-tuning: full and LoRA
    configs = {"DistilBERT (full fine-tune)": FTConfig(epochs=4, lr=5e-5),
               "DistilBERT + LoRA (r=8)": FTConfig(epochs=6, lr=5e-4, lora=True)}
    if smoke:
        configs = {k: FTConfig(epochs=1, lr=c.lr, lora=c.lora, batch_size=16, lora_targets=c.lora_targets)
                   for k, c in configs.items()}
    for name, cfg in configs.items():
        print(f"== {name}", flush=True)
        model, tok = load_transformer(BASE_MODEL, sp.n_classes, smoke)
        model, info = train(model, tok, sp.train.text, sp.train.label, sp.val.text, yv, cfg)
        pv = predict_proba(model, tok, sp.val.text, cfg)
        temp = temperature_scale(pv, yv)
        p, lat = timed_proba(lambda x, m=model, tk=tok, c=cfg: predict_proba(m, tk, list(x), c), sp.test.text)
        ece_cal = expected_calibration_error(apply_temperature(p, temp), yt)
        record(name, p, lat, info["train_seconds"], info["trainable_params"],
               ece_after_temp_scaling=round(ece_cal, 4), temperature=round(temp, 2))
        extra[name] = {"history": info["history"], "total_params": info["total_params"], "temperature": temp}

    results = pd.DataFrame(rows)
    best = results.sort_values("macro_f1").iloc[-1]["model"]

    # Data efficiency: how many labelled examples per intent do you need?
    curve = []
    for n in SHOTS:
        sub = data.subsample_per_class(sp.train, n)
        for name, clf in [("TF-IDF + LogReg", TfidfClassifier(Cs=(10.0,))),
                          ("MiniLM embeddings + LogReg", EmbeddingClassifier(encode))]:
            pr = clf.fit(sub.text, sub.label).predict_proba(sp.test.text)
            full = np.zeros((len(pr), sp.n_classes))
            full[:, clf.clf.classes_] = pr
            curve.append({"shots_per_class": n, "n_train": len(sub), "model": name,
                          "macro_f1": round(f1_score(yt, full.argmax(1), average="macro"), 4)})
    curve = pd.DataFrame(curve)

    # Error analysis for the best model
    pb = probas[best]
    pred = pb.argmax(1)
    cm = confusion_matrix(yt, pred, labels=range(sp.n_classes))
    np.fill_diagonal(cm, 0)
    pairs = sorted(((cm[i, j] + cm[j, i], sp.labels[i], sp.labels[j]) for i in range(sp.n_classes)
                    for j in range(i + 1, sp.n_classes)), reverse=True)[:10]
    per_class = f1_score(yt, pred, average=None, labels=range(sp.n_classes))
    worst = sorted(zip(per_class, sp.labels, strict=True))[:10]
    errs = sp.test.assign(predicted=[sp.labels[i] for i in pred], confidence=pb.max(1).round(3))
    errs = errs[errs["label"] != pred].sort_values("confidence", ascending=False)
    errs[["text", "category", "predicted", "confidence"]].head(25).to_csv(out / "error_examples.csv", index=False)

    summary = {"best_model": best, "results": results.to_dict(orient="records"),
               "data_efficiency": curve.to_dict(orient="records"),
               "top_confused_pairs": [{"count": int(c), "a": a, "b": b} for c, a, b in pairs],
               "worst_classes": [{"intent": n, "f1": round(float(f), 4)} for f, n in worst],
               "training": extra}
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (out / "results.md").write_text(results.to_markdown(index=False) + "\n")

    # ---------------- figures ----------------
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(results))
    ax.bar(x - 0.2, results["accuracy"], 0.4, label="accuracy", color="#2563eb")
    ax.bar(x + 0.2, results["macro_f1"], 0.4, label="macro-F1", color="#16a34a")
    lo = max(0.0, results[["accuracy", "macro_f1"]].min().min() - 0.08)
    ax.set(ylim=(lo, 1.0), title="Banking77 test set (3,080 queries, 77 intents)")
    ax.set_xticks(x, [m.replace(" (", "\n(").replace(" + ", "\n+ ") for m in results["model"]], fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "model_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, g in curve.groupby("model"):
        ax.plot(g["shots_per_class"].clip(upper=150), g["macro_f1"], marker="o", label=name)
    ft_rows = results[results["model"].str.contains("DistilBERT")]
    for (_, r), ls in zip(ft_rows.iterrows(), ["--", ":"], strict=False):
        ax.axhline(r["macro_f1"], ls=ls, lw=1.2, color="grey", label=f"{r['model']} (all data)")
    ax.set(xscale="log", xlabel="labelled examples per intent (full data ≈ 117)", ylabel="test macro-F1",
           title="Data efficiency: how much labelling do you need?")
    ax.set_xticks([5, 10, 20, 50, 150], ["5", "10", "20", "50", "all"])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "data_efficiency.png")

    ft_name = "DistilBERT (full fine-tune)"
    if ft_name in probas:
        p_raw = probas[ft_name]
        p_cal = apply_temperature(p_raw, extra[ft_name]["temperature"])
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for p, lab, col in [(p_raw, "raw", "#dc2626"), (p_cal, "temperature-scaled", "#2563eb")]:
            conf, ok = p.max(1), p.argmax(1) == yt
            edges = np.quantile(conf, np.linspace(0, 1, 11))
            idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, 9)
            xs = [conf[idx == b].mean() for b in range(10) if (idx == b).any()]
            ys = [ok[idx == b].mean() for b in range(10) if (idx == b).any()]
            ax.plot(xs, ys, "o-", label=f"{lab} (ECE={expected_calibration_error(p, yt):.3f})", color=col)
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set(xlabel="confidence", ylabel="accuracy", title="Reliability diagram (fine-tuned DistilBERT)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / "calibration.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([f"{a}  ↔  {b}" for _, a, b in pairs][::-1], [c for c, _, _ in pairs][::-1], color="#7c3aed")
    ax.set(title=f"Most-confused intent pairs ({best})", xlabel="errors (both directions)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(figs / "confused_pairs.png")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke, out=Path("/tmp/textclf_smoke") if args.smoke else None)
