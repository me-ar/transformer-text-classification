"""Offline tests with tiny random models — no downloads needed."""
import numpy as np
import pandas as pd
import pytest

from textclf.classic import EmbeddingClassifier, TfidfClassifier
from textclf.data import subsample_per_class
from textclf.finetune import (
    FTConfig,
    apply_temperature,
    count_params,
    expected_calibration_error,
    predict_proba,
    temperature_scale,
    train,
)
from textclf.tiny import hashing_encoder, tiny_distilbert

TEXTS = ["my card has not arrived", "where is my new card", "card still not here",
         "how do i top up", "top up failed", "add money to my account",
         "exchange rate is wrong", "bad exchange rate applied", "currency rate charged"] * 4
LABELS = [0, 0, 0, 1, 1, 1, 2, 2, 2] * 4


def test_subsample_per_class_caps_counts():
    df = pd.DataFrame({"text": TEXTS, "label": LABELS})
    sub = subsample_per_class(df, 2)
    assert sub["label"].value_counts().max() == 2 and set(sub.columns) == {"text", "label"}


def test_tfidf_learns_simple_intents():
    clf = TfidfClassifier(Cs=(10.0,)).fit(TEXTS, LABELS)
    pred = clf.predict_proba(["card not arrived yet", "top up money", "exchange rate wrong"]).argmax(1)
    assert list(pred) == [0, 1, 2]


def test_embedding_classifier_with_hashing_encoder():
    clf = EmbeddingClassifier(hashing_encoder()).fit(TEXTS, LABELS)
    assert clf.predict_proba(["where is my card"]).argmax(1)[0] == 0


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    perfect = np.eye(2)[y]
    assert expected_calibration_error(perfect, y) == pytest.approx(0.0)
    wrong = np.eye(2)[1 - y]
    assert expected_calibration_error(wrong, y) == pytest.approx(1.0)


def test_temperature_scaling_softens_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    logits = rng.normal(size=(300, 3))
    logits[np.arange(300), y] += 1.0
    proba = np.exp(4 * logits) / np.exp(4 * logits).sum(1, keepdims=True)   # deliberately overconfident
    t = temperature_scale(proba, y)
    assert t > 1.5
    assert expected_calibration_error(apply_temperature(proba, t), y) < expected_calibration_error(proba, y)


@pytest.mark.parametrize("lora", [False, True])
def test_finetune_loop_runs_and_lora_freezes_backbone(lora):
    model, tok = tiny_distilbert(3)
    _, total = count_params(model)
    cfg = FTConfig(epochs=2, lr=5e-3, batch_size=8, lora=lora)
    model, info = train(model, tok, TEXTS, LABELS, TEXTS[:9], LABELS[:9], cfg, log=lambda *_: None)
    assert len(info["history"]) >= 1
    if lora:
        assert info["trainable_params"] < 0.5 * info["total_params"]
    else:
        assert info["trainable_params"] == total
    p = predict_proba(model, tok, TEXTS[:5], cfg)
    assert p.shape == (5, 3) and np.allclose(p.sum(1), 1, atol=1e-5)
