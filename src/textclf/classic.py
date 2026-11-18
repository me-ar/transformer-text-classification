"""Non-neural and frozen-embedding baselines."""
from __future__ import annotations

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


class TfidfClassifier:
    """Word 1–2-grams + character 3–5-grams, logistic regression. C chosen on validation."""

    def __init__(self, Cs=(1.0, 5.0, 20.0)):
        self.Cs = Cs
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)

    def _X(self, texts, fit=False):
        if fit:
            return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, texts, y, val_texts=None, val_y=None):
        X = self._X(texts, fit=True)
        best = (-1.0, None)
        for C in self.Cs if val_texts is not None else self.Cs[:1]:
            clf = LogisticRegression(C=C, max_iter=3000).fit(X, y)
            score = f1_score(val_y, clf.predict(self._X(val_texts)), average="macro") if val_texts is not None else 0
            if score > best[0]:
                best = (score, clf)
        self.clf = best[1]
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self._X(texts))


class EmbeddingClassifier:
    """Frozen sentence embeddings + logistic regression (no fine-tuning)."""

    def __init__(self, encode, C: float = 10.0):
        self.encode, self.C = encode, C

    def fit(self, texts, y, **_):
        self.clf = LogisticRegression(C=self.C, max_iter=3000).fit(self.encode(list(texts)), y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self.encode(list(texts)))


def sentence_transformer_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    return lambda texts: model.encode(texts, batch_size=128, normalize_embeddings=True,
                                      show_progress_bar=False, convert_to_numpy=True)
