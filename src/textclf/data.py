"""Banking77: 13,083 customer-support queries across 77 fine-grained banking intents."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    labels: list[str]

    @property
    def n_classes(self) -> int:
        return len(self.labels)


def download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        path = DATA_DIR / f"{split}.csv"
        if not path.exists():
            urllib.request.urlretrieve(BASE + f"{split}.csv", path)


def load(val_size: float = 0.1, seed: int = 42) -> Splits:
    download()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    labels = sorted(train["category"].unique())
    idx = {c: i for i, c in enumerate(labels)}
    for df in (train, test):
        df["text"] = df["text"].astype(str).str.strip()
        df["label"] = df["category"].map(idx)
    train, val = train_test_split(train, test_size=val_size, stratify=train["label"], random_state=seed)
    return Splits(train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), labels)


def subsample_per_class(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """At most n examples per class — used for the data-efficiency (few-shot) curve."""
    return df.sample(frac=1.0, random_state=seed).groupby("label").head(n).reset_index(drop=True)
