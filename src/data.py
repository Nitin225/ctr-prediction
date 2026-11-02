"""Loading, cleaning and encoding of the Criteo display-ads CTR data.

Design rules
------------
* Everything that is *learned* from data (category vocabularies, rare-category
  threshold, frequencies) is fitted on the training split only, then applied
  to the validation split. Nothing is computed on validation rows.
* Categorical values are mapped to integer codes:
      0 = "other"  (rare in train, or never seen in train)
      1 = "missing"
      2.. = frequent categories
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NUM_COLS = [f"I{i}" for i in range(1, 14)]
CAT_COLS = [f"C{i}" for i in range(1, 27)]
HF_REPO = "reczoo/Criteo_x4"
HF_FILE = "Criteo_x4.zip"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def download_criteo() -> Path:
    """Download the Criteo_x4 zip (~4 GB) from Hugging Face (cached after the first call)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset"))


def _read_member(zip_path: Path, member: str, nrows: int) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(member))
        with z.open(name) as f:
            df = pd.read_csv(f, nrows=nrows)
    return df.rename(columns={"Label": "label"})


def load_criteo(zip_path: Path, n_train: int, n_valid: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the first n_train rows of train.csv and the first n_valid rows of valid.csv."""
    return _read_member(zip_path, "train.csv", n_train), _read_member(zip_path, "valid.csv", n_valid)


def make_synthetic(n: int, seed: int = 0) -> pd.DataFrame:
    """Criteo-shaped fake data with a learnable signal. Only for smoke-testing the pipeline."""
    world = np.random.default_rng(1234)          # fixed: same "world" for train and valid
    rng = np.random.default_rng(seed)
    logit = np.full(n, -1.4)
    cols = {}

    for j, c in enumerate(NUM_COLS):
        x = rng.lognormal(1.0, 1.0, n).round()
        if j < 4:
            logit += 0.25 * np.log1p(x)
        cols[c] = pd.Series(x).mask(rng.random(n) < 0.2)

    for j, c in enumerate(CAT_COLS):
        card = int(world.integers(50, 3000))
        vocab = np.array([f"{v:08x}" for v in world.integers(0, 2**32, card)])
        effect = world.normal(0, 0.6 if j < 8 else 0.0, card)
        p = 1.0 / (np.arange(card) + 1) ** 1.1
        idx = rng.choice(card, n, p=p / p.sum())
        logit += effect[idx]
        cols[c] = pd.Series(vocab[idx]).mask(rng.random(n) < 0.1)

    label = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"label": label, **cols})


# --------------------------------------------------------------------------- #
# Cleaning / encoding
# --------------------------------------------------------------------------- #
class Preprocessor:
    """Turns raw Criteo columns into (float numeric matrix, int categorical-code matrix)."""

    def __init__(self, min_count: int = 10):
        self.min_count = min_count
        self.vocabs: dict[str, dict] = {}
        self.cardinalities: list[int] = []

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.vocabs = {}
        for c in CAT_COLS:
            counts = df[c].value_counts()
            frequent = counts[counts >= self.min_count].index
            self.vocabs[c] = {v: i + 2 for i, v in enumerate(frequent)}
        self.cardinalities = [len(self.vocabs[c]) + 2 for c in CAT_COLS]
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # numeric: negatives -> 0, log1p to tame heavy tails, missing -> 0
        num = np.log1p(df[NUM_COLS].astype("float64").clip(lower=0)).fillna(0.0)
        x_num = num.to_numpy(dtype=np.float32)

        x_cat = np.zeros((len(df), len(CAT_COLS)), dtype=np.int64)
        for j, c in enumerate(CAT_COLS):
            s = df[c]
            codes = s.map(self.vocabs[c]).fillna(0).astype(np.int64).to_numpy()
            x_cat[:, j] = np.where(s.isna().to_numpy(), 1, codes)
        return x_num, x_cat


@dataclass
class Dataset:
    X_num_train: np.ndarray
    X_cat_train: np.ndarray
    y_train: np.ndarray
    X_num_valid: np.ndarray
    X_cat_valid: np.ndarray
    y_valid: np.ndarray
    cardinalities: list[int]


def prepare_dataset(train_df: pd.DataFrame, valid_df: pd.DataFrame, min_count: int = 10) -> Dataset:
    prep = Preprocessor(min_count).fit(train_df)
    xn_tr, xc_tr = prep.transform(train_df)
    xn_va, xc_va = prep.transform(valid_df)
    return Dataset(
        xn_tr, xc_tr, train_df["label"].to_numpy(),
        xn_va, xc_va, valid_df["label"].to_numpy(),
        prep.cardinalities,
    )


def summarize(train_df: pd.DataFrame, valid_df: pd.DataFrame, data: Dataset) -> dict:
    """Numbers for the README's data section."""
    raw_card = train_df[CAT_COLS].nunique()
    return {
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "click_rate_train": float(train_df["label"].mean()),
        "click_rate_valid": float(valid_df["label"].mean()),
        "top_missing_pct": (train_df.drop(columns="label").isna().mean() * 100).round(1)
        .sort_values(ascending=False).head(8).to_dict(),
        "raw_unique_categories_total": int(raw_card.sum()),
        "unique_categories_after_cleaning_total": int(sum(data.cardinalities)),
        "highest_cardinality_raw": raw_card.sort_values(ascending=False).head(5).to_dict(),
    }
