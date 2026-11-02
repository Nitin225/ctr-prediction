"""Feature engineering. Statistics are fitted on the training split only."""
from __future__ import annotations

import numpy as np


class FrequencyEncoder:
    """Replace each category code by how often it occurs in the training data (0..1)."""

    def fit(self, x_cat: np.ndarray, cardinalities: list[int]) -> "FrequencyEncoder":
        n = len(x_cat)
        self.freqs = [np.bincount(x_cat[:, j], minlength=card) / n for j, card in enumerate(cardinalities)]
        return self

    def transform(self, x_cat: np.ndarray) -> np.ndarray:
        out = np.empty(x_cat.shape, dtype=np.float32)
        for j, f in enumerate(self.freqs):
            out[:, j] = f[x_cat[:, j]]
        return out


def tree_matrix(x_num: np.ndarray, x_cat: np.ndarray, freq: np.ndarray | None = None) -> np.ndarray:
    """Dense matrix for tree models: numeric | integer category codes | (optional) frequencies."""
    parts = [x_num, x_cat.astype(np.float32)]
    if freq is not None:
        parts.append(freq)
    return np.hstack(parts)


def tree_feature_names(use_freq: bool) -> list[str]:
    from .data import CAT_COLS, NUM_COLS

    names = NUM_COLS + CAT_COLS
    return names + [f"{c}_freq" for c in CAT_COLS] if use_freq else names
