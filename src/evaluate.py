"""Metrics, calibration and interpretability helpers."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

LABELS = {
    "logreg": "Logistic Regression",
    "xgb": "XGBoost (integer codes)",
    "xgb_freq": "XGBoost (+ frequency encoding)",
    "deepfm": "DeepFM (PyTorch)",
}


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p))}


def calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability curve (equal-count bins) and expected calibration error (ECE)."""
    order = np.argsort(p)
    mean_pred, frac_pos, weights = [], [], []
    for chunk in np.array_split(order, n_bins):
        mean_pred.append(float(p[chunk].mean()))
        frac_pos.append(float(y[chunk].mean()))
        weights.append(len(chunk) / len(p))
    ece = float(sum(w * abs(a - b) for w, a, b in zip(weights, mean_pred, frac_pos)))
    return {"mean_pred": mean_pred, "frac_pos": frac_pos, "ece": ece}


def plot_calibration(models: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    for name, m in models.items():
        c = m["calibration"]
        ax.plot(c["mean_pred"], c["frac_pos"], marker="o", ms=3, label=f"{LABELS.get(name, name)} (ECE {c['ece']:.3f})")
    ax.set_xlabel("mean predicted click probability")
    ax.set_ylabel("observed click rate")
    ax.set_title("Calibration on validation set")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def shap_summary(model, x: np.ndarray, feature_names: list[str], path: Path, n: int = 3000, seed: int = 0) -> None:
    import shap

    idx = np.random.default_rng(seed).choice(len(x), min(n, len(x)), replace=False)
    sv = shap.TreeExplainer(model).shap_values(x[idx])
    plt.figure()
    shap.summary_plot(sv, x[idx], feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def write_results_table(report: dict, path: Path) -> None:
    rows = ["| Model | AUC | Log loss | ECE | Fit time (s) |", "|---|---|---|---|---|"]
    base = report.get("constant_baseline")
    if base:
        rows.append(f"| Constant (train click rate) | {base['auc']:.4f} | {base['logloss']:.4f} | - | - |")
    for name in LABELS:
        m = report["models"].get(name)
        if m:
            rows.append(f"| {LABELS[name]} | {m['auc']:.4f} | {m['logloss']:.4f} | {m['calibration']['ece']:.4f} | {m['fit_seconds']:.0f} |")
    path.write_text("\n".join(rows) + "\n")


def load_report(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"models": {}}
