"""Train and evaluate CTR models.

Examples
--------
    python -m src.train --synthetic                      # 1-minute smoke test, no download
    python -m src.train                                  # all models on real Criteo data
    python -m src.train --model xgb_freq --rows 200000   # one model, smaller run
    python -m src.train --scale-ablation                 # AUC vs. training-set size
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
import yaml

from .data import (download_criteo, load_criteo, make_synthetic, prepare_dataset, summarize)
from .evaluate import (LABELS, calibration, compute_metrics, load_report, plot_calibration,
                       shap_summary, write_results_table)

ALL_MODELS = list(LABELS)


def run_model(name: str, data, cfg: dict, seed: int) -> dict:
    if name == "logreg":
        from .models import logreg
        return logreg.run(data, cfg["models"]["logreg"], seed)
    if name in ("xgb", "xgb_freq"):
        from .models import xgb
        return xgb.run(data, cfg["models"]["xgb"], seed, use_freq=(name == "xgb_freq"))
    if name == "deepfm":
        from .models import deepfm
        return deepfm.run(data, cfg["models"]["deepfm"], seed)
    raise ValueError(name)


def load_frames(cfg: dict, synthetic: bool, rows: int | None):
    d = cfg["data"]
    if synthetic:
        n_train = rows or 50_000
        n_valid = max(n_train // 4, 2_000)
        return make_synthetic(n_train, cfg["seed"]), make_synthetic(n_valid, cfg["seed"] + 1)
    n_train = rows or d["n_train"]
    n_valid = max(rows // 4, 2_000) if rows else d["n_valid"]
    print("Locating / downloading Criteo_x4 (first run downloads ~4 GB)...")
    return load_criteo(download_criteo(), n_train, n_valid)


def scale_ablation(train_df, valid_df, cfg: dict, out_dir: Path) -> None:
    sizes = [s for s in cfg["ablation"]["train_sizes"] if s <= len(train_df)] or [len(train_df)]
    results = []
    for n in sizes:
        print(f"[ablation] training xgb_freq on {n:,} rows")
        data = prepare_dataset(train_df.head(n), valid_df, cfg["data"]["min_count"])
        out = run_model("xgb_freq", data, cfg, cfg["seed"])
        m = compute_metrics(data.y_valid, out["pred"])
        results.append({"train_rows": n, **m})
        print(f"           AUC={m['auc']:.4f}  logloss={m['logloss']:.4f}")
    (out_dir / "scale_ablation.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot([r["train_rows"] for r in results], [r["auc"] for r in results], marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("training rows")
    ax.set_ylabel("validation AUC")
    ax.set_title("XGBoost (+freq): effect of training-set size")
    fig.tight_layout()
    fig.savefig(out_dir / "scale_ablation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="all", choices=ALL_MODELS + ["all"])
    ap.add_argument("--rows", type=int, help="override number of training rows")
    ap.add_argument("--synthetic", action="store_true", help="use fake data (pipeline smoke test)")
    ap.add_argument("--scale-ablation", action="store_true")
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = cfg["seed"]
    np.random.seed(seed)

    out_dir = Path(cfg["paths"]["results_dir"]) / ("synthetic" if args.synthetic else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df = load_frames(cfg, args.synthetic, args.rows)
    print(f"train={len(train_df):,} rows  valid={len(valid_df):,} rows")

    if args.scale_ablation:
        scale_ablation(train_df, valid_df, cfg, out_dir)
        return

    data = prepare_dataset(train_df, valid_df, cfg["data"]["min_count"])
    (out_dir / "data_summary.json").write_text(json.dumps(summarize(train_df, valid_df, data), indent=2))

    report_path = out_dir / "metrics.json"
    report = load_report(report_path)
    report["constant_baseline"] = compute_metrics(
        data.y_valid, np.full(len(data.y_valid), data.y_train.mean())
    )

    for name in (ALL_MODELS if args.model == "all" else [args.model]):
        print(f"\n== {LABELS[name]} ==")
        t = time.time()
        out = run_model(name, data, cfg, seed)
        secs = time.time() - t

        m = compute_metrics(data.y_valid, out["pred"])
        m["calibration"] = calibration(data.y_valid, out["pred"])
        m["fit_seconds"] = secs
        report["models"][name] = m
        print(f"AUC={m['auc']:.4f}  logloss={m['logloss']:.4f}  ECE={m['calibration']['ece']:.4f}  ({secs:.0f}s)")

        if name == "xgb_freq" and not args.no_shap:
            try:
                shap_summary(out["model"], out["x_valid"], out["feature_names"], out_dir / "shap_summary.png")
            except Exception as e:  # SHAP is a nice-to-have, never block the run
                print("SHAP skipped:", e)

        report_path.write_text(json.dumps(report, indent=2))   # save after every model

    write_results_table(report, out_dir / "results.md")
    plot_calibration(report["models"], out_dir / "calibration.png")
    print(f"\nSaved metrics, table and plots to {out_dir}/")


if __name__ == "__main__":
    main()
