"""Baseline 2: gradient-boosted trees (XGBoost), with or without frequency-encoding features."""
from __future__ import annotations

import xgboost as xgb

from ..features import FrequencyEncoder, tree_feature_names, tree_matrix


def run(data, params: dict, seed: int, use_freq: bool = False) -> dict:
    f_tr = f_va = None
    if use_freq:
        enc = FrequencyEncoder().fit(data.X_cat_train, data.cardinalities)   # fitted on train only
        f_tr, f_va = enc.transform(data.X_cat_train), enc.transform(data.X_cat_valid)

    x_tr = tree_matrix(data.X_num_train, data.X_cat_train, f_tr)
    x_va = tree_matrix(data.X_num_valid, data.X_cat_valid, f_va)

    model = xgb.XGBClassifier(
        **params, tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=-1
    )
    model.fit(x_tr, data.y_train)
    return {
        "pred": model.predict_proba(x_va)[:, 1],
        "model": model,
        "x_valid": x_va,
        "feature_names": tree_feature_names(use_freq),
    }
