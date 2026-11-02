"""Baseline 1: logistic regression on one-hot categories + scaled numeric features."""
from __future__ import annotations

from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def run(data, params: dict, seed: int) -> dict:
    enc = OneHotEncoder(handle_unknown="ignore")
    scaler = StandardScaler()
    x_tr = hstack([enc.fit_transform(data.X_cat_train), csr_matrix(scaler.fit_transform(data.X_num_train))]).tocsr()
    x_va = hstack([enc.transform(data.X_cat_valid), csr_matrix(scaler.transform(data.X_num_valid))]).tocsr()

    model = LogisticRegression(C=params["C"], max_iter=params["max_iter"], random_state=seed)
    model.fit(x_tr, data.y_train)
    return {"pred": model.predict_proba(x_va)[:, 1], "model": model}
