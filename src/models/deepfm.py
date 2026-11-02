"""DeepFM (Guo et al., 2017) implemented from scratch in PyTorch.

    logit = bias + first-order terms          (linear part)
          + second-order field interactions   (FM part, sum over embedding pairs)
          + MLP(concat(embeddings, numeric))  (deep part)

The embeddings are shared between the FM and deep components.
Epoch selection uses a slice of the TRAIN split, so the validation split is only
used for the final report.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, cardinalities, n_num, emb_dim=16, hidden=(256, 128), dropout=0.2):
        super().__init__()
        offsets = np.concatenate([[0], np.cumsum(cardinalities)[:-1]])
        self.register_buffer("offsets", torch.as_tensor(offsets, dtype=torch.long))
        total, n_fields = int(sum(cardinalities)), len(cardinalities)

        self.emb = nn.Embedding(total, emb_dim)
        self.lin = nn.Embedding(total, 1)
        self.num_lin = nn.Linear(n_num, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        layers, d = [], n_fields * emb_dim + n_num
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.lin.weight)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        idx = x_cat + self.offsets                       # (B, F) global embedding ids
        e = self.emb(idx)                                # (B, F, K)
        first = self.lin(idx).sum(dim=(1, 2)) + self.num_lin(x_num).squeeze(-1) + self.bias
        fm = 0.5 * (e.sum(dim=1).pow(2) - e.pow(2).sum(dim=1)).sum(dim=1)
        deep = self.mlp(torch.cat([e.flatten(1), x_num], dim=1)).squeeze(-1)
        return first + fm + deep


@torch.no_grad()
def _predict(model, x_num, x_cat, device, batch_size=16384) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x_num), batch_size):
        xn = x_num[i : i + batch_size].to(device)
        xc = x_cat[i : i + batch_size].to(device)
        out.append(torch.sigmoid(model(xn, xc)).cpu())
    return torch.cat(out).numpy()


def run(data, params: dict, seed: int) -> dict:
    from sklearn.metrics import log_loss

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = data.X_num_train.mean(0), data.X_num_train.std(0) + 1e-6
    to_t = lambda a, dt: torch.as_tensor(a, dtype=dt)
    xn_tr = to_t((data.X_num_train - mean) / std, torch.float32)
    xn_va = to_t((data.X_num_valid - mean) / std, torch.float32)
    xc_tr, xc_va = to_t(data.X_cat_train, torch.long), to_t(data.X_cat_valid, torch.long)
    y_tr = to_t(data.y_train, torch.float32)

    # hold out part of train for epoch selection
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(xn_tr))
    n_hold = int(len(perm) * params["holdout_frac"])
    hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]

    model = DeepFM(data.cardinalities, xn_tr.shape[1], params["emb_dim"], tuple(params["hidden"]), params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    bs = params["batch_size"]

    best_loss, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(params["epochs"]):
        model.train()
        order = rng.permutation(fit_idx)
        for i in range(0, len(order), bs):
            b = torch.as_tensor(order[i : i + bs])
            opt.zero_grad()
            loss = loss_fn(model(xn_tr[b].to(device), xc_tr[b].to(device)), y_tr[b].to(device))
            loss.backward()
            opt.step()

        p_hold = _predict(model, xn_tr[hold_idx], xc_tr[hold_idx], device)
        hold_loss = float(log_loss(data.y_train[hold_idx], np.clip(p_hold, 1e-7, 1 - 1e-7)))
        history.append(hold_loss)
        print(f"  [deepfm] epoch {epoch + 1}: holdout log loss = {hold_loss:.4f}")
        if hold_loss < best_loss:
            best_loss, bad = hold_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > params["patience"]:
                break

    model.load_state_dict(best_state)
    return {"pred": _predict(model, xn_va, xc_va, device), "model": model, "holdout_history": history}
