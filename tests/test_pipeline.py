import numpy as np
import pandas as pd
import pytest

from src.data import CAT_COLS, NUM_COLS, Preprocessor, make_synthetic, prepare_dataset
from src.features import FrequencyEncoder


def tiny_frames():
    train = pd.DataFrame({c: ["a", "a", "a", "b", "b", "rare", None] for c in CAT_COLS})
    for c in NUM_COLS:
        train[c] = [1.0, 2.0, -1.0, np.nan, 0.0, 5.0, 3.0]
    valid = pd.DataFrame({c: ["a", "never_seen", None] for c in CAT_COLS})
    for c in NUM_COLS:
        valid[c] = [1.0, 2.0, 3.0]
    return train, valid


def test_category_codes():
    train, valid = tiny_frames()
    prep = Preprocessor(min_count=2).fit(train)
    _, cat_tr = prep.transform(train)
    _, cat_va = prep.transform(valid)
    assert cat_tr[5, 0] == 0                       # "rare" (count 1 < 2) -> other
    assert cat_tr[6, 0] == 1                       # missing
    assert cat_tr[0, 0] >= 2 and cat_tr[3, 0] >= 2  # frequent categories
    assert cat_va[1, 0] == 0                       # unseen in train -> other (no leakage)
    assert cat_va[2, 0] == 1


def test_numeric_transform():
    train, _ = tiny_frames()
    num, _ = Preprocessor(2).fit(train).transform(train)
    assert num.shape == (7, 13) and np.isfinite(num).all()
    assert num[2, 0] == 0.0                        # negative clipped to 0
    assert num[3, 0] == 0.0                        # NaN -> 0
    assert num[1, 0] == pytest.approx(np.log1p(2.0))


def test_frequency_encoder_uses_train_only():
    train, valid = tiny_frames()
    prep = Preprocessor(2).fit(train)
    _, cat_tr = prep.transform(train)
    _, cat_va = prep.transform(valid)
    enc = FrequencyEncoder().fit(cat_tr, prep.cardinalities)
    assert enc.freqs[0].sum() == pytest.approx(1.0)
    f = enc.transform(cat_va)
    assert f.shape == cat_va.shape and (f >= 0).all() and (f <= 1).all()


def test_synthetic_end_to_end_logreg_beats_chance():
    from sklearn.metrics import roc_auc_score
    from src.models import logreg

    tr, va = make_synthetic(20_000, 0), make_synthetic(5_000, 1)
    data = prepare_dataset(tr, va, min_count=5)
    out = logreg.run(data, {"C": 0.1, "max_iter": 200}, seed=0)
    assert roc_auc_score(data.y_valid, out["pred"]) > 0.6


def test_deepfm_forward_shape():
    torch = pytest.importorskip("torch")
    from src.models.deepfm import DeepFM

    cards = [5, 7, 3]
    model = DeepFM(cards, n_num=4, emb_dim=8, hidden=(16,), dropout=0.0)
    x_num = torch.randn(10, 4)
    x_cat = torch.stack([torch.randint(0, c, (10,)) for c in cards], dim=1)
    assert model(x_num, x_cat).shape == (10,)
