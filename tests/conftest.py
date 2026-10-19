from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def blobs():
    rng = np.random.default_rng(7)
    a = rng.normal([0.0, 0.0], 0.4, size=(300, 2))
    b = rng.normal([6.0, 6.0], 0.4, size=(300, 2))
    return np.vstack([a, b])


@pytest.fixture()
def mixed():
    """Three latent groups over 4 continuous + 3 binary features."""
    rng = np.random.default_rng(42)
    sizes = [250, 200, 150]
    centers = [[0.0, -1.0, 2.0, 0.5], [4.0, 3.0, -2.0, 0.5], [-4.0, 2.0, 0.0, 8.0]]
    parts = []
    labels = []
    n_groups = len(sizes)
    for start, (size, center) in enumerate(zip(sizes, centers)):
        cont = rng.normal(center, 0.5, size=(size, 4))
        p = 0.05 + 0.9 * (start / n_groups)
        binom = (rng.random((size, 3)) < p).astype(float)
        parts.append(np.hstack([cont, binom]))
        labels.append(np.full(size, start))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    return X, y


@pytest.fixture()
def mixed_params():
    """SegTree kwargs suited to the `mixed` fixture."""
    return dict(min_gain=0.02, min_leaf_frac=0.05, min_leaf_abs=10, max_depth=4, quantiles=6)


@pytest.fixture()
def mixed_x(mixed):
    """Feature matrix only."""
    return mixed[0]


def asymmetric_data():
    """Root split is strong; afterwards the LEFT child hides a much stronger
    second split than the RIGHT child's weak one.
    """
    rng = np.random.default_rng(0)
    n = 800
    half = n // 2
    quarter = n // 4
    x0 = np.r_[np.zeros(half), np.full(n - half, 10.0)]
    x1 = np.r_[np.zeros(quarter), np.full(quarter, 8.0), np.full(n - half, 3.0)]
    x1 += rng.normal(0, 0.1, n)
    x2 = np.r_[np.full(half, 0.5), np.zeros(quarter), np.full(quarter, 1.6)]
    x2 += rng.normal(0, 0.05, n)
    return np.column_stack([x0, x1, x2])


STRONG_PARAMS = dict(
    min_gain=0.02,
    min_leaf_abs=50,
    min_leaf_frac=0.01,
    quantiles=6,
    splitter="exact",
    min_gain_mode="normalized",
)


def make_mixed_cat(n_per_group=200, seed=3):
    """Three latent groups over 2 continuous + 1 binary + 1 categorical col."""
    rng = np.random.default_rng(seed)
    centers = [[0.0, 0.0], [6.0, 6.0], [-6.0, 6.0]]
    parts = []
    for k, center in enumerate(centers):
        cont = rng.normal(center, 0.5, (n_per_group, 2))
        flag = (rng.random((n_per_group, 1)) < 0.1 + 0.4 * k).astype(float)
        code = np.full(n_per_group, k, dtype=float)
        flip = rng.random(n_per_group) < 0.15
        code[flip] = rng.integers(0, 3, int(flip.sum()))
        parts.append(np.hstack([cont, flag, code[:, None]]))
    return np.vstack(parts)
