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
