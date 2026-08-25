"""Cross-validate the optimized tree against an independent naive implementation.

The reference (tests/_naive_reference.py) recomputes gains with two-pass
variance on boolean masks and loops over candidates in Python, sharing only
the algorithmic specification with the production code. On data without
near-tied split gains both must produce the identical partition.
"""

from __future__ import annotations

import numpy as np
import pytest

from segmtree import SegTree, adjusted_rand_score
from tests._naive_reference import NaiveSegTree


def make_dataset(seed):
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    centers = [[0.0, 0.0, 5.0], [7.0, 6.0, -3.0], [-6.0, 8.0, 2.0]]
    for k, center in enumerate(centers):
        cont = rng.normal(center, 0.6, size=(120, 3))
        p = [0.1, 0.5, 0.9][k]
        binary = (rng.random((120, 2)) < p).astype(np.float64)
        parts.append(np.hstack([cont, binary]))
        labels.append(np.full(120, k))
    noise = rng.uniform(-10, 10, size=(360, 1))
    return np.hstack([np.vstack(parts), noise]), np.concatenate(labels)


PARAMS = dict(min_gain=0.03, min_leaf=30, max_depth=4, quantiles=np.linspace(0.05, 0.95, 6))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_exact_agreement_with_reference(seed):
    X, _ = make_dataset(seed)
    mine = SegTree(
        min_gain=PARAMS["min_gain"],
        min_leaf_frac=0.0,
        min_leaf_abs=PARAMS["min_leaf"],
        max_depth=PARAMS["max_depth"],
        quantiles=PARAMS["quantiles"],
    ).fit(X)
    ref = NaiveSegTree(**PARAMS).fit(X)

    assert mine.n_leaves_ == len(ref.leaves_)
    assert len(mine.cuts_) == len(ref.cuts_)
    assert np.array_equal(mine.labels_, ref.labels_)

    my_gains = np.array([c.gain for c in mine.cuts_])
    ref_gains = np.array([c[2] for c in ref.cuts_])
    assert np.allclose(my_gains, ref_gains, atol=1e-10)


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_partition_quality_agreement_randomized(seed):
    """On noisy data require ARI ≈ 1 rather than bitwise identity.

    Noisy datasets occasionally contain two candidates whose gains agree to
    ~1e-12; prefix-sum (one-pass) and mask (two-pass) variance then may pick
    different-but-equivalent cuts, so exact equality is not required here.
    """
    X, _ = make_dataset(seed)
    X += np.random.default_rng(seed + 100).normal(0, 0.05, X.shape)
    n = X.shape[0]
    mine = SegTree(min_gain=0.02, min_leaf_frac=0.05, min_leaf_abs=20, quantiles=5).fit(X)
    ref_min_leaf = max(20, int(0.05 * n))
    ref = NaiveSegTree(
        min_gain=0.02, min_leaf=ref_min_leaf, max_depth=None, quantiles=np.linspace(0.05, 0.95, 5)
    ).fit(X)
    assert adjusted_rand_score(mine.labels_, ref.labels_) == pytest.approx(1.0, abs=0.02)


def test_binary_only_data_matches_reference():
    rng = np.random.default_rng(21)
    n = 400
    a = rng.random(n) < 0.7
    b = np.where(a, rng.random(n) < 0.9, rng.random(n) < 0.2)
    X = np.column_stack([a.astype(float), b.astype(float), (rng.random(n) < 0.4).astype(float)])
    mine = SegTree(min_gain=0.01, min_leaf_frac=0.05, min_leaf_abs=20).fit(X)
    ref = NaiveSegTree(min_gain=0.01, min_leaf=max(20, int(0.05 * n))).fit(X, binary_set={0, 1, 2})
    assert np.array_equal(mine.labels_, ref.labels_)
