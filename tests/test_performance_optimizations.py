"""Regression tests for the performance work on the histogram path.

These tests pin the optimized code paths to the same results as the
straightforward implementations they replaced:

- manual linear-interpolation quantiles vs ``numpy.quantile``;
- on-demand centered values vs the persistent centered matrix (float64 and
  float32);
- blockwise association row sums vs a single full matrix;
- the batched cross-feature scorer vs its single-candidate wrapper.
"""

from __future__ import annotations

import numpy as np
import pytest

import segmtree.tree as tree_module
from segmtree import SegTree


def _structured(n=900, m=7, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m)).astype(dtype)
    groups = rng.integers(0, 4, n)
    for g in range(4):
        X[groups == g, :3] += (g - 1.5) * 2.0
    X[:, 3] = (X[:, 3] > 0).astype(dtype)
    return X


def _assert_same_tree(a, b):
    assert a.n_leaves_ == b.n_leaves_
    assert a.cuts_ == b.cuts_
    assert a.leaves_ == b.leaves_
    assert np.array_equal(a.labels_, b.labels_)
    assert a.reduction_ == pytest.approx(b.reduction_, abs=1e-12)
    assert np.array_equal(a.feature_importances_, b.feature_importances_)


def test_quantile_edges_match_numpy_quantile():
    rng = np.random.default_rng(3)
    tree = SegTree(max_bins=11)
    tree._rng_ = tree._resolve_rng()
    for values in (
        rng.normal(size=257),
        rng.integers(-5, 6, size=257).astype(float),
        np.r_[np.zeros(50), np.ones(50), np.full(50, 2.0)],
    ):
        edges = tree._quantile_edges(values, 11)
        qs = np.linspace(0.0, 1.0, 12)[1:-1]
        expected = np.unique(np.quantile(values.astype(np.float64), qs))
        np.testing.assert_array_equal(edges, expected)


def test_quantile_edges_constant_column_is_empty():
    tree = SegTree()
    tree._rng_ = tree._resolve_rng()
    assert tree._quantile_edges(np.full(100, 2.5), 8).size == 0


def test_on_demand_centering_matches_centered_matrix(monkeypatch):
    X = _structured()
    params = dict(min_leaf_abs=20, max_leaf_nodes=6, random_state=7)
    centered = SegTree(**params).fit(X)
    assert centered._centered_numeric_ is True

    monkeypatch.setattr(tree_module, "_CENTERED_COPY_BYTES", 0)
    on_demand = SegTree(**params).fit(X)
    assert on_demand._centered_numeric_ is False
    assert on_demand._XcT_num_ is None
    assert on_demand._Xraw is None
    _assert_same_tree(centered, on_demand)


def test_low_memory_parameter_matches_cached_path():
    X = _structured()
    params = dict(min_leaf_abs=20, max_leaf_nodes=6, random_state=7)
    cached = SegTree(**params).fit(X)
    low = SegTree(low_memory=True, **params).fit(X)
    assert low._centered_numeric_ is False
    assert low._Xraw is None
    _assert_same_tree(cached, low)


def test_low_memory_validation():
    with pytest.raises(ValueError, match="low_memory"):
        SegTree(low_memory=1)._check_params(2)


def test_on_demand_centering_matches_float32(monkeypatch):
    X = _structured(dtype=np.float32)
    params = dict(min_leaf_abs=20, max_leaf_nodes=6, random_state=7, dtype=np.float32)
    centered = SegTree(**params).fit(X)
    monkeypatch.setattr(tree_module, "_CENTERED_COPY_BYTES", 0)
    on_demand = SegTree(**params).fit(X)
    assert on_demand._centered_numeric_ is False
    _assert_same_tree(centered, on_demand)


def test_blockwise_association_matches_single_block(monkeypatch):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 41))
    X[:, 1] = X[:, 0] * 0.7 + rng.normal(0, 0.4, X.shape[0])
    params = dict(min_gain=1e9, min_leaf_abs=20)  # association only, no growth
    full = SegTree(**params).fit(X)

    # Force five features per numeric block (the default would use one block).
    monkeypatch.setattr(tree_module, "_ASSOC_BLOCK_BYTES", 8 * 41 * 5)
    blocked = SegTree(**params).fit(X)
    # The association machinery multiplies float32 matrices; blocking changes
    # BLAS accumulation order, so allow float32 round-off.
    np.testing.assert_allclose(blocked.association_, full.association_, rtol=1e-5, atol=1e-8)


def test_batched_cross_gain_matches_single_candidate(monkeypatch):
    """Every candidate's batched score equals the one-candidate wrapper."""
    X = _structured(n=700, m=8, seed=11)
    tree = SegTree(min_gain=0.0, min_leaf_abs=20, max_leaf_nodes=4)
    captured = []
    original = SegTree._hist_local_cross_gains

    def spy(self, cands, idx, n, sw_node):
        gains = original(self, cands, idx, n, sw_node)
        # Call ``original`` rather than the public wrapper to avoid recursion
        # through this monkeypatch.
        singles = [original(self, [cand], idx, n, sw_node)[0] for cand in cands]
        captured.append((gains, singles))
        return gains

    monkeypatch.setattr(SegTree, "_hist_local_cross_gains", spy)
    tree.fit(X)
    assert captured
    for batched, singles in captured:
        np.testing.assert_allclose(batched, singles, rtol=1e-12, atol=1e-15)
