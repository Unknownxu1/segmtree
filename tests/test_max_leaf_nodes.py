"""max_leaf_nodes: hard leaf-budget semantics and interactions."""

from __future__ import annotations

import numpy as np
import pytest

from segmtree import SegTree, replay_check
from tests.conftest import STRONG_PARAMS, asymmetric_data, make_mixed_cat


@pytest.fixture()
def rich_tree_data():
    rng = np.random.default_rng(42)
    parts = []
    for center in ([0.0, -1.0, 2.0], [4.0, 3.0, -2.0], [-4.0, 2.0, 0.0], [8.0, -4.0, 5.0]):
        parts.append(rng.normal(center, 0.5, size=(150, 3)))
    return np.vstack(parts)


RICH_PARAMS = dict(min_gain=0.001, min_leaf_frac=0.02, min_leaf_abs=15, quantiles=6)


@pytest.mark.parametrize("cap", [2, 3, 4, 6])
def test_leaf_budget_respected(rich_tree_data, cap):
    tree = SegTree(**RICH_PARAMS, max_leaf_nodes=cap).fit(rich_tree_data)
    assert tree.n_leaves_ <= cap
    assert len(tree.cuts_) == tree.n_leaves_ - 1  # every split adds one leaf
    assert set(np.unique(tree.labels_)) == set(range(tree.n_leaves_))


def test_budget_reached_exactly_on_rich_data(rich_tree_data):
    tree = SegTree(**RICH_PARAMS, max_leaf_nodes=4).fit(rich_tree_data)
    assert tree.n_leaves_ == 4  # data supports far more leaves than the cap
    uncapped = SegTree(**RICH_PARAMS).fit(rich_tree_data)
    assert uncapped.n_leaves_ > 4
    # executed cuts are the greedy prefix of the unconstrained run
    key = [(c.feature_index, c.threshold) for c in uncapped.cuts_]
    assert [(c.feature_index, c.threshold) for c in tree.cuts_] == key[:3]


def test_none_keeps_previous_default_behavior(rich_tree_data):
    a = SegTree(**RICH_PARAMS).fit(rich_tree_data)
    b = SegTree(**RICH_PARAMS, max_leaf_nodes=None).fit(rich_tree_data)
    assert np.array_equal(a.labels_, b.labels_)
    assert len(a.cuts_) == len(b.cuts_)
    assert a.total_gain_ == b.total_gain_


def test_strictest_stop_condition_wins():
    X = asymmetric_data()
    # depth cap binds before the leaf cap would
    t1 = SegTree(**STRONG_PARAMS, max_depth=1, max_leaf_nodes=8).fit(X)
    assert all(lf.depth <= 1 for lf in t1.leaves_)
    # leaf cap binds before depth
    t2 = SegTree(**STRONG_PARAMS, max_depth=10, max_leaf_nodes=2).fit(X)
    assert t2.n_leaves_ == 2
    # min_gain binds: no splits at all regardless of generous caps
    t3 = SegTree(min_gain=10.0, min_leaf_abs=10, max_leaf_nodes=8).fit(
        np.random.default_rng(1).normal(size=(200, 3))
    )
    assert t3.n_leaves_ == 1


def test_cap_two_selects_single_best_root_cut(rich_tree_data):
    capped = SegTree(**RICH_PARAMS, max_leaf_nodes=2).fit(rich_tree_data)
    free = SegTree(**RICH_PARAMS).fit(rich_tree_data)
    root_free, root_capped = free.cuts_[0], capped.cuts_[0]
    assert len(capped.cuts_) == 1
    assert (root_capped.feature_index, root_capped.threshold) == (
        root_free.feature_index,
        root_free.threshold,
    )
    assert root_capped.gain == pytest.approx(root_free.gain)


def test_deterministic_at_the_limit(rich_tree_data):
    kw = {**RICH_PARAMS, "max_leaf_nodes": 5}
    a = SegTree(**kw).fit(rich_tree_data)
    b = SegTree(**kw).fit(rich_tree_data.copy())
    assert np.array_equal(a.labels_, b.labels_)
    assert [(c.feature_index, c.threshold) for c in a.cuts_] == [
        (c.feature_index, c.threshold) for c in b.cuts_
    ]


def test_cap_with_categorical_rules_replay_and_additivity():
    X = make_mixed_cat()
    params = dict(
        min_gain=0.02,
        min_leaf_frac=0.05,
        min_leaf_abs=10,
        max_depth=3,
        quantiles=5,
        categorical_features=[3],
        max_leaf_nodes=3,
    )
    tree = SegTree(**params).fit(X)
    assert tree.n_leaves_ == 3
    assert np.array_equal(tree.predict(X), tree.labels_)
    assert replay_check(tree, X)
    gap = abs(tree.total_gain_ - (tree.h_root_ - tree.h_avg_))
    assert gap / max(1e-12, tree.h_root_ - tree.h_avg_) < 1e-9


def test_invalid_max_leaf_nodes_values():
    for bad in (1, 0, True, 2.5, "3", -4):
        with pytest.raises(ValueError):
            SegTree(max_leaf_nodes=bad)._check_params(3)
