"""categorical_splitter: exact subset enumeration vs one_vs_rest."""

from __future__ import annotations

import numpy as np
import pytest

from segmtree import (
    SegTree,
    adjusted_rand_score,
    extract_rules,
    replay_check,
    replay_labels,
)
from tests._naive_reference import NaiveSegTree


def make_high_card_data(k=16, n=800, seed=4):
    """Group structure leaks into a high-cardinality code column."""
    rng = np.random.default_rng(seed)
    cont = rng.normal([0.0, 5.0], 0.6, (n, 2))
    third = n // 3
    group = np.repeat([0, 1, 2], [third, third, n - 2 * third])
    base = np.array([0, k // 3, 2 * (k // 3)])[group]
    codes = np.where(rng.random(n) < 0.75, base, rng.integers(0, k, n))
    return np.column_stack([cont, codes.astype(float)]), codes


OVR_PARAMS = dict(
    min_gain=0.02,
    splitter="exact",
    min_gain_mode="normalized",
    min_leaf_frac=0.05,
    min_leaf_abs=20,
    max_depth=3,
    quantiles=5,
    categorical_features=[2],
)


def test_invalid_splitter_rejected():
    with pytest.raises(ValueError, match="categorical_splitter"):
        SegTree(categorical_splitter="fuzzy")._check_params(3)


def test_exact_still_rejects_above_max_categories():
    X, _ = make_high_card_data(k=16)
    with pytest.raises(ValueError, match="max_categories"):
        SegTree(**OVR_PARAMS, categorical_splitter="exact", max_categories=12).fit(X)
    # error message must explain the exponential cost and name the alternative
    with pytest.raises(ValueError, match=r"2\^\(k-1\).*one_vs_rest"):
        SegTree(
            min_leaf_abs=20,
            categorical_features=[2],
            categorical_splitter="exact",
            min_gain_mode="normalized",
        ).fit(X[:, :])


def test_one_vs_rest_ignores_max_categories_and_fits():
    X, _ = make_high_card_data(k=16, seed=9)
    tree = SegTree(**OVR_PARAMS, categorical_splitter="one_vs_rest", max_categories=2).fit(X)
    assert tree.n_leaves_ >= 2
    assert len(tree.categories_["x2"]) == 16
    assert np.array_equal(tree.predict(X), tree.labels_)
    assert replay_check(tree, X)
    rules_text = "\n".join(str(r) for r in extract_rules(tree))
    assert " in {" in rules_text or " not in {" in rules_text


def test_ovr_equals_exact_when_best_split_is_singleton():
    """Category 0 is its own cluster; categories 1..3 share another region:
    the optimal split is single-vs-rest, so both splitters must agree."""
    rng = np.random.default_rng(15)
    n_per = 150
    cont = rng.normal(0.0, 0.5, (n_per * 4, 1))
    codes = np.repeat([0, 1, 2, 3], n_per).astype(float)
    signal = np.where(codes == 0, 12.0, rng.normal(0, 0.5, n_per * 4))
    X = np.column_stack([cont, signal, codes])
    params = dict(
        min_gain=0.05,
        min_leaf_frac=0.0,
        min_leaf_abs=60,
        quantiles=4,
        categorical_features=[2],
        splitter="exact",
        min_gain_mode="normalized",
    )
    exact = SegTree(**params, categorical_splitter="exact").fit(X)
    ovr = SegTree(**params, categorical_splitter="one_vs_rest").fit(X)
    assert adjusted_rand_score(exact.labels_, ovr.labels_) == pytest.approx(1.0)
    assert exact.n_leaves_ == ovr.n_leaves_
    first_exact, first_ovr = exact.cuts_[0], ovr.cuts_[0]
    # same winning feature; side conventions may differ ({0}|rest vs rest|{0})
    assert first_exact.feature_index == first_ovr.feature_index == 2
    all_codes = {0, 1, 2, 3}
    pair_exact = frozenset(first_exact.categories)
    pair_ovr = frozenset(first_ovr.categories)
    same_partition = (pair_exact == pair_ovr) or (
        pair_exact | pair_ovr == frozenset(all_codes) and not (pair_exact & pair_ovr)
    )
    assert same_partition


def test_ovr_matches_naive_reference():
    X, _ = make_high_card_data(k=10, n=600, seed=21)
    mine = SegTree(
        min_gain=0.02,
        min_leaf_frac=0.0,
        min_leaf_abs=25,
        max_depth=3,
        quantiles=5,
        categorical_features=[2],
        categorical_splitter="one_vs_rest",
        splitter="exact",
        min_gain_mode="normalized",
    ).fit(X)
    ref = NaiveSegTree(
        min_gain=0.02,
        min_leaf=25,
        max_depth=3,
        quantiles=np.linspace(0.05, 0.95, 5),
        cat_splitter="one_vs_rest",
    ).fit(X, cat_set={2})
    assert mine.n_leaves_ == len(ref.leaves_)
    assert np.array_equal(mine.labels_, ref.labels_)
    assert np.allclose([c.gain for c in mine.cuts_], [c[2] for c in ref.cuts_], atol=1e-10)


def test_ovr_float32_large_codes_stay_distinct():
    big_a, big_b = 16_777_216.0, 16_777_217.0  # adjacent float32 grid points
    rng = np.random.default_rng(9)
    n = 300
    codes = np.where(rng.random(n) < 0.5, big_a, big_b)
    signal = np.where(codes == big_a, 0.0, 9.0) + rng.normal(0, 0.3, n)
    X = np.column_stack([signal, codes])
    trees = {}
    for dt in (np.float64, np.float32):
        tr = SegTree(
            min_leaf_abs=20,
            categorical_features=[1],
            dtype=dt,
            categorical_splitter="one_vs_rest",
            splitter="exact",
            min_gain_mode="normalized",
        ).fit(X)
        trees[dt] = tr
        assert sorted(tr.categories_["x1"]) == [int(big_a), int(big_b)]
        cat_cuts = [c for c in tr.cuts_ if c.feature_index == 1]
        assert cat_cuts
        assert all(c.op == "in" and len(c.categories) == 1 for c in cat_cuts)
    assert adjusted_rand_score(trees[np.float64].labels_, trees[np.float32].labels_) == 1.0


def test_ovr_invariant_under_code_permutation():
    X, _ = make_high_card_data(k=8, seed=30)
    params = {
        **OVR_PARAMS,
        "categorical_splitter": "one_vs_rest",
        "min_leaf_abs": 25,
        "max_depth": 2,
    }
    base = SegTree(**params).fit(X)
    rng = np.random.default_rng(0)
    for _ in range(3):
        perm = rng.permutation(8)
        Xp = X.copy()
        Xp[:, 2] = perm[X[:, 2].astype(int)]
        tp = SegTree(**params).fit(Xp)
        assert adjusted_rand_score(base.labels_, tp.labels_) == pytest.approx(1.0)


def test_handle_unknown_complement_works_with_both_splitters():
    X, _ = make_high_card_data(k=6, seed=12)
    for splitter in ("exact", "one_vs_rest"):
        tree = SegTree(
            min_leaf_abs=15,
            categorical_features=[2],
            categorical_splitter=splitter,
        ).fit(X)
        fresh = np.column_stack([np.zeros((40, 2)), np.full(40, 5555.0)])
        pred = tree.predict(fresh)
        assert (pred >= 0).all() and (pred < tree.n_leaves_).all()
        assert np.array_equal(replay_labels(tree, fresh), pred)


def test_handle_unknown_error_raises_for_both_splitters():
    X, _ = make_high_card_data(k=6, seed=13)
    for splitter in ("exact", "one_vs_rest"):
        tree = SegTree(
            min_leaf_abs=15,
            categorical_features=[2],
            categorical_splitter=splitter,
            handle_unknown="error",
        ).fit(X)
        fresh = np.column_stack([np.zeros((20, 2)), np.full(20, 999.0)])
        with pytest.raises(ValueError, match=r"x2.*999"):
            tree.predict(fresh)
        ok = np.column_stack([np.zeros((10, 2)), np.zeros(10)])
        assert len(tree.predict(ok)) == 10
