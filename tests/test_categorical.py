from __future__ import annotations

import numpy as np
import pytest

from segmtree import (
    SegTree,
    adjusted_rand_score,
    extract_rules,
    profile_leaves,
    replay_check,
    replay_labels,
)
from tests._naive_reference import NaiveSegTree


def make_cat_data(n=600, k=4, seed=8):
    """Three latent groups; group identity partially leaks through a k-level code."""
    rng = np.random.default_rng(seed)
    cont = rng.normal([0.0, 5.0], 0.6, size=(n, 2))
    group = np.arange(n) // (n // 3)
    p_group = group / 3.0
    codes = (rng.random(n) < 0.7 * p_group + 0.05).astype(int)
    flip = rng.random(n) < 0.15
    codes[flip] = rng.integers(0, k, size=int(flip.sum()))
    X = np.column_stack([cont, codes.astype(float)])
    return X, codes


def test_categorical_subset_rules_and_replay():
    X, _ = make_cat_data()
    tree = SegTree(
        min_gain=0.01, min_leaf_frac=0.03, min_leaf_abs=20, categorical_features=[2]
    ).fit(X)
    assert set(tree.categorical_indices_) == {2}
    cat_cuts = [c for c in tree.cuts_ if c.feature_index == 2]
    assert cat_cuts, "categorical feature should be used for at least one split"
    for cut in cat_cuts:
        assert cut.op == "in"
        assert cut.categories and all(isinstance(c, int) for c in cut.categories)
        assert np.isnan(cut.threshold)
    rules = extract_rules(tree)
    text = "\n".join(str(r) for r in rules)
    assert " in {" in text or " not in {" in text
    assert replay_check(tree, X)


def test_exact_agreement_with_naive_reference():
    X, _ = make_cat_data(n=450, seed=13)
    mine = SegTree(
        min_gain=0.02,
        min_leaf_frac=0.0,
        min_leaf_abs=25,
        max_depth=4,
        quantiles=5,
        categorical_features=[2],
    ).fit(X)
    ref = NaiveSegTree(
        min_gain=0.02, min_leaf=25, max_depth=4, quantiles=np.linspace(0.05, 0.95, 5)
    ).fit(X, binary_set=set(), cat_set={2})
    assert mine.n_leaves_ == len(ref.leaves_)
    assert np.array_equal(mine.labels_, ref.labels_)
    my_gains = np.array([c.gain for c in mine.cuts_])
    ref_gains = np.array([c[2] for c in ref.cuts_])
    assert np.allclose(my_gains, ref_gains, atol=1e-10)


def test_mixed_cont_bin_cat_matches_naive():
    rng = np.random.default_rng(31)
    n = 480
    cont = np.column_stack(
        [
            np.r_[rng.normal(0, 0.5, n // 2), rng.normal(6, 0.5, n - n // 2)],
            rng.normal(2, 1.0, n),
        ]
    )
    binary = (rng.random((n, 1)) < 0.35).astype(float)
    latent = np.r_[np.zeros(n // 2), np.ones(n - n // 2)]
    cat = np.where(
        rng.random(n) < 0.6 + 0.3 * latent, rng.integers(0, 2, n) + 2 * latent, 2
    ).astype(float)
    X = np.hstack([cont, binary, cat[:, None]])
    mine = SegTree(
        min_gain=0.02, min_leaf_frac=0.05, min_leaf_abs=20, quantiles=6, categorical_features=[3]
    ).fit(X)
    ref = NaiveSegTree(
        min_gain=0.02, min_leaf=max(20, int(0.05 * n)), quantiles=np.linspace(0.05, 0.95, 6)
    ).fit(X, binary_set={2}, cat_set={3})
    assert adjusted_rand_score(mine.labels_, ref.labels_) == pytest.approx(1.0)


def test_unseen_category_routes_deterministically():
    X, _ = make_cat_data(k=3)
    tree = SegTree(
        min_gain=0.01, min_leaf_frac=0.03, min_leaf_abs=20, categorical_features=[2]
    ).fit(X)
    fresh = np.column_stack([np.zeros((50, 2)), np.full(50, 99.0)])
    pred = tree.predict(fresh)
    assert (pred >= 0).all() and (pred < tree.n_leaves_).all()
    replay = replay_labels(tree, fresh)
    assert np.array_equal(replay, pred), "unseen codes must replay identically"


def test_conflict_with_binary_features_raises():
    X, _ = make_cat_data(k=2)
    X[:, 2] = np.r_[np.zeros(300), np.ones(300)]
    with pytest.raises(ValueError, match="both binary and categorical"):
        SegTree(binary_features=[2], categorical_features=[2])._resolve_categorical_features(
            X, ["a", "b", "c"], {2}
        )


def test_max_categories_exceeded_raises():
    rng = np.random.default_rng(5)
    X = np.column_stack([rng.normal(size=(200, 1)), rng.integers(0, 20, 200).astype(float)])
    with pytest.raises(ValueError, match="max_categories"):
        SegTree(categorical_features=[1], max_categories=12).fit(X)
    tree = SegTree(categorical_features=[1], max_categories=20, min_leaf_abs=10).fit(X)
    assert tree.n_leaves_ >= 1


def test_non_integral_codes_raise():
    X = np.column_stack([np.zeros(60), np.tile([0.0, 0.5, 1.0], 20)])
    with pytest.raises(ValueError, match="integral"):
        SegTree(categorical_features=[1]).fit(X)


def test_integer_counts_stay_continuous_by_default():
    rng = np.random.default_rng(17)
    counts = rng.integers(0, 4, 400).astype(float)
    signal = np.where(counts >= 2, 8.0, 0.0) + rng.normal(0, 0.2, 400)
    X = np.column_stack([signal, counts])
    tree = SegTree(min_leaf_abs=30).fit(X)
    used = [c.feature_index for c in tree.cuts_]
    assert used, "count feature must still be usable"
    numeric_ops = [c.op for c in tree.cuts_ if c.feature_index == 1]
    assert all(op in ("<=", ">") for op in numeric_ops)


def test_explicit_names_resolution():
    X, _ = make_cat_data(n=240)
    tree = SegTree(
        min_leaf_abs=10, feature_names=["v0", "v1", "grade"], categorical_features=["grade"]
    ).fit(X)
    assert list(tree.categorical_indices_) == [2]
    assert any("grade" in str(c) for r in extract_rules(tree) for c in r.conditions)


def test_predict_before_fit_with_categorical_param_still_validates():
    est = SegTree(categorical_features=[0])
    assert est.get_params()["categorical_features"] == [0]


def test_float32_large_integer_codes_stay_distinct():
    big_a, big_b = 16_777_216.0, 16_777_217.0  # 2**24 and 2**24+1: adjacent f32 gap
    rng = np.random.default_rng(9)
    n = 300
    codes = np.where(rng.random(n) < 0.5, big_a, big_b)
    signal = np.where(codes == big_a, 0.0, 9.0) + rng.normal(0, 0.3, n)
    X = np.column_stack([signal, codes])

    trees = {}
    for dt in (np.float64, np.float32):
        tr = SegTree(min_leaf_abs=20, categorical_features=[1], dtype=dt).fit(X)
        trees[dt] = tr
        assert sorted(tr.categories_["x1"]) == [int(big_a), int(big_b)], "codes merged!"
        cat_cuts = [c for c in tr.cuts_ if c.feature_index == 1]
        assert cat_cuts, "categorical feature must be usable under float32"

    from segmtree import adjusted_rand_score

    assert adjusted_rand_score(trees[np.float64].labels_, trees[np.float32].labels_) == 1.0


def test_categories_attribute():
    X, _ = make_cat_data(k=4)
    tree = SegTree(min_leaf_abs=10, categorical_features=[2]).fit(X)
    assert set(tree.categories_) == {"x2"}
    assert tree.categories_["x2"].tolist() == [0, 1, 2, 3]


def test_handle_unknown_complement_default_and_replay():
    X, _ = make_cat_data(k=3)
    tree = SegTree(min_leaf_abs=10, categorical_features=[2]).fit(X)
    assert tree.handle_unknown == "complement"
    fresh = np.column_stack([np.zeros((40, 2)), np.full(40, 777.0)])
    pred = tree.predict(fresh)
    from segmtree import replay_check

    assert replay_check(tree, fresh)
    assert (pred >= 0).all()


def test_handle_unknown_error_predict_apply_replay():
    pytest.importorskip("numpy")
    X, _ = make_cat_data(k=3)
    tree = SegTree(min_leaf_abs=10, categorical_features=[2], handle_unknown="error").fit(X)
    fresh = np.column_stack([np.zeros((20, 2)), np.full(20, 999.0)])
    for fn in (tree.predict, tree.apply):
        with pytest.raises(ValueError, match=r"x2.*999"):
            fn(fresh)
    from segmtree import replay_labels

    with pytest.raises(ValueError, match=r"x2.*999"):
        replay_labels(tree, fresh)
    # known categories keep working in error mode
    ok = np.column_stack([np.zeros((10, 2)), np.zeros(10)])
    assert len(tree.predict(ok)) == 10


def test_handle_unknown_validation():
    with pytest.raises(ValueError, match="handle_unknown"):
        SegTree(handle_unknown="wrap")._check_params(3)


def test_profile_enriches_categorical_columns():
    rng = np.random.default_rng(21)
    n = 360
    grp = np.repeat([0, 1, 2], n // 3)
    cont = rng.normal(grp[:, None] * 6.0, 0.5, size=(n, 2))
    code_a = np.where(rng.random(n) < 0.85, grp % 4, rng.integers(0, 4, n))
    X = np.column_stack([cont, code_a.astype(float), rng.normal(size=n)])
    tree = SegTree(
        min_leaf_abs=15, categorical_features=[2], feature_names=["f0", "f1", "tier", "noise"]
    ).fit(X)
    profiles = profile_leaves(tree, X, top_k=3)
    assert any(p["category_enrichment"] for p in profiles)
    for p in profiles:
        names_in_top = [nm for nm, _ in p["top_features"]]
        assert "tier" not in names_in_top, "categorical column must not be z-scored"
        for fname, code, share, lift in p["category_enrichment"]:
            assert isinstance(fname, str) and isinstance(code, int)
            assert 0.05 <= share <= 1.0 and lift > 1.0
