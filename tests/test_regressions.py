"""Regression tests for edge cases and new API surface (v0.5.0)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from segmtree import (
    Condition,
    LeafProfile,
    NotFittedError,
    Rule,
    SegTree,
    adjusted_rand_score,
    profile_leaves,
    replay_check,
    replay_labels,
    rule_mask,
)

# ── correctness regressions ──────────────────────────────────────────────


def test_single_value_categorical_column_does_not_crash():
    """C1: a categorical column with one unique code must not divide by zero."""
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=200), np.full(200, 4.0)])
    tree = SegTree(min_leaf_abs=10, categorical_features=[1], binary_features=None).fit(X)
    assert tree.n_leaves_ >= 1
    assert tree.categories_["x1"].tolist() == [4]


def test_negative_category_codes_handled():
    """C2: negative codes (e.g. factorize -1) must not crash bincount."""
    rng = np.random.default_rng(1)
    codes = np.where(rng.random(300) < 0.5, -1, 3)
    X = np.column_stack([rng.normal(size=300), codes.astype(float)])
    tree = SegTree(min_leaf_abs=15, categorical_features=[1]).fit(X)
    assert sorted(tree.categories_["x1"].tolist()) == [-1, 3]
    assert replay_check(tree, X)


def test_large_category_codes_handled():
    """C2: very large codes must not trigger massive bincount allocation."""
    rng = np.random.default_rng(2)
    codes = np.where(rng.random(300) < 0.5, 10**10, 10**10 + 1)
    X = np.column_stack([rng.normal(size=300), codes.astype(float)])
    tree = SegTree(min_leaf_abs=15, categorical_features=[1]).fit(X)
    assert tree.categories_["x1"].tolist() == [10**10, 10**10 + 1]


def test_float32_predict_consistent_with_labels():
    """C3: predict on training data must equal labels_ even under float32."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    X[:, 0] += 1e-10  # adversarial near-boundary values
    X[::7, 0] += 1.0000000001
    tree = SegTree(min_leaf_abs=20, dtype=np.float32).fit(X)
    assert np.array_equal(tree.predict(X), tree.labels_)
    assert replay_check(tree, X)


def test_complex_input_rejected():
    """C4: complex dtype must raise, not silently drop the imaginary part."""
    X = np.array([[1 + 2j, 3], [4, 5 + 6j]])
    with pytest.raises(TypeError, match="complex"):
        SegTree().fit(X)


def test_masked_array_rejected():
    X = np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=[[False, True], [False, False]])
    with pytest.raises(TypeError, match="masked"):
        SegTree().fit(X)


def test_min_gain_nan_rejected():
    with pytest.raises(ValueError, match="min_gain"):
        SegTree(min_gain=float("nan"))._check_params(3)


def test_min_gain_bool_rejected():
    with pytest.raises(ValueError, match="min_gain"):
        SegTree(min_gain=True)._check_params(3)


def test_replay_labels_validates_feature_count():
    tree = SegTree(min_leaf_abs=5).fit(np.random.default_rng(0).normal(size=(50, 3)))
    X5 = np.random.default_rng(1).normal(size=(10, 5))
    with pytest.raises(ValueError, match="features"):
        replay_labels(tree, X5)


def test_ari_single_sample_returns_one():
    assert adjusted_rand_score([0], [0]) == 1.0
    assert adjusted_rand_score([5], [3]) == 1.0


def test_ari_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        adjusted_rand_score([], [])


# ── sklearn compatibility ────────────────────────────────────────────────


def test_not_fitted_error_type():
    tree = SegTree()
    with pytest.raises(NotFittedError):
        tree.predict(np.zeros((3, 2)))
    with pytest.raises(NotFittedError):
        tree.export_dict()
    with pytest.raises(NotFittedError):
        tree.export_text()
    with pytest.raises(NotFittedError):
        tree.summary()
    with pytest.raises(NotFittedError):
        tree.score(np.zeros((3, 2)))


def test_score_method():
    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal(0, 0.5, (200, 3)), rng.normal(5, 0.5, (200, 3))])
    tree = SegTree(min_leaf_abs=20).fit(X)
    s = tree.score(X)
    assert 0.0 < s <= 1.0
    assert s == pytest.approx(tree.reduction_, abs=0.05)


def test_score_before_fit_raises():
    with pytest.raises(NotFittedError):
        SegTree().score(np.zeros((3, 2)))


def test_feature_importances():
    rng = np.random.default_rng(8)
    X = np.vstack([rng.normal(0, 0.5, (150, 3)), rng.normal(6, 0.5, (150, 3))])
    tree = SegTree(min_leaf_abs=15).fit(X)
    fi = tree.feature_importances_
    assert fi.shape == (3,)
    assert fi.sum() == pytest.approx(1.0) or fi.sum() == 0.0
    assert (fi >= 0).all()


def test_transform_one_hot():
    rng = np.random.default_rng(9)
    X = np.vstack([rng.normal(0, 0.5, (100, 2)), rng.normal(5, 0.5, (100, 2))])
    tree = SegTree(min_leaf_abs=10).fit(X)
    T = tree.transform(X)
    assert T.shape == (200, tree.n_leaves_)
    assert (T.sum(axis=1) == 1.0).all()
    assert np.array_equal(T.argmax(axis=1), tree.labels_)


def test_feature_names_in_present():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(50, 3))
    tree = SegTree(min_leaf_abs=5, feature_names=["a", "b", "c"]).fit(X)
    assert tree.feature_names_in_.tolist() == ["a", "b", "c"]
    assert tree.feature_names_ == ["a", "b", "c"]


def test_predict_column_reorder_aligned():
    """DataFrame columns in a different order are reindexed, not mis-scored."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {"a": np.r_[np.zeros(50), np.ones(50) * 5], "b": np.r_[np.ones(50), np.zeros(50)]}
    )
    tree = SegTree(min_leaf_abs=10).fit(df)
    df_swapped = df[["b", "a"]]
    assert np.array_equal(tree.predict(df), tree.predict(df_swapped))


def test_pickle_round_trip():
    rng = np.random.default_rng(11)
    X = np.vstack([rng.normal(0, 0.5, (100, 3)), rng.normal(5, 0.5, (100, 3))])
    tree = SegTree(min_leaf_abs=10).fit(X)
    blob = pickle.dumps(tree)
    tree2 = pickle.loads(blob)
    assert np.array_equal(tree.predict(X), tree2.predict(X))
    assert tree.export_text() == tree2.export_text()


# ── persistence ──────────────────────────────────────────────────────────


def test_export_dict_from_dict_round_trip():
    rng = np.random.default_rng(12)
    X = np.vstack([rng.normal(0, 0.5, (150, 3)), rng.normal(5, 0.5, (150, 3))])
    tree = SegTree(min_leaf_abs=15).fit(X)
    d = tree.export_dict()
    tree2 = SegTree.from_dict(d)
    assert np.array_equal(tree.predict(X), tree2.predict(X))
    assert tree.export_text() == tree2.export_text()


def test_from_dict_categorical():
    rng = np.random.default_rng(13)
    codes = rng.integers(0, 3, 300)
    X = np.column_stack([rng.normal(size=300), codes.astype(float)])
    tree = SegTree(min_leaf_abs=15, categorical_features=[1]).fit(X)
    d = tree.export_dict()
    tree2 = SegTree.from_dict(d)
    assert np.array_equal(tree.predict(X), tree2.predict(X))
    assert replay_check(tree2, X)


# ── API consistency ──────────────────────────────────────────────────────


def test_rule_mask_exported():
    from segmtree import rule_mask as rm

    assert rm is rule_mask


def test_condition_validation():
    with pytest.raises(ValueError, match="op"):
        Condition("x", "??")
    with pytest.raises(ValueError, match="threshold"):
        Condition("x", "<=")
    with pytest.raises(ValueError, match="categories"):
        Condition("x", "in")
    c = Condition("x", "<=", threshold=3.0)
    assert str(c) == "x <= 3"
    c2 = Condition("y", "in", categories=(1, 2))
    assert "in" in str(c2)


def test_rule_mask_unknown_feature_raises():
    r = Rule(
        leaf_id=0,
        conditions=(Condition("nope", "<=", threshold=1.0),),
        n_samples=1,
        fraction=1.0,
        depth=1,
    )
    with pytest.raises(ValueError, match="nope"):
        rule_mask(r, np.zeros((5, 2)), ["a", "b"])


def test_stability_result_eq():
    from segmtree import StabilityResult

    a = StabilityResult(scores=np.array([1.0, 2.0]), mean=1.5, std=0.5, n_replicates=2)
    b = StabilityResult(scores=np.array([1.0, 2.0]), mean=1.5, std=0.5, n_replicates=2)
    assert a == b
    assert hash(a) == hash(b)


def test_profile_top_k_validation():
    tree = SegTree(min_leaf_abs=5).fit(np.random.default_rng(0).normal(size=(50, 2)))
    X = np.random.default_rng(1).normal(size=(50, 2))
    with pytest.raises(ValueError, match="top_k"):
        profile_leaves(tree, X, top_k=-1)
    with pytest.raises(ValueError, match="top_k"):
        profile_leaves(tree, X, top_k=0)


def test_summary_keys():
    tree = SegTree(min_leaf_abs=5).fit(np.random.default_rng(0).normal(size=(80, 2)))
    s = tree.summary()
    assert "n_leaves" in s
    assert "reduction" in s
    assert "leaf_size_min" in s


def test_export_graphviz():
    rng = np.random.default_rng(14)
    X = np.vstack([rng.normal(0, 0.5, (100, 2)), rng.normal(5, 0.5, (100, 2))])
    tree = SegTree(min_leaf_abs=10).fit(X)
    dot = tree.export_graphviz()
    assert dot.startswith("digraph SegTree")
    assert "leaf #" in dot
    assert "}" in dot


def test_method_wrappers():
    rng = np.random.default_rng(15)
    X = np.vstack([rng.normal(0, 0.5, (100, 2)), rng.normal(5, 0.5, (100, 2))])
    tree = SegTree(min_leaf_abs=10).fit(X)
    assert tree.replay_check(X)
    assert len(tree.extract_rules()) == tree.n_leaves_
    profiles = tree.profile_leaves(X, top_k=2)
    assert len(profiles) == tree.n_leaves_
    assert isinstance(profiles[0], LeafProfile)


def test_quantiles_float_scalar_gives_clear_error():
    with pytest.raises(ValueError, match="quantiles"):
        SegTree(quantiles=8.0)._check_params(3)


def test_binary_features_scalar_rejected():
    with pytest.raises(TypeError, match="scalar"):
        SegTree(binary_features=3)._check_params(3)


def test_categorical_features_scalar_rejected():
    with pytest.raises(TypeError, match="scalar"):
        SegTree(categorical_features=3)._check_params(3)


def test_sample_weight_basic():
    rng = np.random.default_rng(16)
    X = np.vstack([rng.normal(0, 0.5, (100, 2)), rng.normal(5, 0.5, (100, 2))])
    sw = np.ones(200)
    sw[:50] = 10.0  # upweight first cluster
    tree_a = SegTree(min_leaf_abs=10).fit(X)
    tree_b = SegTree(min_leaf_abs=10).fit(X, sample_weight=sw)
    assert tree_a.n_leaves_ >= 1
    assert tree_b.n_leaves_ >= 1


def test_sample_weight_validation():
    X = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.raises(ValueError, match="sample_weight"):
        SegTree().fit(X, sample_weight=np.ones(19))
    with pytest.raises(ValueError, match="non-negative"):
        SegTree().fit(X, sample_weight=-np.ones(20))
