from __future__ import annotations

import numpy as np
import pytest

from segmtree import NotFittedError, SegTree, adjusted_rand_score, extract_rules


def test_basic_fit_shapes_and_labels(blobs):
    tree = SegTree(min_leaf_abs=30).fit(blobs)
    assert tree.n_leaves_ >= 2
    assert tree.labels_.shape == (600,)
    assert set(np.unique(tree.labels_)) == set(range(tree.n_leaves_))
    sizes = np.bincount(tree.labels_, minlength=tree.n_leaves_)
    assert (sizes == [lf.n_samples for lf in tree.leaves_]).all()
    assert abs(sum(lf.fraction for lf in tree.leaves_) - 1.0) < 1e-9
    assert 0.0 < tree.reduction_ <= 1.0
    assert tree.h_root_ == 1.0


def test_gain_additivity_self_consistency(mixed_x, mixed_params):
    """Weighted sum of gains must equal H(root) - H(avg)."""
    tree = SegTree(**mixed_params).fit(mixed_x)
    gap = abs(tree.total_gain_ - (tree.h_root_ - tree.h_avg_))
    assert gap / max(1e-12, tree.h_root_ - tree.h_avg_) < 1e-9


def test_recorded_cut_gain_matches_committed_partition():
    """Regression: the gain recorded for a cut must be the gain of the exact
    partition that was committed (guards against evaluation/commit drift when
    a quantile knot coincides with a data point)."""
    rng = np.random.default_rng(3)
    X = rng.integers(0, 50, size=(500, 3)).astype(float)
    X[:, 0] += rng.normal(0, 0.01, 500)
    tree = SegTree(min_leaf_frac=0.05, min_leaf_abs=25, quantiles=8).fit(X)

    rv = np.var(X, axis=0)
    rv[rv < 1e-12] = 1e-12

    def brute_gain(node_idx, j, thr):
        node = X[node_idx]
        hs = float(np.mean(np.var(node, axis=0) / rv))
        L = node[node[:, j] <= thr]
        R = node[node[:, j] > thr]
        return (
            hs
            - len(L) / len(node) * float(np.mean(np.var(L, axis=0) / rv))
            - len(R) / len(node) * float(np.mean(np.var(R, axis=0) / rv))
        )

    def split_of(region, cut):
        if cut.op == "in":
            gl = np.isin(X[region, cut.feature_index], cut.categories)
        else:
            gl = X[region, cut.feature_index] <= cut.threshold
        return region[gl], region[~gl]

    regions = [np.arange(len(X))]
    for cut in tree.cuts_:

        def matches(region, cut=cut):
            left, right = split_of(region, cut)
            return len(left) == cut.n_left and len(right) == cut.n_right

        candidates = [r for r in regions if len(r) == cut.n_samples]
        parent = next(r for r in candidates if matches(r))
        del regions[next(i for i, r in enumerate(regions) if r is parent)]
        left, right = split_of(parent, cut)
        if cut.op != "in":  # subset-cut gains use category blocks; covered by equivalence tests
            assert abs(cut.gain - brute_gain(parent, cut.feature_index, cut.threshold)) < 1e-9
        regions.extend([left, right])
    assert not regions or all(len(r) >= tree.min_leaf_ for r in regions)


def test_predict_matches_train_labels(blobs):
    tree = SegTree(min_leaf_abs=30).fit(blobs)
    assert np.array_equal(tree.predict(blobs), tree.labels_)
    assert np.array_equal(tree.fit_predict(blobs), tree.labels_)
    assert np.array_equal(tree.apply(blobs), tree.labels_)


def test_min_leaf_respected(mixed_x, mixed_params):
    tree = SegTree(**mixed_params).fit(mixed_x)
    for lf in tree.leaves_:
        assert lf.n_samples >= tree.min_leaf_
        assert lf.n_samples >= max(
            mixed_params["min_leaf_abs"], int(mixed_params["min_leaf_frac"] * 600)
        )


def test_max_depth_respected(mixed_x):
    tree = SegTree(min_leaf_abs=10, min_leaf_frac=0.01, max_depth=2).fit(mixed_x)
    assert all(lf.depth <= 2 for lf in tree.leaves_)


def test_deterministic_refit(mixed_x, mixed_params):
    a = SegTree(**mixed_params).fit(mixed_x).labels_
    b = SegTree(**mixed_params).fit(mixed_x.copy()).labels_
    assert np.array_equal(a, b)


def test_binary_auto_detection(mixed_x, mixed_params):
    tree = SegTree(**mixed_params).fit(mixed_x)
    assert set(tree.binary_indices_) == {4, 5, 6}
    binary_cuts = [c for c in tree.cuts_ if c.feature_index in (4, 5, 6)]
    for cut in binary_cuts:
        assert cut.threshold == pytest.approx(0.5)


def test_binary_explicit_indices_and_names():
    X = np.column_stack([np.r_[np.zeros(50), np.ones(50)], np.arange(100) / 100])
    by_index = SegTree(min_leaf_abs=10, binary_features=[0]).fit(X)
    assert list(by_index.binary_indices_) == [0]
    by_name = SegTree(min_leaf_abs=10, binary_features=["flag"], feature_names=["flag", "v"]).fit(X)
    assert list(by_name.binary_indices_) == [0]


def test_constant_feature_ignored():
    rng = np.random.default_rng(3)
    X = np.hstack([rng.normal([0, 8], 0.4, size=(200, 2)), np.full((200, 1), 3.7)])
    tree = SegTree(min_leaf_abs=20).fit(X)
    assert tree.n_leaves_ >= 2
    assert all(cut.feature_index != 2 for cut in tree.cuts_)


def test_single_leaf_when_no_split_worthwhile():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(120, 3))
    tree = SegTree(min_gain=10.0, min_leaf_abs=10).fit(X)
    assert tree.n_leaves_ == 1
    assert tree.reduction_ == pytest.approx(0.0)
    assert np.all(tree.labels_ == 0)
    assert tree.leaves_[0].conditions == ()


def test_float32_mode_agrees_with_float64(mixed_x, mixed_params):
    a = SegTree(**mixed_params).fit(mixed_x)
    b = SegTree(**{**mixed_params, "dtype": np.float32}).fit(mixed_x)
    assert adjusted_rand_score(a.labels_, b.labels_) > 0.95


def test_dataframe_input_preserves_names():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "signal": np.r_[np.zeros(40), np.ones(40) * 5, np.ones(40) * 5],
            "flag": np.r_[np.ones(40), np.ones(40), np.zeros(40)],
        }
    )
    tree = SegTree(min_leaf_abs=10).fit(df)
    assert tree.feature_names_ == ["signal", "flag"]
    rules_text = " | ".join(str(r) for r in extract_rules(tree))
    assert "signal" in rules_text and "flag" in rules_text


def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="NaN"):
        SegTree().fit(np.array([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError, match=r"NaN|infinite"):
        SegTree().fit(np.array([[np.inf, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="2-D"):
        SegTree().fit(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError, match="numeric"):
        SegTree().fit(np.array([["a", "b"], ["c", "d"]]))
    with pytest.raises(ValueError, match="empty"):
        SegTree().fit(np.empty((0, 2)))


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError, match="not fitted"):
        SegTree().predict(np.zeros((3, 2)))


def test_predict_validates_feature_count(blobs):
    tree = SegTree(min_leaf_abs=30).fit(blobs)
    with pytest.raises(ValueError, match="features"):
        tree.predict(np.zeros((5, 3)))


def test_param_validation():
    for kwargs, exc in [
        (dict(min_gain=-0.1), ValueError),
        (dict(min_leaf_frac=1.0), ValueError),
        (dict(min_leaf_abs=0), ValueError),
        (dict(max_depth=0), ValueError),
        (dict(quantiles=1), ValueError),
        (dict(quantiles=[0.5, 1.5]), ValueError),
        (dict(binary_features="yes"), ValueError),
        (dict(dtype=np.int64), ValueError),
    ]:
        with pytest.raises(exc):
            SegTree(**kwargs)._check_params(3)


def test_sklearn_compat_clone_and_grid():
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV

    est = SegTree(min_leaf_abs=5)
    params = est.get_params()
    assert params["min_gain"] == 0.01
    cloned = clone(est)
    assert cloned.get_params() == params
    cloned.set_params(min_gain=0.2)
    assert cloned.min_gain == 0.2

    X = np.r_[np.zeros((60, 2)) + 0.1, np.ones((60, 2)) + 9.9]
    search = GridSearchCV(
        SegTree(min_leaf_abs=10),
        {"min_gain": [0.001, 0.05]},
        cv=3,
        scoring=lambda est_, X_: float(est_.reduction_),
    ).fit(X)
    assert isinstance(search.best_estimator_, SegTree)


def test_summary_dict(blobs):
    tree = SegTree(min_leaf_abs=30).fit(blobs)
    info = tree.summary()
    assert info["n_samples"] == 600 and info["n_leaves"] == tree.n_leaves_
    assert info["n_splits"] == len(tree.cuts_)
    assert info["depth_max"] <= info["n_splits"]


def test_heldout_points_lands_in_correct_region(blobs):
    tree = SegTree(min_leaf_abs=30).fit(blobs)
    rng = np.random.default_rng(99)
    fresh = np.vstack([rng.normal(0, 0.4, (50, 2)), rng.normal(6, 0.4, (50, 2))])
    pred = tree.predict(fresh)
    counts = np.bincount(pred, minlength=tree.n_leaves_)
    dominant = sum(s for s in counts if s > 25)
    assert dominant >= 80  # most fresh points fall into the two big segments
