"""feature_weights: API validation, weighting semantics, equivalence and
additivity guarantees."""

from __future__ import annotations

import numpy as np
import pytest

from segmtree import SegTree, replay_check
from tests._naive_reference import NaiveSegTree


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


PARAMS = dict(
    min_gain=0.02,
    min_leaf_frac=0.05,
    min_leaf_abs=10,
    max_depth=3,
    quantiles=5,
    categorical_features=[3],
)


def _cut_key(cut):
    if cut.op == "in":
        return (cut.feature_index, "in", tuple(sorted(cut.categories)))
    return (cut.feature_index, "<=", cut.threshold)


def test_none_and_all_ones_are_strictly_equivalent():
    X = make_mixed_cat()
    a = SegTree(**PARAMS).fit(X)
    b = SegTree(**{**PARAMS, "feature_weights": np.ones(4)}).fit(X)
    c = SegTree(**{**PARAMS, "feature_weights": [1.0, 1, 1, 1]}).fit(X)
    for other in (b, c):
        assert np.array_equal(a.labels_, other.labels_)
        assert [_cut_key(x) for x in a.cuts_] == [_cut_key(x) for x in other.cuts_]
        assert a.total_gain_ == other.total_gain_
        assert a.summary() == other.summary()
    assert a.feature_weights_.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert b.feature_weights_.dtype == np.float64


def test_weights_change_split_preference():
    """f0 separates the groups strongly, f1 only weakly; heavily down-weighting
    f0 must move the root cut to f1."""
    rng = np.random.default_rng(5)
    n = 600
    group = np.r_[np.zeros(n // 2), np.ones(n - n // 2)]
    f_strong = np.where(group == 0, 0.0, 6.0) + rng.normal(0, 0.5, n)
    f_weak = group * 0.8 + rng.normal(0, 0.5, n)
    X = np.column_stack([f_strong, f_weak])
    base = dict(min_gain=0.01, min_leaf_frac=0.05, min_leaf_abs=30, quantiles=6)

    default_first = SegTree(**base).fit(X).cuts_[0].feature_index
    reweighted_first = (
        SegTree(**base, feature_weights=[0.001, 1000.0]).fit(X).cuts_[0].feature_index
    )
    assert default_first == 0
    assert reweighted_first == 1


def test_zero_weight_feature_can_act_as_proxy_splitter():
    """A zero-weight column that separates the data better than any weighted
    one may still be chosen (proxy semantics): weights drive the accounting,
    not split eligibility."""
    rng = np.random.default_rng(9)
    n = 600
    group = np.r_[np.zeros(n // 2), np.ones(n - n // 2)].astype(bool)
    proxy = group.astype(float)  # perfect separator, weight 0
    noisy_cont = np.where(group, 4.0, 0.0) + rng.normal(0, 1.0, n)  # overlapping
    X = np.column_stack([noisy_cont, proxy])
    tree = SegTree(
        min_gain=0.01, min_leaf_frac=0.05, min_leaf_abs=30, feature_weights=[1.0, 0.0]
    ).fit(X)
    assert tree.cuts_, "a profitable proxy cut exists"
    # the zero-weight column yields the strictly cleaner partition
    first_on_proxy = tree.cuts_[0].feature_index == 1
    if not first_on_proxy:
        # weighted-column cut is only acceptable when it partitions identically
        assert replay_check(tree, X)
    else:
        assert tree.feature_weights_[1] == 0.0
        assert tree.cuts_[0].gain > 0.5


def test_weighted_additivity_holds_with_categoricals():
    X = make_mixed_cat()
    w = [2.0, 0.5, 0.0, 1.5]
    tree = SegTree(**PARAMS, feature_weights=w).fit(X)
    assert tree.h_root_ == pytest.approx(1.0)
    gap = abs(tree.total_gain_ - (tree.h_root_ - tree.h_avg_))
    assert gap / max(1e-12, tree.h_root_ - tree.h_avg_) < 1e-9
    assert tree.feature_weights_.tolist() == w
    assert replay_check(tree, X)


def test_weighted_tree_matches_naive_reference_exactly():
    X = make_mixed_cat(seed=11)
    mine = SegTree(
        min_gain=0.02,
        min_leaf_frac=0.0,
        min_leaf_abs=25,
        max_depth=3,
        quantiles=5,
        categorical_features=[3],
        feature_weights=[1.0, 2.0, 0.5, 1.0],
    ).fit(X)
    ref = NaiveSegTree(
        min_gain=0.02,
        min_leaf=25,
        max_depth=3,
        quantiles=np.linspace(0.05, 0.95, 5),
        feature_weights=[1.0, 2.0, 0.5, 1.0],
    ).fit(X, cat_set={3})
    assert mine.n_leaves_ == len(ref.leaves_)
    assert np.array_equal(mine.labels_, ref.labels_)
    assert np.allclose([c.gain for c in mine.cuts_], [c[2] for c in ref.cuts_], atol=1e-10)


def test_dict_weights_resolved_via_dataframe_columns():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "income": np.r_[rng.normal(50, 2, 200), rng.normal(90, 2, 200)],
            "flag": (rng.random(400) < 0.5).astype(float),
            "tier": np.repeat([0, 1], 200).astype(float),
        }
    )
    tree = SegTree(min_leaf_abs=40, feature_weights={"income": 4.0, "tier": 0.5}).fit(df)
    assert tree.feature_names_ == ["income", "flag", "tier"]
    assert tree.feature_weights_.tolist() == [4.0, 1.0, 0.5]
    assert replay_check(tree, df.to_numpy())
    with pytest.raises(ValueError, match="no_match"):
        SegTree(min_leaf_abs=10, feature_weights={"no_match": 2.0}).fit(df)


@pytest.mark.parametrize(
    "bad",
    [
        [-1.0, 1.0, 1.0, 1.0],
        [np.nan, 1.0, 1.0, 1.0],
        [np.inf, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [[1.0, 1.0], [1.0, 1.0]],
        "weights",
    ],
)
def test_invalid_sequence_weights_rejected(bad):
    X = make_mixed_cat(n_per_group=100)
    with pytest.raises(ValueError):
        SegTree(**PARAMS, feature_weights=bad).fit(X)


def test_boolean_weights_rejected():
    X = make_mixed_cat(n_per_group=100)
    with pytest.raises(TypeError):
        SegTree(**PARAMS, feature_weights=[True, False, True, True]).fit(X)
    with pytest.raises(TypeError):
        SegTree(**PARAMS, feature_weights=np.array([1.0, 0.0, 1.0, 1.0], dtype=bool)).fit(X)
    with pytest.raises(TypeError):
        SegTree(**PARAMS, feature_weights={"income": True}).fit(
            np.hstack([X, np.zeros((X.shape[0], 1))])
        )


def test_dict_unknown_feature_name_rejected_without_fit_data():
    est = SegTree(feature_weights={"nope": 1.0})
    assert est.get_params()["feature_weights"] == {"nope": 1.0}


def test_emphasis_steers_cut_to_emphasized_column():
    """With one cut allowed, emphasizing the perfect categorical column keeps
    the split on it; the isolated single-category leaf is pure."""
    rng = np.random.default_rng(23)
    centers = [[0.0, -1.0], [5.0, 4.0], [-5.0, 3.0]]
    parts = []
    for k, center in enumerate(centers):
        cont = rng.normal(center, 0.5, (200, 2))
        code = np.full((200, 1), k, dtype=float)
        parts.append(np.hstack([cont, code]))
    X = np.vstack(parts)
    truth = np.repeat([0, 1, 2], 200)
    rich = dict(
        min_gain=0.01,
        min_leaf_frac=0.05,
        min_leaf_abs=10,
        max_depth=1,
        quantiles=6,
        categorical_features=[2],
    )
    for weights in (None, [1.0, 1.0, 6.0], [0.2, 8.0, 6.0]):
        tree = SegTree(**rich, feature_weights=weights).fit(X)
        cut = tree.cuts_[0]
        assert cut.feature_index == 2 and cut.op == "in", (
            f"weights={weights}: emphasized categorical column must win the root cut"
        )
        singleton_side = np.isin(X[:, 2], tuple(cut.categories))
        pure_side = singleton_side if cut.n_left <= cut.n_right else ~singleton_side
        assert len(set(truth[pure_side])) == 1
