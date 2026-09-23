"""Category-recoding invariance tests.

The heterogeneity contribution of a declared categorical feature is its Gini
impurity relative to the root Gini — a function of the code *distribution*
only — so any one-to-one remapping of the codes must yield an equivalent
partition (leaf ids may permute).
"""

from __future__ import annotations

import numpy as np
import pytest

from segmtree import SegTree, adjusted_rand_score


def make_nominal_data(n_per_group=140, seed=5):
    """Three latent groups over 4 continuous + 2 nominal features."""
    rng = np.random.default_rng(seed)
    centers = [[0.0, -1.0, 2.0, 0.5], [5.0, 4.0, -3.0, 0.5], [-5.0, 3.0, 1.0, 8.0]]
    parts = []
    for k, center in enumerate(centers):
        cont = rng.normal(center, 0.6, size=(n_per_group, 4))
        own = np.full(n_per_group, k)
        col_a = np.where(rng.random(n_per_group) < 0.78, own, rng.integers(0, 4, n_per_group))
        col_b = np.where(
            rng.random(n_per_group) < 0.72,
            (k * 2 + 1) % 3,
            rng.integers(0, 3, n_per_group),
        )
        parts.append(np.column_stack([cont, col_a.astype(float), col_b.astype(float)]))
    return np.vstack(parts)


PARAMS = dict(
    min_gain=0.02,
    min_leaf_frac=0.04,
    min_leaf_abs=15,
    max_depth=4,
    quantiles=6,
    categorical_features=[4, 5],
)


@pytest.fixture()
def base_tree():
    X = make_nominal_data()
    return X, SegTree(**PARAMS).fit(X)


def _permute(X, perm_a, perm_b):
    Xp = X.copy()
    Xp[:, 4] = perm_a[X[:, 4].astype(int)]
    Xp[:, 5] = perm_b[X[:, 5].astype(int)]
    return Xp


def test_partition_invariant_under_code_permutations(base_tree):
    X, tree = base_tree
    rng = np.random.default_rng(0)
    for trial in range(5):
        perm_a = rng.permutation(4)
        perm_b = rng.permutation(3)
        Xp = _permute(X, perm_a, perm_b)
        tp = SegTree(**PARAMS).fit(Xp)
        assert adjusted_rand_score(tree.labels_, tp.labels_) == pytest.approx(1.0), (
            f"permutation trial {trial}: partitions must be equivalent"
        )
        assert list(tp.categories_["x4"]) == sorted(perm_a.tolist())


def test_gain_additivity_with_categorical_features(base_tree):
    _, tree = base_tree
    gap = abs(tree.total_gain_ - (tree.h_root_ - tree.h_avg_))
    assert gap / max(1e-12, tree.h_root_ - tree.h_avg_) < 1e-9


def test_rules_replay_after_recoding(base_tree):
    X, tree = base_tree
    from segmtree import replay_check

    rng = np.random.default_rng(3)
    perm_a = rng.permutation(4)
    Xp = _permute(X, perm_a, np.arange(3))
    tp = SegTree(**PARAMS).fit(Xp)
    assert replay_check(tp, Xp)
    # same partition => same rule semantics under mapping
    from segmtree import extract_rules

    def rule_sets(rules):
        out = set()
        for r in rules:
            key = []
            for c in r.conditions:
                if c.op in ("in", "not in"):
                    mapped = tuple(sorted(c.categories))
                    key.append(f"{c.feature} {c.op} {mapped}")
                else:
                    key.append(f"{c.feature} {c.op} {c.threshold:.4g}")
            out.add(tuple(key))
        return out

    base_rules = extract_rules(tree)
    assert len(base_rules) == len(extract_rules(tp))
    assert rule_sets(extract_rules(tp))  # extraction works on recoded fit


def test_exact_agreement_with_reference_on_nominal_data():
    from tests._naive_reference import NaiveSegTree

    X = make_nominal_data(seed=11)
    mine = SegTree(**{**PARAMS, "min_leaf_abs": 25}).fit(X)
    # the reference must see the *effective* minimum leaf size mine used
    ref = NaiveSegTree(
        min_gain=PARAMS["min_gain"],
        min_leaf=mine.min_leaf_,
        max_depth=PARAMS["max_depth"],
        quantiles=np.linspace(0.05, 0.95, PARAMS["quantiles"]),
    ).fit(X, binary_set=set(), cat_set={4, 5})
    assert adjusted_rand_score(mine.labels_, ref.labels_) == pytest.approx(1.0)
