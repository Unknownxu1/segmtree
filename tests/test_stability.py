from __future__ import annotations

import numpy as np
import pytest

from segmtree import NotFittedError, SegTree, adjusted_rand_score, bootstrap_stability


def test_ari_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(5, 300))
        a = rng.integers(0, int(rng.integers(1, 8)), n)
        b = rng.permutation(a) if rng.random() < 0.5 else rng.integers(0, 6, n)
        assert adjusted_rand_score(a, b) == pytest.approx(
            sklearn_metrics.adjusted_rand_score(a, b), abs=1e-12
        )


def test_ari_degenerate_cases():
    single = np.zeros(10, dtype=int)
    assert adjusted_rand_score(single, single) == 1.0
    two = np.r_[np.zeros(5), np.ones(5)].astype(int)
    assert adjusted_rand_score(single, two) == pytest.approx(0.0)
    assert adjusted_rand_score(two, two) == 1.0
    assert adjusted_rand_score(two, 1 - two) == 1.0  # permutation invariant
    with pytest.raises(ValueError):
        adjusted_rand_score(single, two[:5])


def test_stability_high_for_separated_blobs(blobs):
    tree = SegTree(min_leaf_abs=30).fit(blobs)
    result = bootstrap_stability(tree, blobs, n_replicates=5, random_state=0)
    assert result.mean > 0.9
    assert result.scores.shape == (5,)
    assert result.n_replicates == 5
    assert ((result.scores >= -1) & (result.scores <= 1)).all()
    assert result.std == pytest.approx(result.scores.std())


def test_stability_deterministic_given_seed(mixed_x, mixed_params):
    tree = SegTree(**mixed_params).fit(mixed_x)
    a = bootstrap_stability(tree, mixed_x, n_replicates=3, random_state=123)
    b = bootstrap_stability(tree, mixed_x, n_replicates=3, random_state=123)
    assert np.array_equal(a.scores, b.scores)


def test_stability_input_validation():
    tree = SegTree(min_leaf_abs=5)
    X = np.zeros((20, 2))
    with pytest.raises(NotFittedError, match="fitted"):
        bootstrap_stability(tree, X)

    train = np.r_[np.zeros((10, 2)), np.ones((10, 2)) * 5]
    fitted = SegTree(min_leaf_abs=5).fit(train)
    with pytest.raises(ValueError, match="shape"):
        bootstrap_stability(fitted, np.zeros((15, 2)))
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError):
            bootstrap_stability(fitted, train, n_replicates=bad)
