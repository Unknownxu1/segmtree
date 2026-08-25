"""Bootstrap stability assessment for fitted trees."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._base import as_2d_array
from .tree import SegTree

__all__ = ["StabilityResult", "adjusted_rand_score", "bootstrap_stability"]


@dataclass(frozen=True)
class StabilityResult:
    """Bootstrap stability of a segmentation.

    ``scores[i]`` is the Adjusted Rand Index between the original partition
    and the partition induced on the full data by a tree refit on bootstrap
    replicate ``i``. Values above ~0.7 indicate a stable structure.
    """

    scores: np.ndarray
    mean: float
    std: float
    n_replicates: int


def adjusted_rand_score(labels_a, labels_b):
    """Adjusted Rand Index between two labelings (pure numpy).

    Matches :func:`sklearn.metrics.adjusted_rand_score` numerically; provided
    to keep the package dependency-free.
    """
    a = np.asarray(labels_a).ravel()
    b = np.asarray(labels_b).ravel()
    if a.shape[0] != b.shape[0]:
        raise ValueError("Label arrays must have the same length.")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    na, nb, n = int(ai.max()) + 1, int(bi.max()) + 1, a.shape[0]
    contingency = np.bincount(ai * nb + bi, minlength=na * nb).astype(np.int64).reshape(na, nb)

    def comb2(x):
        return x * (x - 1) // 2

    sum_comb = comb2(contingency.astype(np.float64)).sum()
    sum_a = comb2(contingency.sum(axis=1).astype(np.float64)).sum()
    sum_b = comb2(contingency.sum(axis=0).astype(np.float64)).sum()
    expected = sum_a * sum_b / comb2(float(n))
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0.0:
        return 1.0
    return float((sum_comb - expected) / denom)


def bootstrap_stability(tree, X, *, n_replicates=12, random_state=None):
    """Refit the tree on bootstrap resamples and measure agreement with the fit.

    Parameters
    ----------
    tree : SegTree
        A fitted tree; its hyperparameters are reused for every replicate.
    X : array-like, shape (n_samples, n_features)
        The training matrix used to fit ``tree``.
    n_replicates : int, default=12
        Number of bootstrap replicates.
    random_state : int or None, default=None
        Seed for reproducibility.

    Returns
    -------
    StabilityResult
    """
    if not hasattr(tree, "labels_"):
        raise RuntimeError("tree must be fitted before calling bootstrap_stability().")
    if isinstance(n_replicates, bool) or not isinstance(n_replicates, int) or n_replicates < 1:
        raise ValueError("n_replicates must be an int >= 1.")
    Xarr, _ = as_2d_array(X, context="X")
    if Xarr.shape != (tree.n_samples_in_, tree.n_features_in_):
        raise ValueError(
            f"X must be the training matrix with shape "
            f"{(tree.n_samples_in_, tree.n_features_in_)}, got {Xarr.shape}."
        )
    params = tree.get_params()
    base_labels = tree.labels_
    rng = np.random.default_rng(random_state)
    scores = np.empty(n_replicates, dtype=np.float64)
    for b in range(n_replicates):
        idx = rng.integers(0, Xarr.shape[0], Xarr.shape[0])
        replica = SegTree(**params).fit(Xarr[idx])
        scores[b] = adjusted_rand_score(base_labels, replica.predict(Xarr))
    return StabilityResult(
        scores=scores,
        mean=float(scores.mean()),
        std=float(scores.std()),
        n_replicates=n_replicates,
    )
