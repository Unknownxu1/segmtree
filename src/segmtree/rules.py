"""Rule extraction, replay validation and segment profiling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from ._base import as_2d_array, check_is_fitted, format_threshold

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

    from .tree import SegTree

__all__ = [
    "Condition",
    "LeafProfile",
    "Rule",
    "extract_rules",
    "profile_leaves",
    "replay_check",
    "replay_labels",
    "rule_mask",
]


def _fmt_codes(categories: Sequence[int]) -> str:
    return "{" + ", ".join(format_threshold(float(c)) for c in sorted(categories)) + "}"


@dataclass(frozen=True)
class Condition:
    """One axis-aligned test.

    ``op`` is ``"<="`` / ``">"`` with a finite float ``threshold``, or
    ``"in"`` / ``"not in"`` with a non-empty tuple of integer category codes
    in ``categories``.
    """

    feature: str
    op: str
    threshold: float = float("nan")
    categories: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.op not in ("<=", ">", "in", "not in"):
            raise ValueError(f"Unknown condition op: {self.op!r}.")
        if self.op in ("<=", ">"):
            if not np.isfinite(self.threshold):
                raise ValueError(
                    f"Condition {self.op!r} requires a finite threshold; got {self.threshold!r}."
                )
            if self.categories:
                raise ValueError(
                    f"Condition {self.op!r} must not carry categories; got {self.categories!r}."
                )
        else:
            if not self.categories:
                raise ValueError(f"Condition {self.op!r} requires a non-empty categories tuple.")

    def __str__(self) -> str:
        if self.op in ("in", "not in"):
            return f"{self.feature} {self.op} {_fmt_codes(self.categories)}"
        return f"{self.feature} {self.op} {format_threshold(self.threshold)}"


@dataclass(frozen=True)
class Rule:
    """The full rule chain defining one segment (leaf)."""

    leaf_id: int
    conditions: tuple[Condition, ...]
    n_samples: int
    fraction: float
    depth: int

    @property
    def text(self) -> str:
        return " AND ".join(str(c) for c in self.conditions) if self.conditions else "TRUE"

    def __str__(self) -> str:
        return f"#{self.leaf_id}: {self.text}"


@dataclass(frozen=True)
class LeafProfile:
    """Distinctive-feature profile of one segment.

    ``top_features`` holds ``(feature_name, z_score)`` for continuous/binary
    features ranked by absolute z-score versus the whole dataset.
    ``category_enrichment`` holds ``(feature_name, code, leaf_share, lift)``
    for categorical columns (empty when the tree has none).
    """

    leaf_id: int
    n_samples: int
    fraction: float
    top_features: tuple[tuple[str, float], ...]
    category_enrichment: tuple[tuple[str, int, float, float], ...]


def extract_rules(tree: SegTree) -> list[Rule]:
    """Return the :class:`Rule` for every fitted segment, in leaf-id order."""
    check_is_fitted(tree)
    rules = []
    for lf in tree.leaves_:
        conds = []
        for name, op, payload in lf.conditions:
            if op in ("in", "not in"):
                conds.append(Condition(name, op, categories=tuple(sorted(payload))))
            else:
                conds.append(Condition(name, op, threshold=float(payload)))
        rules.append(
            Rule(
                leaf_id=lf.id,
                conditions=tuple(conds),
                n_samples=lf.n_samples,
                fraction=lf.fraction,
                depth=lf.depth,
            )
        )
    return rules


def rule_mask(rule: Rule, X: ArrayLike, feature_names: Sequence[str]) -> NDArray[np.bool_]:
    """Boolean mask of rows satisfying ``rule``.

    Evaluated independently of the tree. ``feature_names`` maps column
    positions to the names used in the rule's conditions.
    """
    Xarr = np.asarray(X, dtype=np.float64)
    if Xarr.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {Xarr.shape}.")
    mask = np.ones(Xarr.shape[0], dtype=bool)
    index = {str(nm): i for i, nm in enumerate(feature_names)}
    for cond in rule.conditions:
        if cond.feature not in index:
            raise ValueError(
                f"Rule references feature {cond.feature!r} which is not in "
                f"feature_names {list(feature_names)!r}."
            )
        j = index[cond.feature]
        col = Xarr[:, j]
        if cond.op == "<=":
            mask &= col <= cond.threshold
        elif cond.op == ">":
            mask &= col > cond.threshold
        elif cond.op == "in":
            mask &= np.isin(col, cond.categories)
        else:  # "not in"
            mask &= ~np.isin(col, cond.categories)
    return mask


def replay_labels(tree: SegTree, X: ArrayLike) -> NDArray[np.intp]:
    """Assign labels by replaying every rule chain on ``X``.

    Independent of :meth:`SegTree.predict` (no tree traversal), so agreement
    between the two validates that stored rules reproduce the partition.
    Rows matching no rule receive -1. Honors ``tree.handle_unknown``: with
    ``"error"`` unseen category codes raise ``ValueError``.
    """
    check_is_fitted(tree)
    Xarr, _ = as_2d_array(X, context="X")
    if Xarr.shape[1] != tree.n_features_in_:
        raise ValueError(f"X has {Xarr.shape[1]} features, expected {tree.n_features_in_}.")
    if (
        getattr(tree, "handle_unknown", "complement") == "error"
        and getattr(tree, "categorical_indices_", np.array([])).size
    ):
        cat_codes = tree._encode_input_categories(Xarr)
        tree._check_unknown(Xarr, cat_codes)
    out = np.full(Xarr.shape[0], -1, dtype=np.intp)
    covered = np.zeros(Xarr.shape[0], dtype=bool)
    for rule in extract_rules(tree):
        m = rule_mask(rule, Xarr, tree.feature_names_)
        out[m & ~covered] = rule.leaf_id
        covered |= m
    return out


def replay_check(tree: SegTree, X: ArrayLike) -> bool:
    """True if rule replay reproduces ``tree.predict(X)`` exactly."""
    return bool(np.array_equal(replay_labels(tree, X), tree.predict(X)))


def _category_share(col: NDArray, known: NDArray) -> NDArray[np.float64]:
    """Share of each known category in ``col`` (unknown codes are ignored)."""
    local = np.searchsorted(known, np.rint(col).astype(np.int64))
    counts = np.bincount(local, minlength=known.size + 1)[: known.size]
    return counts / max(counts.sum(), 1)


def profile_leaves(tree: SegTree, X: ArrayLike, top_k: int = 3) -> list[LeafProfile]:
    """Profile each segment by its most distinctive features.

    For every leaf, computes the centroid of ``X`` rows assigned to it.
    Continuous/binary features are ranked by absolute z-score versus the
    whole dataset (positive = above average). Categorical columns are *not*
    z-scored; instead each leaf receives ``category_enrichment``: the most
    enriched (feature, code) pairs by lift of in-leaf share over global share
    (only pairs covering >= 5% of the leaf, lift > 1, capped at ``top_k``,
    sorted by descending lift).

    Parameters
    ----------
    tree : SegTree
        A fitted tree.
    X : array-like, shape (n_samples, n_features)
        Data to profile (typically the training matrix).
    top_k : int, default=3
        Maximum number of top features / enriched categories per leaf
        (must be >= 1).

    Returns
    -------
    list of LeafProfile
    """
    if isinstance(top_k, bool) or not isinstance(top_k, Integral):
        raise ValueError("top_k must be an int >= 1.")
    if int(top_k) < 1:
        raise ValueError("top_k must be an int >= 1.")
    top_k = int(top_k)
    check_is_fitted(tree)
    Xarr, _ = as_2d_array(X, context="X")
    labels = tree.predict(Xarr)
    cat_idx = set(int(j) for j in getattr(tree, "categorical_indices_", []))
    num_cols = [j for j in range(Xarr.shape[1]) if j not in cat_idx]
    mu = Xarr.mean(axis=0)
    sd = Xarr.std(axis=0)
    sd[sd < 1e-12] = 1e-12

    global_dist = {}
    for j in cat_idx:
        known = np.asarray(tree.categories_[tree.feature_names_[j]])
        global_dist[j] = _category_share(Xarr[:, j], known)

    profiles = []
    for lf in tree.leaves_:
        rows_mask = labels == lf.id
        sub = Xarr[rows_mask]
        enrichment = []
        for j in cat_idx:
            known = np.asarray(tree.categories_[tree.feature_names_[j]])
            leaf_share = _category_share(sub[:, j], known)
            if leaf_share.sum() == 0:
                continue
            lift = leaf_share / np.maximum(global_dist[j], 1e-12)
            eligible = np.flatnonzero((leaf_share >= 0.05) & (lift > 1.0))
            for pos in eligible:
                enrichment.append(
                    (
                        tree.feature_names_[j],
                        int(known[pos]),
                        float(leaf_share[pos]),
                        float(lift[pos]),
                    )
                )
        enrichment.sort(key=lambda item: -item[3])
        zcols = sub[:, num_cols] if num_cols else np.empty((len(sub), 0))
        znum = (zcols.mean(axis=0) - mu[num_cols]) / sd[num_cols] if num_cols else ()
        top_order = np.argsort(-np.abs(znum))[:top_k] if num_cols else []
        profiles.append(
            LeafProfile(
                leaf_id=lf.id,
                n_samples=int(sub.shape[0]),
                fraction=sub.shape[0] / Xarr.shape[0],
                top_features=tuple(
                    (tree.feature_names_[num_cols[p]], float(znum[p])) for p in top_order
                ),
                category_enrichment=tuple(enrichment[:top_k]),
            )
        )
    return profiles
