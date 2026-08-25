"""Rule extraction, replay validation and segment profiling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._base import as_2d_array

__all__ = [
    "Condition",
    "Rule",
    "extract_rules",
    "profile_leaves",
    "replay_check",
    "replay_labels",
    "rule_mask",
]


def _fmt(value):
    if float(value).is_integer() and abs(value) < 2**53:
        return str(int(value))
    return f"{value:g}"


def _fmt_codes(categories):
    return "{" + ", ".join(_fmt(float(c)) for c in sorted(categories)) + "}"


@dataclass(frozen=True)
class Condition:
    """One axis-aligned test.

    ``op`` is ``"<="``/``">"`` with a float ``threshold``, or ``"in"`` /
    ``"not in"`` with integer category codes in ``categories``.
    """

    feature: str
    op: str
    threshold: float = float("nan")
    categories: tuple = field(default=())

    def __post_init__(self):
        if self.op not in ("<=", ">", "in", "not in"):
            raise ValueError(f"Unknown condition op: {self.op!r}.")

    def __str__(self):
        if self.op in ("in", "not in"):
            return f"{self.feature} {self.op} {_fmt_codes(self.categories)}"
        return f"{self.feature} {self.op} {_fmt(self.threshold)}"


@dataclass(frozen=True)
class Rule:
    """The full rule chain defining one segment (leaf)."""

    leaf_id: int
    conditions: tuple
    n_samples: int
    fraction: float
    depth: int

    @property
    def text(self):
        return " AND ".join(str(c) for c in self.conditions) if self.conditions else "TRUE"

    def __str__(self):
        return f"#{self.leaf_id}: {self.text}"


def extract_rules(tree):
    """Return the :class:`Rule` for every fitted segment, in leaf-id order."""
    if not hasattr(tree, "labels_"):
        raise RuntimeError("This SegTree instance is not fitted yet; call fit() first.")
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


def rule_mask(rule, X, feature_names):
    """Boolean mask of rows satisfying ``rule``; evaluated independently of the tree."""
    mask = np.ones(X.shape[0], dtype=bool)
    index = {nm: i for i, nm in enumerate(feature_names)}
    for cond in rule.conditions:
        j = index[cond.feature]
        col = X[:, j]
        if cond.op == "<=":
            mask &= col <= cond.threshold
        elif cond.op == ">":
            mask &= col > cond.threshold
        elif cond.op == "in":
            mask &= np.isin(col, cond.categories)
        else:  # "not in"
            mask &= ~np.isin(col, cond.categories)
    return mask


def replay_labels(tree, X):
    """Assign labels by replaying every rule chain on ``X``.

    Independent of :meth:`SegTree.predict` (no tree traversal), so agreement
    between the two validates that stored rules reproduce the partition.
    Rows matching no rule receive -1. Honors ``tree.handle_unknown``: with
    ``"error"`` unseen category codes raise ``ValueError``.
    """
    Xarr, _ = as_2d_array(X, context="X")
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


def replay_check(tree, X):
    """True if rule replay reproduces ``tree.predict(X)`` exactly."""
    return bool(np.array_equal(replay_labels(tree, X), tree.predict(X)))


def profile_leaves(tree, X, top_k=3):
    """Profile each segment by its most distinctive features.

    For every leaf, computes the centroid of ``X`` rows assigned to it.
    Continuous/binary features are ranked by absolute z-score versus the
    whole dataset (positive = above average). Categorical columns are *not*
    z-scored; instead each leaf receives ``category_enrichment``: the most
    enriched (feature, code) pairs by lift of in-leaf share over global share
    (only pairs covering >= 5% of the leaf, lift > 1, capped at ``top_k``,
    sorted by descending lift).

    Returns a list of dicts with keys ``leaf_id``, ``n_samples``, ``fraction``,
    ``top_features`` (tuple of ``(feature_name, z_score)`` for non-categorical
    features only) and ``category_enrichment`` (tuple of
    ``(feature_name, code, leaf_share, lift)``; empty when the tree has no
    categorical features).
    """
    Xarr, _ = as_2d_array(X, context="X")
    labels = tree.predict(Xarr)
    cat_idx = set(int(j) for j in getattr(tree, "categorical_indices_", []))
    num_cols = [j for j in range(Xarr.shape[1]) if j not in cat_idx]
    mu = Xarr.mean(axis=0)
    sd = Xarr.std(axis=0)
    sd[sd < 1e-12] = 1e-12

    # global category shares for lift computation
    global_dist = {}
    for j in cat_idx:
        known = np.asarray(tree.categories_[tree.feature_names_[j]])
        local = np.searchsorted(known, np.rint(Xarr[:, j]).astype(np.int64))
        counts = np.bincount(local, minlength=known.size + 1)[: known.size]
        global_dist[j] = counts / max(counts.sum(), 1)

    profiles = []
    for lf in tree.leaves_:
        rows_mask = labels == lf.id
        sub = Xarr[rows_mask]
        enrichment = []
        for j in cat_idx:
            known = np.asarray(tree.categories_[tree.feature_names_[j]])
            local = np.searchsorted(known, np.rint(sub[:, j]).astype(np.int64))
            counts = np.bincount(local, minlength=known.size + 1)[: known.size]
            if counts.sum() == 0:
                continue
            leaf_share = counts / counts.sum()
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
            {
                "leaf_id": lf.id,
                "n_samples": int(sub.shape[0]),
                "fraction": sub.shape[0] / Xarr.shape[0],
                "top_features": tuple(
                    (tree.feature_names_[num_cols[p]], float(znum[p])) for p in top_order
                ),
                "category_enrichment": tuple(enrichment[:top_k]),
            }
        )
    return profiles
