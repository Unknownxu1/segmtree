"""Unsupervised decision tree (clustering tree) for rule-based segmentation.

Grows a binary tree top-down: every split is the single-feature rule that most
reduces normalized within-node heterogeneity, and nodes are expanded
best-first. Each leaf is a segment described by an interpretable rule chain.

Continuous and binary features contribute their variance relative to the root;
categorical features contribute their Gini impurity relative to the root Gini,
which makes results invariant to any relabeling of category codes.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np

from ._base import BaseEstimator, as_2d_array, check_is_fitted, format_threshold

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

__all__ = ["Cut", "Leaf", "SegTree"]

_VAR_EPS = 1e-12
_GINI_EPS = 1e-12
_MAX_EXACT_CANDIDATES = 1 << 20  # 1M subset candidates before refusing exact mode


@dataclass(frozen=True, eq=False)
class Cut:
    """An executed split.

    ``op`` is ``"<="`` for numeric splits (``threshold`` meaningful,
    ``categories`` empty) or ``"in"`` for categorical splits (``categories``
    holds the integer codes routed left).  ``__eq__`` treats NaN thresholds
    as equal so that categorical-cut copies compare correctly.
    """

    depth: int
    feature_index: int
    feature: str
    threshold: float
    n_samples: int
    n_left: int
    n_right: int
    gain: float
    op: str = "<="
    categories: tuple[int, ...] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cut):
            return NotImplemented
        thr_eq = self.threshold == other.threshold or (
            math.isnan(self.threshold) and math.isnan(other.threshold)
        )
        return (
            self.depth == other.depth
            and self.feature_index == other.feature_index
            and self.feature == other.feature
            and thr_eq
            and self.n_samples == other.n_samples
            and self.n_left == other.n_left
            and self.n_right == other.n_right
            and self.gain == other.gain
            and self.op == other.op
            and self.categories == other.categories
        )

    def __hash__(self) -> int:
        thr_key = None if math.isnan(self.threshold) else self.threshold
        return hash(
            (
                self.depth,
                self.feature_index,
                self.feature,
                thr_key,
                self.n_samples,
                self.n_left,
                self.n_right,
                self.gain,
                self.op,
                self.categories,
            )
        )


@dataclass(frozen=True)
class Leaf:
    """A terminal segment.

    ``conditions`` holds ``(feature_name, op, payload)`` triples describing
    the path from the root: ``payload`` is a float threshold for ``"<="`` /
    ``">"`` splits or a frozenset of category codes for ``"in"`` / ``"not in"``
    splits. An empty tuple means the tree never split.
    """

    id: int
    n_samples: int
    fraction: float
    depth: int
    heterogeneity: float
    conditions: tuple[tuple[str, str, Any], ...]


def _resolve_quantiles(quantiles: int | ArrayLike) -> NDArray[np.float64]:
    if isinstance(quantiles, bool):
        raise ValueError("quantiles must be an int >= 2 or an array of fractions in (0, 1).")
    if isinstance(quantiles, Integral):
        if quantiles < 2:
            raise ValueError("quantiles must be an int >= 2 or an array of fractions in (0, 1).")
        return np.linspace(0.05, 0.95, int(quantiles))
    if isinstance(quantiles, Real):
        raise ValueError(
            f"quantiles must be an int >= 2 (candidate count) or a 1-D array of "
            f"fractions in (0, 1); got scalar {quantiles!r}. Pass e.g. 8 or "
            f"np.linspace(0.05, 0.95, 8)."
        )
    arr = np.asarray(quantiles, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("quantiles must be a 1-D, non-empty array of fractions in (0, 1).")
    if not np.all((arr > 0.0) & (arr < 1.0)):
        raise ValueError("All quantile fractions must lie strictly between 0 and 1.")
    return np.unique(arr)


def _sorted_linear_quantiles(s: NDArray, qs: NDArray) -> NDArray:
    """Quantiles of an ascending array; identical to ``np.quantile(s, qs)``."""
    h = qs * (len(s) - 1)
    lo = np.floor(h).astype(np.intp)
    hi = np.minimum(lo + 1, len(s) - 1)
    frac = h - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _filter_order(order_parent_local: NDArray, mask: NDArray) -> NDArray:
    """Map a parent's sorted-position array onto the child defined by ``mask``.

    Returns child-local positions enumerating the child's rows in the same
    ascending feature order as ``order_parent_local``.
    """
    sel = mask[order_parent_local]
    ranks = np.cumsum(mask)[order_parent_local] - 1
    return ranks[sel]


class SegTree(BaseEstimator):
    """Interpretable unsupervised decision tree for segmentation.

    Partitions rows of ``X`` into segments ("clusters") such that rows within
    a segment are similar and every segment is described by a short
    axis-aligned rule chain. No target variable is used.

    Node heterogeneity is a weighted average of per-feature impurity ratios::

        H(S) = sum_j( w_j * h_j(S) ) / sum_j(w_j)

    where ``w`` is ``feature_weights`` (all ones by default), numeric/binary
    features contribute ``h_j(S) = Var_j(S)/Var_j(root)`` and categorical
    features contribute ``h_j(S) = Gini_j(S)/Gini_j(root)``, so
    ``H(root) = 1`` regardless of scale or weights. Categorical impurity is
    invariant to any one-to-one recoding of category labels. The gain of a
    split into L/R is::

        Gain = H(S) - (n_L/n_S * H(L) + n_R/n_S * H(R))

    (with ``n`` replaced by the sum of ``sample_weight`` when provided).

    Candidate rules: requested quantiles of each continuous feature inside the
    node, one test at 0.5 for binary features, and — for categorical features —
    membership in subsets of categories (``categorical_splitter="exact"``) or
    single-category-vs-rest tests (``"one_vs_rest"``). Among all candidates the
    highest-gain leaf is always expanded next: the frontier lives in a min-heap
    keyed by *negative* gain, so every pop extracts the current best split
    (ties broken deterministically by insertion order).

    Parameters
    ----------
    min_gain : float, default=0.01
        Minimum heterogeneity reduction (relative to total heterogeneity)
        required to accept a split.  Must be finite and >= 0.  Larger values
        yield fewer segments.
    min_leaf_frac : float, default=0.01
        Minimum segment size as a fraction of the training sample size.
    min_leaf_abs : int, default=10
        Absolute floor for the minimum segment size.  The effective minimum
        is ``max(min_leaf_abs, int(min_leaf_frac * n_samples))``.
    max_depth : int or None, default=None
        Maximum number of consecutive splits along any path.  ``None``
        disables the limit.
    quantiles : int or array-like, default=8
        Either an integer ``q >= 2`` (candidate thresholds become the
        ``linspace(0.05, 0.95, q)`` quantiles of each feature within the
        node) or an explicit array of fractions in the open interval (0, 1).
    binary_features : "auto", None or sequence, default="auto"
        Which columns are binary (split once at threshold 0.5).
        ``"auto"`` detects columns whose observed values are a subset of
        {0, 1}; ``None`` treats everything as continuous; otherwise pass
        column indices, column names (requires ``feature_names`` or a
        DataFrame input), or a boolean mask of length ``n_features``.
    categorical_features : sequence or None, default=None
        Columns holding integer category codes (e.g. ``0..k-1`` from an
        ordinal or nominal encoding); they are split by membership in subsets
        of categories rather than by thresholds.  Pass column indices or column
        names.  Codes must be integral-valued and are handled exactly at any
        magnitude, including negative values (internal storage is int64).
        A column with more than ``max_categories`` distinct codes is rejected
        (group levels or use one-hot instead).  Not auto-detected: integer
        counts stay continuous unless listed here.  Mutually exclusive per
        column with ``binary_features``.
    max_categories : int, default=12
        Maximum number of distinct codes enumerated exactly per categorical
        feature (subset enumeration grows as 2^(k-1)).  Only enforced with
        ``categorical_splitter="exact"``; ``"one_vs_rest"`` accepts any
        cardinality.
    handle_unknown : {"complement", "error"}, default="complement"
        Policy for category codes unseen during training when predicting.
        ``"complement"`` routes them to the complement side of every
        categorical test (rule regions stay exhaustive and ``replay_check``
        holds); ``"error"`` makes ``predict`` / ``apply`` / ``replay_labels``
        raise ``ValueError`` naming the feature and the unknown codes.
    feature_names : sequence of str or None, default=None
        Names used in rules and reports.  Inferred from DataFrame columns
        when available; otherwise ``x0 .. x{m-1}``.
    dtype : numpy floating type, default=numpy.float64
        Storage precision for continuous/binary columns
        (``numpy.float32`` saves memory on large inputs).  Split statistics
        accumulate in float64, and categorical codes are always kept exactly.
    feature_weights : sequence, dict or None, default=None
        Per-feature weights ``w_j >= 0`` (not all zero) scaling each column's
        contribution to heterogeneity and gain.  ``None`` means uniform
        weights (bit-identical to the unweighted code path).  A sequence must
        be 1-D with one finite weight per column; a ``dict`` maps feature
        names (or DataFrame columns) to weights, missing names default to 1.
        Zero-weight features may still be *chosen as splitting variables*
        (acting as proxies for weighted features) — weights influence the
        heterogeneity/gain accounting only, not split eligibility.
    max_leaf_nodes : int or None, default=None
        Hard cap on the total number of leaves.  Growth is best-first, so when
        the cap is reached no further splits are executed and remaining
        frontier nodes become leaves.  ``None`` disables the limit.
    categorical_splitter : {"exact", "one_vs_rest"}, default="exact"
        Categorical candidate generation.  ``"exact"`` enumerates all
        canonical subsets (O(2^(k-1)) per node, capped by ``max_categories``);
        ``"one_vs_rest"`` tests each category alone against the rest (linear
        in k, supports arbitrarily high cardinality but can miss splits that
        merge several categories onto one side).

    Attributes
    ----------
    labels_ : ndarray of int, shape (n_samples,)
        Segment (leaf) index per training row.
    n_leaves_ : int
        Number of segments.
    leaves_ : tuple of Leaf
        Per-segment summary including the defining conditions.
    cuts_ : tuple of Cut
        Executed splits in best-first order.
    total_gain_ : float
        Size-weighted sum of split gains; equals ``h_root_ - h_avg_`` up to
        floating-point error (useful self-consistency check).
    h_root_, h_avg_ : float
        Root heterogeneity (= 1.0) and size-weighted mean leaf heterogeneity.
    reduction_ : float
        Fraction of total heterogeneity explained: ``1 - h_avg_``.
    feature_importances_ : ndarray of float64, shape (n_features_in_,)
        Normalized sum of split gains per feature (sklearn convention).
    feature_names_ : list of str
        Feature names in input column order.
    feature_names_in_ : ndarray of str, shape (n_features_in_,)
        sklearn-compatible alias of ``feature_names_``.
    binary_indices_ : ndarray of int
        Column indices treated as binary.
    categorical_indices_ : ndarray of int
        Column indices treated as categorical.
    categories_ : dict
        Maps each categorical feature name to the sorted ``ndarray`` of codes
        seen during fitting.
    min_leaf_ : int
        Effective minimum segment size used during fitting.
    feature_weights_ : ndarray of float64, shape (n_features_in_,)
        Effective per-feature weights in input column order (all ones when
        ``feature_weights=None``).
    leaf_chains_ : tuple of tuple
        Per-leaf chain of ``(feature_index, op, payload)`` triples in index
        form (the name-based form is in ``leaves_[i].conditions``).
    n_samples_in_, n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> from segmtree import SegTree
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(0, 0.3, (300, 4)), rng.normal(5, 0.3, (300, 4))])
    >>> tree = SegTree(min_leaf_abs=30).fit(X)
    >>> tree.n_leaves_ >= 2
    True

    Categorical column at index 1 (codes 0..2):

    >>> X[:, 1] = rng.integers(0, 3, X.shape[0])
    >>> tree = SegTree(min_leaf_abs=30, categorical_features=[1]).fit(X)
    """

    _estimator_type = "clusterer"

    def __init__(
        self,
        min_gain: float = 0.01,
        min_leaf_frac: float = 0.01,
        min_leaf_abs: int = 10,
        max_depth: int | None = None,
        quantiles: int | ArrayLike = 8,
        binary_features: str | Sequence | None = "auto",
        categorical_features: Sequence | None = None,
        max_categories: int = 12,
        handle_unknown: str = "complement",
        feature_names: Sequence[str] | None = None,
        dtype: Any = np.float64,
        feature_weights: Sequence | dict | None = None,
        max_leaf_nodes: int | None = None,
        categorical_splitter: str = "exact",
    ):
        self.min_gain = min_gain
        self.min_leaf_frac = min_leaf_frac
        self.min_leaf_abs = min_leaf_abs
        self.max_depth = max_depth
        self.quantiles = quantiles
        self.binary_features = binary_features
        self.categorical_features = categorical_features
        self.max_categories = max_categories
        self.handle_unknown = handle_unknown
        self.feature_names = feature_names
        self.dtype = dtype
        self.feature_weights = feature_weights
        self.max_leaf_nodes = max_leaf_nodes
        self.categorical_splitter = categorical_splitter

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        X: ArrayLike,
        y: Any = None,
        sample_weight: ArrayLike | None = None,
    ) -> SegTree:
        """Grow the tree from a tabular matrix (or DataFrame).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for API compatibility.
        sample_weight : array-like of shape (n_samples,), default=None
            Per-row weights scaling each sample's contribution to impurity
            and gain.  ``None`` means uniform weights (bit-identical to the
            unweighted path).  Weights must be finite and non-negative.

        Returns
        -------
        self
        """
        Xraw, columns = as_2d_array(X)
        n, m = Xraw.shape
        self._check_params(m)

        names = self._resolve_feature_names(Xraw, columns)
        binary_set = self._resolve_binary_features(Xraw, names)
        cat_codes, cat_set = self._resolve_categorical_features(Xraw, names, binary_set)

        sw = self._resolve_sample_weight(sample_weight, n)
        self._sw_ = sw  # None or float64 (n,)

        self.n_samples_in_ = n
        self.n_features_in_ = m
        self.feature_names_ = list(names)
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.binary_indices_ = np.array(sorted(binary_set), dtype=np.intp)
        self.categorical_indices_ = np.array(sorted(cat_set), dtype=np.intp)
        self.min_leaf_ = max(int(self.min_leaf_abs), int(self.min_leaf_frac * n))
        self.categories_ = {names[j]: cat_codes[j]["values"] for j in sorted(cat_set)}

        num_feats = np.array([j for j in range(m) if j not in cat_set], dtype=np.intp)
        cat_feats = np.array(sorted(cat_set), dtype=np.intp)
        self._num_feats_ = num_feats
        self._cat_feats_ = cat_feats
        self._num_pos_ = {int(j): i for i, j in enumerate(num_feats)}
        n_num = int(num_feats.size)

        w = self._resolve_feature_weights(names, m)
        self.feature_weights_ = np.ones(m, dtype=np.float64) if w is None else w
        self._weighted_ = w is not None
        if w is not None:
            self._w_all_ = w
            self._w_num_ = w[num_feats].copy()
            self._w_sum_ = float(w.sum())

        # Continuous/binary columns are centered for numerical robustness of
        # prefix sums; committed thresholds live in raw space.  Categorical
        # codes are stored separately as int64 so any magnitude (including
        # negative codes) is represented exactly regardless of ``dtype``.
        mean64 = (
            Xraw[:, num_feats].astype(np.float64, copy=False).mean(axis=0) if n_num else np.zeros(0)
        )
        self._mean_ = mean64
        self._Xraw = np.ascontiguousarray(Xraw.astype(np.dtype(self.dtype), copy=False))
        self._cat_codes_ = {j: cat_codes[j]["column"] for j in sorted(cat_set)}  # int64, exact
        dtype = np.dtype(self.dtype)
        XcT_num = (
            np.ascontiguousarray(
                ((Xraw[:, num_feats].astype(np.float64, copy=False) - mean64).astype(dtype)).T
            )
            if n_num
            else np.zeros((0, n), dtype=dtype)
        )
        XcT_cat = (
            np.ascontiguousarray(
                np.stack([self._cat_codes_[int(j)].astype(np.float64) for j in cat_feats])
            )
            if cat_feats.size
            else np.zeros((0, n))
        )
        self._XcT_num_ = XcT_num
        self._XcT_cat_ = XcT_cat
        self._stage64 = dtype == np.float64

        buf_rows = max(n_num, 1)
        self._bufs_ = {
            "mat": np.empty((buf_rows, n), dtype=np.float64),
            "stage": np.empty((buf_rows, n), dtype=dtype),
            "cs": np.empty((buf_rows, n + 1), dtype=np.float64),
            "cs2": np.empty((buf_rows, n + 1), dtype=np.float64),
            "sq": np.empty((buf_rows, n), dtype=np.float64),
            "cw": np.empty(n + 1, dtype=np.float64),
        }

        root_var = np.var(XcT_num.astype(np.float64, copy=False), axis=1) if n_num else np.zeros(0)
        root_var[root_var < _VAR_EPS] = _VAR_EPS
        self._root_var_ = root_var
        self._root_gini_ = {}
        for j in sorted(cat_set):
            g = self._gini_codes(
                self._cat_codes_[j],
                {"codes": cat_codes[j]["values"], "k": int(cat_codes[j]["values"].size)},
            )
            self._root_gini_[j] = max(g, _GINI_EPS)
        self._cat_meta_ = {
            j: {
                "pos": int(np.flatnonzero(cat_feats == j)[0]),
                "codes": cat_codes[j]["values"],
                "k": int(cat_codes[j]["values"].size),
            }
            for j in sorted(cat_set)
        }
        self._quantile_grid_ = _resolve_quantiles(self.quantiles)
        self._binary_set_ = binary_set
        self._cat_set_ = cat_set
        self._n_num_ = n_num
        del XcT_num, XcT_cat

        tick = count()

        def new_node(idx, depth, chain, orders):
            return {
                "idx": idx,
                "orders": orders,
                "depth": depth,
                "chain": chain,
                "best": None,
                "leaf": False,
                "leaf_id": -1,
                "cut_feat": -1,
                "cut_thr": 0.0,
                "cut_cats": None,
                "cut_gain": 0.0,
                "cut_n": 0,
                "left": None,
                "right": None,
            }

        root_orders = [np.argsort(self._column_values(j), kind="mergesort") for j in range(m)]
        root = new_node(np.arange(n, dtype=np.intp), 0, [], root_orders)
        root["best"] = self._best_split(root)
        # heapq is a min-heap: keys are NEGATIVE gains so each pop extracts the
        # frontier node with the highest gain (best-first growth).  Nodes with
        # no split get key=0.0 (popping order among them is by insertion tick).
        root_key = -(root["best"]["gain"]) if root["best"] else 0.0
        heap = [(root_key, next(tick), root)]
        cuts, chains = [], []
        labels = np.full(n, -1, dtype=np.intp)
        leaves = []
        total_gain = 0.0

        while heap:
            _, _, node = heapq.heappop(heap)
            best = node["best"]
            can_split = best is not None and best["gain"] >= self.min_gain
            if (
                can_split
                and self.max_leaf_nodes is not None
                and len(cuts) >= self.max_leaf_nodes - 1
            ):
                can_split = False
            if not can_split:
                self._finalize_leaf(node, leaves, chains, labels)
                continue

            j = best["feat"]
            if best["kind"] == "subset":
                cats = best["cats"]
                go_left = np.isin(self._cat_codes_[j][node["idx"]], tuple(cats))
            else:
                thr = float(best["thr"])
                # Commit on raw values with the same raw threshold the
                # evaluation used, so stored rules exactly reproduce the
                # partition.  ``self._Xraw`` is stored in ``self.dtype`` so
                # the comparison matches the one at predict time.
                go_left = self._Xraw[:, j][node["idx"]] <= thr
            il, ir = node["idx"][go_left], node["idx"][~go_left]
            if il.size == 0 or ir.size == 0:
                self._finalize_leaf(node, leaves, chains, labels)
                continue

            if best["kind"] == "subset":
                chain_l = [(j, "in", cats)]
                chain_r = [(j, "not in", cats)]
                cut_kwargs = {"op": "in", "categories": tuple(sorted(cats))}
            else:
                chain_l = [(j, "<=", thr)]
                chain_r = [(j, ">", thr)]
                cut_kwargs = {}
            cuts.append(
                Cut(
                    depth=node["depth"],
                    feature_index=j,
                    feature=self.feature_names_[j],
                    threshold=float("nan") if best["kind"] == "subset" else thr,
                    n_samples=int(node["idx"].size),
                    n_left=int(il.size),
                    n_right=int(ir.size),
                    gain=float(best["gain"]),
                    **cut_kwargs,
                )
            )
            node_w = self._row_weight(node["idx"])
            total_w = self._total_weight()
            total_gain += float(best["gain"]) * node_w / total_w
            node.update(
                {
                    "cut_feat": j,
                    "cut_thr": thr if best["kind"] != "subset" else 0.0,
                    "cut_cats": cats if best["kind"] == "subset" else None,
                    "cut_gain": float(best["gain"]),
                    "cut_n": int(node["idx"].size),
                    "left": None,
                    "right": None,
                }
            )

            # Children inherit the parent's per-feature sort orders by
            # filtering them: O(n) instead of re-sorting O(n log n).
            nl_orders = [_filter_order(oj, go_left) for oj in node["orders"]]
            nr_orders = [_filter_order(oj, ~go_left) for oj in node["orders"]]
            nl = new_node(il, node["depth"] + 1, node["chain"] + chain_l, nl_orders)
            nr = new_node(ir, node["depth"] + 1, node["chain"] + chain_r, nr_orders)
            node["left"], node["right"] = nl, nr
            for child in (nl, nr):
                child["best"] = self._best_split(child)
                cb = child["best"]
                heapq.heappush(heap, (-(cb["gain"]) if cb else 0.0, next(tick), child))

        sizes = np.array([lf.n_samples for lf in leaves], dtype=np.float64)
        hets = np.array([lf.heterogeneity for lf in leaves], dtype=np.float64)
        self.labels_ = labels
        self.cuts_ = tuple(cuts)
        self.leaves_ = tuple(leaves)
        self.leaf_chains_ = tuple(chains)
        self.n_leaves_ = len(leaves)
        self.total_gain_ = float(total_gain)
        self.h_root_ = 1.0
        if self._sw_ is not None:
            leaf_w = np.array([self._row_weight(np.flatnonzero(labels == lf.id)) for lf in leaves])
            self.h_avg_ = float(np.sum(leaf_w / self._total_weight() * hets))
        else:
            self.h_avg_ = float(np.sum(sizes / n * hets))
        self.reduction_ = 1.0 - self.h_avg_
        self.feature_importances_ = self._compute_feature_importances()

        self._strip_indices(root)
        self._root_ = root
        del self._Xraw, self._bufs_, self._XcT_num_, self._XcT_cat_
        return self

    def fit_predict(
        self,
        X: ArrayLike,
        y: Any = None,
        sample_weight: ArrayLike | None = None,
    ) -> NDArray[np.intp]:
        """Fit and return ``labels_``."""
        return self.fit(X, y, sample_weight=sample_weight).labels_

    def fit_transform(
        self,
        X: ArrayLike,
        y: Any = None,
        sample_weight: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Fit and return the one-hot leaf encoding of the training rows."""
        self.fit(X, y, sample_weight=sample_weight)
        return self.transform(X)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "n_leaves_")

    def __sklearn_tags__(self) -> Any:
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:  # pragma: no cover
            from types import SimpleNamespace

            return SimpleNamespace(estimator_type="clusterer")
        tags.estimator_type = "clusterer"
        return tags

    # --------------------------------------------------------------- predict

    def _encode_input_categories(self, Xarr: NDArray) -> dict[int, NDArray[np.int64]]:
        """Exact int64 codes for the categorical columns of a prediction matrix."""
        out = {}
        for j in self._cat_feats_:
            col = np.asarray(Xarr[:, j], dtype=np.float64)
            if not np.isfinite(col).all() or not np.array_equal(col, np.round(col)):
                raise ValueError(
                    f"Categorical column {self.feature_names_[j]!r} must contain "
                    "finite integer category codes."
                )
            out[int(j)] = np.rint(col).astype(np.int64)
        return out

    def _check_unknown(self, Xarr: NDArray, cat_codes: dict[int, NDArray]) -> None:
        if self.handle_unknown != "error":
            return
        for j, codes in cat_codes.items():
            known = self.categories_[self.feature_names_[j]]
            unseen = np.setdiff1d(codes, known)
            if unseen.size:
                preview = ", ".join(str(int(v)) for v in unseen[:5])
                more = " ..." if unseen.size > 5 else ""
                raise ValueError(
                    f"Unknown category codes for feature "
                    f"{self.feature_names_[j]!r}: [{preview}{more}] "
                    f"(handle_unknown='error')."
                )

    def _validate_predict_input(self, Xarr: NDArray) -> None:
        if Xarr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xarr.shape[1]} features, expected {self.n_features_in_}.")
        if hasattr(self, "feature_names_in_"):
            getattr(Xarr, "columns", None)
            # Column-name validation happens in predict() where raw X is available.

    def predict(self, X: ArrayLike) -> NDArray[np.intp]:
        """Assign each row to the segment whose rule region contains it.

        When ``X`` is a DataFrame whose columns are named, the column names
        must match the training names (order-insensitive): the frame is
        reindexed before scoring to prevent silent misalignment.
        """
        check_is_fitted(self)
        Xarr, columns = as_2d_array(X, context="X")
        self._validate_predict_input(Xarr)
        if columns is not None:
            expected = list(self.feature_names_)
            if sorted(columns) != sorted(expected):
                raise ValueError(f"Feature names mismatch: got {columns!r}, expected {expected!r}.")
            if columns != expected:
                # Reindex to training order to prevent silent misalignment.
                reorder = [columns.index(nm) for nm in expected]
                Xarr = np.ascontiguousarray(Xarr[:, reorder])
        cat_codes = self._encode_input_categories(Xarr)
        self._check_unknown(Xarr, cat_codes)
        Xcmp = Xarr.astype(np.dtype(self.dtype), copy=False)
        out = np.full(Xarr.shape[0], -1, dtype=np.intp)
        stack = [(self._root_, np.arange(Xarr.shape[0], dtype=np.intp))]
        while stack:
            node, pos = stack.pop()
            if node["leaf"]:
                out[pos] = node["leaf_id"]
                continue
            if node["cut_cats"] is not None:
                codes_j = cat_codes[node["cut_feat"]][pos]
                go_left = np.isin(codes_j, tuple(node["cut_cats"]))
            else:
                go_left = Xcmp[pos, node["cut_feat"]] <= node["cut_thr"]
            if go_left.any():
                stack.append((node["left"], pos[go_left]))
            if (~go_left).any():
                stack.append((node["right"], pos[~go_left]))
        return out

    def apply(self, X: ArrayLike) -> NDArray[np.intp]:
        """Return the leaf (segment) index each row falls into.

        Alias of :meth:`predict`; mirrors the sklearn convention where
        ``apply`` returns leaf identifiers rather than cluster labels.
        """
        return self.predict(X)

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """One-hot encode segment membership: shape ``(n_samples, n_leaves_)``.

        Useful as a feature-engineering step in a Pipeline (each segment
        becomes a binary indicator column).
        """
        labels = self.predict(X)
        out = np.zeros((labels.shape[0], self.n_leaves_), dtype=np.float64)
        out[np.arange(labels.shape[0]), labels] = 1.0
        return out

    def score(self, X: ArrayLike, y: Any = None) -> float:
        """Heterogeneity reduction of ``X`` under the fitted partition.

        Returns ``1 - H_avg / H_root`` where ``H`` is the same normalized
        mixed-type impurity used for splitting, computed on ``X`` itself.
        A value of 1.0 means the partition fully explains the structure of
        ``X``; 0.0 means it explains nothing (comparable across folds, so
        the method plugs directly into ``GridSearchCV``).
        """
        check_is_fitted(self)
        Xarr, _columns = as_2d_array(X, context="X")
        self._validate_predict_input(Xarr)
        labels = self.predict(Xarr)
        n = Xarr.shape[0]
        cat_idx = set(int(j) for j in self.categorical_indices_)
        num_cols = [j for j in range(self.n_features_in_) if j not in cat_idx]
        fw = self.feature_weights_
        w_sum = float(fw.sum())

        root_var = (
            np.var(Xarr[:, num_cols].astype(np.float64, copy=False), axis=0)
            if num_cols
            else np.zeros(0)
        )
        root_var = np.maximum(root_var, _VAR_EPS)
        known_gini = {}
        known_codes = {}
        for j in sorted(cat_idx):
            codes = np.rint(Xarr[:, j]).astype(np.int64)
            uniq = np.unique(codes)
            known_codes[j] = uniq
            known_gini[j] = max(
                self._gini_codes(codes, {"codes": uniq, "k": int(uniq.size)}), _GINI_EPS
            )

        def het_of(idx: NDArray) -> float:
            num_term = 0.0
            if num_cols:
                sub = (
                    Xarr[np.ix_(idx, num_cols)].astype(np.float64, copy=False)
                    if idx.size
                    else np.zeros((0, len(num_cols)))
                )
                vars_ = np.var(sub, axis=0) if idx.size else np.zeros(len(num_cols))
                ratios = np.maximum(vars_, 0.0) / root_var
                if self._weighted_:
                    num_term = float(np.dot(fw[num_cols], ratios))
                else:
                    num_term = float(np.mean(ratios))
            cat_term = 0.0
            for j in sorted(cat_idx):
                codes = (
                    np.rint(Xarr[idx, j]).astype(np.int64)
                    if idx.size
                    else np.zeros(0, dtype=np.int64)
                )
                ratio = (
                    self._gini_codes(
                        codes, {"codes": known_codes[j], "k": int(known_codes[j].size)}
                    )
                    / known_gini[j]
                )
                if self._weighted_:
                    cat_term += ratio * fw[j]
                else:
                    cat_term += ratio
            return (
                (num_term * self._n_num_ + cat_term) / self.n_features_in_
                if not self._weighted_
                else (num_term + cat_term) / w_sum
            )

        h_root = het_of(np.arange(n))
        if h_root <= 0:
            return 1.0
        h_avg = 0.0
        for leaf_id in range(self.n_leaves_):
            idx = np.flatnonzero(labels == leaf_id)
            if idx.size == 0:
                continue
            h_avg += (idx.size / n) * het_of(idx)
        return float(1.0 - h_avg / h_root)

    # ---------------------------------------------------------------- export

    def _require_fitted(self) -> None:
        check_is_fitted(self)

    @staticmethod
    def _fmt_threshold(value: float) -> str:
        return format_threshold(value)

    def _export_node_dict(self, nd: dict) -> dict:
        """Recursively convert a fitted internal node into plain-Python dicts."""
        if nd["leaf"]:
            lf = self.leaves_[nd["leaf_id"]]
            conditions = []
            for name, op, payload in lf.conditions:
                if op in ("in", "not in"):
                    conditions.append(
                        {
                            "feature": name,
                            "op": op,
                            "categories": [int(c) for c in sorted(payload)],
                        }
                    )
                else:
                    conditions.append({"feature": name, "op": op, "threshold": float(payload)})
            return {
                "type": "leaf",
                "leaf_id": int(nd["leaf_id"]),
                "depth": int(nd["depth"]),
                "n_samples": int(lf.n_samples),
                "fraction": float(lf.fraction),
                "heterogeneity": float(lf.heterogeneity),
                "conditions": conditions,
            }
        is_subset = nd["cut_cats"] is not None
        node = {
            "type": "split",
            "depth": int(nd["depth"]),
            "feature": self.feature_names_[nd["cut_feat"]],
            "feature_index": int(nd["cut_feat"]),
            "op": "in" if is_subset else "<=",
            "threshold": None if is_subset else float(nd["cut_thr"]),
            "categories": [int(c) for c in sorted(nd["cut_cats"])] if is_subset else None,
            "gain": float(nd["cut_gain"]),
            "n_samples": int(nd["cut_n"]),
        }
        node["left"] = self._export_node_dict(nd["left"])
        node["right"] = self._export_node_dict(nd["right"])
        return node

    def export_dict(self) -> dict:
        """Export the fitted tree as a JSON-serializable nested dict.

        The root split/leaf starts the structure; parent-child relations come
        from the tree itself, not from cut execution order.  Split nodes carry
        ``type/depth/feature/feature_index/op/threshold/categories/gain/
        n_samples/left/right``; leaf nodes carry ``type/leaf_id/depth/
        n_samples/fraction/heterogeneity/conditions``.  All numbers are native
        Python floats/ints and categories are sorted lists, so
        ``json.dumps(tree.export_dict())`` works out of the box.

        A ``_meta`` key on the root node records feature names, handle_unknown
        policy and category code sets so :meth:`from_dict` can fully
        reconstruct the tree.
        """
        check_is_fitted(self)
        root = self._export_node_dict(self._root_)
        meta: dict[str, Any] = {
            "feature_names": list(self.feature_names_),
            "n_features": int(self.n_features_in_),
            "handle_unknown": self.handle_unknown,
            "categories": {
                nm: [int(c) for c in sorted(codes)] for nm, codes in self.categories_.items()
            },
            "binary_indices": [int(j) for j in self.binary_indices_],
            "categorical_indices": [int(j) for j in self.categorical_indices_],
            "feature_weights": [float(v) for v in self.feature_weights_],
            "n_samples_in": int(self.n_samples_in_),
            "min_leaf": int(self.min_leaf_),
        }
        root["_meta"] = meta
        return root

    def export_text(self) -> str:
        """Render the fitted tree as a stable, human-readable indented text.

        Each split line shows the left-branch condition plus its gain and
        sample count; the complementary branch (``>`` / ``not in {...}``) is
        prefixed on the right subtree's first line.  Leaves show their id,
        size, fraction, depth and heterogeneity.
        """
        check_is_fitted(self)
        lines = []

        def walk(nd, tag):
            indent = "    " * nd["depth"]
            if nd["leaf"]:
                lf = self.leaves_[nd["leaf_id"]]
                lines.append(
                    f"{indent}{tag}leaf #{lf.id}: n={lf.n_samples}"
                    f" ({lf.fraction:.1%}), depth={lf.depth},"
                    f" heterogeneity={lf.heterogeneity:.4f}"
                )
                return
            name = self.feature_names_[nd["cut_feat"]]
            if nd["cut_cats"] is not None:
                codes = ", ".join(str(int(c)) for c in sorted(nd["cut_cats"]))
                cond = f"{name} in {{{codes}}}"
                rtag = f"not in {{{codes}}} "
            else:
                ts = format_threshold(nd["cut_thr"])
                cond = f"{name} <= {ts}"
                rtag = f"> {ts} "
            lines.append(f"{indent}{tag}{cond}  [gain={nd['cut_gain']:.4f}, n={nd['cut_n']}]")
            walk(nd["left"], "")
            walk(nd["right"], rtag)

        walk(self._root_, "")
        return "\n".join(lines)

    def export_graphviz(self, *, filled: bool = True) -> str:
        """Export the tree as a Graphviz DOT string (zero dependencies).

        Render with ``dot -Tpng tree.dot -o tree.png`` or any Graphviz
        consumer.  Nodes show the splitting condition, gain and sample count;
        leaves show id, size and heterogeneity.
        """
        check_is_fitted(self)
        lines = ["digraph SegTree {", '  node [shape=box, fontname="Helvetica"];']
        counter = [0]

        def walk(nd):
            nid = counter[0]
            counter[0] += 1
            if nd["leaf"]:
                lf = self.leaves_[nd["leaf_id"]]
                label = f"leaf #{lf.id}\\nn={lf.n_samples}\\nh={lf.heterogeneity:.3f}"
                style = 'style="filled", fillcolor="#d4edda"' if filled else ""
                lines.append(f'  {nid} [label="{label}" {style}];')
                return nid
            name = self.feature_names_[nd["cut_feat"]]
            if nd["cut_cats"] is not None:
                codes = ", ".join(str(int(c)) for c in sorted(nd["cut_cats"]))
                cond = f"{name} in {{{codes}}}"
            else:
                cond = f"{name} <= {format_threshold(nd['cut_thr'])}"
            label = f"{cond}\\ngain={nd['cut_gain']:.3f}\\nn={nd['cut_n']}"
            style = 'style="filled", fillcolor="#cce5ff"' if filled else ""
            lines.append(f'  {nid} [label="{label}" {style}];')
            left_id = walk(nd["left"])
            right_id = walk(nd["right"])
            lines.append(f'  {nid} -> {left_id} [label="yes"];')
            lines.append(f'  {nid} -> {right_id} [label="no"];')
            return nid

        walk(self._root_)
        lines.append("}")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict, **overrides: Any) -> SegTree:
        """Reconstruct a fitted tree from :meth:`export_dict` output.

        The rebuilt tree can ``predict``, ``apply``, ``transform``,
        ``export_text``, ``export_dict`` and ``export_graphviz``.  Attributes
        that depend on the original training matrix (``labels_``,
        ``feature_importances_``, ``reduction_`` etc.) are not restored.

        Parameters
        ----------
        data : dict
            Output of :meth:`export_dict` (with or without the ``_meta`` key).
        **overrides
            Explicit metadata overriding whatever is in ``_meta``:
            ``feature_names``, ``handle_unknown``, ``categories`` etc.

        Returns
        -------
        SegTree
        """
        meta = dict(data.get("_meta", {}))
        meta.update(overrides)
        feature_names = meta.get("feature_names")
        if feature_names is None:
            feature_names = sorted(
                {node["feature"] for node in _iter_split_nodes(data)}
                | {
                    cond["feature"]
                    for node in _iter_leaf_nodes(data)
                    for cond in node.get("conditions", [])
                }
            )
        feature_names = [str(nm) for nm in feature_names]
        n_features = int(meta.get("n_features", len(feature_names)))
        handle_unknown = meta.get("handle_unknown", "complement")
        categories = {
            str(nm): np.asarray(codes, dtype=np.int64)
            for nm, codes in meta.get("categories", {}).items()
        }
        if not categories:
            # Infer the codes that appear in categorical splits.
            collected: dict[str, set] = {}
            for node in _iter_split_nodes(data):
                if node.get("op") == "in" and node.get("categories"):
                    collected.setdefault(node["feature"], set()).update(
                        int(c) for c in node["categories"]
                    )
            categories = {
                nm: np.asarray(sorted(codes), dtype=np.int64) for nm, codes in collected.items()
            }

        tree = cls(
            handle_unknown=handle_unknown,
            feature_names=feature_names,
            min_leaf_abs=meta.get("min_leaf", 1),
        )
        tree.n_features_in_ = n_features
        tree.n_samples_in_ = int(meta.get("n_samples_in", 0))
        tree.feature_names_ = list(feature_names)
        tree.feature_names_in_ = np.asarray(feature_names, dtype=object)
        tree.categories_ = categories
        tree.binary_indices_ = np.asarray(meta.get("binary_indices", []), dtype=np.intp)
        tree.categorical_indices_ = np.asarray(meta.get("categorical_indices", []), dtype=np.intp)
        tree.handle_unknown = handle_unknown
        fw = meta.get("feature_weights")
        tree.feature_weights_ = (
            np.ones(n_features, dtype=np.float64)
            if fw is None
            else np.asarray(fw, dtype=np.float64)
        )
        tree._cat_set_ = set(int(j) for j in tree.categorical_indices_)
        tree._cat_feats_ = tree.categorical_indices_
        tree._num_feats_ = np.array(
            [j for j in range(n_features) if j not in tree._cat_set_], dtype=np.intp
        )
        tree._n_num_ = int(tree._num_feats_.size)
        tree._weighted_ = not np.all(tree.feature_weights_ == 1.0)
        if tree._weighted_:
            tree._w_all_ = tree.feature_weights_
            tree._w_num_ = tree.feature_weights_[tree._num_feats_]
            tree._w_sum_ = float(tree.feature_weights_.sum())
        tree.min_leaf_ = int(meta.get("min_leaf", 1))

        # Rebuild node tree and leaves.
        leaves: list[Leaf] = []

        def build(node_data: dict) -> dict:
            if node_data["type"] == "leaf":
                conds = []
                for cond in node_data.get("conditions", []):
                    if cond["op"] in ("in", "not in"):
                        conds.append(
                            (
                                cond["feature"],
                                cond["op"],
                                frozenset(int(c) for c in cond["categories"]),
                            )
                        )
                    else:
                        conds.append((cond["feature"], cond["op"], float(cond["threshold"])))
                lf = Leaf(
                    id=int(node_data["leaf_id"]),
                    n_samples=int(node_data["n_samples"]),
                    fraction=float(node_data["fraction"]),
                    depth=int(node_data["depth"]),
                    heterogeneity=float(node_data["heterogeneity"]),
                    conditions=tuple(conds),
                )
                leaves.append(lf)
                return {
                    "leaf": True,
                    "leaf_id": lf.id,
                    "depth": lf.depth,
                    "cut_feat": -1,
                    "cut_thr": 0.0,
                    "cut_cats": None,
                    "cut_gain": 0.0,
                    "cut_n": lf.n_samples,
                    "left": None,
                    "right": None,
                }
            is_subset = node_data.get("op") == "in"
            nd = {
                "leaf": False,
                "leaf_id": -1,
                "depth": int(node_data["depth"]),
                "cut_feat": int(node_data["feature_index"]),
                "cut_thr": 0.0 if is_subset else float(node_data["threshold"]),
                "cut_cats": (
                    frozenset(int(c) for c in node_data["categories"]) if is_subset else None
                ),
                "cut_gain": float(node_data.get("gain", 0.0)),
                "cut_n": int(node_data.get("n_samples", 0)),
                "left": build(node_data["left"]),
                "right": build(node_data["right"]),
            }
            return nd

        root = build(data)
        leaves.sort(key=lambda lf: lf.id)
        tree.leaves_ = tuple(leaves)
        tree.n_leaves_ = len(leaves)
        tree._root_ = root
        tree.cuts_ = tuple()  # execution order not recoverable from export_dict
        tree.leaf_chains_ = tuple(
            tuple(
                (feature_names.index(nm) if nm in feature_names else 0, op, payload)
                for nm, op, payload in lf.conditions
            )
            for lf in leaves
        )
        return tree

    # -------------------------------------------------------- rule helpers

    def extract_rules(self):
        """Return the :class:`~segmtree.Rule` for every segment (thin wrapper)."""
        from .rules import extract_rules

        return extract_rules(self)

    def replay_labels(self, X: ArrayLike) -> NDArray[np.intp]:
        """Assign labels by replaying rule chains (thin wrapper)."""
        from .rules import replay_labels

        return replay_labels(self, X)

    def replay_check(self, X: ArrayLike) -> bool:
        """True if rule replay reproduces ``predict(X)`` (thin wrapper)."""
        from .rules import replay_check

        return replay_check(self, X)

    def profile_leaves(self, X: ArrayLike, top_k: int = 3):
        """Profile each segment by distinctive features (thin wrapper)."""
        from .rules import profile_leaves

        return profile_leaves(self, X, top_k=top_k)

    def bootstrap_stability(self, X: ArrayLike, **kwargs):
        """Bootstrap ARI stability (thin wrapper)."""
        from .stability import bootstrap_stability

        return bootstrap_stability(self, X, **kwargs)

    # --------------------------------------------------------------- helpers

    def _check_params(self, m: int) -> None:
        if isinstance(self.min_gain, bool) or not isinstance(self.min_gain, Real):
            raise ValueError("min_gain must be a float >= 0.")
        if not np.isfinite(self.min_gain) or self.min_gain < 0:
            raise ValueError("min_gain must be a finite float >= 0.")
        if isinstance(self.min_leaf_frac, bool) or not isinstance(self.min_leaf_frac, Real):
            raise ValueError("min_leaf_frac must be a float in [0, 1).")
        if not 0 <= self.min_leaf_frac < 1:
            raise ValueError("min_leaf_frac must be a float in [0, 1).")
        if isinstance(self.min_leaf_abs, bool) or not isinstance(self.min_leaf_abs, Integral):
            raise ValueError("min_leaf_abs must be an int >= 1.")
        if self.min_leaf_abs < 1:
            raise ValueError("min_leaf_abs must be an int >= 1.")
        if self.max_depth is not None and (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, Integral)
            or self.max_depth < 1
        ):
            raise ValueError("max_depth must be a positive int or None.")
        _resolve_quantiles(self.quantiles)
        spec = self.binary_features
        if isinstance(spec, str) and spec != "auto":
            raise ValueError('binary_features must be "auto", None, or a sequence.')
        if spec is not None and not isinstance(spec, str) and isinstance(spec, (int, float, bool)):
            raise TypeError(
                "binary_features must be 'auto', None, or a sequence of "
                "indices/names/bool mask; got a scalar."
            )
        if self.categorical_features is not None:
            if isinstance(self.categorical_features, str):
                raise ValueError(
                    "categorical_features must be None or a sequence of names/indices."
                )
            if isinstance(self.categorical_features, (int, float, bool)):
                raise TypeError(
                    "categorical_features must be None or a sequence of "
                    "names/indices; got a scalar."
                )
        if isinstance(self.max_categories, bool) or not isinstance(self.max_categories, Integral):
            raise ValueError("max_categories must be an int >= 2.")
        if self.max_categories < 2:
            raise ValueError("max_categories must be an int >= 2.")
        if self.handle_unknown not in ("complement", "error"):
            raise ValueError('handle_unknown must be "complement" or "error".')
        if self.max_leaf_nodes is not None and (
            isinstance(self.max_leaf_nodes, bool)
            or not isinstance(self.max_leaf_nodes, Integral)
            or self.max_leaf_nodes < 2
        ):
            raise ValueError("max_leaf_nodes must be an int >= 2 or None.")
        if self.categorical_splitter not in ("exact", "one_vs_rest"):
            raise ValueError('categorical_splitter must be "exact" or "one_vs_rest".')
        dtype = np.dtype(self.dtype) if not isinstance(self.dtype, np.dtype) else self.dtype
        if dtype.kind != "f" or dtype.itemsize not in (4, 8):
            raise ValueError("dtype must be numpy.float32 or numpy.float64.")

    def _resolve_feature_names(self, Xraw: NDArray, columns: list | None) -> list[str]:
        if self.feature_names is not None:
            names = [str(nm) for nm in self.feature_names]
        elif columns is not None:
            names = list(columns)
        else:
            names = [f"x{j}" for j in range(Xraw.shape[1])]
        if len(names) != Xraw.shape[1]:
            raise ValueError(
                f"feature_names has {len(names)} entries but X has {Xraw.shape[1]} features."
            )
        if len(set(names)) != len(names):
            dupes = sorted({nm for nm in names if names.count(nm) > 1})
            raise ValueError(f"Feature names must be unique; duplicated: {dupes}.")
        return names

    @staticmethod
    def _resolve_sequence_spec(spec: Sequence, names: list[str], m: int, kind: str) -> set:
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        resolved = set()
        for item in spec:
            if isinstance(item, str):
                if item not in name_to_idx:
                    raise ValueError(f"Unknown {kind} feature name: {item!r}.")
                resolved.add(name_to_idx[item])
            elif isinstance(item, bool):
                raise TypeError("Pass a plain int index or str name, not a Python bool.")
            elif isinstance(item, (Integral, np.bool_, np.integer)):
                idx = int(item)
                if not 0 <= idx < m:
                    raise ValueError(f"{kind} feature index {idx} out of range for {m} columns.")
                resolved.add(idx)
            else:
                raise TypeError(f"Unsupported entry in {kind}_features: {item!r}.")
        return resolved

    def _resolve_feature_weights(self, names: list[str], m: int) -> NDArray | None:
        """Validate ``feature_weights`` and return a float64 array or ``None``.

        ``None`` keeps the unweighted fast path (bit-identical results).  An
        all-ones explicit spec is also normalized to that fast path so both
        forms are strictly equivalent to the default behavior.
        """
        spec = self.feature_weights
        if spec is None:
            return None
        if isinstance(spec, dict):
            name_to_idx = {nm: i for i, nm in enumerate(names)}
            w = np.ones(m, dtype=np.float64)
            for key, value in spec.items():
                if isinstance(value, (bool, np.bool_)):
                    raise TypeError(f"feature_weights[{key!r}] must be a number, not a bool.")
                if not isinstance(value, Real):
                    raise TypeError(f"feature_weights[{key!r}] must be a real number.")
                if key not in name_to_idx:
                    raise ValueError(
                        f"feature_weights key {key!r} does not match any feature name."
                    )
                w[name_to_idx[key]] = float(value)
        else:
            if isinstance(spec, (str, bytes)):
                raise ValueError("feature_weights must be a 1-D numeric sequence, a dict, or None.")
            entries = list(spec)
            if any(isinstance(v, (bool, np.bool_)) for v in entries):
                raise TypeError("feature_weights must not contain boolean values.")
            try:
                w = np.asarray(entries, dtype=np.float64)
            except (TypeError, ValueError) as e:
                raise ValueError(f"feature_weights must be numeric; {e}") from e
            if w.ndim != 1:
                raise ValueError(
                    f"feature_weights must be a 1-D sequence with one weight per feature "
                    f"(got {w.ndim}-D)."
                )
            if w.size != m:
                raise ValueError(f"feature_weights has {w.size} entries but X has {m} features.")
        if not np.isfinite(w).all():
            raise ValueError("feature_weights must be finite (no NaN/inf).")
        if (w < 0).any():
            raise ValueError("feature_weights must be non-negative.")
        if not (w > 0).any():
            raise ValueError("feature_weights must contain at least one positive weight.")
        if np.all(w == 1.0):
            return None  # exactly equivalent to the default uniform path
        return np.array(w, dtype=np.float64, copy=True)

    @staticmethod
    def _resolve_sample_weight(sample_weight: ArrayLike | None, n: int) -> NDArray | None:
        if sample_weight is None:
            return None
        sw = np.asarray(sample_weight, dtype=np.float64).ravel()
        if sw.shape[0] != n:
            raise ValueError(f"sample_weight has {sw.shape[0]} entries but X has {n} samples.")
        if not np.isfinite(sw).all():
            raise ValueError("sample_weight must be finite (no NaN/inf).")
        if (sw < 0).any():
            raise ValueError("sample_weight must be non-negative.")
        return sw

    def _resolve_binary_features(self, Xraw: NDArray, names: list[str]) -> set:
        spec = self.binary_features
        if spec is None:
            return set()
        if isinstance(spec, str):
            if spec != "auto":
                raise ValueError('binary_features must be "auto", None, or a sequence.')
            return {
                j for j in range(Xraw.shape[1]) if np.isin(np.unique(Xraw[:, j]), (0.0, 1.0)).all()
            }
        return self._resolve_sequence_spec(spec, names, Xraw.shape[1], "binary")

    def _resolve_categorical_features(
        self, Xraw: NDArray, names: list[str], binary_set: set
    ) -> tuple[dict, set]:
        spec = self.categorical_features
        if spec is None:
            return {}, set()
        resolved = self._resolve_sequence_spec(spec, names, Xraw.shape[1], "categorical")
        overlap = resolved & binary_set
        if overlap:
            dupes = sorted(names[j] for j in overlap)
            raise ValueError(f"Columns cannot be both binary and categorical: {dupes}.")
        codes = {}
        for j in sorted(resolved):
            col = np.asarray(Xraw[:, j], dtype=np.float64)
            if not np.isfinite(col).all():
                raise ValueError(f"Categorical column {names[j]!r} contains NaN/infinite values.")
            if not np.array_equal(col, np.round(col)):
                raise ValueError(
                    f"Categorical column {names[j]!r} must hold integral category codes "
                    "(e.g. 0..k-1); encode non-numeric labels first (e.g. pandas factorize)."
                )
            codes_j = np.rint(col).astype(np.int64)
            uniq = np.unique(codes_j)
            if self.categorical_splitter == "exact" and uniq.size > self.max_categories:
                raise ValueError(
                    f"Categorical column {names[j]!r} has {uniq.size} distinct codes > "
                    f"max_categories={self.max_categories}: exact subset enumeration costs "
                    f"O(2^(k-1)) per node. Group rare levels, raise max_categories, or set "
                    f"categorical_splitter='one_vs_rest' for high-cardinality columns."
                )
            codes[j] = {"column": codes_j, "values": uniq}
        return codes, resolved

    def _column_values(self, j: int) -> NDArray:
        """Full-column training values used for the initial sort orders."""
        if j in self._cat_set_:
            return self._cat_codes_[j]
        return self._XcT_num_[self._num_pos_[j]]

    def _row_weight(self, idx: NDArray) -> float:
        if self._sw_ is None:
            return float(idx.size)
        return float(self._sw_[idx].sum()) if idx.size else 0.0

    def _total_weight(self) -> float:
        if self._sw_ is None:
            return float(self.n_samples_in_)
        return float(self._sw_.sum())

    @staticmethod
    def _gini_of(codes: NDArray, known: NDArray | None = None) -> float:
        """Gini impurity of integer codes (handles negative / large values)."""
        if known is not None:
            local = np.searchsorted(known, codes)
            counts = np.bincount(local, minlength=int(known.size))
        else:
            _, local = np.unique(codes, return_inverse=True)
            counts = np.bincount(local)
        counts = counts[counts > 0]
        p = counts / counts.sum()
        return float(1.0 - np.sum(p * p))

    def _gini_codes(
        self,
        codes: NDArray,
        meta: dict,
        sw: NDArray | None = None,
    ) -> float:
        """Gini impurity of a code sample against one categorical column."""
        if codes.size == 0:
            return 0.0
        local = np.searchsorted(meta["codes"], codes)
        if sw is None:
            counts = np.bincount(local, minlength=meta["k"])
        else:
            counts = np.bincount(local, weights=sw, minlength=meta["k"])
        total = counts.sum()
        if total <= 0:
            return 0.0
        p = counts[counts > 0] / total
        return float(1.0 - np.sum(p * p))

    def _cat_counts_by_block(
        self,
        vals: NDArray,
        meta: dict,
        block_ids: NDArray,
        n_blocks: int,
        sw: NDArray | None = None,
    ) -> NDArray:
        """Contingency counts of one categorical column across j-blocks."""
        local = np.searchsorted(meta["codes"], vals)
        flat = block_ids * meta["k"] + local
        if sw is None:
            table = np.bincount(flat, minlength=n_blocks * meta["k"])
        else:
            table = np.bincount(flat, weights=sw, minlength=n_blocks * meta["k"])
        return table.reshape(n_blocks, meta["k"]).astype(np.float64)

    def _combined_heterogeneity(self, num_term: float, cat_term: float) -> float:
        """Combine the numeric and categorical terms into ``H``.

        Unweighted: ``(mean_numeric * n_num + sum_categorical) / m``.
        Weighted (callers pass weighted sums for both domains):
        ``(sum_j w_j * term_j) / sum_j w_j``.
        """
        if self._weighted_:
            return (num_term + cat_term) / self._w_sum_
        return (num_term * self._n_num_ + cat_term) / self.n_features_in_

    def _num_reduce(self, M: NDArray, *, axis: int) -> NDArray:
        """Reduce per-numeric-feature terms along the feature ``axis``.

        Unweighted this is a plain mean; with feature weights it is the
        weighted sum over that axis (division by the total weight happens in
        :meth:`_combined_heterogeneity`).  ``M`` may hold its feature axis at
        position 0 or last.
        """
        if self._weighted_:
            ax = axis % M.ndim
            return M @ self._w_num_ if ax == M.ndim - 1 else self._w_num_ @ M
        return M.mean(axis=axis)

    def _node_cat_ratios_from_idx(self, idx: NDArray, sw_node: NDArray | None = None) -> float:
        total = 0.0
        for j in self._cat_set_:
            meta = self._cat_meta_[j]
            codes_j = self._cat_codes_[j][idx]
            ratio = self._gini_codes(codes_j, meta, sw_node) / self._root_gini_[j]
            total += ratio * self._w_all_[j] if self._weighted_ else ratio
        return total

    def _compute_feature_importances(self) -> NDArray:
        importances = np.zeros(self.n_features_in_, dtype=np.float64)
        for cut in self.cuts_:
            importances[cut.feature_index] += cut.gain
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

    # ------------------------------------------------------------- splitting

    def _best_split(self, node: dict) -> dict | None:
        """Find the highest-gain single-feature rule inside a node."""
        idx = node["idx"]
        n = int(idx.size)
        depth = node["depth"]
        if self.max_depth is not None and depth >= self.max_depth:
            return None
        if n < 2 * self.min_leaf_:
            return None
        m = self.n_features_in_
        rv = self._root_var_
        has_num = self._n_num_ > 0
        sw_node = self._sw_[idx] if self._sw_ is not None else None
        if has_num:
            mat = self._bufs_["mat"][: self._n_num_, :n]
            cs = self._bufs_["cs"][: self._n_num_, : n + 1]
            cs2 = self._bufs_["cs2"][: self._n_num_, : n + 1]
            sq = self._bufs_["sq"][: self._n_num_, :n]
            stage = self._bufs_["stage"][: self._n_num_, :n]

        def gather(rows):
            if not has_num:
                return
            if self._stage64:
                np.take(self._XcT_num_, rows, axis=1, out=mat)
            else:
                np.take(self._XcT_num_, rows, axis=1, out=stage)
                np.copyto(mat, stage)

        hs = None
        best = None
        for j in range(m):
            rows = idx[node["orders"][j]]  # global row ids sorted by feature j
            is_cat = j in self._cat_set_
            scol = (
                self._XcT_cat_[self._cat_meta_[j]["pos"]][rows]
                if is_cat
                else self._XcT_num_[self._num_pos_[j]][rows]
            )
            if scol[-1] - scol[0] < _VAR_EPS:
                continue  # feature constant within this node

            gather(rows)
            if hs is None:
                # Impurity is row-order invariant, so compute it from this
                # first gathered matrix instead of gathering twice.
                if has_num:
                    if sw_node is None:
                        num_term = float(self._num_reduce(np.var(mat, axis=1) / rv, axis=0))
                    else:
                        mean_w = np.average(mat, weights=sw_node, axis=1)
                        var_w = np.average((mat - mean_w[:, None]) ** 2, weights=sw_node, axis=1)
                        num_term = float(self._num_reduce(np.maximum(var_w, 0.0) / rv, axis=0))
                else:
                    num_term = 0.0
                cat_term = 0.0
                sw_sorted = sw_node[node["orders"][j]] if sw_node is not None else None
                for cj in self._cat_set_:
                    codes_cj = self._cat_codes_[cj][rows]
                    ratio = (
                        self._gini_codes(codes_cj, self._cat_meta_[cj], sw_sorted)
                        / self._root_gini_[cj]
                    )
                    cat_term += ratio * self._w_all_[cj] if self._weighted_ else ratio
                hs = self._combined_heterogeneity(num_term, cat_term)
            if has_num:
                cs[:, 0] = 0.0
                cs2[:, 0] = 0.0
                if sw_node is None:
                    np.cumsum(mat, axis=1, out=cs[:, 1:])
                    np.multiply(mat, mat, out=sq)
                else:
                    sw_sorted = sw_node[node["orders"][j]]
                    wmat = sw_sorted[None, :] * mat
                    np.cumsum(wmat, axis=1, out=cs[:, 1:])
                    np.multiply(wmat, mat, out=sq)
                np.cumsum(sq, axis=1, out=cs2[:, 1:])
                tot, tot2 = cs[:, n], cs2[:, n]
                if sw_node is not None:
                    cw = self._bufs_["cw"]
                    cw_slice = cw[: n + 1]
                    cw_slice[0] = 0.0
                    np.cumsum(sw_node[node["orders"][j]], out=cw_slice[1:])
                else:
                    cw = None
            else:
                tot = tot2 = cw = None

            # Per-category cumulative one-hot counts along this gather order,
            # needed to score children Gini for every candidate cut position.
            # Skipped when ``j`` itself is categorical: ``_best_cat_split``
            # rebuilds the contingency tables it needs from block boundaries.
            ccs = []
            if self._cat_set_ and not is_cat:
                pos_ar = np.arange(n)
                sw_sorted = sw_node[node["orders"][j]] if sw_node is not None else None
                for cj in sorted(self._cat_set_):
                    meta = self._cat_meta_[cj]
                    vals = self._cat_codes_[cj][rows]
                    local = np.searchsorted(meta["codes"], vals)
                    oh = np.zeros((n, meta["k"]))
                    if sw_sorted is None:
                        oh[pos_ar, local] = 1.0
                    else:
                        oh[pos_ar, local] = sw_sorted
                    wc = self._w_all_[cj] if self._weighted_ else 1.0
                    ccs.append(
                        (
                            np.vstack([np.zeros((1, meta["k"])), np.cumsum(oh, axis=0)]),
                            float(n),
                            self._root_gini_[cj],
                            wc,
                        )
                    )

            if is_cat:
                cand_best = self._best_cat_split(j, rows, scol, tot, tot2, rv, hs, n, sw_node)
            else:
                # Candidate thresholds live in raw space and cut positions are
                # located on raw values too: evaluating and committing must
                # agree on which rows go left.
                rcol = self._Xraw[:, j][rows]
                cand_best = self._best_numeric_split(
                    j, rcol, cs, cs2, tot, tot2, rv, hs, n, ccs, cw
                )
            if cand_best is not None and (best is None or cand_best["gain"] > best["gain"]):
                best = cand_best
        return best

    def _best_numeric_split(
        self,
        j: int,
        rcol: NDArray,
        cs: NDArray,
        cs2: NDArray,
        tot: NDArray,
        tot2: NDArray,
        rv: NDArray,
        hs: float,
        n: int,
        ccs: list,
        cw: NDArray | None,
    ) -> dict | None:
        """Best threshold split; ``rcol`` is the node's raw column, ascending."""
        if j in self._binary_set_:
            cand = np.array([0.5])
        else:
            cand = np.unique(_sorted_linear_quantiles(rcol, self._quantile_grid_))

        ks_all = np.searchsorted(rcol, cand, side="right")
        uniq_k, first_pos = np.unique(ks_all, return_index=True)
        ok = (uniq_k >= self.min_leaf_) & (n - uniq_k >= self.min_leaf_)
        if not ok.any():
            return None
        k = uniq_k[ok]
        t = cand[first_pos[ok]].astype(np.float64)
        kf = k.astype(np.float64)
        nf = float(n)

        # Weight sums: use cumulative weights when sample_weight is active.
        if cw is not None:
            kw = cw[k]
            nw = cw[n]
            kw_safe = np.maximum(kw, _VAR_EPS)
            nwf = np.maximum(nw - kw, _VAR_EPS)
        else:
            kw = kf
            nw = nf
            kw_safe = kf
            nwf = np.maximum(nf - kf, _VAR_EPS)

        hL_num, hR_num = self._children_numeric(cs, cs2, tot, tot2, rv, k, kw_safe, nwf, nw)
        # children Gini ratios of every categorical column at each candidate
        sum_gL = np.zeros(k.size)
        sum_gR = np.zeros(k.size)
        for cc, _total_n, gr, wc in ccs:
            NLc = cc[k]  # (nk, Kc)
            NRc = cc[n][None, :] - NLc
            gL = 1.0 - ((NLc / kw_safe[:, None]) ** 2).sum(axis=1)
            gR = 1.0 - ((NRc / nwf[:, None]) ** 2).sum(axis=1)
            sum_gL += wc * gL / gr
            sum_gR += wc * gR / gr
        hL = self._combined_heterogeneity(hL_num, sum_gL)
        hR = self._combined_heterogeneity(hR_num, sum_gR)
        gains = hs - (kw / nw) * hL - ((nw - kw) / nw) * hR
        local = int(np.argmax(gains))
        return {
            "gain": float(gains[local]),
            "feat": j,
            "thr": float(t[local]),
            "kind": "threshold",
            "cats": None,
        }

    def _children_numeric(
        self,
        cs: NDArray,
        cs2: NDArray,
        tot: NDArray,
        tot2: NDArray,
        rv: NDArray,
        k: NDArray,
        kw_safe: NDArray,
        nwf: NDArray,
        nw: float,
    ) -> tuple[NDArray, NDArray]:
        """Mean normalized-variance of left/right children over numeric features."""
        if not self._n_num_:
            z = np.zeros(k.size)
            return z, z
        csL, cs2L = cs[:, k], cs2[:, k]
        mL = csL / kw_safe
        vL = np.maximum(cs2L / kw_safe - mL**2, 0.0)
        mR = (tot[:, None] - csL) / nwf
        vR = np.maximum((tot2[:, None] - cs2L) / nwf - mR**2, 0.0)
        return self._num_reduce(vL / rv[:, None], axis=0), self._num_reduce(
            vR / rv[:, None], axis=0
        )

    def _best_cat_split(
        self,
        j: int,
        rows: NDArray,
        scol: NDArray,
        tot: NDArray | None,
        tot2: NDArray | None,
        rv: NDArray,
        hs: float,
        n: int,
        sw_node: NDArray | None = None,
    ) -> dict | None:
        """Best categorical split of column ``j`` under ``categorical_splitter``.

        Categories form contiguous blocks in ``scol`` (ascending code order).
        Both splitters share the same scoring machinery: numeric children via
        prefix-sum block differences, categorical children via contingency
        tables against the blocks.  They differ only in the candidate matrix:

        - ``"exact"``: all canonical partitions (smallest code pinned right),
          ``2^(k-1)-1`` candidates — exponential in k.
        - ``"one_vs_rest"``: each category alone vs the complement, k linear
          candidates built by direct slicing (no enumeration matrix).
        """
        edges = np.flatnonzero(np.diff(scol)) + 1
        bounds = np.concatenate(([0], edges, [n]))
        sizes_i = np.diff(bounds)  # int block sizes (row counts)
        sizes_f = sizes_i.astype(np.float64)
        k_blocks = sizes_i.size
        block_codes = scol[bounds[:-1]].astype(np.int64)
        ovr = self.categorical_splitter == "one_vs_rest"

        # Weight sums per block (equals sizes_f when sample_weight is None).
        if sw_node is not None:
            sw_sorted = sw_node[sw_node.argsort(kind="mergesort")] if False else None
            # sw_node is in idx order; rows maps sorted positions to idx positions
            # Actually sw_node = self._sw_[idx] and rows = idx[orders[j]],
            # so sw at sorted positions is self._sw_[rows] = self._sw_[idx[orders[j]]]
            sw_at_sorted = self._sw_[rows]
            block_w = np.zeros(k_blocks)
            for b in range(k_blocks):
                block_w[b] = sw_at_sorted[bounds[b] : bounds[b + 1]].sum()
        else:
            block_w = sizes_f

        if ovr:  # candidate c puts block c alone on the left; linear in k
            cand_left = None
            nL = sizes_f
            nL_w = block_w
        else:  # all canonical partitions (smallest code pinned right)
            half = 1 << (k_blocks - 1)
            if half > _MAX_EXACT_CANDIDATES:
                raise ValueError(
                    f"Exact categorical split on {k_blocks} categories requires "
                    f"{half} candidate subsets (> {_MAX_EXACT_CANDIDATES}). "
                    f"Use categorical_splitter='one_vs_rest' or reduce the "
                    f"number of categories in this node."
                )
            bits = ((np.arange(half)[:, None] >> np.arange(k_blocks - 1)) & 1).astype(np.float64)
            cand_left = np.concatenate([np.zeros((half, 1)), bits], axis=1)  # block 0 -> right
            del bits
            nL = cand_left @ sizes_f
            nL_w = cand_left @ block_w

        valid = (nL >= self.min_leaf_) & (n - nL >= self.min_leaf_)
        if not valid.any():
            return None
        nw_total = float(block_w.sum())
        nL_safe = np.maximum(nL_w, _VAR_EPS)[:, None]
        nR_w = nw_total - nL_w
        nR_safe = np.maximum(nR_w, _VAR_EPS)[:, None]

        # numeric children impurity (weighted mean over numeric features)
        if self._n_num_:
            cs = self._bufs_["cs"][: self._n_num_, : n + 1]
            cs2 = self._bufs_["cs2"][: self._n_num_, : n + 1]
            bsums = cs[:, bounds[1:]] - cs[:, bounds[:-1]]
            bsqs = cs2[:, bounds[1:]] - cs2[:, bounds[:-1]]
            if ovr:  # direct slices: candidate c selects block c's column
                SL, QL = bsums.T, bsqs.T
            else:
                SL, QL = cand_left @ bsums.T, cand_left @ bsqs.T
            vL = np.maximum(QL / nL_safe - (SL / nL_safe) ** 2, 0.0)
            hL_num = self._num_reduce(vL / rv, axis=-1)
            vR = np.maximum(
                (tot2[None, :] - QL) / nR_safe - ((tot[None, :] - SL) / nR_safe) ** 2,
                0.0,
            )
            hR_num = self._num_reduce(vR / rv, axis=-1)
        else:
            hL_num = hR_num = np.zeros(nL.size)

        # contingency counts of every categorical column against the blocks
        block_ids = np.repeat(np.arange(k_blocks), sizes_i)
        sw_sorted = self._sw_[rows] if sw_node is not None else None
        cnts = []
        for cj in self._cat_set_:
            meta = self._cat_meta_[cj]
            if cj == j:
                # left counts of column j under a partition = union of blocks
                cnt = np.zeros((k_blocks, meta["k"]))
                left_cols = np.searchsorted(meta["codes"], block_codes)
                cnt[np.arange(k_blocks), left_cols] = block_w
            else:
                vals = self._XcT_cat_[meta["pos"]][rows]
                cnt = self._cat_counts_by_block(vals, meta, block_ids, k_blocks, sw_sorted)
            cnts.append(cnt)

        sum_gL = np.zeros(nL.size)
        sum_gR = np.zeros(nL.size)
        for cj, cnt in zip(self._cat_set_, cnts):
            NL = cnt if ovr else cand_left @ cnt  # (n_cand, k)
            NR = cnt.sum(axis=0)[None, :] - NL
            gL = 1.0 - ((NL / nL_safe) ** 2).sum(axis=1)
            gR = 1.0 - ((NR / nR_safe) ** 2).sum(axis=1)
            gr = self._root_gini_[cj]
            wc = self._w_all_[cj] if self._weighted_ else 1.0
            sum_gL += wc * gL / gr
            sum_gR += wc * gR / gr

        hL = self._combined_heterogeneity(hL_num, sum_gL)
        hR = self._combined_heterogeneity(hR_num, sum_gR)
        gains = hs - (nL_w / nw_total) * hL - ((nw_total - nL_w) / nw_total) * hR
        gains[~valid] = -np.inf
        local = int(np.argmax(gains))
        if not np.isfinite(gains[local]):
            return None
        if ovr:
            cats = frozenset((int(block_codes[local]),))
        else:
            cats = frozenset(int(block_codes[c]) for c in np.flatnonzero(cand_left[local]))
        return {
            "gain": float(gains[local]),
            "feat": j,
            "thr": float("nan"),
            "kind": "subset",
            "cats": cats,
        }

    def _finalize_leaf(self, node: dict, leaves: list, chains: list, labels: NDArray) -> None:
        node["leaf"] = True
        node["leaf_id"] = len(leaves)
        node["n"] = int(node["idx"].size)
        idx = node["idx"]
        sw_node = self._sw_[idx] if self._sw_ is not None else None
        if self._n_num_:
            if sw_node is None:
                vars_ = np.var(self._XcT_num_[:, idx], axis=1)
            else:
                sub = self._XcT_num_[:, idx].astype(np.float64, copy=False)
                mean_w = np.average(sub, weights=sw_node, axis=1)
                vars_ = np.average((sub - mean_w[:, None]) ** 2, weights=sw_node, axis=1)
            num_ratio = float(self._num_reduce(np.maximum(vars_, 0.0) / self._root_var_, axis=0))
        else:
            num_ratio = 0.0
        cat_sum = self._node_cat_ratios_from_idx(idx, sw_node)
        het = self._combined_heterogeneity(num_ratio, cat_sum)
        leaves.append(
            Leaf(
                id=node["leaf_id"],
                n_samples=node["n"],
                fraction=node["n"] / self.n_samples_in_,
                depth=node["depth"],
                heterogeneity=het,
                conditions=tuple(
                    (self.feature_names_[j], op, payload) for j, op, payload in node["chain"]
                ),
            )
        )
        chains.append(tuple(node["chain"]))
        labels[idx] = node["leaf_id"]
        node["idx"] = None

    @staticmethod
    def _strip_indices(node: dict) -> None:
        stack = [node]
        while stack:
            nd = stack.pop()
            nd.pop("idx", None)
            nd.pop("orders", None)
            nd.pop("chain", None)
            nd.pop("best", None)
            for key in ("left", "right"):
                child = nd.get(key)
                if child is not None:
                    stack.append(child)

    # ---------------------------------------------------------------- report

    def summary(self) -> dict:
        """Return a compact dict of key fitting statistics."""
        check_is_fitted(self)
        sizes = np.array([lf.n_samples for lf in self.leaves_])
        depths = [lf.depth for lf in self.leaves_]
        return {
            "n_samples": self.n_samples_in_,
            "n_features": self.n_features_in_,
            "n_leaves": self.n_leaves_,
            "n_splits": len(self.cuts_),
            "reduction": self.reduction_,
            "total_gain": self.total_gain_,
            "leaf_size_min": int(sizes.min()),
            "leaf_size_median": float(np.median(sizes)),
            "leaf_size_max": int(sizes.max()),
            "depth_min": int(min(depths)),
            "depth_max": int(max(depths)),
            "min_leaf": self.min_leaf_,
        }


def _iter_split_nodes(data: dict):
    """Yield every split node in a export_dict tree (root first)."""
    stack = [data]
    while stack:
        nd = stack.pop()
        if nd.get("type") == "split":
            yield nd
            stack.append(nd["right"])
            stack.append(nd["left"])


def _iter_leaf_nodes(data: dict):
    """Yield every leaf node in a export_dict tree."""
    stack = [data]
    while stack:
        nd = stack.pop()
        if nd.get("type") == "leaf":
            yield nd
        else:
            stack.append(nd.get("right"))
            stack.append(nd.get("left"))
