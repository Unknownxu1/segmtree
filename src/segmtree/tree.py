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
from dataclasses import dataclass
from itertools import count
from numbers import Integral, Real

import numpy as np

from ._base import BaseEstimator, as_2d_array

__all__ = ["Cut", "Leaf", "SegTree"]

_VAR_EPS = 1e-12


@dataclass(frozen=True)
class Cut:
    """An executed split.

    ``op`` is ``"<="``/``">"`` for numeric splits (``threshold`` meaningful,
    ``categories`` empty) or ``"in"`` for categorical splits (``categories``
    holds the integer codes routed left).
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
    categories: tuple = ()


@dataclass(frozen=True)
class Leaf:
    """A terminal segment.

    ``conditions`` holds ``(feature_name, op, payload)`` triples describing
    the path from the root: ``payload`` is a float threshold for ``"<="`` /
    ``">"`` splits or a frozenset of category codes for ``"in"`` splits. An
    empty tuple means the tree never split.
    """

    id: int
    n_samples: int
    fraction: float
    depth: int
    heterogeneity: float
    conditions: tuple


def _resolve_quantiles(quantiles):
    if isinstance(quantiles, Integral) and not isinstance(quantiles, bool):
        if quantiles < 2:
            raise ValueError("quantiles must be an int >= 2 or an array of fractions in (0, 1).")
        return np.linspace(0.05, 0.95, int(quantiles))
    arr = np.asarray(quantiles, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("quantiles must be a 1-D, non-empty array of fractions in (0, 1).")
    if not np.all((arr > 0.0) & (arr < 1.0)):
        raise ValueError("All quantile fractions must lie strictly between 0 and 1.")
    return np.unique(arr)


def _sorted_linear_quantiles(s, qs):
    """Quantiles of an ascending array; identical to ``np.quantile(s, qs)``."""
    h = qs * (len(s) - 1)
    lo = np.floor(h).astype(np.intp)
    hi = np.minimum(lo + 1, len(s) - 1)
    frac = h - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _filter_order(order_parent_local, mask):
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
        required to accept a split. Larger values yield fewer segments.
    min_leaf_frac : float, default=0.01
        Minimum segment size as a fraction of the training sample size.
    min_leaf_abs : int, default=50
        Absolute floor for the minimum segment size. The effective minimum
        is ``max(min_leaf_abs, min_leaf_frac * n_samples)``.
    max_depth : int or None, default=None
        Maximum number of consecutive splits along any path. ``None``
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
        of categories rather than by thresholds. Pass column indices or column
        names. Codes must be integral-valued and are handled exactly at any
        magnitude (internal storage is int64/float64 regardless of
        ``dtype``). A column with more than ``max_categories`` distinct codes
        is rejected (group levels or use one-hot instead). Not auto-detected:
        integer counts stay continuous unless listed here. Mutually exclusive
        per column with ``binary_features``.
    max_categories : int, default=12
        Maximum number of distinct codes enumerated exactly per categorical
        feature (subset enumeration grows as 2^(k-1)). Only enforced with
        ``categorical_splitter="exact"``; ``"one_vs_rest"`` accepts any
        cardinality.
    handle_unknown : {"complement", "error"}, default="complement"
        Policy for category codes unseen during training when predicting.
        ``"complement"`` routes them to the complement side of every
        categorical test (rule regions stay exhaustive and ``replay_check``
        holds); ``"error"`` makes ``predict`` / ``apply`` / ``replay_labels``
        raise ``ValueError`` naming the feature and the unknown codes.
    feature_names : sequence of str or None, default=None
        Names used in rules and reports. Inferred from DataFrame columns
        when available; otherwise ``x0 .. x{m-1}``.
    dtype : numpy floating type, default=numpy.float64
        Storage precision for continuous/binary columns
        (``numpy.float32`` saves memory on large inputs). Split statistics
        accumulate in float64, and categorical codes are always kept exactly.
    feature_weights : sequence, dict or None, default=None
        Per-feature weights ``w_j >= 0`` (not all zero) scaling each column's
        contribution to heterogeneity and gain. ``None`` means uniform
        weights (bit-identical to the unweighted code path). A sequence must
        be 1-D with one finite weight per column; a ``dict`` maps feature
        names (or DataFrame columns) to weights, missing names default to 1.
        Zero-weight features may still be *chosen as splitting variables*
        (acting as proxies for weighted features) — weights influence the
        heterogeneity/gain accounting only, not split eligibility.
    max_leaf_nodes : int or None, default=None
        Hard cap on the total number of leaves. Growth is best-first, so when
        the cap is reached no further splits are executed and remaining
        frontier nodes become leaves. ``None`` disables the limit.
    categorical_splitter : {"exact", "one_vs_rest"}, default="exact"
        Categorical candidate generation. ``"exact"`` enumerates all
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
    feature_names_ : list of str
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

    def __init__(
        self,
        min_gain=0.01,
        min_leaf_frac=0.01,
        min_leaf_abs=50,
        max_depth=None,
        quantiles=8,
        binary_features="auto",
        categorical_features=None,
        max_categories=12,
        handle_unknown="complement",
        feature_names=None,
        dtype=np.float64,
        feature_weights=None,
        max_leaf_nodes=None,
        categorical_splitter="exact",
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

    def fit(self, X, y=None):
        """Grow the tree from a tabular matrix (or DataFrame). Returns ``self``."""
        Xraw, columns = as_2d_array(X)
        n, m = Xraw.shape
        self._check_params(m)

        names = self._resolve_feature_names(Xraw, columns)
        binary_set = self._resolve_binary_features(Xraw, names)
        cat_codes, cat_set = self._resolve_categorical_features(Xraw, names, binary_set)

        self.n_samples_in_ = n
        self.n_features_in_ = m
        self.feature_names_ = list(names)
        self.binary_indices_ = np.array(sorted(binary_set), dtype=np.intp)
        self.categorical_indices_ = np.array(sorted(cat_set), dtype=np.intp)
        self.min_leaf_ = max(int(self.min_leaf_abs), int(self.min_leaf_frac * n))
        self.categories_ = {names[j]: cat_codes[j]["values"] for j in sorted(cat_set)}

        num_feats = np.array([j for j in range(m) if j not in cat_set], dtype=np.intp)
        cat_feats = np.array(sorted(cat_set), dtype=np.intp)
        self._num_feats_ = num_feats
        self._cat_feats_ = cat_feats
        n_num = int(num_feats.size)

        w = self._resolve_feature_weights(names, m)
        self.feature_weights_ = np.ones(m, dtype=np.float64) if w is None else w
        self._weighted_ = w is not None
        if w is not None:
            self._w_all_ = w
            self._w_num_ = w[num_feats].copy()
            self._w_sum_ = float(w.sum())

        # Continuous/binary columns are centered for numerical robustness of
        # prefix sums; committed thresholds live in raw space. Categorical
        # codes are stored separately as float64/int64 so any magnitude is
        # represented exactly regardless of ``dtype``.
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
            np.ascontiguousarray(np.stack([Xraw[:, j].astype(np.float64) for j in cat_feats]))
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
        }

        root_var = np.var(XcT_num.astype(np.float64, copy=False), axis=1) if n_num else np.zeros(0)
        root_var[root_var < _VAR_EPS] = _VAR_EPS
        self._root_var_ = root_var
        self._root_gini_ = {j: self._gini_of(cat_codes[j]["column"]) for j in sorted(cat_set)}
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
        # frontier node with the highest gain (best-first growth). Ties break
        # by insertion order via the monotonic tick, keeping runs deterministic.
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
                # Leaf budget exhausted: every successful split adds exactly one
                # leaf, so with max_leaf_nodes reached no further split may run;
                # remaining frontier nodes become ordinary leaves.
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
                # partition.
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
            total_gain += float(best["gain"]) * node["idx"].size / n
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
        self.h_avg_ = float(np.sum(sizes / n * hets))
        self.reduction_ = 1.0 - self.h_avg_

        self._strip_indices(root)
        self._root_ = root
        del self._Xraw, self._bufs_, self._XcT_num_, self._XcT_cat_
        return self

    def fit_predict(self, X, y=None):
        """Fit and return ``labels_``."""
        return self.fit(X, y).labels_

    def __sklearn_is_fitted__(self):
        return hasattr(self, "labels_")

    # --------------------------------------------------------------- predict

    def _encode_input_categories(self, Xarr):
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

    def _check_unknown(self, Xarr, cat_codes):
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

    def predict(self, X):
        """Assign each row to the segment whose rule region contains it."""
        if not hasattr(self, "labels_"):
            raise RuntimeError("This SegTree instance is not fitted yet; call fit() first.")
        Xarr, _ = as_2d_array(X, context="X")
        if Xarr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {Xarr.shape[1]} features, expected {self.n_features_in_}.")
        cat_codes = self._encode_input_categories(Xarr)
        self._check_unknown(Xarr, cat_codes)
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
                go_left = Xarr[pos, node["cut_feat"]] <= node["cut_thr"]
            if go_left.any():
                stack.append((node["left"], pos[go_left]))
            if (~go_left).any():
                stack.append((node["right"], pos[~go_left]))
        return out

    apply = predict

    # ---------------------------------------------------------------- export

    def _require_fitted(self):
        if not hasattr(self, "labels_"):
            raise RuntimeError("This SegTree instance is not fitted yet; call fit() first.")

    @staticmethod
    def _fmt_threshold(value):
        value = float(value)
        if value.is_integer() and abs(value) < 2**53:
            return str(int(value))
        return f"{value:g}"

    def _export_node_dict(self, nd):
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
            # ``threshold`` is meaningful for numeric splits, ``categories``
            # for categorical splits; the other field is None (JSON null).
            "threshold": None if is_subset else float(nd["cut_thr"]),
            "categories": [int(c) for c in sorted(nd["cut_cats"])] if is_subset else None,
            "gain": float(nd["cut_gain"]),
            "n_samples": int(nd["cut_n"]),
        }
        node["left"] = self._export_node_dict(nd["left"])
        node["right"] = self._export_node_dict(nd["right"])
        return node

    def export_dict(self):
        """Export the fitted tree as a JSON-serializable nested dict.

        The root split/leaf starts the structure; parent-child relations come
        from the tree itself, not from cut execution order. Split nodes carry
        ``type/depth/feature/feature_index/op/threshold/categories/gain/
        n_samples/left/right``; leaf nodes carry ``type/leaf_id/depth/
        n_samples/fraction/heterogeneity/conditions``. All numbers are native
        Python floats/ints and categories are sorted lists, so
        ``json.dumps(tree.export_dict())`` works out of the box.
        """
        self._require_fitted()
        return self._export_node_dict(self._root_)

    def export_text(self):
        """Render the fitted tree as a stable, human-readable indented text.

        Each split line shows the left-branch condition plus its gain and
        sample count; the complementary branch (``>`` / ``not in {...}``) is
        prefixed on the right subtree's first line. Leaves show their id,
        size, fraction, depth and heterogeneity.
        """
        self._require_fitted()
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
                ts = self._fmt_threshold(nd["cut_thr"])
                cond = f"{name} <= {ts}"
                rtag = f"> {ts} "
            lines.append(f"{indent}{tag}{cond}  [gain={nd['cut_gain']:.4f}, n={nd['cut_n']}]")
            walk(nd["left"], "")
            walk(nd["right"], rtag)

        walk(self._root_, "")
        return "\n".join(lines)

    # --------------------------------------------------------------- helpers

    def _check_params(self, m):
        if not isinstance(self.min_gain, Real) or self.min_gain < 0:
            raise ValueError("min_gain must be a float >= 0.")
        if not isinstance(self.min_leaf_frac, Real) or not 0 <= self.min_leaf_frac < 1:
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
        if self.categorical_features is not None and isinstance(self.categorical_features, str):
            raise ValueError("categorical_features must be None or a sequence of names/indices.")
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

    def _resolve_feature_names(self, Xraw, columns):
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
    def _resolve_sequence_spec(spec, names, m, kind):
        name_to_idx = {nm: i for i, nm in enumerate(names)}
        resolved = set()
        for item in spec:
            if isinstance(item, str):
                if item not in name_to_idx:
                    raise KeyError(f"Unknown {kind} feature name: {item!r}.")
                resolved.add(name_to_idx[item])
            elif isinstance(item, bool):
                raise TypeError("Pass a plain int index or str name, not a Python bool.")
            elif isinstance(item, Integral):
                idx = int(item)
                if not 0 <= idx < m:
                    raise IndexError(f"{kind} feature index {idx} out of range for {m} columns.")
                resolved.add(idx)
            else:
                raise TypeError(f"Unsupported entry in {kind}_features: {item!r}.")
        return resolved

    def _resolve_feature_weights(self, names, m):
        """Validate ``feature_weights`` and return a float64 array or ``None``.

        ``None`` keeps the unweighted fast path (bit-identical results). An
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
                raise ValueError(f"feature_weights must be numeric; got {e}") from e
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

    def _resolve_binary_features(self, Xraw, names):
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

    def _resolve_categorical_features(self, Xraw, names, binary_set):
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

    def _column_values(self, j):
        """Full-column training values used for the initial sort orders."""
        if j in self._cat_set_:
            return self._cat_codes_[j]
        return self._XcT_num_[int(np.flatnonzero(self._num_feats_ == j)[0])]

    @staticmethod
    def _gini_of(codes):
        counts = np.bincount(codes)
        counts = counts[counts > 0]
        p = counts / counts.sum()
        return float(1.0 - np.sum(p * p))

    def _gini_codes(self, codes, meta):
        """Gini impurity of a code sample against one categorical column."""
        local = np.searchsorted(meta["codes"], codes)
        counts = np.bincount(local, minlength=meta["k"])
        p = counts[counts > 0] / codes.size
        return float(1.0 - np.sum(p * p))

    def _cat_counts_by_block(self, vals, meta, block_ids, n_blocks):
        """Contingency counts of one categorical column across j-blocks."""
        local = np.searchsorted(meta["codes"], vals)
        flat = block_ids * meta["k"] + local
        table = np.bincount(flat, minlength=n_blocks * meta["k"])
        return table.reshape(n_blocks, meta["k"]).astype(np.float64)

    def _combined_heterogeneity(self, num_term, cat_term):
        """Combine the numeric and categorical terms into ``H``.

        Unweighted: ``(mean_numeric * n_num + sum_categorical) / m``.
        Weighted (callers pass weighted sums for both domains):
        ``(sum_j w_j * term_j) / sum_j w_j``.
        """
        if self._weighted_:
            return (num_term + cat_term) / self._w_sum_
        return (num_term * self._n_num_ + cat_term) / self.n_features_in_

    def _num_reduce(self, M, *, axis):
        """Reduce per-numeric-feature terms along the feature ``axis``.

        Unweighted this is a plain mean; with feature weights it is the
        weighted sum over that axis (division by the total weight happens in
        :meth:`_combined_heterogeneity`). ``M`` may hold its feature axis at
        position 0 or last.
        """
        if self._weighted_:
            ax = axis % M.ndim
            return M @ self._w_num_ if ax == M.ndim - 1 else self._w_num_ @ M
        return M.mean(axis=axis)

    def _node_cat_ratios_from_idx(self, idx):
        total = 0.0
        for j in self._cat_set_:
            meta = self._cat_meta_[j]
            ratio = self._gini_codes(self._cat_codes_[j][idx], meta) / self._root_gini_[j]
            total += ratio * self._w_all_[j] if self._weighted_ else ratio
        return total

    # ------------------------------------------------------------- splitting

    def _best_split(self, node):
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
                else self._XcT_num_[int(np.flatnonzero(self._num_feats_ == j)[0])][rows]
            )
            if scol[-1] - scol[0] < _VAR_EPS:
                continue  # feature constant within this node

            gather(rows)
            if hs is None:
                # Impurity is row-order invariant, so compute it from this
                # first gathered matrix instead of gathering twice.
                num_term = (
                    float(self._num_reduce(np.var(mat, axis=1) / rv, axis=0)) if has_num else 0.0
                )
                cat_term = 0.0
                for cj in self._cat_set_:
                    ratio = (
                        self._gini_codes(self._cat_codes_[cj][rows], self._cat_meta_[cj])
                        / self._root_gini_[cj]
                    )
                    cat_term += ratio * self._w_all_[cj] if self._weighted_ else ratio
                hs = self._combined_heterogeneity(num_term, cat_term)
            if has_num:
                cs[:, 0] = 0.0
                cs2[:, 0] = 0.0
                np.cumsum(mat, axis=1, out=cs[:, 1:])
                np.multiply(mat, mat, out=sq)
                np.cumsum(sq, axis=1, out=cs2[:, 1:])
                tot, tot2 = cs[:, n], cs2[:, n]
            else:
                tot = tot2 = None

            # Per-category cumulative one-hot counts along this gather order,
            # needed to score children Gini for every candidate cut position.
            ccs = []
            if self._cat_set_:
                pos_ar = np.arange(n)
                for cj in sorted(self._cat_set_):
                    meta = self._cat_meta_[cj]
                    vals = self._cat_codes_[cj][rows] if cj != j else scol.astype(np.int64)
                    local = np.searchsorted(meta["codes"], vals)
                    oh = np.zeros((n, meta["k"]))
                    oh[pos_ar, local] = 1.0
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
                cand_best = self._best_cat_split(j, rows, scol, tot, tot2, rv, hs, n)
            else:
                # Candidate thresholds live in raw space and cut positions are
                # located on raw values too: evaluating and committing must
                # agree on which rows go left.
                rcol = self._Xraw[:, j][rows]
                cand_best = self._best_numeric_split(j, rcol, cs, cs2, tot, tot2, rv, hs, n, ccs)
            if cand_best is not None and (best is None or cand_best["gain"] > best["gain"]):
                best = cand_best
        return best

    def _best_numeric_split(self, j, rcol, cs, cs2, tot, tot2, rv, hs, n, ccs):
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

        hL_num, hR_num = self._children_numeric(cs, cs2, tot, tot2, rv, k, kf, nf, n)
        # children Gini ratios of every categorical column at each candidate
        sum_gL = np.zeros(k.size)
        sum_gR = np.zeros(k.size)
        for cc, _total_n, gr, wc in ccs:
            NLc = cc[k]  # (nk, Kc)
            NRc = cc[n][None, :] - NLc
            gL = 1.0 - ((NLc / kf[:, None]) ** 2).sum(axis=1)
            gR = 1.0 - ((NRc / (nf - kf)[:, None]) ** 2).sum(axis=1)
            sum_gL += wc * gL / gr
            sum_gR += wc * gR / gr
        hL = self._combined_heterogeneity(hL_num, sum_gL)
        hR = self._combined_heterogeneity(hR_num, sum_gR)
        gains = hs - (kf / nf) * hL - ((nf - kf) / nf) * hR
        local = int(np.argmax(gains))
        return {
            "gain": float(gains[local]),
            "feat": j,
            "thr": float(t[local]),
            "kind": "threshold",
            "cats": None,
        }

    def _children_numeric(self, cs, cs2, tot, tot2, rv, k, kf, nf, n):
        """Mean normalized-variance of left/right children over numeric features."""
        if not self._n_num_:
            z = np.zeros(k.size)
            return z, z
        csL, cs2L = cs[:, k], cs2[:, k]
        mL = csL / kf
        vL = np.maximum(cs2L / kf - mL**2, 0.0)
        nrf = (n - k).astype(np.float64)
        mR = (tot[:, None] - csL) / nrf
        vR = np.maximum((tot2[:, None] - cs2L) / nrf - mR**2, 0.0)
        return self._num_reduce(vL / rv[:, None], axis=0), self._num_reduce(
            vR / rv[:, None], axis=0
        )

    def _best_cat_split(self, j, rows, scol, tot, tot2, rv, hs, n):
        """Best categorical split of column ``j`` under ``categorical_splitter``.

        Categories form contiguous blocks in ``scol`` (ascending code order).
        Both splitters share the same scoring machinery: numeric children via
        prefix-sum block differences, categorical children via contingency
        tables against the blocks. They differ only in the candidate matrix:

        - ``"exact"``: all canonical partitions (smallest code pinned right),
          ``2^(k-1)-1`` candidates — exponential in k.
        - ``"one_vs_rest"``: each category alone vs the complement, k linear
          candidates built by direct slicing (no enumeration matrix).
        """
        edges = np.flatnonzero(np.diff(scol)) + 1
        bounds = np.concatenate(([0], edges, [n]))
        sizes_i = np.diff(bounds)  # int block sizes
        sizes_f = sizes_i.astype(np.float64)
        k_blocks = sizes_i.size
        block_codes = scol[bounds[:-1]].astype(np.int64)
        ovr = self.categorical_splitter == "one_vs_rest"

        if ovr:  # candidate c puts block c alone on the left; linear in k
            cand_left = None
            nL = sizes_f
        else:  # all canonical partitions (smallest code pinned right)
            half = 1 << (k_blocks - 1)
            bits = ((np.arange(half)[:, None] >> np.arange(k_blocks - 1)) & 1).astype(np.float64)
            cand_left = np.concatenate([np.zeros((half, 1)), bits], axis=1)  # block 0 -> right
            del bits
            nL = cand_left @ sizes_f

        valid = (nL >= self.min_leaf_) & (n - nL >= self.min_leaf_)
        if not valid.any():
            return None
        nL_safe = np.maximum(nL, 1.0)[:, None]
        nR_safe = np.maximum(float(n) - nL, 1.0)[:, None]

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
            nRf = float(n) - nL
            vR = np.maximum(
                (tot2[None, :] - QL) / nRf[:, None] - ((tot[None, :] - SL) / nRf[:, None]) ** 2,
                0.0,
            )
            hR_num = self._num_reduce(vR / rv, axis=-1)
        else:
            hL_num = hR_num = np.zeros(nL.size)

        # contingency counts of every categorical column against the blocks
        block_ids = np.repeat(np.arange(k_blocks), sizes_i)
        cnts = []
        for cj in self._cat_set_:
            meta = self._cat_meta_[cj]
            if cj == j:
                # left counts of column j under a partition = union of blocks
                cnt = np.zeros((k_blocks, meta["k"]))
                left_cols = np.searchsorted(meta["codes"], block_codes)
                cnt[np.arange(k_blocks), left_cols] = sizes_f
            else:
                vals = self._XcT_cat_[meta["pos"]][rows]
                cnt = self._cat_counts_by_block(vals, meta, block_ids, k_blocks)
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

        nf = float(n)
        hL = self._combined_heterogeneity(hL_num, sum_gL)
        hR = self._combined_heterogeneity(hR_num, sum_gR)
        gains = hs - (nL / nf) * hL - ((nf - nL) / nf) * hR
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

    def _finalize_leaf(self, node, leaves, chains, labels):
        node["leaf"] = True
        node["leaf_id"] = len(leaves)
        node["n"] = int(node["idx"].size)
        idx = node["idx"]
        num_ratio = (
            float(
                self._num_reduce(np.var(self._XcT_num_[:, idx], axis=1) / self._root_var_, axis=0)
            )
            if self._n_num_
            else 0.0
        )
        cat_sum = self._node_cat_ratios_from_idx(idx)
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
    def _strip_indices(node):
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

    def summary(self):
        """Return a compact dict of key fitting statistics."""
        if not hasattr(self, "labels_"):
            raise RuntimeError("This SegTree instance is not fitted yet; call fit() first.")
        sizes = np.array([lf.n_samples for lf in self.leaves_])
        depths = [lf.depth for lf in self.leaves_]
        return {
            "n_samples": self.n_samples_in_,
            "n_features": self.n_features_in_,
            "n_segments": self.n_leaves_,
            "n_splits": len(self.cuts_),
            "heterogeneity_reduction": self.reduction_,
            "total_gain": self.total_gain_,
            "segment_size_min": int(sizes.min()),
            "segment_size_median": float(np.median(sizes)),
            "segment_size_max": int(sizes.max()),
            "depth_min": int(min(depths)),
            "depth_max": int(max(depths)),
            "min_leaf": self.min_leaf_,
        }
