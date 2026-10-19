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
import time
import warnings
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

# -- scalability budgets -----------------------------------------------------
# The histogram path is the default.  ``splitter="exact"`` remains as an
# explicit legacy/reference mode; these budgets bound its working set.
_EXACT_BUFFER_BYTES = 192 * 1024**2  # numeric prefix-sum buffers per chunk
_EXACT_CAT_MATRIX_BYTES = 512 * 1024**2  # exact categorical candidate matrix
_ASSOC_SAMPLE_ROWS = 50_000  # rows used to estimate pairwise association
_DENSE_CAT_CELLS = 4_000_000  # above this, categorical cont tables go sparse


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
    features contribute ``h_j(S) = Gini_j(S)/Gini_j(root)``, so ``H(root) = 1``
    when no column is constant (``h_root_`` reports the actual value).
    Categorical impurity is invariant to any one-to-one recoding of category
    labels. The gain of a split into L/R is::

        Gain = H(S) - (n_L/n_S * H(L) + n_R/n_S * H(R))

    (with ``n`` replaced by the sum of ``sample_weight`` when provided).

    The default ``splitter="hist"`` path bins each continuous feature into at
    most ``max_bins`` quantile bins, proposes candidates per node with an
    association-gated proxy, and re-scores the top candidates with the
    node-local heterogeneity drop of the other features.  The explicit
    ``splitter="exact"`` legacy mode instead evaluates the in-node quantiles
    with sort + prefix sums.  Binary features get a single 0.5 test;
    categorical features use subset membership (``categorical_splitter=
    "exact"``) or single-category-vs-rest tests (``"one_vs_rest"``).  Among
    all candidates the highest-gain leaf is always expanded next: the frontier
    lives in a min-heap keyed by *negative* gain, so every pop extracts the
    current best split (ties broken deterministically by insertion order).

    Parameters
    ----------
    min_gain : float, default=0.01
        Minimum heterogeneity reduction required to accept a split.  Must be
        finite and >= 0.  Larger values yield fewer segments.  The unit is
        controlled by ``min_gain_mode``: on the default histogram path the
        threshold is decoupled from the number of columns, so adding weak
        features does not silently stop the tree at the root.
    min_gain_mode : {"auto", "sqrt", "total", "normalized"}, default="auto"
        Controls how the normalized gain is compared with ``min_gain``.
        ``"normalized"`` is the legacy behaviour (gain averaged over all
        features), whose effective threshold grows with the number of columns:
        adding noise columns can silently stop the tree at the root.
        ``"sqrt"`` compares ``gain * sqrt(n_features)``, keeping the threshold
        meaningful at any width; ``"total"`` compares the raw sum of
        per-feature reductions (most permissive on wide data).  ``"auto"``
        (default) uses ``"sqrt"`` on the default histogram path (whose gain
        is already association-gated) and ``"normalized"`` only for the
        explicit ``splitter="exact"`` legacy mode, preserving its original
        semantics.  Weighted fits use the total feature weight in place of
        ``n_features``.
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
        Used by the explicit ``splitter="exact"`` mode; the default histogram
        path uses ``max_bins`` instead.
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
        Not auto-detected: integer counts stay continuous unless listed here.
        Mutually exclusive per column with ``binary_features``.
    max_categories : int, default=12
        Maximum number of distinct codes enumerated exactly per categorical
        feature (subset enumeration grows as 2^(k-1)).  With the default
        ``categorical_splitter="auto"`` columns above this limit fall back to
        ``"one_vs_rest"`` with a warning instead of failing the fit.
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
    categorical_splitter : {"auto", "exact", "one_vs_rest"} or dict, default="auto"
        Categorical candidate generation.  ``"exact"`` enumerates all
        canonical subsets (O(2^(k-1)) per node); ``"one_vs_rest"`` tests each
        category alone against the rest (linear in k, supports arbitrarily
        high cardinality but can miss splits that merge several categories
        onto one side); ``"auto"`` uses exact when the observed cardinality
        and candidate-matrix memory fit the budget, otherwise falls back to
        one-vs-rest with a warning.  A dict maps feature names (or indices)
        to one of these strategies, so different columns can use different
        splitters; unlisted columns use ``"auto"``.
    splitter : {"hist", "exact"}, default="hist"
        Split-search implementation.  ``"hist"`` (the default for every data
        size) bins each continuous feature once (``max_bins`` quantile bins)
        and scores splits with the per-feature impurity reduction weighted by
        that feature's mean squared association with the other columns
        (``association_`` stores the sum of squared associations; the score
        divides it by ``n_features - 1``); per-node cost is
        ``O(n_node · n_features)``, the full-data buffers are ``n x F`` values
        plus one byte per bin code (two bytes above 256 bins), and no
        per-feature sort orders are kept.  A pairwise ``O(F²)`` association
        matrix exists during fit.  Its search first ranks candidates by an
        association-gated proxy, then scores the top few with the actual
        node-local heterogeneity drop of the *other* features — this avoids
        the median-cut self-variance bias and keeps multi-column pure-noise
        data at a single leaf (a single-column problem has no corroboration
        and uses the exact-path self-variance criterion).
        ``"exact"`` is the original algorithm, kept as an explicit
        legacy/reference mode: every node evaluates the full mixed-type
        heterogeneity decrease with prefix sums, costing
        ``O(n_features² · n_node)`` and several ``(n_numeric, n_node)``
        buffers.  It is not recommended for production use; it exists for
        validation against the independent naive reference and for users who
        need its original criterion semantics.
        On the histogram path, leaf heterogeneities / ``reduction_`` are
        always exact, but split thresholds are bin boundaries and
        ``Cut.gain`` is a node-local cross-feature score.
    max_bins : int, default=32
        Maximum number of quantile bins per continuous feature on the
        ``splitter="hist"`` path (2..65536).  More bins give finer thresholds
        at proportionally more per-node work.  Ignored by the exact path.
    random_state : int, numpy Generator or None, default=None
        Seed for reproducible randomness: association subsampling on the
        histogram path and random tie-breaking between equally good splits.
        With ``None`` sampling is seeded deterministically and ties keep the
        legacy first-seen order, so fits are reproducible.
    verbose : int, default=0
        ``0`` is silent.  ``>= 1`` prints one line per executed split
        (depth, node size, gain, children, elapsed time) to stdout, which
        makes long fits on large data observable.

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
        On the exact path, the size-weighted sum of split gains; equals
        ``h_root_ - h_avg_`` up to floating-point error.  On the histogram
        path this is set to the exact ``h_root_ - h_avg_`` because split
        gains are search scores and do not telescope.
    h_root_, h_root_normalized_, h_avg_ : float
        ``h_root_`` is the actual root heterogeneity (1.0 when no column is
        constant, lower when constant columns are present, 0 for all-constant
        input).  ``h_root_normalized_`` is the 1.0 normalization baseline by
        construction.  Neither is a variance: use ``root_variances_`` /
        ``root_ginis_`` for the raw per-feature root statistics.  ``h_avg_``
        is the size-weighted mean leaf heterogeneity (always exact, also on
        the histogram path).
    reduction_ : float
        Fraction of total heterogeneity explained: ``1 - h_avg_``.
    root_variances_ : ndarray of float64, shape (n_numeric,)
        Variance of each continuous/binary column at the root (raw units).
    root_ginis_ : dict
        Maps each categorical feature name to its root Gini impurity.
    splitter_ : {"exact", "hist"}
        Split-search implementation actually used by ``fit``.
    max_bins_ : int
        Effective bin cap used by the histogram path.
    association_ : ndarray of float64, shape (n_features_in_,)
        Sum of squared associations of each column with the other columns
        (Pearson r² / correlation ratio η² / Cramér's V²), estimated on at
        most 50k rows.  Empty on the exact path.  The pairwise matrix is
        O(n_features²) during fit (0.7 MB at 300 columns).
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
        categorical_splitter: str | dict = "auto",
        splitter: str = "hist",
        max_bins: int = 32,
        min_gain_mode: str = "auto",
        random_state: int | np.random.Generator | None = None,
        verbose: int = 0,
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
        self.splitter = splitter
        self.max_bins = max_bins
        self.min_gain_mode = min_gain_mode
        self.random_state = random_state
        self.verbose = verbose

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
        self._w_sum_ = float(w.sum()) if w is not None else float(m)
        if w is not None:
            self._w_all_ = w
            self._w_num_ = w[num_feats].copy()

        # Resolve scalability choices *before* allocating any (n_num x n)
        # arrays, so an explicit exact request fails early.
        self.splitter_ = self.splitter
        self._cat_splitter_ = self._resolve_cat_splitters(names, cat_codes, n_num, self.splitter_)
        self._min_gain_mode_ = (
            ("normalized" if self.splitter_ == "exact" else "sqrt")
            if self.min_gain_mode == "auto"
            else self.min_gain_mode
        )
        self.max_bins_ = int(self.max_bins)
        self._gain_scale_ = self._w_sum_ if self._weighted_ else float(m)
        self._rng_ = self._resolve_rng()
        self._random_ties_ = self.random_state is not None

        # Continuous/binary columns are centered for numerical robustness of
        # prefix sums; committed thresholds live in raw space.  Categorical
        # codes are stored separately as int64 so any magnitude (including
        # negative codes) is represented exactly regardless of ``dtype``.
        #
        # Everything is built column by column: a vectorized
        # ``Xraw[:, num_feats].astype(float64)`` would allocate an extra
        # n x n_num x 8-byte temporary (multiple GB on 800k x 300).
        dtype = np.dtype(self.dtype)
        mean64 = np.empty(n_num, dtype=np.float64) if n_num else np.zeros(0)
        for pos, j in enumerate(num_feats):
            mean64[pos] = np.mean(Xraw[:, int(j)], dtype=np.float64)
        self._mean_ = mean64
        self._cat_codes_ = {j: cat_codes[j]["column"] for j in sorted(cat_set)}  # int64, exact
        XcT_num = np.zeros((n_num, n), dtype=dtype) if n_num else np.zeros((0, n), dtype=dtype)
        for pos, j in enumerate(num_feats):
            XcT_num[pos] = Xraw[:, int(j)].astype(np.float64, copy=False) - mean64[pos]
        self._XcT_num_ = XcT_num
        # Categorical columns are accessed through the exact int64 code
        # arrays; no float64 copy is kept (it would collapse codes above
        # 2**53 and cost extra memory).
        self._XcT_cat_ = None
        self._stage64 = dtype == np.float64

        root_var = np.zeros(n_num, dtype=np.float64)
        for pos in range(n_num):
            root_var[pos] = np.var(XcT_num[pos], dtype=np.float64)
        self.root_variances_ = root_var.copy()
        root_var = np.maximum(root_var, _VAR_EPS)
        self._root_var_ = root_var
        self._root_gini_ = {}
        self.root_ginis_ = {}
        for j in sorted(cat_set):
            g = self._gini_codes(
                self._cat_codes_[j],
                {"codes": cat_codes[j]["values"], "k": int(cat_codes[j]["values"].size)},
            )
            self.root_ginis_[names[j]] = g
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

        if self.splitter_ == "hist":
            # The histogram path reads the caller's matrix directly while
            # building bins; no full dtype-cast copy is kept afterwards.
            self._Xraw = None
            self._build_histogram_state(Xraw, num_feats, cat_feats)
            self._bufs_ = {}
            self._num_blocks_ = []
        else:
            self._Xraw = np.ascontiguousarray(Xraw.astype(dtype, copy=False))
            self.association_ = np.empty(0, dtype=np.float64)
            # Five persistent float64 buffers (mat/cs/cs2/sq/wmat) plus
            # transient float64 temporaries in the per-block reductions.
            buf_rows = max(1, min(n_num, int(_EXACT_BUFFER_BYTES // max(1, 8 * 8 * n))))
            rows_buf = buf_rows if n_num else 0
            self._bufs_ = {
                "mat": np.empty((rows_buf, n), dtype=np.float64),
                "stage": (np.empty((rows_buf, n), dtype=dtype) if dtype != np.float64 else None),
                "cs": np.empty((rows_buf, n + 1), dtype=np.float64),
                "cs2": np.empty((rows_buf, n + 1), dtype=np.float64),
                "sq": np.empty((rows_buf, n), dtype=np.float64),
                "wmat": np.empty((rows_buf, n), dtype=np.float64),
                "cw": np.empty(n + 1, dtype=np.float64),
            }
            self._num_blocks_ = [
                slice(s, min(s + buf_rows, n_num)) for s in range(0, n_num, buf_rows)
            ]
        del XcT_num

        tick = count()
        fit_start = time.perf_counter()

        def new_node(idx, depth, chain, orders=None):
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

        root_orders = (
            [np.argsort(self._column_values(j), kind="mergesort") for j in range(m)]
            if self.splitter_ == "exact"
            else None
        )
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
            can_split = best is not None and self._accept_gain(best["gain"])
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
                if best.get("bin") is not None:
                    # Histogram path: reproduce the evaluated partition from
                    # the cached bin codes (no raw matrix is kept in memory).
                    pos = self._num_pos_[j]
                    go_left = self._bin_codes_[pos][node["idx"]] <= int(best["bin"])
                else:
                    # Exact path: commit on raw values with the same raw
                    # threshold the evaluation used.  ``self._Xraw`` is stored
                    # in ``self.dtype`` so this matches predict time.
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

            if self.splitter_ == "exact":
                # Children inherit the parent's per-feature sort orders by
                # filtering them: O(n) instead of re-sorting O(n log n).
                nl_orders = [_filter_order(oj, go_left) for oj in node["orders"]]
                nr_orders = [_filter_order(oj, ~go_left) for oj in node["orders"]]
            else:
                nl_orders = nr_orders = None
            nl = new_node(il, node["depth"] + 1, node["chain"] + chain_l, nl_orders)
            nr = new_node(ir, node["depth"] + 1, node["chain"] + chain_r, nr_orders)
            node["left"], node["right"] = nl, nr
            if self.verbose:
                elapsed = time.perf_counter() - fit_start
                print(
                    f"[segmtree] split {len(cuts)}: depth={node['depth']} "
                    f"n={node['idx'].size} {self.feature_names_[j]} "
                    f"gain={best['gain']:.6g} left={il.size} right={ir.size} "
                    f"leaves={len(cuts) + 1} elapsed={elapsed:.2f}s "
                    f"({self.splitter_})",
                    flush=True,
                )
            for child in (nl, nr):
                child["best"] = self._best_split(child)
                cb = child["best"]
                heapq.heappush(heap, (-(cb["gain"]) if cb else 0.0, next(tick), child))

        sizes = np.array([lf.n_samples for lf in leaves], dtype=np.float64)
        hets = np.array([lf.heterogeneity for lf in leaves], dtype=np.float64)
        self.labels_ = labels
        self.cuts_ = tuple(cuts)
        self._used_cat_feats_ = np.array(
            sorted({int(cut.feature_index) for cut in cuts if cut.op == "in"}),
            dtype=np.intp,
        )
        self.leaves_ = tuple(leaves)
        self.leaf_chains_ = tuple(chains)
        self.n_leaves_ = len(leaves)
        # Actual root heterogeneity: constant columns contribute 0, so this
        # is 1.0 only when no column is constant (and 0 for all-constant X).
        # ``h_root_normalized_`` remains the 1.0 normalization baseline.
        num_root = 0.0
        if n_num:
            ratios = (self.root_variances_ > _VAR_EPS).astype(np.float64)
            num_root = (
                float(np.dot(self._w_num_, ratios)) if self._weighted_ else float(ratios.mean())
            )
        cat_root = 0.0
        for j in sorted(cat_set):
            if self.root_ginis_[names[j]] > _GINI_EPS:
                cat_root += self._w_all_[j] if self._weighted_ else 1.0
        self.h_root_ = self._combined_heterogeneity(num_root, cat_root)
        self.h_root_normalized_ = 1.0
        if self._sw_ is not None:
            leaf_w = np.array([self._row_weight(np.flatnonzero(labels == lf.id)) for lf in leaves])
            self.h_avg_ = float(np.sum(leaf_w / self._total_weight() * hets))
        else:
            self.h_avg_ = float(np.sum(sizes / n * hets))
        # Exact split gains telescope to h_root_ - h_avg_.  Histogram gains
        # are search scores, so report the exact partition improvement.
        self.total_gain_ = (
            float(total_gain) if self.splitter_ == "exact" else self.h_root_ - self.h_avg_
        )
        self.reduction_ = 1.0 - self.h_avg_ / self.h_root_ if self.h_root_ > 0 else 0.0
        self.feature_importances_ = self._compute_feature_importances()

        self._strip_indices(root)
        self._root_ = root
        self._Xraw = None
        self._bufs_ = {}
        self._XcT_num_ = None
        self._XcT_cat_ = None
        self._bin_codes_ = None
        # Training-only state; predict/rules/export never read these.
        self._cat_codes_ = None
        self._cat_meta_ = None
        self._sw_ = None
        if self.verbose:
            print(
                f"[segmtree] fitted {self.n_leaves_} leaves with "
                f"{len(self.cuts_)} splits in {time.perf_counter() - fit_start:.2f}s "
                f"({self.splitter_} path, reduction={self.reduction_:.6g})",
                flush=True,
            )
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

    def _encode_input_categories(
        self, Xarr: NDArray, columns: NDArray | None = None
    ) -> dict[int, NDArray[np.int64]]:
        """Exact int64 codes for the categorical columns of a prediction matrix.

        ``columns`` defaults to every declared categorical column; ``predict``
        passes only the columns actually used by a cut (unless
        ``handle_unknown="error"`` requires checking all of them).
        """
        out = {}
        for j in self._cat_feats_ if columns is None else columns:
            out[int(j)] = self._integral_codes(Xarr[:, j], self.feature_names_[j], prediction=True)
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
        if self.handle_unknown == "error":
            cat_codes = self._encode_input_categories(Xarr)
            self._check_unknown(Xarr, cat_codes)
        else:
            # Only cut features need codes; encoding every declared column
            # costs n x 8 bytes each even when it is never used.
            used = getattr(self, "_used_cat_feats_", None)
            cat_codes = self._encode_input_categories(
                Xarr, self._cat_feats_ if used is None else used
            )
        cmp_dtype = np.dtype(self.dtype)
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
                # Cast only the sliced column (not the whole matrix): a
                # float64 prediction matrix must not cost an n*F copy just
                # because training used float32 (or vice versa).
                col = Xarr[pos, node["cut_feat"]]
                if col.dtype != cmp_dtype:
                    col = col.astype(cmp_dtype)
                go_left = col <= node["cut_thr"]
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
            np.array([np.var(Xarr[:, j], dtype=np.float64) for j in num_cols])
            if num_cols
            else np.zeros(0)
        )
        root_var = np.maximum(root_var, _VAR_EPS)
        known_gini = {}
        known_codes = {}
        for j in sorted(cat_idx):
            codes = self._integral_codes(Xarr[:, j], self.feature_names_[j], prediction=True)
            uniq = np.unique(codes)
            known_codes[j] = uniq
            known_gini[j] = max(
                self._gini_codes(codes, {"codes": uniq, "k": int(uniq.size)}), _GINI_EPS
            )

        def het_of(idx: NDArray) -> float:
            num_term = 0.0
            if num_cols:
                # Column loop avoids a dense (n_node x n_num) float64 copy.
                vars_ = (
                    np.array([np.var(Xarr[idx, j], dtype=np.float64) for j in num_cols])
                    if idx.size
                    else np.zeros(len(num_cols))
                )
                ratios = np.maximum(vars_, 0.0) / root_var
                if self._weighted_:
                    num_term = float(np.dot(fw[num_cols], ratios))
                else:
                    num_term = float(np.mean(ratios))
            cat_term = 0.0
            for j in sorted(cat_idx):
                codes = (
                    self._integral_codes(Xarr[idx, j], self.feature_names_[j], prediction=True)
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
        used_cat: set[int] = set()
        stack = [root]
        while stack:
            nd = stack.pop()
            if nd["cut_cats"] is not None:
                used_cat.add(int(nd["cut_feat"]))
            if nd.get("left") is not None:
                stack.extend((nd["left"], nd["right"]))
        tree._used_cat_feats_ = np.array(sorted(used_cat), dtype=np.intp)
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
        valid_cat_splitters = ("auto", "exact", "one_vs_rest")
        if isinstance(self.categorical_splitter, str):
            if self.categorical_splitter not in valid_cat_splitters:
                raise ValueError(
                    'categorical_splitter must be "auto", "exact", "one_vs_rest", '
                    "or a dict mapping feature names/indices to one of those."
                )
        elif isinstance(self.categorical_splitter, dict):
            for key, value in self.categorical_splitter.items():
                if not isinstance(key, (str, Integral)) or isinstance(key, bool):
                    raise ValueError(
                        "categorical_splitter dict keys must be feature names or indices."
                    )
                if value not in valid_cat_splitters:
                    raise ValueError(
                        f"categorical_splitter[{key!r}] must be one of {valid_cat_splitters}."
                    )
        else:
            raise TypeError(
                "categorical_splitter must be a string or a dict mapping "
                "feature names/indices to a strategy."
            )
        if self.splitter not in ("hist", "exact"):
            raise ValueError('splitter must be "hist" or "exact".')
        if isinstance(self.max_bins, bool) or not isinstance(self.max_bins, Integral):
            raise ValueError("max_bins must be an int in [2, 65536].")
        if not 2 <= int(self.max_bins) <= 65536:
            raise ValueError("max_bins must be an int in [2, 65536].")
        if self.min_gain_mode not in ("auto", "sqrt", "total", "normalized"):
            raise ValueError('min_gain_mode must be "auto", "sqrt", "total", or "normalized".')
        if isinstance(self.verbose, bool) or not isinstance(self.verbose, Integral):
            raise ValueError("verbose must be an int >= 0.")
        if self.verbose < 0:
            raise ValueError("verbose must be an int >= 0.")
        rs = self.random_state
        if rs is not None and not (
            isinstance(rs, (Integral, np.random.Generator)) and not isinstance(rs, bool)
        ):
            raise ValueError("random_state must be an int, numpy Generator, or None.")
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
            # Vectorized min/max filter first: running np.unique on every
            # column is O(F * n log n) and took ~25 s at 829k x 300.
            try:
                mn = np.min(Xraw, axis=0)
                mx = np.max(Xraw, axis=0)
            except (TypeError, ValueError):  # mixed object data
                mn = mx = None
            if mn is not None:
                candidates = [j for j in range(Xraw.shape[1]) if mn[j] >= 0.0 and mx[j] <= 1.0]
                return {j for j in candidates if np.all((Xraw[:, j] == 0) | (Xraw[:, j] == 1))}
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
            codes_j = self._integral_codes(Xraw[:, j], names[j], prediction=False)
            uniq = np.unique(codes_j)
            codes[j] = {"column": codes_j, "values": uniq}
        return codes, resolved

    @staticmethod
    def _integral_codes(col: NDArray, name: str, *, prediction: bool) -> NDArray[np.int64]:
        """Exact int64 category codes without a lossy float64 round-trip.

        Integer-typed columns are cast directly, so codes above ``2**53``
        (where float64 can no longer distinguish adjacent integers) remain
        distinct.  Float columns must be finite and integral-valued.
        """
        arr = np.asarray(col)
        if np.issubdtype(arr.dtype, np.integer):
            if arr.dtype == np.uint64 and arr.size and int(arr.max()) > np.iinfo(np.int64).max:
                raise ValueError(
                    f"Categorical column {name!r} contains codes above the int64 range."
                )
            return arr.astype(np.int64, copy=False)
        fcol = arr.astype(np.float64, copy=False)
        if prediction:
            if not np.isfinite(fcol).all() or not np.array_equal(fcol, np.round(fcol)):
                raise ValueError(
                    f"Categorical column {name!r} must contain finite integer category codes."
                )
        else:
            if not np.isfinite(fcol).all():
                raise ValueError(f"Categorical column {name!r} contains NaN/infinite values.")
            if not np.array_equal(fcol, np.round(fcol)):
                raise ValueError(
                    f"Categorical column {name!r} must hold integral category codes "
                    "(e.g. 0..k-1); encode non-numeric labels first (e.g. pandas factorize)."
                )
        return np.rint(fcol).astype(np.int64)

    # -------------------------------------------------- scalability helpers

    def _resolve_cat_splitters(
        self, names: list[str], cat_codes: dict, n_num: int, splitter: str
    ) -> dict[int, str]:
        """Resolve ``categorical_splitter`` (possibly per column) for this fit.

        ``"auto"`` keeps exact enumeration only when both the cardinality cap
        and the candidate-matrix memory budget allow it; otherwise the column
        degrades to one-vs-rest with a warning.  Explicitly requested
        ``"exact"`` raises *before* any node is built when it cannot fit.
        """
        spec = self.categorical_splitter
        cat_set = set(cat_codes)
        explicit: dict[int, str] = {}
        default_strategy = spec if isinstance(spec, str) else "auto"
        if isinstance(spec, dict):
            name_to_idx = {nm: i for i, nm in enumerate(names)}
            for key, value in spec.items():
                if isinstance(key, str):
                    if key not in name_to_idx:
                        raise ValueError(
                            f"categorical_splitter key {key!r} does not match any feature name."
                        )
                    j = name_to_idx[key]
                else:
                    j = int(key)
                    if not 0 <= j < len(names):
                        raise ValueError(
                            f"categorical_splitter index {j} is out of range for "
                            f"{len(names)} columns."
                        )
                if j not in cat_set:
                    raise ValueError(
                        f"categorical_splitter entry {key!r} refers to column "
                        f"{names[j]!r}, which is not categorical. Add it to "
                        "categorical_features first."
                    )
                explicit[j] = value

        resolved: dict[int, str] = {}
        for j in sorted(cat_set):
            k = int(cat_codes[j]["values"].size)
            requested = explicit.get(j, default_strategy)
            from_auto = requested == "auto"
            if from_auto:
                strat = "exact" if k <= self.max_categories else "one_vs_rest"
                if k > self.max_categories:
                    warnings.warn(
                        f"Categorical column {names[j]!r} has {k} distinct codes > "
                        f"max_categories={self.max_categories}; using "
                        "categorical_splitter='one_vs_rest' for this column. "
                        "Set categorical_splitter='exact' explicitly to raise instead.",
                        UserWarning,
                        stacklevel=3,
                    )
            else:
                strat = requested

            if strat == "exact" and k > self.max_categories:
                message = (
                    f"Categorical column {names[j]!r} has {k} distinct codes > "
                    f"max_categories={self.max_categories}: exact subset enumeration "
                    f"evaluates 2^(k-1) = {1 << max(k - 1, 0):,} candidates per node. "
                    "Group rare levels, raise max_categories, use "
                    "categorical_splitter='one_vs_rest', or use the default 'auto' "
                    "to degrade automatically."
                )
                if from_auto:  # pragma: no cover - guarded above
                    warnings.warn(message, UserWarning, stacklevel=3)
                    strat = "one_vs_rest"
                else:
                    raise ValueError(message)

            if strat == "exact":
                half = 1 << max(k - 1, 0)
                # Budget only the path actually selected.  Both paths hold
                # candidate/left/right/probability temporaries of shape
                # (half, k); the exact path adds two (half, n_numeric)
                # float64 matrices.  Counting 7 k-sized arrays matches the
                # peak observed in practice.
                if splitter == "exact":
                    budget_bytes = half * (max(n_num, 1) * 16 + k * 40)
                    shape_desc = f"x ({max(n_num, 1)} numeric x 2 + {k} categories x 7) x 8 bytes"
                else:
                    budget_bytes = half * k * 40
                    shape_desc = f"x {k} category counts x 8 bytes x 5"
                over = half > _MAX_EXACT_CANDIDATES or budget_bytes > _EXACT_CAT_MATRIX_BYTES
                if over:
                    message = (
                        f"Exact categorical split on column {names[j]!r} ({k} codes, "
                        f"node up to {k} levels) needs {half:,} candidate subsets "
                        f"({shape_desc}), "
                        f"~{budget_bytes / 1024**2:,.0f} MiB per node. "
                        f"Use categorical_splitter='one_vs_rest' (or 'auto') or "
                        "reduce the number of categories."
                    )
                    if from_auto:
                        warnings.warn(
                            message + " Falling back to 'one_vs_rest'.",
                            UserWarning,
                            stacklevel=3,
                        )
                        strat = "one_vs_rest"
                    else:
                        raise ValueError(message)
            resolved[j] = strat
        return resolved

    def _resolve_rng(self) -> np.random.Generator:
        """Generator used for association sampling and optional tie-breaking."""
        rs = self.random_state
        if isinstance(rs, np.random.Generator):
            return rs
        return np.random.default_rng(0 if rs is None else int(rs))

    def _quantile_edges(self, col: NDArray, max_bins: int) -> NDArray[np.float64]:
        """Quantile bin boundaries; ``searchsorted(edges, x, 'left')`` bins.

        Large columns use a *random* subsample (plus the extremes).  A
        deterministic stride aliases with periodic columns — e.g. alternating
        0/1 with an even step would sample one phase only and collapse every
        value into a single bin.
        """
        n = int(col.size)
        if n < 2:
            return np.empty(0, dtype=np.float64)
        lo = float(np.min(col))
        hi = float(np.max(col))
        if not hi - lo > 0.0:
            return np.empty(0, dtype=np.float64)  # constant column
        sample = col
        if n > _ASSOC_SAMPLE_ROWS:
            idx = self._rng_.integers(0, n, size=_ASSOC_SAMPLE_ROWS)
            sample = np.concatenate((col[idx], np.array([lo, hi], dtype=col.dtype)))
        qs = np.linspace(0.0, 1.0, int(max_bins) + 1)[1:-1]
        if qs.size == 0:
            return np.empty(0, dtype=np.float64)
        edges = np.unique(np.quantile(sample.astype(np.float64, copy=False), qs))
        return np.asarray(edges, dtype=np.float64)

    def _build_histogram_state(self, Xraw: NDArray, num_feats: NDArray, cat_feats: NDArray) -> None:
        """Precompute bin codes and association weights for the hist path."""
        n = int(Xraw.shape[0])
        n_num = int(num_feats.size)
        max_bins = int(self.max_bins)
        code_dtype = np.uint8 if max_bins <= 256 else np.uint16
        bin_codes = np.empty((n_num, n), dtype=code_dtype)
        edges_list: list[NDArray[np.float64]] = []
        dtype = np.dtype(self.dtype)
        for pos, j in enumerate(num_feats):
            # Bin on the same dtype-cast values used by ``predict`` so a
            # float64 -> float32 boundary never routes differently.
            col = np.asarray(Xraw[:, int(j)]).astype(dtype, copy=False)
            col64 = col.astype(np.float64, copy=False)
            if int(j) in self._binary_set_:
                lo = float(col64.min())
                hi = float(col64.max())
                if hi > lo:
                    # Match the exact path's single 0.5 test for {0, 1}
                    # columns; fall back to the midpoint for other binary
                    # encodings.
                    mid = 0.5 if lo < 0.5 < hi else (lo + hi) / 2.0
                    edges = np.array([mid], dtype=np.float64)
                else:
                    edges = np.empty(0, dtype=np.float64)
            else:
                edges = self._quantile_edges(col64, max_bins)
            # Quantize edges to the working dtype *before* computing bins and
            # before storing thresholds: predict compares dtype-cast values
            # against these thresholds, so an interpolated float64 quantile
            # that is not representable in float32 would route differently.
            edges = np.unique(edges.astype(dtype)).astype(np.float64)
            edges_dtype = edges.astype(dtype)
            edges_list.append(edges)
            bin_codes[pos] = np.searchsorted(edges_dtype, col, side="left").astype(code_dtype)
        self._bin_codes_ = bin_codes
        self._hist_edges_ = edges_list
        self._hist_code_dtype_ = code_dtype
        self.association_ = self._compute_association(Xraw, num_feats, cat_feats)

    def _compute_association(
        self, Xraw: NDArray, num_feats: NDArray, cat_feats: NDArray
    ) -> NDArray[np.float64]:
        """Mean squared pairwise association of every column with the others.

        Numeric-numeric uses Pearson ``r**2``; numeric-categorical uses the
        correlation ratio ``eta**2``; categorical-categorical uses squared
        Cramér's V.  Estimated on at most ``_ASSOC_SAMPLE_ROWS`` rows, further
        capped so the sampled matrix stays near 4M cells on very wide data
        (deterministic for a fixed ``random_state``).
        """
        n, m = Xraw.shape
        n_num = int(num_feats.size)
        assoc = np.zeros((m, m), dtype=np.float64)
        rng = self._resolve_rng()
        # Bound the sample matrix (rows x n_num) to a few million cells.
        max_rows = min(_ASSOC_SAMPLE_ROWS, max(1_000, 4_000_000 // max(n_num, 1)))
        rows = np.arange(n) if n <= max_rows else rng.integers(0, n, size=max_rows)
        Ns = int(rows.size)
        # Center each column in float64 *before* the float32 cast: values
        # like 1e8 + noise would otherwise lose their low bits, and the
        # uncentered sum-of-squares formula would suffer catastrophic
        # cancellation.
        Xn = np.zeros((Ns, n_num), dtype=np.float32)
        for pos, j in enumerate(num_feats):
            vals = np.asarray(Xraw[rows, int(j)], dtype=np.float64)
            Xn[:, pos] = vals - vals.mean()

        if n_num > 1:
            mu = Xn.mean(axis=0, dtype=np.float64)
            sd = Xn.std(axis=0, dtype=np.float64)
            safe = np.where(sd > _VAR_EPS, sd, 1.0).astype(np.float32)
            Z = Xn - mu.astype(np.float32)
            Z /= safe
            Z[:, sd <= _VAR_EPS] = 0.0
            corr = (Z.T @ Z).astype(np.float64) / Ns
            np.clip(corr, -1.0, 1.0, out=corr)
            r2 = corr * corr
            np.fill_diagonal(r2, 0.0)
            assoc[np.ix_(num_feats, num_feats)] = r2
            del Z

        # Column totals are independent of the categorical column; compute
        # them once instead of per (cat, numeric) pair.
        totals_sq = np.empty(n_num, dtype=np.float64)
        for pos in range(n_num):
            x = Xn[:, pos].astype(np.float64, copy=False)
            totals_sq[pos] = float(np.dot(x, x))

        locals_: dict[int, NDArray[np.intp]] = {}
        for j in cat_feats:
            j = int(j)
            meta = self._cat_meta_[j]
            k = int(meta["k"])
            codes = self._cat_codes_[j][rows]
            local = np.searchsorted(meta["codes"], codes)
            locals_[j] = local
            if n_num == 0:
                continue
            counts = np.bincount(local, minlength=k).astype(np.float64)
            nz = counts > 0
            for pos, fj in enumerate(num_feats):
                x = Xn[:, pos].astype(np.float64, copy=False)
                sums = np.bincount(local, weights=x, minlength=k)
                total = float(sums.sum())
                total_sq = float(totals_sq[pos])
                between = float(np.sum(sums[nz] ** 2 / counts[nz]))
                denom = total_sq - total * total / Ns
                eta2 = 0.0 if denom <= _VAR_EPS else (between - total * total / Ns) / denom
                eta2 = float(np.clip(eta2, 0.0, 1.0))
                assoc[j, int(fj)] = eta2
                assoc[int(fj), j] = eta2

        cats = [int(j) for j in cat_feats]
        for a_idx, j1 in enumerate(cats):
            for j2 in cats[a_idx + 1 :]:
                k1 = int(self._cat_meta_[j1]["k"])
                k2 = int(self._cat_meta_[j2]["k"])
                if k1 < 2 or k2 < 2:
                    continue
                combined = locals_[j1].astype(np.int64) * k2 + locals_[j2]
                uniq, observed = np.unique(combined, return_counts=True)
                row_ids = (uniq // k2).astype(np.intp)
                col_ids = (uniq % k2).astype(np.intp)
                row_sum = np.bincount(row_ids, weights=observed, minlength=k1)
                col_sum = np.bincount(col_ids, weights=observed, minlength=k2)
                expected = row_sum[row_ids] * col_sum[col_ids] / Ns
                chi2 = float(np.sum((observed - expected) ** 2 / np.maximum(expected, 1e-12)))
                # Unobserved cells contribute (0 - E)^2 / E = E each; summing
                # only observed cells underestimates chi-square (noticeably
                # for high-cardinality pairs).
                chi2 += float(Ns - expected.sum())
                denom = Ns * min(k1 - 1, k2 - 1)
                v2 = float(np.clip(chi2 / denom, 0.0, 1.0)) if denom > 0 else 0.0
                assoc[j1, j2] = v2
                assoc[j2, j1] = v2

        if m <= 1:
            return np.zeros(m, dtype=np.float64)
        # Sum (not mean) of squared associations: it approximates how many
        # other features a split on column j also explains, matching the
        # scale of the exact sum-over-features heterogeneity gain.
        return assoc.sum(axis=1)

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

    def _accept_gain(self, gain: float) -> bool:
        """Apply ``min_gain`` in the configured unit."""
        if self._min_gain_mode_ == "normalized":
            return gain >= self.min_gain
        if self._min_gain_mode_ == "total":
            return gain * self._gain_scale_ >= self.min_gain
        return gain * math.sqrt(self._gain_scale_) >= self.min_gain

    def _best_split(self, node: dict) -> dict | None:
        """Find the highest-gain single-feature rule inside a node."""
        if self.splitter_ == "hist":
            return self._hist_best_split(node)
        return self._exact_best_split(node)

    # ---------------------------------------------------------- hist search

    def _hist_best_split(self, node: dict) -> dict | None:
        """Scalable split search: association proposal + local verification.

        Per node, each feature needs one pass over the node's rows to build
        per-bin counts/sums; candidates are ranked cheaply by the
        association-gated proxy.  The top candidates are then re-scored with
        the *node-local cross-feature* heterogeneity drop: how much the split
        actually explains the other columns inside this node.  This removes
        the "any median cut removes 64% of its own variance" bias and stops
        within-cluster over-splitting that a purely global proxy would miss.
        Unlike the exact path there are no per-feature sort orders and no
        ``(n_num x n_node)`` buffers.
        """
        idx = node["idx"]
        n = int(idx.size)
        depth = node["depth"]
        if self.max_depth is not None and depth >= self.max_depth:
            return None
        if n < 2 * self.min_leaf_:
            return None
        sw_node = self._sw_[idx] if self._sw_ is not None else None
        candidates: list[dict] = []
        for j in range(self.n_features_in_):
            if j in self._cat_set_:
                cand = self._hist_cat_candidate(j, idx, n, sw_node)
            else:
                cand = self._hist_numeric_candidate(j, idx, n, sw_node)
            if cand is not None:
                candidates.append(cand)
        if not candidates:
            return None
        if self.n_features_in_ <= 1:
            best = None
            for cand in candidates:
                if self._better_gain(cand, best):
                    best = cand
            # A zero-gain "split" is meaningless even when min_gain == 0.
            return best if best is not None and best["gain"] > 0.0 else None
        candidates.sort(key=lambda c: c["gain"], reverse=True)
        best = None
        for cand in candidates[:4]:
            verified = dict(cand)
            verified["proxy_gain"] = cand["gain"]
            verified["gain"] = self._hist_local_cross_gain(cand, idx, n, sw_node)
            if self._better_gain(verified, best):
                best = verified
        return best if best is not None and best["gain"] > 0.0 else None

    def _hist_local_cross_gain(
        self, cand: dict, idx: NDArray, n: int, sw_node: NDArray | None
    ) -> float:
        """Node-local heterogeneity drop contributed by features *other* than
        the split feature.

        A variance-based criterion always credits a median cut with ~64% of
        the split column's variance, even under pure noise.  Requiring that
        the split also reduces the other columns' heterogeneity inside the
        node is what distinguishes a real segment boundary from an arbitrary
        cut.  The result is normalized like the exact mixed gain (divided by
        the number of features or total feature weight).
        """
        j = int(cand["feat"])
        if cand["kind"] == "subset":
            go_left = np.isin(self._cat_codes_[j][idx], tuple(cand["cats"]))
        else:
            go_left = self._bin_codes_[self._num_pos_[j]][idx] <= int(cand["bin"])
        n_left = int(go_left.sum())
        if n_left == 0 or n_left == n:
            return 0.0
        if sw_node is None:
            w_l, w_r, w_tot = float(n_left), float(n - n_left), float(n)
            sw_l = sw_r = None
        else:
            sw_l = sw_node[go_left]
            sw_r = sw_node[~go_left]
            w_l = float(sw_l.sum())
            w_r = float(sw_r.sum())
            w_tot = w_l + w_r
        if w_tot <= _VAR_EPS or w_l <= _VAR_EPS or w_r <= _VAR_EPS:
            return 0.0

        cross = 0.0
        for pos, fj in enumerate(self._num_feats_):
            fj = int(fj)
            if fj == j:
                continue
            vals = self._XcT_num_[pos][idx].astype(np.float64, copy=False)
            vl = vals[go_left]
            vr = vals[~go_left]
            if sw_node is None:
                sl, sr = float(vl.sum()), float(vr.sum())
                ql, qr = float(np.dot(vl, vl)), float(np.dot(vr, vr))
            else:
                sl, sr = float(np.dot(vl, sw_l)), float(np.dot(vr, sw_r))
                ql, qr = float(np.dot(vl * vl, sw_l)), float(np.dot(vr * vr, sw_r))
            s_tot, q_tot = sl + sr, ql + qr
            var_tot = max(q_tot / w_tot - (s_tot / w_tot) ** 2, 0.0)
            var_l = max(ql / w_l - (sl / w_l) ** 2, 0.0)
            var_r = max(qr / w_r - (sr / w_r) ** 2, 0.0)
            delta = var_tot - (w_l / w_tot) * var_l - (w_r / w_tot) * var_r
            ratio = max(delta, 0.0) / self._root_var_[pos]
            cross += ratio * (self._w_all_[fj] if self._weighted_ else 1.0)

        for cj in self._cat_set_:
            cj = int(cj)
            if cj == j:
                continue
            meta = self._cat_meta_[cj]
            local = np.searchsorted(meta["codes"], self._cat_codes_[cj][idx])
            if sw_node is None:
                cnt = np.bincount(local, minlength=meta["k"]).astype(np.float64)
                cnt_l = np.bincount(local[go_left], minlength=meta["k"]).astype(np.float64)
                cnt_r = np.bincount(local[~go_left], minlength=meta["k"]).astype(np.float64)
            else:
                cnt = np.bincount(local, weights=sw_node, minlength=meta["k"])
                cnt_l = np.bincount(local[go_left], weights=sw_l, minlength=meta["k"])
                cnt_r = np.bincount(local[~go_left], weights=sw_r, minlength=meta["k"])

            def _gini(c: NDArray, total: float) -> float:
                c = c[c > 0]
                if total <= _VAR_EPS:
                    return 0.0
                p = c / total
                return float(1.0 - np.sum(p * p))

            g_tot = _gini(cnt, w_tot)
            g_l = _gini(cnt_l, w_l)
            g_r = _gini(cnt_r, w_r)
            delta = g_tot - (w_l / w_tot) * g_l - (w_r / w_tot) * g_r
            ratio = max(delta, 0.0) / self._root_gini_[cj]
            cross += ratio * (self._w_all_[cj] if self._weighted_ else 1.0)

        denom = self._w_sum_ if self._weighted_ else float(self.n_features_in_)
        gain = float(cross) / denom
        # Under a random split, every *other* feature loses ~1/n_eff of its
        # variance to between-group noise.  Summed over features this is a
        # predictable floor, not structure; subtract it so deeper nodes
        # (where 1/n_node grows) do not start splitting noise.
        if sw_node is None:
            n_eff = float(n)
        else:
            n_eff = float(w_tot * w_tot / max(float(np.dot(sw_node, sw_node)), _VAR_EPS))
        return max(gain - 1.0 / max(n_eff, 1.0), 0.0)

    def _hist_assoc(self, j: int) -> float:
        """Sum of squared associations of feature ``j`` with the others."""
        if self.association_.size != self.n_features_in_:
            return 0.0
        return float(self.association_[j])

    def _hist_score(self, j: int, delta_ratio: NDArray) -> NDArray:
        """Association-gated split score.

        The split feature's own variance reduction is *not* evidence of
        structure: cutting any continuous column at its median removes about
        64% of its variance even under pure noise.  What distinguishes a real
        segment boundary is that the split also separates the other columns,
        so the score is ``delta_ratio * mean(squared association with the
        other columns)``.  For a single-column problem there is nothing to
        corroborate, and the plain variance reduction is used.
        """
        m = self.n_features_in_
        if m <= 1:
            return delta_ratio
        return delta_ratio * (self._hist_assoc(j) / float(m - 1))

    def _hist_numeric_candidate(
        self, j: int, idx: NDArray, n: int, sw_node: NDArray | None
    ) -> dict | None:
        pos = self._num_pos_[j]
        edges = self._hist_edges_[pos]
        q = int(edges.size) + 1
        if q < 2:
            return None
        bins = self._bin_codes_[pos][idx]
        vals = self._XcT_num_[pos][idx].astype(np.float64, copy=False)
        counts = np.bincount(bins, minlength=q).astype(np.float64)
        if not counts.any():
            return None
        if sw_node is None:
            weights = counts
            sums = np.bincount(bins, weights=vals, minlength=q)
            sqs = np.bincount(bins, weights=vals * vals, minlength=q)
        else:
            weights = np.bincount(bins, weights=sw_node, minlength=q)
            sums = np.bincount(bins, weights=sw_node * vals, minlength=q)
            sqs = np.bincount(bins, weights=sw_node * (vals * vals), minlength=q)
        cw = np.cumsum(weights)
        cs = np.cumsum(sums)
        cq = np.cumsum(sqs)
        cc = np.cumsum(counts)
        total_w = float(cw[-1])
        if total_w <= _VAR_EPS:
            return None
        lw, ls, lq, lc = cw[:-1], cs[:-1], cq[:-1], cc[:-1]
        rw = total_w - lw
        rs = float(cs[-1]) - ls
        rq = float(cq[-1]) - lq
        valid = (
            (lc >= self.min_leaf_) & (n - lc >= self.min_leaf_) & (lw > _VAR_EPS) & (rw > _VAR_EPS)
        )
        if not valid.any():
            return None
        var_node = max(float(cq[-1]) / total_w - (float(cs[-1]) / total_w) ** 2, 0.0)
        lw_safe = np.maximum(lw, _VAR_EPS)
        rw_safe = np.maximum(rw, _VAR_EPS)
        var_l = np.maximum(lq / lw_safe - (ls / lw_safe) ** 2, 0.0)
        var_r = np.maximum(rq / rw_safe - (rs / rw_safe) ** 2, 0.0)
        delta = (var_node - (lw / total_w) * var_l - (rw / total_w) * var_r) / (
            self._root_var_[pos]
        )
        delta = np.maximum(delta, 0.0)
        score = self._hist_score(j, delta)
        score[~valid] = -np.inf
        local = self._argmax_with_ties(score)
        if not np.isfinite(score[local]):
            return None
        return {
            "gain": float(score[local]),
            "feat": j,
            "thr": float(edges[local]),
            "kind": "threshold",
            "cats": None,
            "bin": int(local),
        }

    def _hist_cat_candidate(
        self, j: int, idx: NDArray, n: int, sw_node: NDArray | None
    ) -> dict | None:
        meta = self._cat_meta_[j]
        k = int(meta["k"])
        if k < 2:
            return None
        codes = self._cat_codes_[j][idx]
        local_codes = np.searchsorted(meta["codes"], codes)
        raw_counts = np.bincount(local_codes, minlength=k).astype(np.float64)
        present = raw_counts > 0
        if int(present.sum()) < 2:
            return None
        if sw_node is None:
            counts = raw_counts
        else:
            counts = np.bincount(local_codes, weights=sw_node, minlength=k)
        total_w = float(counts.sum())
        if total_w <= _VAR_EPS:
            return None
        p = counts / total_w
        gini_parent = float(1.0 - np.sum(p * p))
        if gini_parent <= _GINI_EPS:
            return None

        present_codes = meta["codes"][present]
        present_raw = raw_counts[present]
        present_w = counts[present]
        kk = int(present_codes.size)
        strat = self._cat_splitter_.get(j, "one_vs_rest")
        if strat == "one_vs_rest":
            cand_left = None
            n_l_raw = present_raw
            n_l_w = present_w
        else:
            half = 1 << (kk - 1)
            if half > _MAX_EXACT_CANDIDATES:
                raise ValueError(
                    f"Exact categorical split on column {self.feature_names_[j]!r} "
                    f"needs {half:,} candidate subsets at this node. Use "
                    "categorical_splitter='one_vs_rest' or reduce categories."
                )
            bits = ((np.arange(half)[:, None] >> np.arange(kk - 1)) & 1).astype(np.float64)
            cand_left = np.concatenate([np.zeros((half, 1)), bits], axis=1)
            del bits
            n_l_raw = cand_left @ present_raw
            n_l_w = cand_left @ present_w
        valid = (n_l_raw >= self.min_leaf_) & (n - n_l_raw >= self.min_leaf_)
        if not valid.any():
            return None
        n_l_safe = np.maximum(n_l_w, _VAR_EPS)
        n_r_w = total_w - n_l_w
        n_r_safe = np.maximum(n_r_w, _VAR_EPS)
        if strat == "one_vs_rest":
            # Candidate c puts category c alone on the left: its Gini is 0.
            # Right-side squared proportions are computed analytically as
            # (sum(w^2) - w_c^2) / (T - w_c)^2, avoiding a k x k matrix.
            g_l = np.zeros(kk, dtype=np.float64)
            sum_w2 = float(np.dot(present_w, present_w))
            right_total = total_w - present_w
            right_safe = np.maximum(right_total, _VAR_EPS)
            g_r = 1.0 - (sum_w2 - present_w * present_w) / (right_safe * right_safe)
            g_r = np.clip(g_r, 0.0, 1.0)
        else:
            left_cat = cand_left * present_w[None, :]
            right_cat = present_w[None, :] - left_cat
            p_l = left_cat / n_l_safe[:, None]
            p_r = right_cat / n_r_safe[:, None]
            g_l = 1.0 - np.sum(p_l * p_l, axis=1)
            g_r = 1.0 - np.sum(p_r * p_r, axis=1)
        delta = (
            gini_parent - (n_l_w / total_w) * g_l - (n_r_w / total_w) * g_r
        ) / self._root_gini_[j]
        delta = np.maximum(delta, 0.0)
        score = self._hist_score(j, delta)
        score[~valid] = -np.inf
        local = self._argmax_with_ties(score)
        if not np.isfinite(score[local]):
            return None
        if strat == "one_vs_rest":
            cats = frozenset((int(present_codes[local]),))
        else:
            cats = frozenset(int(present_codes[c]) for c in np.flatnonzero(cand_left[local]))
        return {
            "gain": float(score[local]),
            "feat": j,
            "thr": float("nan"),
            "kind": "subset",
            "cats": cats,
            "bin": None,
        }

    def _exact_best_split(self, node: dict) -> dict | None:
        """Exact mixed-heterogeneity search with chunked prefix sums.

        Numeric features are processed in blocks sized so that the four
        working buffers stay within ``_EXACT_BUFFER_BYTES``.  Each feature's
        sort order is still evaluated exactly; only the *storage* is chunked,
        so partitions and thresholds match a single-buffer implementation
        (gains can differ at floating-point round-off because the reduction
        over features is accumulated block by block).
        """
        idx = node["idx"]
        n = int(idx.size)
        depth = node["depth"]
        if self.max_depth is not None and depth >= self.max_depth:
            return None
        if n < 2 * self.min_leaf_:
            return None
        m = self.n_features_in_
        sw_node = self._sw_[idx] if self._sw_ is not None else None
        hs = self._exact_node_heterogeneity(idx, sw_node)
        best = None
        for j in range(m):
            rows = idx[node["orders"][j]]  # global row ids sorted by feature j
            is_cat = j in self._cat_set_
            if is_cat:
                # Exact int64 codes: never route block detection through
                # float64, which would merge distinct codes above 2**53.
                scol = self._cat_codes_[j][rows]
                if scol[0] == scol[-1]:
                    continue  # constant within this node
            else:
                scol = self._XcT_num_[self._num_pos_[j]][rows]
                if scol[-1] - scol[0] < _VAR_EPS:
                    continue  # feature constant within this node

            sw_sorted = self._sw_[rows] if self._sw_ is not None else None
            if is_cat:
                cand_best = self._best_cat_split(j, rows, scol, hs, n, sw_node, sw_sorted)
            else:
                # Candidate thresholds live in raw space and cut positions are
                # located on raw values too: evaluating and committing must
                # agree on which rows go left.
                rcol = self._Xraw[:, j][rows]
                cand_best = self._best_numeric_split(j, rows, rcol, hs, n, sw_node, sw_sorted)
            if cand_best is not None and self._better_gain(cand_best, best):
                best = cand_best
        return best

    def _gather_num_block(self, block: slice, rows: NDArray) -> NDArray[np.float64]:
        """Gather ``self._XcT_num_[block][:, rows]`` into a float64 buffer."""
        width = int(block.stop - block.start)
        n_node = int(rows.size)
        mat = self._bufs_["mat"][:width, :n_node]
        if self._stage64:
            np.take(self._XcT_num_[block], rows, axis=1, out=mat)
        else:
            stage = self._bufs_["stage"][:width, :n_node]
            np.take(self._XcT_num_[block], rows, axis=1, out=stage)
            np.copyto(mat, stage)
        return mat

    def _exact_node_heterogeneity(self, idx: NDArray, sw_node: NDArray | None) -> float:
        """Exact mixed heterogeneity of a node (chunked over numeric blocks)."""
        n_num = self._n_num_
        if n_num:
            if sw_node is None:
                total = 0.0
                for block in self._num_blocks_:
                    mat = self._gather_num_block(block, idx)
                    ratio = np.var(mat, axis=1) / self._root_var_[block]
                    if self._weighted_:
                        total += float(self._w_num_[block] @ ratio)
                    else:
                        # Sum then divide once: matches the single-buffer
                        # reduction order instead of averaging per block.
                        total += float(ratio.sum())
                num_term = total if self._weighted_ else total / n_num
            else:
                num_term = 0.0
                for block in self._num_blocks_:
                    mat = self._gather_num_block(block, idx).astype(np.float64, copy=False)
                    mean_w = np.average(mat, weights=sw_node, axis=1)
                    var_w = np.average((mat - mean_w[:, None]) ** 2, weights=sw_node, axis=1)
                    ratio = np.maximum(var_w, 0.0) / self._root_var_[block]
                    if self._weighted_:
                        num_term += float(np.dot(self._w_num_[block], ratio))
                    else:
                        num_term += float(ratio.sum())
                if not self._weighted_:
                    num_term /= n_num
        else:
            num_term = 0.0
        cat_term = 0.0
        for cj in self._cat_set_:
            ratio = (
                self._gini_codes(self._cat_codes_[cj][idx], self._cat_meta_[cj], sw_node)
                / self._root_gini_[cj]
            )
            cat_term += ratio * self._w_all_[cj] if self._weighted_ else ratio
        return self._combined_heterogeneity(num_term, cat_term)

    def _better_gain(self, candidate: dict, best: dict | None) -> bool:
        """Strict improvement, with optional random tie-breaking."""
        if best is None:
            return True
        diff = float(candidate["gain"]) - float(best["gain"])
        if diff > 1e-12:
            return True
        if diff < -1e-12:
            return False
        if self._random_ties_:
            return bool(self._rng_.random() < 0.5)
        return False

    @staticmethod
    def _prefix_counts_at(
        local: NDArray[np.intp],
        positions: NDArray,
        k_cat: int,
        weights: NDArray | None = None,
    ) -> NDArray[np.float64]:
        """Category counts in ``local[:pos]`` for each sorted position.

        Uses one pass over the rows instead of allocating an ``(n, k)``
        one-hot prefix matrix, which keeps exact-path memory bounded even
        when a node has many categories.
        """
        out = np.zeros((int(positions.size), int(k_cat)), dtype=np.float64)
        running = np.zeros(int(k_cat), dtype=np.float64)
        start = 0
        for i, raw_pos in enumerate(positions):
            pos = int(raw_pos)
            if pos > start:
                segment = local[start:pos]
                if weights is None:
                    running += np.bincount(segment, minlength=int(k_cat))
                else:
                    running += np.bincount(
                        segment, weights=weights[start:pos], minlength=int(k_cat)
                    )
            out[i] = running
            start = pos
        return out

    def _argmax_with_ties(self, values: NDArray) -> int:
        """argmax; with ``random_state`` set, ties are broken at random."""
        if not self._random_ties_:
            return int(np.argmax(values))
        best = float(np.max(values))
        ties = np.flatnonzero(values >= best - 1e-12)
        if ties.size <= 1:
            return int(ties[0]) if ties.size else 0
        return int(self._rng_.choice(ties))

    def _best_numeric_split(
        self,
        j: int,
        rows: NDArray,
        rcol: NDArray,
        hs: float,
        n: int,
        sw_node: NDArray | None,
        sw_sorted: NDArray | None,
    ) -> dict | None:
        """Best threshold split; ``rcol`` is the node's raw column, ascending."""
        if j in self._binary_set_:
            cand = np.array([0.5])
        else:
            cand = np.unique(_sorted_linear_quantiles(rcol, self._quantile_grid_))

        ks_all = np.searchsorted(rcol, cand, side="right")
        uniq_k = np.unique(ks_all)
        ok = (uniq_k >= self.min_leaf_) & (n - uniq_k >= self.min_leaf_)
        if not ok.any():
            return None
        k = uniq_k[ok]
        # Commit a threshold that is *exactly representable* in the working
        # dtype: the largest value on the left side (or 0.5 for binary
        # columns).  Using the interpolated quantile instead can round onto
        # the right side under float32 (adjacent values such as 1e6 and
        # 1e6 + 0.0625), making evaluation and commit disagree.
        t = rcol[k - 1].astype(np.float64)
        if j in self._binary_set_:
            t = np.full(k.shape, 0.5, dtype=np.float64)
        kf = k.astype(np.float64)
        nf = float(n)

        # Weight sums: use cumulative weights when sample_weight is active.
        if sw_sorted is not None:
            cw = np.empty(n + 1, dtype=np.float64)
            cw[0] = 0.0
            np.cumsum(sw_sorted, out=cw[1:])
            kw = cw[k]
            nw = float(cw[n])
            kw_safe = np.maximum(kw, _VAR_EPS)
            nwf = np.maximum(nw - kw, _VAR_EPS)
        else:
            kw = kf
            nw = nf
            kw_safe = kf
            nwf = np.maximum(nf - kf, _VAR_EPS)

        # Accumulate per-block contributions over numeric features.  Blocks
        # keep the four working buffers bounded regardless of n_num.
        n_num = self._n_num_
        hL_num = np.zeros(k.size, dtype=np.float64)
        hR_num = np.zeros(k.size, dtype=np.float64)
        if n_num:
            n_node = int(rows.size)
            for block in self._num_blocks_:
                width = int(block.stop - block.start)
                mat = self._gather_num_block(block, rows)
                cs = self._bufs_["cs"][:width, : n_node + 1]
                cs2 = self._bufs_["cs2"][:width, : n_node + 1]
                sq = self._bufs_["sq"][:width, :n_node]
                cs[:, 0] = 0.0
                cs2[:, 0] = 0.0
                if sw_sorted is None:
                    np.cumsum(mat, axis=1, out=cs[:, 1:])
                    np.multiply(mat, mat, out=sq)
                else:
                    wmat = self._bufs_["wmat"][:width, :n_node]
                    np.multiply(sw_sorted[None, :], mat, out=wmat)
                    np.cumsum(wmat, axis=1, out=cs[:, 1:])
                    np.multiply(wmat, mat, out=sq)
                np.cumsum(sq, axis=1, out=cs2[:, 1:])
                tot, tot2 = cs[:, n], cs2[:, n]
                hL_b, hR_b = self._children_numeric(
                    cs,
                    cs2,
                    tot,
                    tot2,
                    self._root_var_[block],
                    k,
                    kw_safe,
                    nwf,
                    nw,
                    self._w_num_[block] if self._weighted_ else None,
                )
                # Blocks return sums; the unweighted normalization by
                # n_num happens once after the loop (matching the original
                # single-buffer reduction order).
                hL_num += hL_b
                hR_num += hR_b
            if not self._weighted_:
                hL_num /= n_num
                hR_num /= n_num

        # Children Gini of every categorical column at each candidate cut.
        sum_gL = np.zeros(k.size)
        sum_gR = np.zeros(k.size)
        for cj in self._cat_set_:
            meta = self._cat_meta_[cj]
            local = np.searchsorted(meta["codes"], self._cat_codes_[cj][rows])
            NLc = self._prefix_counts_at(local, k, meta["k"], sw_sorted)
            if sw_sorted is None:
                totals = np.bincount(local, minlength=meta["k"]).astype(np.float64)
            else:
                totals = np.bincount(local, weights=sw_sorted, minlength=meta["k"])
            NRc = totals[None, :] - NLc
            gL = 1.0 - ((NLc / kw_safe[:, None]) ** 2).sum(axis=1)
            gR = 1.0 - ((NRc / nwf[:, None]) ** 2).sum(axis=1)
            gr = self._root_gini_[cj]
            wc = self._w_all_[cj] if self._weighted_ else 1.0
            sum_gL += wc * gL / gr
            sum_gR += wc * gR / gr
        hL = self._combined_heterogeneity(hL_num, sum_gL)
        hR = self._combined_heterogeneity(hR_num, sum_gR)
        gains = hs - (kw / nw) * hL - ((nw - kw) / nw) * hR
        local = self._argmax_with_ties(gains)
        return {
            "gain": float(gains[local]),
            "feat": j,
            "thr": float(t[local]),
            "kind": "threshold",
            "cats": None,
            "bin": None,
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
        w_block: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Mean normalized-variance of left/right children over numeric features.

        ``w_block`` carries the feature weights of the current numeric chunk;
        using the full ``_w_num_`` here would have the wrong length once the
        exact path is chunked.
        """
        if not self._n_num_:
            z = np.zeros(k.size)
            return z, z
        csL, cs2L = cs[:, k], cs2[:, k]
        mL = csL / kw_safe
        vL = np.maximum(cs2L / kw_safe - mL**2, 0.0)
        mR = (tot[:, None] - csL) / nwf
        vR = np.maximum((tot2[:, None] - cs2L) / nwf - mR**2, 0.0)
        if self._weighted_:
            wb = w_block if w_block is not None else self._w_num_
            # Features are rows here, so weights multiply from the left.
            return wb @ (vL / rv[:, None]), wb @ (vR / rv[:, None])
        # Return block sums; callers divide by n_num once after all blocks.
        return (vL / rv[:, None]).sum(axis=0), (vR / rv[:, None]).sum(axis=0)

    def _best_cat_split(
        self,
        j: int,
        rows: NDArray,
        scol: NDArray,
        hs: float,
        n: int,
        sw_node: NDArray | None,
        sw_sorted: NDArray | None,
    ) -> dict | None:
        """Best categorical split of column ``j``.

        Categories form contiguous blocks in ``scol`` (ascending code order).
        Both splitters share the same scoring machinery: numeric children via
        chunked prefix-sum block differences, categorical children via
        contingency tables against the blocks.  They differ only in the
        candidate matrix:

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
        ovr = self._cat_splitter_.get(j, "one_vs_rest") == "one_vs_rest"

        # Weight sums per block (equals sizes_f when sample_weight is None).
        if sw_sorted is not None:
            cw = np.empty(n + 1, dtype=np.float64)
            cw[0] = 0.0
            np.cumsum(sw_sorted, out=cw[1:])
            block_w = np.diff(cw[bounds])
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

        # Numeric children impurity: accumulate block sums over numeric
        # feature chunks, then score all candidates from the small
        # (n_num x n_blocks) tables.
        n_num = self._n_num_
        if n_num:
            bsums = np.zeros((n_num, k_blocks), dtype=np.float64)
            bsqs = np.zeros((n_num, k_blocks), dtype=np.float64)
            tot = np.zeros(n_num, dtype=np.float64)
            tot2 = np.zeros(n_num, dtype=np.float64)
            n_node = int(rows.size)
            for block in self._num_blocks_:
                width = int(block.stop - block.start)
                mat = self._gather_num_block(block, rows)
                cs = self._bufs_["cs"][:width, : n_node + 1]
                cs2 = self._bufs_["cs2"][:width, : n_node + 1]
                sq = self._bufs_["sq"][:width, :n_node]
                cs[:, 0] = 0.0
                cs2[:, 0] = 0.0
                if sw_sorted is None:
                    np.cumsum(mat, axis=1, out=cs[:, 1:])
                    np.multiply(mat, mat, out=sq)
                else:
                    wmat = self._bufs_["wmat"][:width, :n_node]
                    np.multiply(sw_sorted[None, :], mat, out=wmat)
                    np.cumsum(wmat, axis=1, out=cs[:, 1:])
                    np.multiply(wmat, mat, out=sq)
                np.cumsum(sq, axis=1, out=cs2[:, 1:])
                bsums[block, :] = cs[:, bounds[1:]] - cs[:, bounds[:-1]]
                bsqs[block, :] = cs2[:, bounds[1:]] - cs2[:, bounds[:-1]]
                tot[block] = cs[:, n]
                tot2[block] = cs2[:, n]
            if ovr:  # direct slices: candidate c selects block c's column
                SL, QL = bsums.T, bsqs.T
            else:
                SL, QL = cand_left @ bsums.T, cand_left @ bsqs.T
            vL = np.maximum(QL / nL_safe - (SL / nL_safe) ** 2, 0.0)
            hL_num = self._num_reduce(vL / self._root_var_, axis=-1)
            vR = np.maximum(
                (tot2[None, :] - QL) / nR_safe - ((tot[None, :] - SL) / nR_safe) ** 2,
                0.0,
            )
            hR_num = self._num_reduce(vR / self._root_var_, axis=-1)
        else:
            hL_num = hR_num = np.zeros(nL.size)

        # Categorical children impurity.  For one-vs-rest the split column
        # and any other column use O(n_node) sparse contingency arithmetic:
        # a dense (k_blocks x k_cat) table is O(k^2) and reaches GBs for the
        # high-cardinality columns this strategy exists to support.
        block_ids = np.repeat(np.arange(k_blocks), sizes_i)
        sum_gL = np.zeros(nL.size)
        sum_gR = np.zeros(nL.size)
        block_w2 = block_w * block_w
        total_w2 = float(block_w2.sum())
        for cj in self._cat_set_:
            cj = int(cj)
            meta = self._cat_meta_[cj]
            gr = self._root_gini_[cj]
            wc = self._w_all_[cj] if self._weighted_ else 1.0
            if cj == j:
                if ovr:
                    # Candidate c = category c alone on the left (Gini 0).
                    g_l = np.zeros(k_blocks)
                    right_tot = np.maximum(nw_total - block_w, _VAR_EPS)
                    right_sq = total_w2 - block_w2
                    g_r = 1.0 - right_sq / (right_tot * right_tot)
                else:
                    left_cat = cand_left * block_w[None, :]
                    right_cat = block_w[None, :] - left_cat
                    g_l = 1.0 - ((left_cat / nL_safe) ** 2).sum(axis=1)
                    g_r = 1.0 - ((right_cat / nR_safe) ** 2).sum(axis=1)
            elif ovr and k_blocks * int(meta["k"]) > _DENSE_CAT_CELLS:
                # Sparse contingency: only observed (block, category) cells.
                local = np.searchsorted(meta["codes"], self._cat_codes_[cj][rows])
                combined = block_ids.astype(np.int64) * int(meta["k"]) + local
                uniq, inv = np.unique(combined, return_inverse=True)
                if sw_sorted is None:
                    counts = np.bincount(inv).astype(np.float64)
                else:
                    counts = np.bincount(inv, weights=sw_sorted)
                rb = (uniq // int(meta["k"])).astype(np.intp)
                cc = (uniq % int(meta["k"])).astype(np.intp)
                counts_sq = counts * counts
                left_sq = np.bincount(rb, weights=counts_sq, minlength=k_blocks)
                left_tot = np.bincount(rb, weights=counts, minlength=k_blocks)
                col_tot = np.bincount(cc, weights=counts, minlength=int(meta["k"]))
                col_sq = np.bincount(cc, weights=counts_sq, minlength=int(meta["k"]))
                cross = np.bincount(rb, weights=counts * col_tot[cc], minlength=k_blocks)
                right_tot = np.maximum(nw_total - left_tot, _VAR_EPS)
                right_sq = float(col_sq.sum()) - 2.0 * cross + left_sq
                g_l = 1.0 - left_sq / np.maximum(left_tot, _VAR_EPS) ** 2
                g_r = 1.0 - right_sq / (right_tot * right_tot)
            else:
                cnt = self._cat_counts_by_block(
                    self._cat_codes_[cj][rows], meta, block_ids, k_blocks, sw_sorted
                )
                NL = cnt if ovr else cand_left @ cnt
                NR = cnt.sum(axis=0)[None, :] - NL
                g_l = 1.0 - ((NL / nL_safe) ** 2).sum(axis=1)
                g_r = 1.0 - ((NR / nR_safe) ** 2).sum(axis=1)
            np.clip(g_l, 0.0, 1.0, out=g_l)
            np.clip(g_r, 0.0, 1.0, out=g_r)
            sum_gL += wc * g_l / gr
            sum_gR += wc * g_r / gr

        hL = self._combined_heterogeneity(hL_num, sum_gL)
        hR = self._combined_heterogeneity(hR_num, sum_gR)
        gains = hs - (nL_w / nw_total) * hL - ((nw_total - nL_w) / nw_total) * hR
        gains[~valid] = -np.inf
        local = self._argmax_with_ties(gains)
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
            "bin": None,
        }

    def _finalize_leaf(self, node: dict, leaves: list, chains: list, labels: NDArray) -> None:
        node["leaf"] = True
        node["leaf_id"] = len(leaves)
        node["n"] = int(node["idx"].size)
        idx = node["idx"]
        sw_node = self._sw_[idx] if self._sw_ is not None else None
        if self._n_num_:
            # Row-by-row: ``self._XcT_num_[:, idx]`` would copy the whole
            # (n_num, n_leaf) block as float64, which can reach hundreds of
            # MiB for a large leaf.
            ratios = np.empty(self._n_num_, dtype=np.float64)
            if sw_node is None:
                for pos in range(self._n_num_):
                    ratios[pos] = np.var(self._XcT_num_[pos][idx], dtype=np.float64)
            else:
                for pos in range(self._n_num_):
                    vals = self._XcT_num_[pos][idx].astype(np.float64, copy=False)
                    mean_w = np.average(vals, weights=sw_node)
                    ratios[pos] = np.average((vals - mean_w) ** 2, weights=sw_node)
            num_ratio = float(self._num_reduce(np.maximum(ratios, 0.0) / self._root_var_, axis=0))
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
