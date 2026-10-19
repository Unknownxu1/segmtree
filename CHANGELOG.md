# Changelog

## 0.6.0 — 2026-10-19

Scalability and robustness release driven by 16 production issues reproduced
on an 829k x 300 customer table.

### New: scalable split search (`splitter="hist"`, the default for all data)

- **The default split-search path is now the histogram path for every data
  size.**  The original exact algorithm remains available only as an explicit
  `splitter="exact"` legacy/reference mode; it is not used automatically and
  is not recommended for production (its per-node cost is
  `O(n_features² · n_node)` with several full-size buffers).
- The histogram path (`splitter="hist"`, `max_bins=32`) bins each continuous
  feature into quantile bins once, then per node:
  1. ranks candidates with an association-gated proxy
     (`delta_variance * mean squared association`, where association is
     Pearson r² / correlation ratio η² / Cramér's V² estimated on a
     deterministic random sample of at most 50k rows, adaptively capped to
     ~4M cells on very wide data);
  2. re-scores the top 4 candidates with the actual **node-local
     cross-feature** heterogeneity drop, subtracting the expected `1/n_eff`
     random-split noise floor.
  This removes the median-cut self-variance bias (any continuous column loses
  ~64% of its variance under a median cut, even for pure noise), and prevents
  within-cluster over-splitting.  Multi-column pure-noise data stops at a
  single leaf when there are enough rows; a single-column problem has no
  cross-feature corroboration and follows the exact-path self-variance
  criterion.
- Histogram is not just a large-data fallback: on 1,200×6 it runs in ~0.011 s
  with 5 leaves versus ~0.089 s / 69 leaves for exact; on 20,000×50 it is
  ~0.39 s / 7 leaves versus ~8.2 s / 65 leaves.  Exact's variance criterion
  over-splits noise on wide data, so the default is now histogram for every
  size.
- The histogram path never allocates the old `(n_numeric x n_samples)`
  float64 buffers, never keeps per-feature `argsort` orders, and releases its
  bin codes after fit.  Working memory is `n x F` values (`dtype`) plus one
  byte per bin code for `max_bins <= 256` (two bytes above), plus an
  `O(F²)` association matrix during fit (0.7 MB at 300 columns).  `predict`
  no longer casts the whole matrix either.
- Measured end-to-end (peak RSS sampled during fit and includes the caller's
  input matrix; float32 for the largest case):
  | shape | path | time | peak RSS | leaves |
  |---|---|---|---|---|
  | 200k x 30 | hist | 2.5 s | 263 MB | 4 |
  | 60k x 300 (3 informative) | hist | 5.5 s | 489 MB | 4 |
  | 300k x 301 (3 informative) | hist | 29 s | 1,015 MB | 4 |
  | 829k x 301 (3 informative, f32) | hist | 84 s | 2,439 MB (input 1,152 MB) | 4 |
  | 60k x 300 pure noise | hist | 3.3 s | 497 MB | 1 |
  | exact reference on 1000 x 202 | exact | — | — | root (issue #6 repro) |
- The exact path itself is now **chunked over numeric features**
  (`_EXACT_BUFFER_BYTES`, 192 MiB cap) instead of allocating four full
  `(n_num x n)` float64 buffers.  Partitions and thresholds are identical to
  the single-buffer implementation; gains may differ at floating-point
  round-off (regression-tested by forcing a 2-row chunk and by a randomized
  chunked-vs-unchunked fuzz).
- Categorical prefix counts on the exact path are computed only at candidate
  positions, avoiding `(n_node x k)` one-hot prefix allocations.

### Fixed: `min_gain` no longer drifts with the number of columns (#6)

- New `min_gain_mode={"auto", "sqrt", "total", "normalized"}`.
  `"auto"` (default) resolves to `"sqrt"` on the default histogram path and
  to `"normalized"` only for the explicit `splitter="exact"` mode, preserving
  its original threshold semantics.
  `"total"`/`"sqrt"` decouple the threshold from `n_features`, so adding weak
  columns no longer silently stops the tree at the root.
- Regression test reproduces the report exactly: two perfectly separating
  columns + 200 noise columns with legacy `min_gain=0.01` → 1 leaf; the same
  data on the default histogram path finds the signal and ignores all noise.

### Fixed: categorical handling (#8, #9, #10, #16)

- **`categorical_splitter` default is now `"auto"`**: columns with more than
  `max_categories` codes automatically fall back to `"one_vs_rest"` with a
  warning instead of failing the whole fit.  Explicit `"exact"` still raises.
- `categorical_splitter` accepts a **dict** mapping feature names or indices
  to strategies, so e.g. a 6-level column can use exact enumeration while a
  14-level column uses one-vs-rest in the same fit.
- Exact categorical candidate matrices are **pre-checked at fit start**
  (cardinality and estimated MiB across all numeric features), before any
  tree is built.  Errors now name the column, cardinality, candidate count,
  estimated memory, current `n_numeric`, and the suggested parameters.
- Node-level exact enumeration still refuses > 1M candidates with context.
- **Int64 category codes above `2**53` are preserved exactly**: integer-typed
  columns are cast directly to int64, and categorical block detection /
  prediction use the int64 code arrays instead of a float64 copy.  Adjacent
  codes such as `2**60` and `2**60 + 1` no longer collapse.
- The unused `(n_cat x n)` float64 code copy (`_XcT_cat_`) and the dead
  `_require_fitted()` wrapper were removed.

### New: observability and reproducibility (#12, #15)

- `random_state` seeds association subsampling and random tie-breaking
  (equal gains / equal candidate thresholds).  `None` keeps deterministic
  first-seen tie-breaks, so default fits stay reproducible.
- `verbose=1` prints one line per executed split (depth, node size, feature,
  gain, child sizes, elapsed) and a final summary.
- `bootstrap_stability(..., subsample=None|float|int)`: draw each replicate
  with replacement from a bounded size (fraction or absolute row count)
  instead of always retraining on the full sample (#7).
- `h_root_` now reports the **actual** root heterogeneity (1.0 when no
  column is constant, lower with constant columns, 0 for all-constant input);
  `h_root_normalized_` is the 1.0 normalization baseline.  Raw root
  statistics are exposed as `root_variances_` and `root_ginis_` (#11).
- New fitted attributes: `splitter_`, `max_bins_`, `association_`.

### Fixed: findings from the adversarial audit

- **`one_vs_rest` categorical scoring on the histogram path**: every
  candidate used to receive the same Gini value (the category proportions
  were summed over the candidate axis instead of evaluated per candidate),
  so the tree always picked the lowest code.  Right-side impurity is now
  computed analytically per candidate.
- **float32 threshold consistency**: exact thresholds are committed as the
  largest left-side *observed* value (representable in `dtype`), while
  histogram edges/bins are quantized to `dtype` before use.  Rows adjacent at
  float32 spacing (e.g. `1e6` vs `1e6 + 0.0625`) no longer route differently
  between `labels_` and `predict`; `replay_labels` casts numeric columns to
  the training dtype like `predict` does.
- **Association estimation** now centers columns in float64 before the
  float32 cast, so large-offset columns (amounts, timestamps, IDs) keep
  meaningful r²/η²; the categorical χ² includes the contribution of
  unobserved contingency cells (correct Cramér's V).
- **Periodic columns**: bin edges use a random (endpoint-inclusive) sample
  instead of a fixed stride, which could alias with alternating data.
- **Binary columns on the histogram path** now get their single midpoint
  (0.5 for `{0, 1}`) candidate even when the quantile grid collapses onto
  the maximum value.
- **High-cardinality `one_vs_rest` no longer builds `(k_blocks x k_cat)`
  dense tables** on the exact path.  The split column uses analytic Gini;
  other columns use sparse observed-cell contingency arithmetic above a
  4M-cell threshold.  A 12,000-category fit went from ~4.4 GB peak / 2.75 s
  to ~0 MB / 0.01 s.
- `score()`, `rule_mask`, replay and rule text now preserve int64 category
  codes above `2**53` (score no longer returns 1.0 for a single-leaf tree,
  replay matches `predict`, and rule text shows exact integers).
- `bootstrap_stability` carries resolved feature names into replicas, so
  name-keyed `categorical_splitter` / `feature_weights` dicts work with
  DataFrame input.
- The categorical memory pre-check budgets only the selected path with
  realistic temporary-matrix factors instead of summing both paths.
- **Performance cleanups found by the audit**: `binary_features="auto"` uses
  vectorized min/max plus a linear `{0,1}` check (0.17 s at 829k x 300,
  previously ~25 s of `np.unique` per column); `as_2d_array` validates
  finiteness in chunks (2 MB instead of an `n x F` mask); `predict` encodes
  only categorical columns used by a cut (unless `handle_unknown="error"`);
  fit frees `_cat_codes_`, `_cat_meta_` and `_sw_`; association reuses
  per-column totals.

### Documentation

- Parameter docs state the histogram memory formula and the exact path's
  `O(n_features² · n_node)` cost.  README (both languages) gained a
  large-data section with the hist-default / exact-opt-in guidance and measured
  numbers.

### Testing

- 33 new tests in `tests/test_scalability.py` (179 total): default-hist and
  explicit-exact selection,
  histogram cluster recovery, pure-noise root stop, issue #6 reproduction,
  fit-state release, per-column categorical strategies, auto degradation,
  pre-fit categorical memory check, chunked-exact identity (deterministic and
  randomized differential fuzz), reproducibility, verbose output, stability
  subsampling sizes, root statistics, weighted/categorical histogram paths,
  `max_bins` code dtype and edge resolution, sample-weight behavior,
  per-candidate one-vs-rest scoring, per-column strategy behavior,
  float32 boundary thresholds, large-offset association, periodic-column
  sampling, imbalanced binary columns, unobserved-cell Cramér's V,
  all-constant input, high-cardinality categorical memory (tracemalloc
  < 100 MiB where the old dense table alone was ~275 MiB), sparse-vs-dense
  contingency equivalence, exact int64 score/rule text, predict encoding only
  used categorical columns, and DataFrame bootstrap.
- `ruff check` and `ruff format --check` clean.

## 0.5.0 — 2026-09-23

### Fixed

- **Single-value categorical column no longer crashes** (`ZeroDivisionError`):
  root Gini is now clamped to a small epsilon; a column with one unique code
  is treated as constant.
- **Negative / very large category codes handled exactly**: Gini computation
  now uses `searchsorted` local codes instead of raw `bincount`, so codes like
  `pd.factorize`'s `-1` or values near `2^63` work correctly without
  massive allocations.
- **`float32` mode: `predict(X_train)` now equals `labels_`**: prediction-side
  threshold comparisons cast input to the stored `dtype`, eliminating
  boundary drift.  `replay_check` was also masking this — now consistent.
- **Complex input rejected** with a clear `TypeError` instead of silently
  discarding the imaginary part.
- **Masked arrays rejected** (mask was silently ignored).
- `adjusted_rand_score([0], [0])` now returns `1.0` (was `NaN`); empty input
  raises `ValueError`.
- `min_gain=nan` and `min_gain=True` now rejected by parameter validation
  (were silently accepted, producing degenerate trees).
- `replay_labels` validates feature count (was silently returning garbage on
  width mismatch).
- `Condition` validates op/payload consistency (finite threshold for `<=`/`>`,
  non-empty categories for `in`/`not in`).
- `profile_leaves` validates `top_k >= 1`.
- `binary_features`/`categorical_features` reject scalar values with a clear
  `TypeError` (was `'int' object is not iterable`).
- `quantiles=8.0` (float scalar) gives a targeted error message.
- `_resolve_sequence_spec` raises `ValueError` (not `KeyError`/`IndexError`).

### Changed

- **`min_leaf_abs` default lowered from 50 to 10** — the old default silently
  produced a single leaf on datasets under ~5000 rows.  Override explicitly
  for large-scale use.
- **Not-fitted errors are now `NotFittedError`** (sklearn-compatible,
  inherits `ValueError` + `AttributeError`) instead of `RuntimeError`.
- `profile_leaves` returns `LeafProfile` dataclass instances (attribute
  access) instead of dicts.
- `summary()` keys renamed: `n_segments`→`n_leaves`,
  `heterogeneity_reduction`→`reduction`, `segment_size_*`→`leaf_size_*`.
- `export_dict()` now includes a `_meta` key on the root node with feature
  names, `handle_unknown`, category sets etc. so `from_dict` can fully
  reconstruct the tree.  Existing keys are unchanged.
- `feature_names_in_` (ndarray, sklearn convention) added alongside
  `feature_names_` (list).
- `replay_labels` raises `ValueError` on unknown feature names in rules
  (was bare `KeyError`).
- `StabilityResult` custom `__eq__`/`__hash__` that handle ndarray fields.
- `n_replicates` accepts any `Integral` (was `int`-only).

### New

- **`score(X, y=None)`** — heterogeneity reduction of `X` under the fitted
  partition; plugs directly into `GridSearchCV`/`cross_val_score`.
- **`from_dict(data)`** classmethod — rebuild a predict-capable tree from
  `export_dict()` output (deploy rules to production without pickling).
- **`transform(X)`** / **`fit_transform(X)`** — one-hot leaf encoding for
  Pipeline intermediate steps.
- **`export_graphviz()`** — zero-dependency Graphviz DOT export.
- **`feature_importances_`** — normalized split gain per feature (sklearn
  convention).
- **`sample_weight`** parameter on `fit`/`fit_predict`/`fit_transform` —
  per-row weights scaling impurity and gain.  `None` (default) is
  bit-identical to the previous unweighted path.
- `apply(X)` is now a real method with its own docstring (was a bare alias).
- Method-level wrappers: `tree.extract_rules()`, `tree.replay_check(X)`,
  `tree.replay_labels(X)`, `tree.profile_leaves(X)`,
  `tree.bootstrap_stability(X)` (thin delegates to the free functions).
- `rule_mask` exported from the package top level.
- `LeafProfile` dataclass exported.
- `NotFittedError` exported.
- `py.typed` (PEP 561) — downstream type checkers now see the annotations.
- Full type annotations on all public API.
- `Condition`/`Cut`/`Leaf`/`Rule`/`StabilityResult` documented with numpydoc.

### Performance

- Categorical-split evaluation no longer builds unused one-hot prefix sums
  (was O(n·Σk) wasted per categorical split candidate).
- `j → numeric row` mapping precomputed at fit (was O(n_num) scan per
  feature per node).
- Exact categorical subset enumeration refuses > 1M candidate subsets with
  a clear error (was silently allocating multi-GB matrices).
- `_fmt`/`format_threshold` de-duplicated into `_base`.

### Testing

- 34 new regression tests for edge cases, new API, and previously uncovered
  branches (negative codes, single-value categories, float32 consistency,
  complex/masked rejection, ARI edge cases, pickle round-trip,
  `from_dict` round-trip, `score`, `transform`, `feature_importances_`,
  `sample_weight`, column-order alignment, etc.).
- Fixed 3 fake/dead assertions (`or True`, self-comparison, dead set
  construction).
- Data factories consolidated into `conftest.py` (test modules no longer
  import from each other).
- CI: `pytest-cov`, `--doctest-modules`, pure-numpy-only job, Python 3.14.
- Release workflow gated on tests + version-tag consistency check.

## 0.4.0 — 2026-08-26

- **Fixed: true best-first growth (heap-key bug)**. The frontier min-heap was
  keyed by the *positive* split gain, so every pop expanded the lowest-gain
  leaf — worst-first growth, contradicting the documented best-first
  behavior. Keys are now the negated gain; ties break deterministically by
  insertion order. Final partitions are unchanged (stopping rules are
  node-local, so every node that passes them is eventually split either
  way), but **cut execution order (`cuts_`) and leaf-id numbering may differ
  from earlier releases**. Both the production tree and the independent
  naive reference share the corrected specification, and a dedicated test
  now verifies mechanically that every pop takes the current maximum-gain
  frontier node.
- **New: `feature_weights`** — per-feature weights ``w_j`` scaling each
  column's contribution to heterogeneity,
  ``H(S) = sum_j(w_j*h_j(S)) / sum_j(w_j)``. Accepts a 1-D numeric sequence,
  a dict keyed by feature/DataFrame-column names (missing names default to
  weight 1), or ``None`` for uniform weights (bit-identical to the previous
  code path). Weights must be finite and non-negative with at least one
  positive entry. Zero-weight features may still serve as splitting
  variables (proxy semantics). Exposed as ``feature_weights_`` after fit;
  gain additivity continues to hold exactly.
- **New: `max_leaf_nodes`** — sklearn-style hard cap on the number of
  leaves. Growth remains best-first: once the budget is reached no further
  splits execute and remaining frontier nodes become ordinary leaves.
  Composes with ``max_depth``, ``min_gain`` and the leaf-size floors under
  the strictest-condition-wins rule.
- **New: public tree export** — ``export_dict()`` returns a fully
  JSON-serializable nested dict of splits (feature/op/threshold/categories/
  gain/n_samples/left/right) and leaves (id/fraction/heterogeneity/
  conditions); ``export_text()`` renders a stable indented listing including
  branch conditions, gains, sample counts and leaf ids. Neither exposes the
  private node objects, and no new runtime dependency is introduced.
- **New: `categorical_splitter="one_vs_rest"`** — for high-cardinality
  categorical columns, evaluate each category alone against the complement
  (linear in k) instead of the exact exponential subset enumeration
  (``"exact"``, still the default, still guarded by ``max_categories``).
  ``max_categories`` only limits exact mode; one-vs-rest accepts any
  cardinality while reusing the same categorical-Gini machinery, keeping
  int64-exact codes and both ``handle_unknown`` policies intact.
- Docs: README is now Chinese-first (``README.md``) with the full English
  version in ``README.en.md``; both cover feature weighting, leaf budgets,
  tree export (with a graphviz conversion sketch) and high-cardinality
  categoricals; tuning tips updated accordingly.

## 0.3.0 — 2026-08-25

- **Encoding-invariant categorical impurity**: declared categorical features
  now contribute *Gini impurity relative to the root Gini* to node
  heterogeneity (continuous/binary keep normalized variance). Results no
  longer depend on how categories happen to be integer-coded; any relabeling
  yields an equivalent partition. Children Gini for every candidate split is
  evaluated exactly via per-column contingency tables.
- **Exact categorical codes under any dtype**: category columns are stored as
  int64/float64 internally, so codes like `16_777_216` vs `16_777_217` stay
  distinct even with `dtype=numpy.float32`.
- **`handle_unknown` parameter** (`"complement"` default, `"error"`): control
  whether unseen category codes at prediction time route deterministically to
  the complement side or raise `ValueError` naming the feature and values.
  New learned attribute ``categories_`` records the training-time code set of
  every categorical column.
- **`profile_leaves` categorical profiles**: categorical columns are excluded
  from z-scored ``top_features`` and instead reported through
  ``category_enrichment``: the most enriched ``(feature, code, leaf_share,
  lift)`` pairs per segment.
- Test suite hardening: package-style imports (`tests/__init__.py`),
  recoding-invariance tests across multiple permutations, float32 large-code
  regression, unknown-category policy tests and categorical profile tests.

## 0.2.0 — 2026-08-25

- **Native categorical features**: pass integer-coded columns via
  `categorical_features`; splits are category-subset memberships
  (`color in {1, 2}`), enumerated exhaustively and scored with the same
  normalized-heterogeneity criterion (`max_categories` guards the 2^(k-1)
  enumeration).
- Candidate thresholds for numeric splits are now computed and located in the
  original feature space, guaranteeing that the evaluated partition is exactly
  the committed one (fixes a rare boundary-point flip when a quantile knot
  coincided with a data point).
- `Condition` / `Cut` gained set-membership semantics (`op="in"`, `categories`);
  rule replay, `predict` on unseen codes (deterministic right branch) and
  profiling all support categorical columns.

## 0.1.0 — 2026-08-25

Initial release.

- `SegTree`: best-first unsupervised decision tree (clustering tree) with
  normalized mixed-type heterogeneity criterion, quantile candidate splits,
  automatic binary-feature detection, sklearn-compatible API and float32/64
  internal precision.
- `extract_rules` / `replay_labels` / `replay_check`: rule extraction with
  independent replay verification.
- `profile_leaves`: per-segment z-score feature profiles.
- `bootstrap_stability`: bootstrap ARI stability assessment.
- `adjusted_rand_score`: pure-numpy ARI matching scikit-learn numerics.
