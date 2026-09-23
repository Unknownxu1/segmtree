# Changelog

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
