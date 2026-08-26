# Changelog

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
