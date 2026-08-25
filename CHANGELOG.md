# Changelog

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
