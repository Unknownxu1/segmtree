# segmtree

Interpretable unsupervised decision trees for segmentation — clusters rows of a
tabular dataset into segments, where every segment is described by a short,
human-readable rule chain. No target variable required.

`segmtree` implements a **clustering tree**: instead of predicting labels, it
recursively splits the data so that within-segment heterogeneity drops as much
as possible at every cut, growing best-first (highest-gain node expanded
first). Think of it as "a decision tree grown without `y`" — you get both the
partition *and* the rules that define it, which plain k-means or GMM cannot
give you.

## Why segmtree?

| | k-means / GMM | segmtree |
|---|---|---|
| Segment definition | centroid coordinates | `income > 50k AND is_active = 0` style rules |
| New-row assignment | nearest centroid | a few threshold/membership tests per rule chain |
| Mixed binary + continuous features | needs scaling tricks | one normalized criterion for both |
| Built-in self-checks | none | gain additivity + rule replay verification |
| Stability assessment | ad hoc | one-call bootstrap ARI |

Typical uses: customer segmentation with behavioral flags, patient phenotyping,
survey respondent typing — any "many 0/1 tags + a few counts" table.

## Installation

```bash
pip install segmtree
```

Requires Python ≥ 3.9 and numpy only.

## Quickstart

```python
import numpy as np
from segmtree import SegTree, extract_rules, profile_leaves, replay_check, bootstrap_stability

rng = np.random.default_rng(0)
# 4 continuous features + 3 binary flags, three latent groups
X = np.vstack([
    np.hstack([rng.normal([0, -1, 2, .5], .5, (250, 4)), (rng.random((250, 3)) < .1)]),
    np.hstack([rng.normal([4, 3, -2, .5], .5, (200, 4)), (rng.random((200, 3)) < .5)]),
    np.hstack([rng.normal([-4, 2, 0, 8], .5, (150, 4)), (rng.random((150, 3)) < .9)]),
])

tree = SegTree(min_gain=0.02, min_leaf_frac=0.05, min_leaf_abs=10)
tree.fit(X)

print(f"{tree.n_leaves_} segments, heterogeneity reduced {tree.reduction_:.0%}")

for rule in extract_rules(tree):
    print(rule)                      # #0: x1 <= -1.24 AND x0 <= 1.83 ...

profile_leaves(tree, X, top_k=3)     # most distinctive features per segment
replay_check(tree, X)                # True: stored rules reproduce the partition exactly
bootstrap_stability(tree, X, n_replicates=12, random_state=0).mean   # e.g. 0.98
```

Binary columns (all values in `{0, 1}`) are detected automatically and split
with a single test; pandas DataFrames are accepted and column names flow into
the rules.

## Categorical features

Columns holding integer category codes (e.g. `0..k-1` from a nominal
encoding) can be declared with `categorical_features`. They are split by the
best **subset** of categories rather than by a threshold, and the rules read
as membership tests:

```python
tree = SegTree(categorical_features=["color"], max_categories=12).fit(df)
# rule example:  #3: color in {1, 2} AND income > 42 ...
```

Impurity contribution: categorical columns use **Gini impurity relative to
the root Gini**, while continuous/binary columns keep normalized variance.
Gini depends only on the category *distribution*, so any one-to-one recoding
of the codes yields an equivalent partition — results are encoding-invariant.

Notes:

- Codes must be integral; non-numeric labels should be factorized first
  (`pd.factorize`, `df["color"].astype("category").cat.codes`).
- Codes are handled exactly at any magnitude (int64/float64 internally), even
  with `dtype=np.float32` for the continuous columns.
- **Candidate subsets**: the default `categorical_splitter="auto"` picks a
  strategy per column. Below `max_categories` it evaluates all
  `2^(k-1)-1` category partitions exactly; above it, the column
  **automatically degrades** to `"one_vs_rest"` (each category alone against
  the complement, linear in k) with a single `UserWarning` instead of failing
  the whole fit. Explicitly requesting `"exact"` still raises a `ValueError`
  that names the column, candidate count and estimated memory.
- **Per-column strategies**: `categorical_splitter={"city": "exact",
  "sku": "one_vs_rest"}` mixes both strategies in one fit (names or column
  indices).
- Unseen codes at prediction time follow `handle_unknown`: `"complement"`
  (default) routes them deterministically to the complement side and
  `replay_check` stays exact; `"error"` raises `ValueError` naming the
  feature and unknown values. The training-time code sets are exposed via
  the ``categories_`` attribute.
- ``profile_leaves`` reports categorical structure through
  ``category_enrichment`` — ``(feature, code, leaf_share, lift)`` pairs —
  and keeps z-scores for continuous/binary features only.

## How it works

Node heterogeneity is a weighted mean of per-feature impurity ratios:

```
H(S) = Σ_j w_j · Var_j(S)/Var_j(root) / Σ_j w_j        H(root) = 1
```

(with `w = 1` for every feature unless you pass `feature_weights`). Because
Bernoulli variance p(1−p) fits the same formula, binary and continuous
features share one comparable criterion; declared categorical features
contribute Gini impurity relative to the root Gini instead. A split into L/R
earns:

```
Gain = H(S) − [ n_L/n_S · H(L) + n_R/n_S · H(R) ]
```

The default `hist` path bins each continuous feature once into at most
`max_bins` global quantile bins, proposes candidates per node, then re-scores
the top few with the node-local heterogeneity drop of the *other* features
(see the next section). Only the explicit `splitter="exact"` mode uses in-node
quantiles (`quantiles=8` → the 5% … 95% quantiles; binary features get 0.5)
with sort + prefix sums. The globally best leaf is always split next: the
frontier lives in a min-heap keyed by the *negative* gain, so every expansion
takes the highest-gain node available (best-first, deterministically
tie-broken).

Splitting stops when no cut gains at least `min_gain`, a node is too small
(`max(min_leaf_abs, min_leaf_frac·n)` per child), `max_depth` is reached, or
`max_leaf_nodes` leaves already exist (each split adds exactly one leaf, so
the budget caps executed splits; remaining frontier nodes simply become
leaves).

Two properties make results trustworthy out of the box:

- **Gain additivity** — `total_gain_ == h_root_ − h_avg_` up to float error;
  it accounts for exactly how much heterogeneity was explained (with any
  valid weights).
- **Rule replay** — `replay_check()` re-evaluates every leaf's rule chain from
  scratch and must reproduce `predict()` bit-for-bit.

The test suite additionally verifies exact agreement against an independent
naive reference implementation.

## Large data and performance

**`splitter="hist"` is the default for every data size.** The original exact
algorithm is kept only as an explicit `splitter="exact"` legacy/reference mode;
it is never selected automatically and is not recommended for production.

| Implementation | When | Per-node cost | Memory |
|---|---|---|---|
| `"hist"` (default) | all data sizes | `O(F · n_node)` | `n×F` values + 1 byte per bin code (2 bytes above 256 bins) + an `O(F²)` association matrix during fit |
| `"exact"` | explicit only: validation against the original algorithm | `O(F² · n_node)` | chunked buffers (≤192 MiB) |

The exact mode exists mainly for bit-level comparison against the independent
naive reference implementation and for cases that need its original criterion
semantics; normal use should keep the default.

The histogram path first ranks all candidates with an **association proxy**
(Pearson r² for numeric pairs, correlation ratio η² for categorical-numeric,
Cramér's V² for categorical pairs, estimated on at most 50k rows), then
re-scores the top four with the **node-local heterogeneity drop of the other
columns**, subtracting the `1/n_eff` random-split noise floor. As a result:

- pure-noise data stops at a single leaf when at least two columns and
  enough rows provide corroboration (a median cut of any continuous column
  removes ~64% of its own variance even under noise, so that term is not used
  as evidence).  A single-column problem has nothing to corroborate and
  behaves like the exact path; see the boundary note below;
- segments are not split further inside homogeneous clusters;
- adding weak columns no longer silently stops the tree at the root.

`min_gain_mode="auto"` uses `sqrt(F)` normalization on the default histogram
path, decoupling `min_gain` from the number of columns; only the explicit
`splitter="exact"` mode keeps the original (feature-averaged) threshold. Set
`"total"` or `"normalized"` explicitly if you need a different unit.

Measured end-to-end (single machine; peak RSS sampled during fit and
**includes the caller's input matrix**; `dtype=np.float32` for the largest
case):

| Data | Path | Time | Peak RSS | Leaves |
|---|---|---|---|---|
| 200k × 30 | hist | 2.5 s | 263 MB | 4 |
| 60k × 300 (3 signal columns) | hist | 5.5 s | 489 MB | 4 |
| 300k × 301 (3 signal columns) | hist | 29 s | 1,015 MB | 4 |
| **829k × 301 (3 signal columns, f32)** | hist | **84 s** | **2,439 MB (input 1,152 MB)** | 4 |
| 60k × 300 (pure noise) | hist | 3.3 s | 497 MB | 1 |

The second-to-last row is the originally reported scale: the old
implementation OOM'd there; the histogram path finishes in ~1.4 minutes, uses
~1.3 GB beyond the input, and splits only on 2 of the 3 signal columns.

Small and medium data benefit too: on 1,200×6 the histogram path takes
~0.011 s / 5 leaves versus ~0.089 s / 69 leaves for exact; on 20,000×50 it is
~0.39 s / 7 leaves versus ~8.2 s / 65 leaves (exact tends to cut noise into
many leaves on wide data). This is why histogram is the default everywhere.

Other related parameters:

- `max_bins=32`: quantile bins per continuous column on the histogram path;
- `random_state`: seeds association subsampling and random tie-breaks
  (`None` keeps deterministic first-seen ties);
- `verbose=1`: one line per split (depth, size, feature, gain, children,
  elapsed) plus a final summary;
- `bootstrap_stability(..., subsample=0.5)`: train each replicate on a bounded
  sample instead of the full data.

**Boundary note**: the histogram path requires corroboration from *other*
columns. If exactly one column carries structure and every other column is
independent of it, the path conservatively stops at the root (such a structure
is indistinguishable from noise without supervision). Use
`splitter="exact"` and tune `min_gain` for that case.

## Feature weighting

Domain knowledge often says some columns matter more. Pass
`feature_weights` as a sequence of length `n_features`, a dict keyed by
column names (unlisted columns keep weight 1), or leave it `None`:

```python
tree = SegTree(feature_weights={"income": 4.0, "churn_flag": 2.0}).fit(df)
```

Weights scale each column's contribution to heterogeneity and gain — they do
not forbid splits on zero-weight columns, which may still be selected as
proxies that separate the weighted features well.

## Exporting and persistence

```python
d = tree.export_dict()        # JSON-serializable nested dict
import json; json.dumps(d)    # works out of the box

print(tree.export_text())     # indented listing with gains, n and leaf ids
print(tree.export_graphviz()) # Graphviz DOT string (zero dependencies)

tree2 = SegTree.from_dict(d)  # rebuild a predict-capable tree from the dict
```

Split nodes carry `feature/feature_index/op/threshold|categories/gain/
n_samples/left/right`; leaves carry `leaf_id/n_samples/fraction/heterogeneity/
conditions`. No private objects are exposed. `from_dict` restores full
prediction capability — ideal for deploying rules to a production scoring
service.

## API overview

```python
SegTree(min_gain=0.01, min_leaf_frac=0.01, min_leaf_abs=10, max_depth=None,
        quantiles=8, binary_features="auto", categorical_features=None,
        max_categories=12, handle_unknown="complement", feature_names=None,
        dtype=np.float64, feature_weights=None, max_leaf_nodes=None,
        categorical_splitter="auto", splitter="hist", max_bins=32,
        min_gain_mode="auto", random_state=None, verbose=0)
```

| Member | Purpose |
|---|---|
| `.fit(X)` / `.fit_predict(X)` / `.predict(X)` | sklearn-style fit & assignment |
| `.score(X)` | heterogeneity reduction (plugs into `GridSearchCV`) |
| `.transform(X)` | one-hot leaf encoding (Pipeline intermediate) |
| `.apply(X)` | leaf indices (sklearn `apply` convention) |
| `.labels_`, `.n_leaves_`, `.leaves_`, `.cuts_` | fitted structure |
| `.reduction_`, `.h_avg_`, `.total_gain_`, `.summary()` | quality statistics |
| `.feature_importances_` | normalized split gain per feature |
| `.feature_weights_` | effective per-feature weights |
| `.splitter_`, `.association_` | search path used and per-feature association |
| `.root_variances_`, `.root_ginis_` | raw root variance / Gini (`h_root_` is the actual root heterogeneity, usually 1.0; `h_root_normalized_` is the baseline) |
| `.export_dict()` / `.export_text()` / `.export_graphviz()` | public tree export |
| `SegTree.from_dict(d)` | rebuild a predict-capable tree from `export_dict` |
| `extract_rules(tree)` | list of `Rule` objects (`str(rule)` → readable chain) |
| `replay_labels(tree, X)` / `replay_check(tree, X)` | rule-replay validation |
| `profile_leaves(tree, X, top_k=3)` | `LeafProfile` per segment |
| `bootstrap_stability(tree, X, n_replicates, random_state)` | mean/std of bootstrap ARI |
| `adjusted_rand_score(a, b)` | pure-numpy ARI |

Tuning tips:

- Fewer, larger segments → raise `min_gain` (e.g. `0.02–0.05`), lower
  `quantiles`, cap `max_depth` (e.g. 4–6), **or set `max_leaf_nodes`
  directly** when you know the target segment count.
- Large / wide data → keep the default `splitter="hist"`; set `dtype=np.float32` beyond
  a million rows / several hundred columns (internal values and the input
  each take one copy, so this roughly halves memory); tune `max_bins` for
  finer or coarser thresholds; set `min_gain_mode="normalized"` for the
  legacy threshold semantics.
- Small datasets (< 500 rows) → lower `min_leaf_abs` further (default is 10).
- Business priority → emphasize key columns via `feature_weights` instead of
  dropping the rest.
- Unstable segments (low bootstrap ARI) → fewer quantiles, higher `min_gain`,
  a depth cap or a leaf budget usually stabilize the structure.

## Comparison with related tools

- **scikit-learn DecisionTree on k-means labels** ("surrogate tree"): two-step,
  rules may contradict the clustering; `segmtree` optimizes split quality and
  rule fidelity simultaneously.
- **forest-clustering / URF**: forest-proximity clustering — accurate but the
  segments themselves are not directly interpretable.
- **CUBT (R)**: closest academic relative; `segmtree` adds best-first growth,
  mixed-type normalization, additivity checks and bootstrap stability in a
  numpy-only package.

## Development

```bash
git clone <your-fork-url> && cd segmtree
pip install -e .[dev]
pytest                       # run the test suite
ruff check src tests         # lint
ruff format --check src tests  # formatting
```

## License

MIT — see [LICENSE](LICENSE).

[中文文档](README.md)
