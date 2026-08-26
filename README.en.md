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
| New-row assignment | nearest centroid | evaluate ≤ 6 threshold tests |
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
- **Candidate subsets**: with the default `categorical_splitter="exact"` all
  `2^(k-1)-1` category partitions of a node are evaluated (exponential —
  guarded by `max_categories`). For high-cardinality columns set
  `categorical_splitter="one_vs_rest"` to test each category alone against
  the complement instead: cost grows linearly with k, `max_categories` no
  longer applies, but splits that merge several categories onto one side can
  be missed.
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

Candidates are the in-node quantiles of each feature (`quantiles=8` → the 5%
… 95% quantiles; binary features get one candidate at 0.5). All candidates are
evaluated per feature via sort + prefix sums in O(n log n), and the globally
best leaf is always split next: the frontier lives in a min-heap keyed by the
*negative* gain, so every expansion takes the highest-gain node available
(best-first, deterministically tie-broken).

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

## Exporting a fitted tree

```python
d = tree.export_dict()      # JSON-serializable nested dict
import json; json.dumps(d)  # works out of the box

print(tree.export_text())   # indented listing with gains, n and leaf ids
```

Split nodes carry `feature/feature_index/op/threshold|categories/gain/
n_samples/left/right`; leaves carry `leaf_id/n_samples/fraction/heterogeneity/
conditions`. No private objects are exposed. To render graphics, convert the
dict — no extra dependency ships with segmtree:

```python
def to_graphviz(node, lines):     # sketch: pip install graphviz separately
    if node["type"] == "split":
        label = f"{node['feature']} <= {node['threshold']}" if node["op"] == "<=" \
            else f"{node['feature']} in {node['categories']}"
        lines.append(f'"{id(node)}" [label="{label}"];')
        for child in (node["left"], node["right"]):
            lines.append(f'"{id(node)}" -> "{id(child)}";')
            to_graphviz(child, lines)
    else:
        lines.append(
            f'"{id(node)}" [shape=box, label="leaf #{node["leaf_id"]} n={node["n_samples"]}"];'
        )
```

(Or simply iterate `export_dict()` yourself — the structure is plain dicts.)

## API overview

```python
SegTree(min_gain=0.01, min_leaf_frac=0.01, min_leaf_abs=50, max_depth=None,
        quantiles=8, binary_features="auto", categorical_features=None,
        max_categories=12, handle_unknown="complement", feature_names=None,
        dtype=np.float64, feature_weights=None, max_leaf_nodes=None,
        categorical_splitter="exact")
```

| Member | Purpose |
|---|---|
| `.fit(X)` / `.fit_predict(X)` / `.predict(X)` | sklearn-style fit & assignment |
| `.labels_`, `.n_leaves_`, `.leaves_`, `.cuts_` | fitted structure |
| `.reduction_`, `.h_avg_`, `.total_gain_`, `.summary()` | quality statistics |
| `.feature_weights_` | effective per-feature weights |
| `.export_dict()` / `.export_text()` | public tree export (JSON-safe dict / text) |
| `extract_rules(tree)` | list of `Rule` objects (`str(rule)` → readable chain) |
| `replay_labels(tree, X)` / `replay_check(tree, X)` | rule-replay validation |
| `profile_leaves(tree, X, top_k=3)` | z-score profiles per segment |
| `bootstrap_stability(tree, X, n_replicates, random_state)` | mean/std of bootstrap ARI |
| `adjusted_rand_score(a, b)` | pure-numpy ARI |

Tuning tips:

- Fewer, larger segments → raise `min_gain` (e.g. `0.02–0.05`), lower
  `quantiles`, cap `max_depth` (e.g. 4–6), **or set `max_leaf_nodes`
  directly** when you know the target segment count.
- Small datasets (< 5000 rows) → lower `min_leaf_abs` (it defaults to 50).
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
pytest            # run the test suite
ruff check src tests
```

## License

MIT — see [LICENSE](LICENSE).

[中文文档](README.md)
