"""Independent naive reference implementation used for cross-validation tests.

Mirrors the algorithm specification directly (per-candidate Python loops,
two-pass np.var / explicit bincount Gini on boolean masks, identical
tie-breaking) without any of the production optimizations (prefix sums,
vectorized threshold search, centered-space evaluation, one-hot cumsums), so
agreement between the two validates the optimized code path.

Shared specification (must match SegTree):
- frontier is a min-heap keyed by *negative* gain, so every pop expands the
  current highest-gain leaf; ties break by insertion order;
- H(S) = sum_j w_j * h_j(S) / sum_j(w_j) with h = Var/Var(root) for
  numeric/binary columns and Gini/Gini(root) for categorical columns;
- a successful split adds exactly one leaf, so ``max_leaf_nodes`` caps the
  number of executed splits at ``max_leaf_nodes - 1``;
- categorical candidates: all canonical subsets ("exact") or singleton vs
  rest ("one_vs_rest").
"""

from __future__ import annotations

import heapq
from itertools import count

import numpy as np


def _gini(values):
    """Gini impurity of an integer-coded sample."""
    _, counts = np.unique(values, return_counts=True)
    p = counts / values.size
    return float(1.0 - np.sum(p * p))


class NaiveSegTree:
    def __init__(
        self,
        min_gain=0.01,
        min_leaf=10,
        max_depth=None,
        quantiles=None,
        feature_weights=None,
        max_leaf_nodes=None,
        cat_splitter="exact",
    ):
        self.min_gain = min_gain
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.quantiles = (
            np.linspace(0.05, 0.95, 8) if quantiles is None else np.asarray(quantiles, float)
        )
        self.feature_weights = feature_weights
        self.max_leaf_nodes = max_leaf_nodes
        self.cat_splitter = cat_splitter

    def _h(self, Xs):
        """Weighted mixed-type heterogeneity per the shared specification."""
        num_idx = [j for j in range(Xs.shape[1]) if j not in self.cat_set_]
        total = 0.0
        denom = 0.0
        for j in range(Xs.shape[1]):
            w = 1.0 if self.w_ is None else float(self.w_[j])
            denom += w
            if j in self.cat_set_:
                total += w * _gini(Xs[:, j].astype(int)) / self.root_gini_[j]
        for j in num_idx:
            w = 1.0 if self.w_ is None else float(self.w_[j])
            total += w * float(np.var(Xs[:, j].astype(float))) / self.root_var_num_[j]
        return total / denom

    def _candidate_cat_partitions(self, vals):
        vals = [int(v) for v in vals]
        if self.cat_splitter == "one_vs_rest":
            return [[v] for v in vals]
        half = 1 << (len(vals) - 1)
        out = []
        for r in range(half):  # smallest code stays right; bits pick left codes
            out.append([vals[1 + b] for b in range(len(vals) - 1) if (r >> b) & 1])
        return out

    def _best_split(self, idx, depth):
        n = idx.size
        if self.max_depth is not None and depth >= self.max_depth:
            return None
        if n < 2 * self.min_leaf:
            return None
        Xs = self.X_[idx]
        hs = self._h(Xs)
        best = None

        def consider(gain, feat, payload, kind):
            nonlocal best
            if best is None or gain > best["gain"]:
                best = {"gain": float(gain), "feat": int(feat), "payload": payload, "kind": kind}

        for j in range(Xs.shape[1]):
            col = Xs[:, j]
            if len(np.unique(col)) < 2:
                continue
            if j in self.cat_set_:
                vals = np.unique(col.astype(int))
                for left_codes in self._candidate_cat_partitions(vals):
                    left = np.isin(col.astype(int), left_codes) if left_codes else np.zeros(n, bool)
                    nl, nr = int(left.sum()), int(n - left.sum())
                    if nl < self.min_leaf or nr < self.min_leaf:
                        continue
                    g = hs - (nl / n) * self._h(Xs[left]) - (nr / n) * self._h(Xs[~left])
                    consider(g, j, frozenset(left_codes), "subset")
                continue
            if j in self.binary_set_:
                cand = [0.5]
            else:
                cand = list(np.unique(np.quantile(col.astype(float), self.quantiles)))
            for t in cand:
                left = col <= t
                nl, nr = int(left.sum()), int(n - left.sum())
                if nl < self.min_leaf or nr < self.min_leaf:
                    continue
                g = hs - (nl / n) * self._h(Xs[left]) - (nr / n) * self._h(Xs[~left])
                consider(g, j, float(t), "threshold")
        return best

    def fit(self, X, binary_set=frozenset(), cat_set=frozenset()):
        self.X_ = np.asarray(X, dtype=np.float64)
        self.binary_set_ = set(binary_set)
        self.cat_set_ = set(cat_set)
        self.w_ = None if self.feature_weights is None else np.asarray(self.feature_weights, float)
        num_idx = [j for j in range(self.X_.shape[1]) if j not in self.cat_set_]
        self.root_var_num_ = {j: max(float(np.var(self.X_[:, j])), 1e-12) for j in num_idx}
        self.root_gini_ = {j: max(_gini(self.X_[:, j].astype(int)), 1e-12) for j in self.cat_set_}
        tick = count()

        node = {
            "idx": np.arange(len(self.X_)),
            "depth": 0,
            "best": None,
            "leaf": False,
            "id": -1,
            "j": -1,
            "t": 0.0,
            "L": None,
            "R": None,
        }
        node["best"] = self._best_split(node["idx"], 0)
        # Negative-gain keys: the min-heap pops the highest-gain frontier node.
        root_key = -(node["best"]["gain"]) if node["best"] else 0.0
        heap = [(root_key, next(tick), node)]
        self.labels_ = np.full(len(self.X_), -1, dtype=int)
        self.leaves_ = []
        self.cuts_ = []

        while heap:
            _, _, nd = heapq.heappop(heap)
            b = nd["best"]
            can_split = b is not None and b["gain"] >= self.min_gain
            if (
                can_split
                and self.max_leaf_nodes is not None
                and len(self.cuts_) >= self.max_leaf_nodes - 1
            ):
                can_split = False
            if not can_split:
                nd["leaf"] = True
                nd["id"] = len(self.leaves_)
                self.leaves_.append(
                    {
                        "id": nd["id"],
                        "idx": nd["idx"],
                        "n": len(nd["idx"]),
                        "h": self._h(self.X_[nd["idx"]]),
                        "depth": nd["depth"],
                        "chain": nd.get("chain", []),
                    }
                )
                self.labels_[nd["idx"]] = nd["id"]
                continue
            j, payload, kind = b["feat"], b["payload"], b["kind"]
            if kind == "subset":
                left = np.isin(self.X_[:, j][nd["idx"]].astype(int), tuple(payload))
                chain_add = [(j, "in", payload), (j, "not in", payload)]
            else:
                left = self.X_[:, j][nd["idx"]] <= payload
                chain_add = [(j, "<=", payload), (j, ">", payload)]
            il, ir = nd["idx"][left], nd["idx"][~left]
            self.cuts_.append((nd["depth"], j, float(b["gain"]), len(il), len(ir), kind))
            chain = nd.get("chain", [])
            for child_idx, branch_op, branch in ((il, chain_add[0], "L"), (ir, chain_add[1], "R")):
                ch = {
                    "idx": child_idx,
                    "depth": nd["depth"] + 1,
                    "best": None,
                    "leaf": False,
                    "id": -1,
                    "j": -1,
                    "t": 0.0,
                    "L": None,
                    "R": None,
                    "chain": [*chain, branch_op],
                }
                ch["best"] = self._best_split(ch["idx"], ch["depth"])
                nd[branch] = ch
                cb = ch["best"]
                heapq.heappush(heap, (-(cb["gain"]) if cb else 0.0, next(tick), ch))
        return self
