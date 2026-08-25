"""Independent naive reference implementation used for cross-validation tests.

Mirrors the algorithm specification directly (per-candidate Python loops,
two-pass np.var / explicit bincount Gini on boolean masks, identical
tie-breaking) without any of the production optimizations (prefix sums,
vectorized threshold search, centered-space evaluation, one-hot cumsums), so
agreement between the two validates the optimized code path.
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
    def __init__(self, min_gain=0.01, min_leaf=10, max_depth=None, quantiles=None):
        self.min_gain = min_gain
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.quantiles = (
            np.linspace(0.05, 0.95, 8) if quantiles is None else np.asarray(quantiles, float)
        )

    def _h(self, Xs):
        """Combined heterogeneity: mean of normalized variances (numeric/binary)
        and normalized Ginis (categorical) over all features."""
        num_idx = [j for j in range(Xs.shape[1]) if j not in self.cat_set_]
        total = 0.0
        for j in num_idx:
            total += float(np.var(Xs[:, j].astype(float)) / self.root_var_num_[j])
        for j in self.cat_set_:
            total += _gini(Xs[:, j].astype(int)) / self.root_gini_[j]
        return total / Xs.shape[1]

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
                half = 1 << (len(vals) - 1)
                for r in range(half):  # smallest code stays right; bits pick left codes
                    left_codes = [int(vals[1 + b]) for b in range(len(vals) - 1) if (r >> b) & 1]
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
        heap = [(node["best"]["gain"] if node["best"] else 0.0, next(tick), node)]
        self.labels_ = np.full(len(self.X_), -1, dtype=int)
        self.leaves_ = []
        self.cuts_ = []

        while heap:
            _, _, nd = heapq.heappop(heap)
            b = nd["best"]
            if b is None or b["gain"] < self.min_gain:
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
                heapq.heappush(heap, (ch["best"]["gain"] if ch["best"] else 0.0, next(tick), ch))
        return self
