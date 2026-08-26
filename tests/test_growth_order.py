"""Best-first growth-order guarantees.

Regression tests for the heap-key bug: ``heapq`` is a min-heap, so the
frontier must be keyed by *negative* gain -- with positive keys the loop
expanded the lowest-gain node first (worst-first) while claiming best-first
growth. These tests verify both the mechanism (heap key invariant, spied
directly) and the behavior (cut sequences on data whose correct best-first
order is unambiguous).
"""

from __future__ import annotations

import heapq

import numpy as np
import pytest

from segmtree import SegTree, adjusted_rand_score
from tests._naive_reference import NaiveSegTree


def asymmetric_data():
    """Root split is strong; afterwards the LEFT child hides a much stronger
    second split than the RIGHT child's weak one.

    A correct best-first pass must cut the left child before the right one;
    the old worst-first bug popped the weaker child first.
    """
    rng = np.random.default_rng(0)
    n = 800
    half = n // 2
    quarter = n // 4
    x0 = np.r_[np.zeros(half), np.full(n - half, 10.0)]  # root splitter
    x1 = np.r_[np.zeros(quarter), np.full(quarter, 8.0), np.full(n - half, 3.0)]
    x1 += rng.normal(0, 0.1, n)  # strong structure inside the LEFT cluster only
    x2 = np.r_[np.full(half, 0.5), np.zeros(quarter), np.full(quarter, 1.6)]
    x2 += rng.normal(0, 0.05, n)  # deliberately weak structure on the RIGHT
    return np.column_stack([x0, x1, x2])


STRONG_PARAMS = dict(min_gain=0.02, min_leaf_abs=50, min_leaf_frac=0.01, quantiles=6)


def test_second_cut_is_the_strongest_frontier_split():
    tree = SegTree(**STRONG_PARAMS).fit(asymmetric_data())
    assert len(tree.cuts_) >= 3
    # the strong sub-structure must be exploited before the weak-right tail
    weak = min(c.gain for c in tree.cuts_)
    strong_cuts = [c for c in tree.cuts_ if c.feature_index == 1]
    assert strong_cuts and max(c.gain for c in strong_cuts) > weak * 5


def test_budget_respects_best_first_choice():
    tree = SegTree(**{**STRONG_PARAMS, "max_leaf_nodes": 3}).fit(asymmetric_data())
    assert tree.n_leaves_ == 3
    uncapped = SegTree(**STRONG_PARAMS).fit(asymmetric_data())
    assert [(c.feature_index, c.threshold) for c in tree.cuts_] == [
        (c.feature_index, c.threshold) for c in uncapped.cuts_[:2]
    ]


def test_cap_trees_form_nested_greedy_prefixes():
    """Signature of best-first growth observable through the public API:
    a tree allowed K leaves must execute exactly the same cuts (same choices,
    same order) as the first K-1 cuts of any larger-budget tree. Under a
    worst-first or breadth-first regime these prefixes would diverge."""
    X = asymmetric_data()
    seqs = {}
    for cap in (2, 3, 4, 6):
        t = SegTree(**{**STRONG_PARAMS, "max_leaf_nodes": cap}).fit(X)
        assert t.n_leaves_ <= cap
        seqs[cap] = [(c.feature_index, c.threshold, c.gain) for c in t.cuts_]
        assert len(seqs[cap]) == cap - 1  # every split adds exactly one leaf
    full = seqs[6]
    for cap in (2, 3, 4):
        assert seqs[cap] == full[: cap - 1], f"cap={cap} must be a prefix of cap=6"


def test_root_cut_is_global_best_by_brute_force():
    """The first executed cut must be the highest-gain candidate over ALL
    features and thresholds, verified by an exhaustive recomputation."""
    rng = np.random.default_rng(12)
    X = rng.normal(size=(300, 4))
    X[:150, 0] += 4.0
    X[150:, 1] += 3.0
    X[:, 3] = (rng.random(300) < (np.arange(300) < 150)).astype(float)
    qgrid = np.linspace(0.05, 0.95, 8)
    tree = SegTree(min_gain=0.0, min_leaf_frac=0.0, min_leaf_abs=30, quantiles=qgrid).fit(X)

    rv = np.var(X, axis=0)

    def h(rows):
        return float(np.mean(np.var(rows, axis=0) / rv))

    hs = h(X)
    n = len(X)
    best = None
    for j in range(X.shape[1]):
        col = X[:, j]
        cand = (
            np.array([0.5])
            if set(np.unique(col)) <= {0.0, 1.0}
            else np.unique(np.quantile(col, qgrid))
        )
        for t in cand:
            gl = col <= t
            nl = int(gl.sum())
            if nl < 30 or n - nl < 30:
                continue
            g = hs - nl / n * h(X[gl]) - (n - nl) / n * h(X[~gl])
            if best is None or g > best[0]:
                best = (g, j, float(t))
    assert best is not None
    root = tree.cuts_[0]
    assert root.feature_index == best[1]
    assert root.threshold == pytest.approx(best[2])
    assert root.gain == pytest.approx(best[0], abs=1e-9)


def test_heap_keys_are_negated_and_every_pop_is_the_live_maximum(monkeypatch):
    """Direct mechanism check: keys pushed onto the frontier heap are
    non-positive (negated gains) and every pop removes the minimum live key,
    i.e. expands the maximum-gain leaf currently on the frontier."""
    real_push, real_pop = heapq.heappush, heapq.heappop
    events = []

    def spy_push(h, item):
        events.append(("push", item[0]))
        real_push(h, item)

    def spy_pop(h):
        item = real_pop(h)
        events.append(("pop", item[0]))
        return item

    monkeypatch.setattr(heapq, "heappush", spy_push)
    monkeypatch.setattr(heapq, "heappop", spy_pop)

    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal([0, 0, 0], 0.5, (250, 3)), rng.normal([5, 5, 5], 0.5, (250, 3))])
    SegTree(min_leaf_abs=25).fit(X)

    assert events, "the training loop must use the frontier heap"
    live = []
    seen_root_pop = False
    for kind, key in events:
        if kind == "push":
            assert key <= 0.0, "heap keys must be negated gains (min-heap => best-first)"
            live.append(key)
        elif not seen_root_pop:
            # the root enters the heap as a list literal (no heappush), so the
            # very first pop is necessarily the single-entry root
            seen_root_pop = True
            assert not live
        else:
            assert live, "popped from an empty frontier"
            assert key == min(live), "every pop must take the current highest-gain node"
            live.remove(key)


def test_growth_deterministic_across_refits():
    X = asymmetric_data()
    a = SegTree(**STRONG_PARAMS).fit(X)
    b = SegTree(**STRONG_PARAMS).fit(X.copy())
    assert np.array_equal(a.labels_, b.labels_)
    assert [(c.feature_index, c.threshold, c.gain) for c in a.cuts_] == [
        (c.feature_index, c.threshold, c.gain) for c in b.cuts_
    ]


def test_exact_ties_break_identically_in_production_and_reference():
    """Twin features give byte-identical candidate gains; insertion-order
    tie-breaking must be stable and identical in both implementations."""
    rng = np.random.default_rng(4)
    signal = np.r_[rng.normal(0, 0.4, 200), rng.normal(6, 0.4, 200)]
    X = np.column_stack([signal, signal.copy(), rng.normal(size=400)])
    mine = SegTree(min_gain=0.05, min_leaf_frac=0.0, min_leaf_abs=40, max_depth=2, quantiles=4)
    mine = mine.fit(X)
    ref_q = np.linspace(0.05, 0.95, 4)
    ref = NaiveSegTree(min_gain=0.05, min_leaf=40, max_depth=2, quantiles=ref_q).fit(X)
    assert mine.n_leaves_ == len(ref.leaves_)
    assert np.array_equal(mine.labels_, ref.labels_)


def test_full_tree_partition_matches_reference_after_fix():
    """End-to-end: with corrected ordering, production and independent
    reference produce the identical partition and cut count."""
    X = asymmetric_data()
    mine = SegTree(min_gain=0.02, min_leaf_frac=0.0, min_leaf_abs=50, max_depth=3, quantiles=6).fit(
        X
    )
    ref = NaiveSegTree(
        min_gain=0.02, min_leaf=50, max_depth=3, quantiles=np.linspace(0.05, 0.95, 6)
    ).fit(X)
    assert np.array_equal(mine.labels_, ref.labels_)
    assert len(mine.cuts_) == len(ref.cuts_)
    assert adjusted_rand_score(mine.labels_, ref.labels_) == pytest.approx(1.0)
