"""Regression tests for the scalability / observability work (0.6.0).

Covers the issues reported in ``segmtree_问题清单.md``:

- automatic exact-vs-histogram selection and the absence of the old
  ``(n_numeric x n_samples)`` buffers / sort orders on the histogram path;
- ``min_gain`` no longer silently stopping at the root when weak columns are
  added, and pure-noise data staying at a single leaf;
- categorical ``auto`` degradation, per-column strategies and the early
  candidate-memory check;
- chunked exact-path buffers producing identical trees;
- ``random_state``, ``verbose`` and stability subsampling.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import segmtree.stability as stability_module
import segmtree.tree as tree_module
from segmtree import SegTree, adjusted_rand_score, replay_check
from segmtree.stability import bootstrap_stability


def _blobs(n=3000, d=6, k=3, seed=10):
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for c in range(k):
        centers = rng.normal(np.full(d, c * 5.0), 0.8)
        parts.append(rng.normal(centers, 0.6, (n // k, d)))
        labels.append(np.full(n // k, c))
    return np.vstack(parts), np.concatenate(labels)


def _signal_plus_noise(n=3000, n_noise=200, seed=3):
    rng = np.random.default_rng(seed)
    half = n // 2
    informative = np.column_stack([np.r_[np.zeros(half), np.full(n - half, 5.0)]] * 2)
    noise = rng.normal(size=(n, n_noise))
    X = np.column_stack([informative, noise]) + rng.normal(0, 0.05, (n, n_noise + 2))
    return X


def test_default_splitter_is_hist_and_exact_is_opt_in():
    small, _ = _blobs(n=600, d=4)
    assert SegTree().fit(small).splitter_ == "hist"
    assert SegTree(splitter="exact").fit(small).splitter_ == "exact"
    with pytest.raises(ValueError, match="splitter"):
        SegTree(splitter="auto")._check_params(3)


def test_hist_recovers_well_separated_clusters():
    X, y = _blobs()
    tree = SegTree(splitter="hist", min_leaf_abs=50, min_gain=0.05).fit(X)
    assert tree.splitter_ == "hist"
    assert tree.n_leaves_ <= 8
    assert adjusted_rand_score(tree.labels_, y) > 0.85
    assert np.array_equal(tree.predict(X), tree.labels_)
    assert replay_check(tree, X)


def test_hist_pure_noise_stops_at_root():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(20_000, 60))
    tree = SegTree(splitter="hist", min_leaf_abs=100).fit(X)
    assert tree.n_leaves_ == 1
    assert tree.cuts_ == ()


def test_noise_columns_do_not_suppress_or_hijack_signal():
    X = _signal_plus_noise()
    # The default hist path must still find the two informative columns (the
    # legacy normalized gain stops at the root: issue #6).
    tree = SegTree(min_leaf_abs=20, min_gain=0.01).fit(X)
    assert tree.splitter_ == "hist"
    assert tree.n_leaves_ >= 2
    # The two informative columns are identical, so one of them is enough;
    # crucially, no noise column may be selected.
    assert {cut.feature_index for cut in tree.cuts_} <= {0, 1}

    # Exact reproduction of issue #6: two perfectly separating columns plus
    # 200 noise columns.  The legacy normalized gain divides by 202, so even
    # a perfect split (total reduction ~2.0) misses min_gain=0.01 * 202.
    rng = np.random.default_rng(5)
    n = 1000
    half = n // 2
    informative = np.column_stack([np.r_[np.zeros(half), np.full(n - half, 5.0)]] * 2)
    X_legacy = np.column_stack([informative, rng.normal(size=(n, 200))])
    X_legacy += rng.normal(0, 0.05, X_legacy.shape)
    legacy = SegTree(
        splitter="exact", min_gain_mode="normalized", min_gain=0.01, min_leaf_abs=20
    ).fit(X_legacy)
    assert legacy.n_leaves_ == 1
    total = SegTree(splitter="exact", min_gain_mode="total", min_gain=0.01, min_leaf_abs=20).fit(
        X_legacy
    )
    assert total.n_leaves_ > legacy.n_leaves_


def test_hist_releases_large_fit_state():
    X, _ = _blobs(n=1200, d=5)
    tree = SegTree(splitter="hist", min_leaf_abs=30).fit(X)
    assert tree._bufs_ == {}
    assert tree._Xraw is None
    assert tree._bin_codes_ is None
    assert tree._XcT_num_ is None
    assert tree.association_.shape == (X.shape[1],)
    assert tree.root_variances_.shape == (X.shape[1],)
    assert tree.root_ginis_ == {}
    assert tree._cat_codes_ is None
    assert tree._cat_meta_ is None
    assert tree._sw_ is None


def test_per_column_categorical_splitter_dict():
    rng = np.random.default_rng(4)
    n = 900
    g = rng.integers(0, 3, n)
    num = rng.normal(g[:, None] * 3, 0.5, (n, 2))
    low = g + (rng.random(n) < 0.1) * rng.integers(0, 6, n)
    high = g * 5 + (rng.random(n) < 0.1) * rng.integers(0, 40, n)
    X = np.column_stack([num, low, high]).astype(float)
    tree = SegTree(
        categorical_features=[2, 3],
        categorical_splitter={2: "exact", 3: "one_vs_rest"},
        max_categories=8,
        min_leaf_abs=20,
        splitter="exact",
        min_gain_mode="normalized",
    ).fit(X)
    assert tree._cat_splitter_ == {2: "exact", 3: "one_vs_rest"}
    assert replay_check(tree, X)
    # The exact column can merge categories; the ovr-only tree cannot.
    assert any(cut.feature_index == 2 and len(cut.categories) > 1 for cut in tree.cuts_)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ovr_only = SegTree(
            categorical_features=[2, 3],
            categorical_splitter={2: "one_vs_rest", 3: "one_vs_rest"},
            max_categories=8,
            min_leaf_abs=20,
            splitter="exact",
            min_gain_mode="normalized",
        ).fit(X)
    assert all(len(cut.categories) == 1 for cut in ovr_only.cuts_ if cut.op == "in")
    assert not np.array_equal(tree.labels_, ovr_only.labels_)


def test_categorical_auto_degrades_on_hist_path():
    rng = np.random.default_rng(6)
    n = 1200
    g = rng.integers(0, 3, n)
    num = rng.normal(g[:, None] * 3, 0.5, (n, 2))
    cat = g * 5 + (rng.random(n) < 0.2) * rng.integers(0, 40, n)
    X = np.column_stack([num, cat]).astype(float)
    with pytest.warns(UserWarning, match="one_vs_rest"):
        tree = SegTree(
            splitter="hist", categorical_features=[2], max_categories=8, min_leaf_abs=20
        ).fit(X)
    assert tree._cat_splitter_[2] == "one_vs_rest"
    assert replay_check(tree, X)


def test_exact_categorical_candidate_memory_checked_before_fit():
    rng = np.random.default_rng(7)
    n, n_num = 120, 300
    X = np.column_stack([rng.normal(size=(n, n_num)), rng.integers(0, 20, n).astype(float)])
    with pytest.raises(ValueError, match=r"candidate subsets.*MiB"):
        SegTree(
            splitter="exact",
            categorical_features=[n_num],
            categorical_splitter="exact",
            max_categories=32,
        ).fit(X)


def test_exact_chunked_buffers_are_result_identical(monkeypatch):
    rng = np.random.default_rng(8)
    X = np.vstack(
        [
            rng.normal(0.0, 0.4, (200, 4)),
            rng.normal(5.0, 0.4, (200, 4)),
            rng.normal(0.0, 0.4, (200, 4)) + np.array([0.0, 6.0, 0.0, 0.0]),
        ]
    )
    params = dict(splitter="exact", min_gain_mode="normalized", min_leaf_abs=20, quantiles=5)
    full = SegTree(**params).fit(X)
    monkeypatch.setattr(tree_module, "_EXACT_BUFFER_BYTES", 8 * 4 * X.shape[0] * 2)
    chunked = SegTree(**params).fit(X)
    assert np.array_equal(full.labels_, chunked.labels_)
    assert [c.threshold for c in full.cuts_] == [c.threshold for c in chunked.cuts_]
    assert len(chunked._num_blocks_) > 1


def test_exact_chunked_weighted_categorical_matches_full(monkeypatch):
    """Chunking must not mix full-feature weights into a block reduction."""
    rng = np.random.default_rng(13)
    n = 600
    X = rng.normal(size=(n, 6))
    X[:, 2] = rng.integers(0, 3, n)
    sw = rng.uniform(0.2, 2.0, n)
    params = dict(
        splitter="exact",
        min_gain_mode="normalized",
        min_leaf_abs=20,
        quantiles=5,
        categorical_features=[2],
        feature_weights=[1.0, 2.0, 1.5, 0.5, 1.0, 2.0],
    )
    full = SegTree(**params).fit(X, sample_weight=sw)
    monkeypatch.setattr(tree_module, "_EXACT_BUFFER_BYTES", 8 * 2 * n * 2)
    chunked = SegTree(**params).fit(X, sample_weight=sw)
    assert len(chunked._num_blocks_) > 1
    assert np.array_equal(full.labels_, chunked.labels_)

    def cut_key(c):
        thr = None if np.isnan(c.threshold) else c.threshold
        return (c.feature_index, c.op, c.categories, thr)

    assert [cut_key(c) for c in full.cuts_] == [cut_key(c) for c in chunked.cuts_]


def test_exact_chunked_randomized_equivalence(monkeypatch):
    """Randomized chunked-vs-full differential check (weights, cats, depths)."""
    rng = np.random.default_rng(17)
    original_bytes = tree_module._EXACT_BUFFER_BYTES
    for _ in range(5):
        n = int(rng.integers(200, 350))
        m = int(rng.integers(3, 8))
        X = rng.normal(size=(n, m))
        X[:, 1] = np.rint(rng.normal(size=n))
        params = dict(
            splitter="exact",
            min_gain_mode="normalized",
            min_leaf_abs=int(rng.integers(5, 15)),
            max_depth=int(rng.integers(2, 5)),
            quantiles=int(rng.integers(3, 6)),
            categorical_features=[1],
        )
        sw = rng.uniform(0.2, 2.0, n) if rng.random() < 0.5 else None
        if rng.random() < 0.5:
            params["feature_weights"] = rng.uniform(0.2, 2.0, m)
        monkeypatch.setattr(tree_module, "_EXACT_BUFFER_BYTES", original_bytes)
        full = SegTree(**params).fit(X, sample_weight=sw)
        monkeypatch.setattr(tree_module, "_EXACT_BUFFER_BYTES", 8 * n * 2)
        chunked = SegTree(**params).fit(X, sample_weight=sw)
        assert len(chunked._num_blocks_) > 1
        assert np.array_equal(full.labels_, chunked.labels_)

        def cut_key(c):
            thr = None if np.isnan(c.threshold) else c.threshold
            return (c.feature_index, c.op, c.categories, thr)

        assert [cut_key(c) for c in full.cuts_] == [cut_key(c) for c in chunked.cuts_]


def test_high_cardinality_categorical_avoids_quadratic_matrices():
    import tracemalloc

    k = 6000
    sizes = [130] + [1] * (k - 1)
    X = np.repeat(np.arange(k), sizes).reshape(-1, 1).astype(float)
    tracemalloc.start()
    try:
        with pytest.warns(UserWarning, match="one_vs_rest"):
            tree = SegTree(
                categorical_features=[0],
                max_leaf_nodes=2,
                max_depth=1,
                min_leaf_abs=10,
            ).fit(X)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert tree.n_leaves_ == 2
    # The old dense (k_blocks x k_cat) count table alone was ~275 MiB here.
    assert peak < 100 * 1024**2


def test_sparse_categorical_contingency_matches_dense(monkeypatch):
    rng = np.random.default_rng(19)
    n = 1200
    X = np.column_stack(
        [rng.normal(size=n), rng.integers(0, 30, n), rng.integers(0, 30, n)]
    ).astype(float)
    params = dict(
        splitter="exact",
        min_gain_mode="normalized",
        categorical_features=[1, 2],
        categorical_splitter="one_vs_rest",
        max_depth=2,
        min_gain=0.0,
        min_leaf_abs=20,
    )
    dense = SegTree(**params).fit(X)
    monkeypatch.setattr(tree_module, "_DENSE_CAT_CELLS", 100)
    sparse = SegTree(**params).fit(X)
    assert np.array_equal(dense.labels_, sparse.labels_)

    def cut_key(c):
        thr = None if np.isnan(c.threshold) else c.threshold
        return (c.feature_index, c.op, c.categories, thr)

    assert [cut_key(c) for c in dense.cuts_] == [cut_key(c) for c in sparse.cuts_]


def test_predict_encodes_only_used_categorical_columns(monkeypatch):
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 3, n)
    X = np.column_stack(
        [y * 4 + rng.normal(0, 0.5, n), rng.integers(0, 3, n), rng.integers(0, 4, n)]
    ).astype(float)
    tree = SegTree(
        categorical_features=[1, 2],
        max_depth=1,
        min_leaf_abs=20,
    ).fit(X)
    assert all(cut.op != "in" for cut in tree.cuts_)
    calls: list[object] = []
    original = SegTree._encode_input_categories

    def spy(self, Xarr, columns=None):
        calls.append("all" if columns is None else [int(c) for c in columns])
        return original(self, Xarr, columns)

    monkeypatch.setattr(SegTree, "_encode_input_categories", spy)
    tree.predict(X)
    assert calls == [[]]
    # handle_unknown="error" must still check every declared categorical column.
    strict = SegTree(
        categorical_features=[1, 2], max_depth=1, min_leaf_abs=20, handle_unknown="error"
    ).fit(X)
    calls.clear()
    strict.predict(X)
    assert calls == ["all"]


def test_random_state_is_reproducible():
    X, _ = _blobs(n=900, d=4)
    a = SegTree(splitter="hist", random_state=7, min_leaf_abs=20).fit(X)
    b = SegTree(splitter="hist", random_state=7, min_leaf_abs=20).fit(X)
    assert np.array_equal(a.labels_, b.labels_)
    assert [c.gain for c in a.cuts_] == [c.gain for c in b.cuts_]
    c = SegTree(splitter="hist", random_state=7, min_leaf_abs=20).fit(X)
    assert np.array_equal(a.labels_, c.labels_)


def test_verbose_reports_progress(capsys):
    X, _ = _blobs(n=900, d=4)
    SegTree(splitter="hist", min_leaf_abs=20, verbose=1).fit(X)
    out = capsys.readouterr().out
    assert "[segmtree] split" in out
    assert "fitted" in out


def test_stability_subsample():
    X, _ = _blobs(n=900, d=4)
    base = SegTree(splitter="hist", min_leaf_abs=20).fit(X)
    result = bootstrap_stability(base, X, n_replicates=2, subsample=0.5, random_state=0)
    assert result.n_replicates == 2
    assert result.scores.shape == (2,)
    assert np.isfinite(result.scores).all()
    with pytest.raises(ValueError, match="subsample"):
        bootstrap_stability(base, X, n_replicates=1, subsample=0.0)
    with pytest.raises(ValueError, match="subsample"):
        bootstrap_stability(base, X, n_replicates=1, subsample=10_000)


def test_root_statistics_and_min_gain_modes():
    X, _ = _blobs(n=900, d=4)
    exact = SegTree(splitter="exact", min_leaf_abs=20).fit(X)
    assert exact.h_root_ == 1.0
    assert exact.h_root_normalized_ == 1.0
    assert exact.root_variances_.shape == (4,)
    assert exact._min_gain_mode_ == "normalized"  # auto on the exact path
    hist = SegTree(splitter="hist", min_leaf_abs=20).fit(X)
    assert hist._min_gain_mode_ == "sqrt"
    for invalid in ("bogus", 3):
        with pytest.raises(ValueError, match="min_gain_mode"):
            SegTree(min_gain_mode=invalid)._check_params(4)
    assert SegTree(min_gain_mode="total").fit(X).n_leaves_ >= 1


def test_hist_weighted_path_runs_and_replays():
    X, _ = _blobs(n=900, d=4)
    rng = np.random.default_rng(11)
    sw = rng.uniform(0.5, 2.0, X.shape[0])
    tree = SegTree(
        splitter="hist",
        min_leaf_abs=20,
        feature_weights=[1.0, 2.0, 1.0, 1.0],
    ).fit(X, sample_weight=sw)
    assert tree.n_leaves_ >= 2
    assert replay_check(tree, X)
    assert np.array_equal(tree.predict(X), tree.labels_)


def test_hist_max_bins_validation():
    for bad in (1, 70_000):
        with pytest.raises(ValueError, match="max_bins"):
            SegTree(max_bins=bad)._check_params(3)


def test_hist_one_vs_rest_scores_each_candidate():
    """Categories are scored independently; the best is not always code 0."""
    sizes = [100, 100, 100, 300, 100, 100]
    codes = np.repeat(np.arange(6), sizes).reshape(-1, 1).astype(float)
    common = dict(
        categorical_features=[0],
        categorical_splitter="one_vs_rest",
        min_leaf_abs=50,
        min_gain=0.0,
        max_depth=1,
    )
    exact = SegTree(splitter="exact", min_gain_mode="normalized", **common).fit(codes)
    hist = SegTree(splitter="hist", **common).fit(codes)
    assert exact.cuts_[0].categories == (3,)
    assert hist.cuts_[0].categories == (3,)
    assert hist.cuts_[0].gain == pytest.approx(exact.cuts_[0].gain, rel=1e-9)


def test_float32_boundary_thresholds_are_representable():
    """A threshold between adjacent float32 values must not flip the split."""
    f32 = np.float32
    X = np.empty((200, 1), dtype=f32)
    X[:20, 0] = 1_000_000.0
    X[20:, 0] = np.nextafter(f32(1_000_000.0), f32(np.inf))
    for splitter in ("exact", "hist"):
        tree = SegTree(
            splitter=splitter,
            dtype=np.float32,
            min_leaf_abs=10,
            min_leaf_frac=0.0,
        ).fit(X)
        assert tree.n_leaves_ == 2
        assert np.bincount(tree.predict(X), minlength=2).tolist() == [20, 180]
        assert replay_check(tree, X)


def test_association_stable_for_large_offsets():
    rng = np.random.default_rng(0)
    n = 20_000
    z = rng.normal(size=n)
    X = np.column_stack([1e8 + z, 1e8 + 2 * z + rng.normal(0, 1, n)])
    tree = SegTree(splitter="hist", min_leaf_abs=100, min_gain=1e9).fit(X)
    # True correlation is ~0.894; squared association should be ~0.8.
    assert tree.association_[0] > 0.6
    assert tree.association_[1] > 0.6


def test_hist_periodic_column_not_aliased():
    n = 99_999
    X = (np.arange(n) % 2).astype(float).reshape(-1, 1)
    tree = SegTree(splitter="hist", max_bins=2, min_leaf_abs=50).fit(X)
    assert tree.n_leaves_ == 2
    assert tree.cuts_[0].n_left + tree.cuts_[0].n_right == n


def test_hist_imbalanced_binary_gets_a_split():
    for n0, n1, max_bins in [(490, 510, 2), (400, 19_600, 32)]:
        X = np.r_[np.zeros(n0), np.ones(n1)].reshape(-1, 1)
        tree = SegTree(splitter="hist", max_bins=max_bins, min_leaf_abs=50, min_leaf_frac=0.0).fit(
            X
        )
        assert tree.n_leaves_ == 2
        assert tree.cuts_[0].threshold == 0.5
        assert tree.cuts_[0].n_left + tree.cuts_[0].n_right == n0 + n1


def test_categorical_association_includes_unobserved_cells():
    rng = np.random.default_rng(1)
    n = 500
    a = rng.integers(0, 20, n)
    b = rng.integers(0, 20, n)
    X = np.column_stack([a, b]).astype(float)
    tree = SegTree(
        splitter="hist",
        categorical_features=[0, 1],
        categorical_splitter="one_vs_rest",
        min_leaf_abs=20,
        min_gain=1e9,
    ).fit(X)
    table = np.zeros((20, 20))
    np.add.at(table, (a, b), 1)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float(np.sum((table - expected) ** 2 / np.maximum(expected, 1e-12)))
    full_v2 = chi2 / (n * 19)
    assert tree.association_[0] == pytest.approx(full_v2, rel=1e-6)


def test_all_constant_input_has_zero_root_heterogeneity():
    X = np.ones((50, 3))
    tree = SegTree().fit(X)
    assert tree.n_leaves_ == 1
    assert tree.h_root_ == 0.0
    assert tree.h_root_normalized_ == 1.0
    assert tree.reduction_ == 0.0
    assert tree.total_gain_ == 0.0


def test_random_state_controls_association_sampling(monkeypatch):
    monkeypatch.setattr(tree_module, "_ASSOC_SAMPLE_ROWS", 100)
    X = np.random.default_rng(2).normal(size=(1000, 5))
    params = dict(splitter="hist", min_gain=1e9)
    a = SegTree(random_state=1, **params).fit(X)
    b = SegTree(random_state=2, **params).fit(X)
    c = SegTree(random_state=1, **params).fit(X)
    assert np.array_equal(a.association_, c.association_)
    assert not np.array_equal(a.association_, b.association_)


def test_bootstrap_preserves_dataframe_feature_names():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(5)
    n = 180
    frame = pd.DataFrame(
        {
            "cat_col": np.repeat([0, 1, 2], n // 3),
            "num": rng.normal(size=n),
        }
    )
    tree = SegTree(
        categorical_features=[0],
        categorical_splitter={"cat_col": "one_vs_rest"},
        min_leaf_abs=10,
    ).fit(frame)
    result = bootstrap_stability(tree, frame, n_replicates=1, random_state=0)
    assert np.isfinite(result.scores).all()


def test_subsample_controls_replicate_size(monkeypatch):
    X, _ = _blobs(n=900, d=4)
    base = SegTree(splitter="hist", min_leaf_abs=20).fit(X)
    sizes: list[int] = []

    class RecordingTree(stability_module.SegTree):
        def fit(self, X, y=None, sample_weight=None):
            sizes.append(int(np.asarray(X).shape[0]))
            return super().fit(X, y, sample_weight=sample_weight)

    monkeypatch.setattr(stability_module, "SegTree", RecordingTree)
    bootstrap_stability(base, X, n_replicates=3, subsample=120, random_state=0)
    assert sizes == [120, 120, 120]
    sizes.clear()
    bootstrap_stability(base, X, n_replicates=2, subsample=0.5, random_state=0)
    assert sizes == [450, 450]


def test_max_bins_effective_code_dtype_and_resolution():
    X, _ = _blobs(n=900, d=4)
    small = SegTree(splitter="hist", max_bins=32, min_leaf_abs=20).fit(X)
    large = SegTree(splitter="hist", max_bins=1024, min_leaf_abs=20).fit(X)
    assert small.max_bins_ == 32
    assert small._hist_code_dtype_ == np.uint8
    assert large.max_bins_ == 1024
    assert large._hist_code_dtype_ == np.uint16
    # More bins must actually produce more candidate boundaries.
    coarse = SegTree(splitter="hist", max_bins=8, min_gain=1e9).fit(X)
    fine = SegTree(splitter="hist", max_bins=64, min_gain=1e9).fit(X)
    assert len(coarse._hist_edges_[0]) <= 7
    assert len(fine._hist_edges_[0]) > len(coarse._hist_edges_[0])


def test_sample_weight_changes_the_partition():
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    group = rng.random(n) < 0.5
    X[group, 0] += 1.2
    X[~group, 1] += 1.0
    weights = np.where(group, 5.0, 0.2)
    params = dict(min_gain=0.001, min_leaf_abs=20, max_depth=3)
    plain = SegTree(**params).fit(X)
    weighted = SegTree(**params).fit(X, sample_weight=weights)
    assert not np.array_equal(plain.labels_, weighted.labels_)


def test_per_column_strategy_is_enforced_in_validation():
    rng = np.random.default_rng(3)
    n = 300
    num = rng.normal(size=(n, 1))
    codes = np.column_stack([rng.integers(0, 16, n), rng.integers(0, 3, n)])
    X = np.column_stack([num, codes]).astype(float)
    with pytest.raises(ValueError, match="max_categories"):
        SegTree(
            splitter="hist",
            categorical_features=[1, 2],
            categorical_splitter={1: "exact", 2: "auto"},
            max_categories=8,
            min_leaf_abs=10,
        ).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = SegTree(
            splitter="hist",
            categorical_features=[1, 2],
            categorical_splitter={1: "auto", 2: "one_vs_rest"},
            max_categories=8,
            min_leaf_abs=10,
        ).fit(X)
    assert tree._cat_splitter_ == {1: "one_vs_rest", 2: "one_vs_rest"}
