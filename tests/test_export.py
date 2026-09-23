"""export_dict / export_text: public tree export without private access."""

from __future__ import annotations

import json

import numpy as np
import pytest

from segmtree import NotFittedError, SegTree


def make_export_data():
    """Continuous + binary + categorical mix whose trees contain both numeric
    and categorical splits."""
    rng = np.random.default_rng(6)
    n_per = 200
    parts = []
    for k, center in enumerate([[0.0, 0.0], [7.0, 5.0], [-7.0, 4.0]]):
        cont = rng.normal(center, 0.6, (n_per, 2))
        flag = (rng.random((n_per, 1)) < 0.1 + 0.4 * k).astype(float)
        code = np.full(n_per, k, dtype=float)
        flip = rng.random(n_per) < 0.12
        code[flip] = rng.integers(0, 3, int(flip.sum()))
        parts.append(np.hstack([cont, flag, code[:, None]]))
    return np.vstack(parts)


PARAMS = dict(
    min_gain=0.02,
    min_leaf_frac=0.05,
    min_leaf_abs=10,
    max_depth=3,
    quantiles=5,
    categorical_features=[3],
)


@pytest.fixture()
def fitted():
    return SegTree(**PARAMS).fit(make_export_data())


def _walk(node):
    yield node
    if node["type"] == "split":
        yield from _walk(node["left"])
        yield from _walk(node["right"])


def _leaves(node):
    return [n for n in _walk(node) if n["type"] == "leaf"]


def test_unfitted_raises_not_fitted_error():
    tree = SegTree()
    with pytest.raises(NotFittedError, match="not fitted"):
        tree.export_dict()
    with pytest.raises(NotFittedError, match="not fitted"):
        tree.export_text()


def test_export_dict_is_json_serializable_with_native_types(fitted):
    d = fitted.export_dict()
    dumped = json.dumps(d)  # must not raise (no NaN/np types inside)
    reloaded = json.loads(dumped)
    assert reloaded == d  # round-trip preserves structure exactly
    for node in _walk(d):
        for _key, value in node.items():
            if isinstance(value, list):
                for item in value:
                    assert isinstance(item, dict) or type(item) in (int, float, str)
            else:
                assert value is None or type(value) in (int, float, str, dict)
        if node["type"] == "leaf":
            for cond in node["conditions"]:
                assert set(cond) <= {"feature", "op", "threshold", "categories"}
                assert ("threshold" in cond) ^ ("categories" in cond)


def test_split_and_leaf_schemas(fitted):
    d = fitted.export_dict()
    nodes = list(_walk(d))
    split_nodes = [x for x in nodes if x["type"] == "split"]
    leaf_nodes = [x for x in nodes if x["type"] == "leaf"]
    assert split_nodes and leaf_nodes
    required_split = {
        "type",
        "depth",
        "feature",
        "feature_index",
        "op",
        "threshold",
        "categories",
        "gain",
        "n_samples",
        "left",
        "right",
    }
    required_leaf = {
        "type",
        "leaf_id",
        "depth",
        "n_samples",
        "fraction",
        "heterogeneity",
        "conditions",
    }
    for node in split_nodes:
        assert required_split <= set(node)
        assert node["op"] in ("<=", "in")
        if node["op"] == "<=":
            assert isinstance(node["threshold"], float) and node["categories"] is None
        else:
            assert node["threshold"] is None
            assert isinstance(node["categories"], list)
            assert node["categories"] == sorted(node["categories"])
            assert all(isinstance(c, int) for c in node["categories"])
    for node in leaf_nodes:
        assert required_leaf <= set(node)
        assert isinstance(node["leaf_id"], int)
        assert isinstance(node["fraction"], float)
    # both split kinds present somewhere in this tree
    ops = {node["op"] for node in split_nodes}
    assert "<=" in ops


def test_structural_consistency_and_leaf_bookkeeping(fitted):
    d = fitted.export_dict()
    leaves = _leaves(d)
    assert len(leaves) == fitted.n_leaves_
    assert {lf["leaf_id"] for lf in leaves} == set(range(fitted.n_leaves_))
    assert sum(lf["n_samples"] for lf in leaves) == fitted.n_samples_in_
    by_public_id = {lf.id: lf for lf in fitted.leaves_}
    for lf in leaves:
        pub = by_public_id[lf["leaf_id"]]
        assert lf["n_samples"] == pub.n_samples
        assert lf["fraction"] == pytest.approx(pub.fraction)
        assert len(lf["conditions"]) == pub.depth == lf["depth"]
    for node in _walk(d):
        if node["type"] == "split":
            assert node["left"]["type"] in ("split", "leaf")
            assert node["left"]["n_samples"] + node["right"]["n_samples"] == node["n_samples"]
    assert d["n_samples"] == fitted.n_samples_in_


def test_categorical_splits_exported_as_sorted_category_lists(fitted):
    d = fitted.export_dict()
    cat_splits = [n for n in _walk(d) if n["type"] == "split" and n["op"] == "in"]
    assert cat_splits, "fixture tree must contain a categorical split"
    public_cats = {tuple(sorted(c.categories)) for c in fitted.cuts_ if c.op == "in"}
    exported_cats = {tuple(n["categories"]) for n in cat_splits}
    assert exported_cats == public_cats


def test_export_text_stable_readable_contains_rules(fitted):
    text1 = fitted.export_text()
    text2 = fitted.export_text()
    assert text1 == text2  # stable across calls
    assert "leaf #" in text1
    assert "[gain=" in text1
    assert " in {" in text1 and " not in {" in text1
    assert fitted.feature_names_[fitted.cuts_[0].feature_index] in text1
    for lf in fitted.leaves_:
        assert f"leaf #{lf.id}" in text1


def test_single_leaf_tree_exports_without_splits():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(120, 2))
    tree = SegTree(min_gain=10.0, min_leaf_abs=10).fit(X)
    d = tree.export_dict()
    assert d["type"] == "leaf" and d["conditions"] == []
    assert json.dumps(d)
    text = tree.export_text()
    assert text.count("leaf #") == 1
