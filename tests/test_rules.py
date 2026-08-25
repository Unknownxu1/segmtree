from __future__ import annotations

import numpy as np
import pytest

from segmtree import SegTree, extract_rules, profile_leaves, replay_check, replay_labels


@pytest.fixture()
def fitted(mixed_x, mixed_params):
    return SegTree(**mixed_params).fit(mixed_x)


def test_rule_structure(fitted):
    rules = extract_rules(fitted)
    assert len(rules) == fitted.n_leaves_
    assert [r.leaf_id for r in rules] == list(range(fitted.n_leaves_))
    assert abs(sum(r.fraction for r in rules) - 1.0) < 1e-9
    for rule in rules:
        assert len(rule.conditions) == rule.depth
        assert all(c.op in ("<=", ">") for c in rule.conditions)


def test_rule_text_formatting(fitted):
    rules = extract_rules(fitted)
    joined = "\n".join(str(r) for r in rules)
    assert " AND " in joined or len(rules) == 1
    assert "#" in joined
    binary_conditions = [
        c for r in rules for c in r.conditions if c.feature.startswith("x") and c.threshold == 0.5
    ]
    if binary_conditions:
        assert any(str(c).endswith("0.5") for c in binary_conditions)


def test_replay_matches_predict_on_training_data(fitted, mixed_x):
    assert replay_check(fitted, mixed_x)


def test_replay_matches_predict_on_fresh_points(fitted):
    rng = np.random.default_rng(5)
    fresh = rng.normal(size=(400, 7))
    labels_pred = fitted.predict(fresh)
    labels_replay = replay_labels(fitted, fresh)
    assert np.array_equal(labels_pred, labels_replay)
    assert (labels_replay >= 0).all()  # every point falls into exactly one rule region


def test_profiles_structure(fitted, mixed_x):
    profiles = profile_leaves(fitted, mixed_x, top_k=3)
    assert len(profiles) == fitted.n_leaves_
    total = sum(p["fraction"] for p in profiles)
    assert abs(total - 1.0) < 1e-9
    for p in profiles:
        assert len(p["top_features"]) == 3
        names, zs = zip(*p["top_features"])
        assert all(isinstance(nm, str) for nm in names)
        order = np.argsort(-np.abs(zs))
        assert list(order) == [0, 1, 2]  # sorted by descending |z|


def test_unfitted_tree_raises():
    tree = SegTree()
    with pytest.raises(RuntimeError, match="not fitted"):
        extract_rules(tree)
    X = np.zeros((10, 2))
    for fn in (replay_labels, replay_check):
        with pytest.raises(RuntimeError, match="not fitted"):
            fn(tree, X)
    with pytest.raises(RuntimeError, match="not fitted"):
        profile_leaves(tree, X)
