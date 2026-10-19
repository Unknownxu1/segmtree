"""segmtree: interpretable unsupervised decision trees for segmentation.

Partitions rows of a feature matrix into segments described by short,
human-readable rule chains, without any target variable. See
:class:`SegTree` for the estimator and the README for a full walkthrough.
"""

from __future__ import annotations

from ._base import NotFittedError
from .rules import (
    Condition,
    LeafProfile,
    Rule,
    extract_rules,
    profile_leaves,
    replay_check,
    replay_labels,
    rule_mask,
)
from .stability import StabilityResult, adjusted_rand_score, bootstrap_stability
from .tree import Cut, Leaf, SegTree

__version__ = "0.6.0"

__all__ = [
    "Condition",
    "Cut",
    "Leaf",
    "LeafProfile",
    "NotFittedError",
    "Rule",
    "SegTree",
    "StabilityResult",
    "__version__",
    "adjusted_rand_score",
    "bootstrap_stability",
    "extract_rules",
    "profile_leaves",
    "replay_check",
    "replay_labels",
    "rule_mask",
]
