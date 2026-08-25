"""Shared infrastructure: sklearn-compatible base class and input validation.

The package's only runtime dependency is numpy. When scikit-learn is
installed we reuse its ``BaseEstimator`` for full ecosystem compatibility
(``clone``, pipelines, hyperparameter search); otherwise a minimal,
API-compatible fallback is used.
"""

from __future__ import annotations

import inspect

import numpy as np

__all__ = ["BaseEstimator", "as_2d_array"]

try:  # pragma: no cover - exercised implicitly depending on environment
    from sklearn.base import BaseEstimator

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover

    class BaseEstimator:
        """Minimal stand-in for :class:`sklearn.base.BaseEstimator`."""

        def get_params(self, deep=True):
            params = {}
            for name, p in inspect.signature(type(self).__init__).parameters.items():
                if name == "self" or p.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                params[name] = getattr(self, name)
            return params

        def set_params(self, **params):
            valid = self.get_params(deep=False)
            for key, value in params.items():
                if key not in valid:
                    raise ValueError(
                        f"Invalid parameter {key!r} for {type(self).__name__}. "
                        f"Valid parameters are: {sorted(valid)}."
                    )
                setattr(self, key, value)
            return self

        def __repr__(self):
            parts = []
            for name in self.get_params(deep=False):
                try:
                    default = inspect.signature(type(self).__init__).parameters[name].default
                    current = getattr(self, name)
                    equal = (
                        bool(np.all(current == default))
                        if isinstance(current, np.ndarray)
                        else (current == default)
                    )
                except Exception:
                    equal = True
                if not equal:
                    parts.append(f"{name}={getattr(self, name)!r}")
            return f"{type(self).__name__}({', '.join(parts)})"


def as_2d_array(X, *, context="X"):
    """Coerce tabular input to a 2-D finite numeric ndarray.

    Accepts anything ``np.asarray`` understands plus objects exposing
    ``to_numpy`` (e.g. pandas DataFrames). Returns ``(array, column_names_or_None)``.
    """
    columns = None
    if hasattr(X, "to_numpy") and hasattr(X, "columns"):
        columns = [str(c) for c in X.columns]
        X = X.to_numpy()
    arr = np.asarray(X)
    if arr.dtype == object:
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError) as e:
            raise TypeError(f"{context} must be numeric; got dtype=object ({e})") from e
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{context} must have a numeric dtype, got {arr.dtype}.")
    if arr.ndim != 2:
        raise ValueError(f"{context} must be 2-D (n_samples, n_features), got shape {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{context} must not be empty, got shape {arr.shape}.")
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise ValueError(
            f"{context} contains {n_bad} NaN/infinite value(s). "
            "Impute or drop them before fitting (segmtree does no imputation)."
        )
    return arr, columns
