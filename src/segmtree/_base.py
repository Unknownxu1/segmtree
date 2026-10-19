"""Shared infrastructure: sklearn-compatible base class and input validation.

The package's only runtime dependency is numpy. When scikit-learn is
installed we reuse its ``BaseEstimator`` and ``NotFittedError`` for full
ecosystem compatibility (``clone``, pipelines, hyperparameter search);
otherwise minimal API-compatible fallbacks are used.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "BaseEstimator",
    "NotFittedError",
    "as_2d_array",
    "check_is_fitted",
    "format_threshold",
]

try:
    from sklearn.base import BaseEstimator
    from sklearn.exceptions import NotFittedError

    _HAS_SKLEARN = True
except ImportError:

    class NotFittedError(ValueError, AttributeError):
        """Raised when an estimator is used before being fitted.

        Mirrors :class:`sklearn.exceptions.NotFittedError` so the same
        ``except`` clause works with or without scikit-learn installed.
        """

    class BaseEstimator:
        """Minimal stand-in for :class:`sklearn.base.BaseEstimator`."""

        def get_params(self, deep: bool = True) -> dict:
            params = {}
            for name, p in inspect.signature(type(self).__init__).parameters.items():
                if name == "self" or p.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                params[name] = getattr(self, name)
            return params

        def set_params(self, **params: Any) -> BaseEstimator:
            valid = self.get_params(deep=False)
            for key, value in params.items():
                if key not in valid:
                    raise ValueError(
                        f"Invalid parameter {key!r} for {type(self).__name__}. "
                        f"Valid parameters are: {sorted(valid)}."
                    )
                setattr(self, key, value)
            return self

        def __repr__(self) -> str:
            parts = []
            for name in self.get_params(deep=False):
                try:
                    default = inspect.signature(type(self).__init__).parameters[name].default
                    current = getattr(self, name)
                    if isinstance(current, np.ndarray):
                        equal = bool(np.array_equal(current, default))
                    elif default is inspect.Parameter.empty:
                        equal = False
                    else:
                        equal = current == default
                        if isinstance(equal, np.ndarray):
                            equal = bool(np.all(equal))
                except Exception:
                    equal = True
                if not equal:
                    parts.append(f"{name}={getattr(self, name)!r}")
            return f"{type(self).__name__}({', '.join(parts)})"

    _HAS_SKLEARN = False


def check_is_fitted(estimator: Any, attribute: str = "n_leaves_") -> None:
    """Raise :class:`NotFittedError` if ``estimator`` lacks ``attribute``."""
    if not hasattr(estimator, attribute):
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' with appropriate arguments before using this estimator."
        )


def format_threshold(value: float) -> str:
    """Render a threshold compactly (integers without decimal point)."""
    value = float(value)
    if value.is_integer() and abs(value) < 2**53:
        return str(int(value))
    return f"{value:g}"


def as_2d_array(X: ArrayLike, *, context: str = "X") -> tuple[NDArray[np.floating], list | None]:
    """Coerce tabular input to a 2-D finite real numeric ndarray.

    Accepts anything ``np.asarray`` understands plus objects exposing
    ``to_numpy`` (e.g. pandas DataFrames). Returns
    ``(array, column_names_or_None)``.

    Raises
    ------
    TypeError
        If the input has complex dtype, object dtype that cannot be cast to
        float, or a non-numeric dtype. Masked arrays are rejected because
        their mask would be silently ignored.
    ValueError
        If the input is not 2-D, is empty, or contains NaN/infinite values.
    """
    if isinstance(X, np.ma.MaskedArray):
        raise TypeError(
            f"{context} must not be a numpy masked array (the mask would be "
            "silently ignored). Pass the underlying data with missing values "
            "imputed or dropped."
        )
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
    if np.issubdtype(arr.dtype, np.complexfloating):
        raise TypeError(
            f"{context} must be real-valued; complex dtypes are not supported "
            "(got {arr.dtype}). Drop the imaginary part explicitly if intended."
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{context} must have a numeric dtype, got {arr.dtype}.")
    if arr.ndim != 2:
        raise ValueError(f"{context} must be 2-D (n_samples, n_features), got shape {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{context} must not be empty, got shape {arr.shape}.")
    if np.issubdtype(arr.dtype, np.floating):
        # Chunked so a valid n x F matrix never allocates a full boolean
        # mask (n x F bytes) just for validation.
        n_bad = 0
        chunk_rows = max(1, 1_000_000 // max(1, arr.shape[1]))
        for start in range(0, arr.shape[0], chunk_rows):
            finite = np.isfinite(arr[start : start + chunk_rows])
            if not finite.all():
                n_bad += int((~finite).sum())
        if n_bad:
            raise ValueError(
                f"{context} contains {n_bad} NaN/infinite value(s). "
                "Impute or drop them before fitting (segmtree does no imputation)."
            )
    return arr, columns
