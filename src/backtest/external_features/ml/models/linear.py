"""Sklearn-backed RidgeModel (T-0207).

Used for the per-ticker Stage-2 residual fit and as a fallback when
LightGBM is not installed. Lazy import keeps the core import path free
of sklearn (Codex D12).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from backtest.external_features.ml.models.base import (
    ForecastModel,
    _stable_param_hash,
)


def _require_sklearn():
    try:
        from sklearn.linear_model import Ridge  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise RuntimeError(
            "RidgeModel requires scikit-learn. Install with "
            "`poetry install --with ml`."
        ) from exc
    return Ridge


class RidgeModel(ForecastModel):
    DEFAULT_PARAMS: Dict[str, Any] = {"alpha": 1.0, "random_state": 42}

    def __init__(self, params: Optional[Mapping[str, Any]] = None) -> None:
        self._ridge_cls = _require_sklearn()
        self._params: Dict[str, Any] = {**self.DEFAULT_PARAMS, **dict(params or {})}
        self._model = None
        self._feature_columns: list[str] = []

    @property
    def engine_code(self) -> str:
        return f"ridge@{_stable_param_hash(self._params)}"

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "RidgeModel":
        self._feature_columns = list(X.columns)
        model = self._ridge_cls(**self._params)
        model.fit(X, pd.to_numeric(y, errors="coerce"), sample_weight=sample_weight)
        self._model = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RidgeModel.fit must be called before predict")
        return self._model.predict(X[self._feature_columns])

    def feature_importance(self) -> Dict[str, float]:
        if self._model is None:
            return {}
        coefs = np.asarray(self._model.coef_, dtype=float)
        if coefs.ndim > 1:
            coefs = coefs.ravel()
        total = float(np.sum(np.abs(coefs))) or 1.0
        return {col: float(abs(val)) / total for col, val in zip(self._feature_columns, coefs)}


__all__ = ["RidgeModel"]
