"""Forecast-model interface (T-0204).

Concrete implementations are lazy in their library imports so the
phase-A/B/C test surface keeps working without ``--with ml`` installed.
"""

from __future__ import annotations

import hashlib
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """Tiny abstract wrapper around a regression model.

    The contract intentionally stays small: ``fit``/``predict``/feature-
    importance plus ``save``/``load``. Implementations may keep extra
    state for ``pred_contrib``-style attributions.
    """

    @property
    @abstractmethod
    def engine_code(self) -> str:
        """Stable identifier such as ``lightgbm@4.0_a1b2c3d4``."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "ForecastModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict_contributions(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Return row-level contributions, shape ``(n_rows, n_features+1)``.

        Default implementation returns ``None`` so callers can fall back
        to a global importance bag.
        """

        return None

    def feature_importance(self) -> Dict[str, float]:
        return {}

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Path) -> "ForecastModel":
        with Path(path).open("rb") as handle:
            obj = pickle.load(handle)
        if not isinstance(obj, ForecastModel):
            raise TypeError(f"pickled object at {path} is not a ForecastModel")
        return obj


def _stable_param_hash(params: Mapping[str, Any]) -> str:
    canonical = "|".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


class MockForecastModel(ForecastModel):
    """Deterministic backup model used by tests and the synthetic adapter.

    The "fit" stores per-column means; ``predict`` returns ``X @ weights +
    bias`` where weights are derived from the feature names — no real
    learning happens, but the output is reproducible.
    """

    def __init__(self, intercept: float = 0.0) -> None:
        self._intercept = float(intercept)
        self._feature_columns: list[str] = []
        self._weights: Dict[str, float] = {}
        self._fitted = False

    @property
    def engine_code(self) -> str:
        return "mock@1.0"

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "MockForecastModel":
        self._feature_columns = list(X.columns)
        # Use a fixed pseudo-random weight pattern keyed off the feature
        # name so identical configs land at identical weights.
        digest = hashlib.sha256("|".join(self._feature_columns).encode("utf-8")).digest()
        self._weights = {
            col: ((digest[i % len(digest)] / 255.0) - 0.5) * 0.1
            for i, col in enumerate(self._feature_columns)
        }
        self._intercept = float(np.nanmean(pd.to_numeric(y, errors="coerce").dropna()))
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MockForecastModel must be fit before predict()")
        cols = [c for c in self._feature_columns if c in X.columns]
        if not cols:
            return np.full(len(X), self._intercept, dtype=float)
        weights = np.array([self._weights[c] for c in cols], dtype=float)
        values = pd.to_numeric(X[cols].stack(), errors="coerce").unstack()
        return (values.to_numpy() @ weights) + self._intercept

    def feature_importance(self) -> Dict[str, float]:
        if not self._weights:
            return {}
        total = sum(abs(v) for v in self._weights.values()) or 1.0
        return {col: abs(weight) / total for col, weight in self._weights.items()}


__all__ = ["ForecastModel", "MockForecastModel", "_stable_param_hash"]
