"""LightGBM wrapper around :class:`ForecastModel` (T-0205).

Lazy import keeps the package usable without the optional ``ml``
extras. Attempting to instantiate without ``lightgbm`` installed raises
``RuntimeError`` with a clear setup hint.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from backtest.external_features.ml.models.base import (
    ForecastModel,
    _stable_param_hash,
)


def _require_lightgbm():
    try:
        import lightgbm  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise RuntimeError(
            "LightGBMModel requires lightgbm. Install with "
            "`poetry install --with ml`."
        ) from exc
    return lightgbm


class LightGBMModel(ForecastModel):
    DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 20,
        "verbose": -1,
    }

    def __init__(self, params: Optional[Mapping[str, Any]] = None) -> None:
        self._lgb = _require_lightgbm()
        self._params: Dict[str, Any] = {**self.DEFAULT_PARAMS, **dict(params or {})}
        self._booster = None
        self._feature_columns: list[str] = []

    @property
    def engine_code(self) -> str:
        version = getattr(self._lgb, "__version__", "0.0")
        return f"lightgbm@{version}_{_stable_param_hash(self._params)}"

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "LightGBMModel":
        self._feature_columns = list(X.columns)
        model = self._lgb.LGBMRegressor(**self._params)
        model.fit(X, pd.to_numeric(y, errors="coerce"), sample_weight=sample_weight)
        self._booster = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("LightGBMModel.fit must be called before predict")
        return self._booster.predict(X[self._feature_columns])

    def predict_contributions(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self._booster is None:
            return None
        booster = self._booster.booster_
        return booster.predict(X[self._feature_columns], pred_contrib=True)

    def feature_importance(self) -> Dict[str, float]:
        if self._booster is None:
            return {}
        importances = self._booster.feature_importances_
        total = float(sum(importances)) or 1.0
        return {col: float(val) / total for col, val in zip(self._feature_columns, importances)}

    # Pickle support: the `_lgb` module object is not picklable.
    # We remove it on dump and re-import it on load.
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lgb", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lgb = _require_lightgbm()


__all__ = ["LightGBMModel"]
