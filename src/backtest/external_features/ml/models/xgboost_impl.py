"""XGBoost wrapper around :class:`ForecastModel` (T-0206).

Lazy import + clear setup hint. Attribution falls back to a sidecar
note if the installed xgboost version cannot produce ``pred_contribs``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from backtest.external_features.ml.models.base import (
    ForecastModel,
    _stable_param_hash,
)


def _require_xgboost():
    try:
        import xgboost  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise RuntimeError(
            "XGBoostModel requires xgboost. Install with "
            "`poetry install --with ml` (macOS: brew install libomp)."
        ) from exc
    return xgboost


class XGBoostModel(ForecastModel):
    DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 6,
        "n_estimators": 200,
        "tree_method": "hist",
        "verbosity": 0,
    }

    def __init__(self, params: Optional[Mapping[str, Any]] = None) -> None:
        self._xgb = _require_xgboost()
        self._params: Dict[str, Any] = {**self.DEFAULT_PARAMS, **dict(params or {})}
        self._regressor = None
        self._feature_columns: list[str] = []
        self._supports_contribs = True

    @property
    def engine_code(self) -> str:
        version = getattr(self._xgb, "__version__", "0.0")
        return f"xgboost@{version}_{_stable_param_hash(self._params)}"

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> "XGBoostModel":
        self._feature_columns = list(X.columns)
        regressor = self._xgb.XGBRegressor(**self._params)
        regressor.fit(X, pd.to_numeric(y, errors="coerce"), sample_weight=sample_weight)
        self._regressor = regressor
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._regressor is None:
            raise RuntimeError("XGBoostModel.fit must be called before predict")
        return self._regressor.predict(X[self._feature_columns])

    def predict_contributions(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if self._regressor is None or not self._supports_contribs:
            return None
        try:
            booster = self._regressor.get_booster()
            dmat = self._xgb.DMatrix(X[self._feature_columns])
            return booster.predict(dmat, pred_contribs=True)
        except Exception:
            self._supports_contribs = False
            return None

    def feature_importance(self) -> Dict[str, float]:
        if self._regressor is None:
            return {}
        importances = self._regressor.feature_importances_
        total = float(sum(importances)) or 1.0
        return {col: float(val) / total for col, val in zip(self._feature_columns, importances)}

    # Pickle support: the `_xgb` module object is not picklable.
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_xgb", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._xgb = _require_xgboost()


__all__ = ["XGBoostModel"]
