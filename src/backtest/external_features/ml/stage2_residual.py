"""Stage-2 per-ticker residual model (T-0209)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd

from backtest.external_features.ml.models.base import ForecastModel, MockForecastModel


@dataclass
class Stage2Result:
    residual_models: Dict[str, ForecastModel] = field(default_factory=dict)
    fallback_tickers: List[str] = field(default_factory=list)


def fit_residual_models(
    feature_frame: pd.DataFrame,
    oof_stage1_predictions: pd.Series,
    actuals: pd.Series,
    *,
    min_ticker_history_days: int = 250,
    model_class: Optional[Type[ForecastModel]] = None,
) -> Stage2Result:
    """Fit one Stage-2 model per ticker on OOF Stage-1 residuals."""

    if not {"ticker"}.issubset(feature_frame.columns):
        raise ValueError("feature_frame must contain a 'ticker' column")
    cls = model_class or MockForecastModel
    residuals = pd.to_numeric(actuals, errors="coerce") - pd.to_numeric(
        oof_stage1_predictions, errors="coerce"
    )
    feature_cols = [c for c in feature_frame.columns if c not in {"as_of", "ticker"}]
    if not feature_cols:
        return Stage2Result(residual_models={}, fallback_tickers=[])
    out = Stage2Result()
    for ticker, group in feature_frame.groupby("ticker"):
        local_residuals = residuals.loc[group.index].dropna()
        if len(local_residuals) < int(min_ticker_history_days):
            out.fallback_tickers.append(str(ticker))
            continue
        X = group.loc[local_residuals.index, feature_cols].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0)
        y = local_residuals.loc[X.index]
        try:
            model = cls()
        except TypeError:
            model = cls({})  # type: ignore[call-arg]
        model.fit(X, y)
        out.residual_models[str(ticker)] = model
    return out


def predict_residuals(
    feature_frame: pd.DataFrame,
    residual_models: Dict[str, ForecastModel],
) -> pd.Series:
    """Apply the per-ticker Stage-2 models. Unknown tickers contribute 0."""

    feature_cols = [c for c in feature_frame.columns if c not in {"as_of", "ticker"}]
    out = pd.Series(0.0, index=feature_frame.index, dtype=float)
    for ticker, group in feature_frame.groupby("ticker"):
        model = residual_models.get(str(ticker))
        if model is None:
            continue
        X = group[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        try:
            preds = model.predict(X)
        except Exception:
            continue
        out.loc[group.index] = preds
    return out


__all__ = ["Stage2Result", "fit_residual_models", "predict_residuals"]
