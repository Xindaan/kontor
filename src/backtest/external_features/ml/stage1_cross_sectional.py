"""Stage-1 cross-sectional fit + OOF prediction (T-0208)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Type

import numpy as np
import pandas as pd

from backtest.external_features.ml.models.base import ForecastModel, MockForecastModel
from backtest.external_features.ml.splits import PurgedDateSplit


@dataclass
class Stage1Result:
    model: ForecastModel
    oof_predictions: pd.Series
    fold_metrics: List[Dict[str, float]] = field(default_factory=list)


def fit_cross_sectional(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_class: Optional[Type[ForecastModel]] = None,
    hyperparams: Optional[Dict[str, object]] = None,
) -> ForecastModel:
    """Train the final cross-sectional model on the full window."""

    cls = model_class or MockForecastModel
    if hyperparams:
        try:
            instance = cls(hyperparams)  # type: ignore[call-arg]
        except TypeError:
            instance = cls()
    else:
        instance = cls()
    instance.fit(X, y)
    return instance


def predict_oof(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    as_of: Sequence,
    label_ends: pd.DataFrame,
    n_splits: int = 5,
    horizon_key: Optional[str] = None,
    model_class: Optional[Type[ForecastModel]] = None,
    hyperparams: Optional[Dict[str, object]] = None,
) -> Stage1Result:
    """Run :class:`PurgedDateSplit` and return out-of-fold predictions."""

    if len(X) != len(y) or len(X) != len(as_of) or len(X) != len(label_ends):
        raise ValueError("X, y, as_of and label_ends must align")

    splitter = PurgedDateSplit(n_splits=n_splits, horizon_key=horizon_key)
    oof = pd.Series(np.nan, index=X.index, dtype=float)
    fold_metrics: List[Dict[str, float]] = []
    final_model: Optional[ForecastModel] = None
    for fold, (train_idx, test_idx) in enumerate(splitter.split(as_of, label_ends)):
        model = fit_cross_sectional(
            X.iloc[train_idx],
            y.iloc[train_idx],
            model_class=model_class,
            hyperparams=hyperparams,
        )
        preds = model.predict(X.iloc[test_idx])
        oof.iloc[test_idx] = preds
        fold_metrics.append(
            {
                "fold": fold,
                "rmse": float(np.sqrt(np.mean((y.iloc[test_idx].to_numpy() - preds) ** 2))),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
        final_model = model
    if final_model is None:
        # No fold was usable (e.g. too little data) — fall back to a
        # full-sample fit so callers always get a model back.
        final_model = fit_cross_sectional(
            X, y, model_class=model_class, hyperparams=hyperparams
        )
    return Stage1Result(model=final_model, oof_predictions=oof, fold_metrics=fold_metrics)


__all__ = ["Stage1Result", "fit_cross_sectional", "predict_oof"]
