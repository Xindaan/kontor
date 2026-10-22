"""Stage-3 stacking meta-model (T-0210, Codex D22/D23).

Inputs: ``combined_h = stage1_oof_h + stage2_oof_h`` for h in 21/63/252.
Target: ``mean(z_21, z_63, z_252)`` where ``z_h`` standardises the
forward return at horizon ``h`` using TRAIN-FOLD-ONLY mean/std.

The Stage-3 frame only keeps rows where all three horizons are
populated (Codex D23). The fitted z-score statistics get stored in the
bundle as ``zscore_stats.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import math

import numpy as np
import pandas as pd

from backtest.external_features.ml.models.base import ForecastModel, MockForecastModel


HORIZON_KEYS = (21, 63, 252)


@dataclass
class ZScoreStats:
    means: Dict[int, float] = field(default_factory=dict)
    stds: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "means": {str(k): float(v) for k, v in self.means.items()},
            "stds": {str(k): float(v) for k, v in self.stds.items()},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Dict[str, object]]) -> "ZScoreStats":
        means = {int(k): float(v) for k, v in (payload.get("means") or {}).items()}
        stds = {int(k): float(v) for k, v in (payload.get("stds") or {}).items()}
        return cls(means=means, stds=stds)

    def standardize(self, horizon: int, value: float) -> float:
        mean = self.means.get(int(horizon), 0.0)
        std = self.stds.get(int(horizon), 1.0)
        if not math.isfinite(std) or std <= 0:
            return 0.0
        return (float(value) - mean) / std


@dataclass
class Stage3Result:
    model: ForecastModel
    zscore_stats: ZScoreStats
    rows_used: int = 0


def fit_z_score_stats(forward_returns: pd.DataFrame) -> ZScoreStats:
    """Fit per-horizon mean and std on the training fold only."""

    stats = ZScoreStats()
    for horizon in HORIZON_KEYS:
        col = f"horizon_{horizon}d"
        if col not in forward_returns.columns:
            continue
        values = pd.to_numeric(forward_returns[col], errors="coerce").dropna()
        if values.empty:
            continue
        stats.means[horizon] = float(values.mean())
        stats.stds[horizon] = float(values.std(ddof=0)) or 1.0
    return stats


def build_stacking_target(
    forward_returns: pd.DataFrame,
    stats: ZScoreStats,
) -> pd.Series:
    """Compute ``mean(z_21, z_63, z_252)`` row-wise."""

    parts = []
    for horizon in HORIZON_KEYS:
        col = f"horizon_{horizon}d"
        if col not in forward_returns.columns:
            return pd.Series(np.nan, index=forward_returns.index, dtype=float)
        std = stats.stds.get(horizon, 1.0) or 1.0
        mean = stats.means.get(horizon, 0.0)
        parts.append((pd.to_numeric(forward_returns[col], errors="coerce") - mean) / std)
    target = pd.concat(parts, axis=1).mean(axis=1)
    return target


def fit_stacking_model(
    combined_oof: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    model_class: Optional[Type[ForecastModel]] = None,
) -> Stage3Result:
    """Fit the meta-model on rows with all three horizons present."""

    required = {f"combined_{h}d" for h in HORIZON_KEYS}
    missing = required - set(combined_oof.columns)
    if missing:
        raise ValueError(f"combined_oof missing columns: {sorted(missing)}")
    # Codex D23: keep rows where all three forward-returns are observed.
    target_cols = [f"horizon_{h}d" for h in HORIZON_KEYS]
    available = forward_returns.dropna(subset=target_cols)
    aligned = combined_oof.loc[available.index].dropna()
    available = available.loc[aligned.index]
    if aligned.empty:
        cls = model_class or MockForecastModel
        return Stage3Result(model=cls(), zscore_stats=ZScoreStats(), rows_used=0)

    stats = fit_z_score_stats(available)
    target = build_stacking_target(available, stats)
    cls = model_class or MockForecastModel
    try:
        model = cls()
    except TypeError:
        model = cls({})  # type: ignore[call-arg]
    feature_cols = [f"combined_{h}d" for h in HORIZON_KEYS]
    model.fit(aligned[feature_cols], target)
    return Stage3Result(model=model, zscore_stats=stats, rows_used=int(len(aligned)))


def predict_stacking(
    combined: pd.DataFrame,
    model: ForecastModel,
) -> np.ndarray:
    feature_cols = [f"combined_{h}d" for h in HORIZON_KEYS]
    return model.predict(combined[feature_cols])


__all__ = [
    "HORIZON_KEYS",
    "Stage3Result",
    "ZScoreStats",
    "build_stacking_target",
    "fit_stacking_model",
    "fit_z_score_stats",
    "predict_stacking",
]
