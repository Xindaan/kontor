"""Phase E4 — `weighting` parameter in MLForecastTilt (T-0327, T-0329)."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_strategy_module():
    spec = importlib.util.spec_from_file_location(
        "ml_forecast_tilt", Path("strategies/ml_forecast_tilt.py").resolve()
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeSnapshot:
    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df


class _FakeProvider:
    """Provides deterministic ML scores so top-N even works without a real
    adapter."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def snapshot_dataset(self, dataset, *, as_of, tickers):
        rows = []
        for t in tickers:
            score = self._scores.get(str(t).upper(), 0.0)
            rows.append(
                {
                    "ticker": str(t).upper(),
                    "feature_name": "ml_forecast_score",
                    "feature_value": float(score),
                }
            )
        return _FakeSnapshot(pd.DataFrame(rows))


def _attach_provider(strat, scores: dict[str, float]) -> None:
    """Attaches a FakeProvider to the strategy."""
    strat._external_features_provider = _FakeProvider(scores)


def _make_price_panel(seed: int = 0, n: int = 252) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"]
    data = {}
    for i, c in enumerate(cols):
        # Different vol levels, so ERC and inverse vol aren't trivial.
        vol = 0.005 + 0.002 * i
        rets = rng.normal(0.0005, vol, n)
        data[c] = (1.0 + rets).cumprod() * 100.0
    return pd.DataFrame(data, index=pd.bdate_range("2024-01-01", periods=n))


def test_strategy_weighting_equal_is_phase_d_bit_identical():
    """weighting='equal' produces the same allocation as Phase D.

    Without an ML provider, the strategy falls back to equal-weight
    over the entire universe (Phase D behavior).
    """
    mod = _load_strategy_module()
    strat = mod.MLForecastTilt(weighting="equal", top_n=3)
    prices = _make_price_panel()
    alloc = strat.signal(date(2024, 12, 1), prices)
    weights = dict(alloc.weights)
    assert weights, "fallback must produce non-empty allocation"
    # all weights identical and sum = 1.0
    sum_w = sum(weights.values())
    assert abs(sum_w - 1.0) < 1e-9
    expected = 1.0 / len(weights)
    for w in weights.values():
        assert abs(w - expected) < 1e-9


def test_strategy_weighting_inverse_vol_differs_from_equal():
    mod = _load_strategy_module()
    scores = {"SPY": 0.9, "QQQ": 0.8, "IWM": 0.7, "EFA": 0.1, "EEM": 0.0,
              "TLT": -0.1, "GLD": -0.2}
    eq = mod.MLForecastTilt(weighting="equal", top_n=3)
    iv = mod.MLForecastTilt(weighting="inverse_vol", top_n=3)
    _attach_provider(eq, scores)
    _attach_provider(iv, scores)
    prices = _make_price_panel()
    alloc_eq = eq.signal(date(2024, 12, 1), prices)
    alloc_iv = iv.signal(date(2024, 12, 1), prices)
    assert alloc_eq.weights and alloc_iv.weights
    diffs = [
        abs(alloc_eq.weights.get(t, 0.0) - alloc_iv.weights.get(t, 0.0))
        for t in set(alloc_eq.weights) | set(alloc_iv.weights)
    ]
    assert max(diffs) > 1e-3


def test_strategy_weighting_erc_converges():
    mod = _load_strategy_module()
    scores = {"SPY": 0.9, "QQQ": 0.8, "IWM": 0.7}
    strat = mod.MLForecastTilt(weighting="erc", top_n=3, target_sum=1.0)
    _attach_provider(strat, scores)
    prices = _make_price_panel()
    alloc = strat.signal(date(2024, 12, 1), prices)
    assert alloc.weights, "erc must produce non-empty allocation"
    total = sum(alloc.weights.values())
    assert abs(total - 1.0) < 1e-4


def test_strategy_weighting_invalid_raises():
    mod = _load_strategy_module()
    with pytest.raises(ValueError, match="weighting"):
        mod.MLForecastTilt(weighting="banana")


def test_strategy_weighting_erc_with_target_sum_partial():
    mod = _load_strategy_module()
    scores = {"SPY": 0.9, "QQQ": 0.8, "IWM": 0.7}
    strat = mod.MLForecastTilt(weighting="erc", top_n=3, target_sum=0.8)
    _attach_provider(strat, scores)
    prices = _make_price_panel()
    alloc = strat.signal(date(2024, 12, 1), prices)
    assert alloc.weights
    total = sum(alloc.weights.values())
    assert abs(total - 0.8) < 1e-4
