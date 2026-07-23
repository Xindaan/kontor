"""Tests for the analyst_momentum_filter demo strategy (Phase B T-0061)."""

from __future__ import annotations

from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from backtest.external_features.loader import ExternalFeatureSnapshot
from backtest.strategy import Allocation
from strategies.analyst_momentum_filter import AnalystMomentumFilter


class _FakeProvider:
    def __init__(self, score_by_ticker):
        self._scores = score_by_ticker

    def snapshot(self, as_of, tickers=None):  # noqa: ARG002
        rows = []
        for ticker, score in self._scores.items():
            rows.append(
                {
                    "ticker": ticker,
                    "release_date": pd.Timestamp("2026-05-01"),
                    "snapshot_ts": pd.Timestamp("2026-05-13T12:00:00Z"),
                    "feature_name": "analyst_score",
                    "feature_value": float(score),
                    "source": "Test",
                    "dataset": "test_ds",
                }
            )
        return ExternalFeatureSnapshot(
            as_of=as_of, dataset="test_ds", data=pd.DataFrame(rows)
        )


def _prices(tickers, length=200, slope_by_ticker=None):
    slope_by_ticker = slope_by_ticker or {}
    idx = pd.date_range(end=pd.Timestamp("2026-05-01"), periods=length, freq="B")
    data = {}
    for i, ticker in enumerate(tickers):
        slope = slope_by_ticker.get(ticker, 0.001 * (i + 1))
        data[ticker] = 100.0 * np.exp(np.linspace(0, slope * length, length))
    return pd.DataFrame(data, index=idx)


def test_strategy_runs_without_provider():
    strategy = AnalystMomentumFilter(universe=["A", "B", "C"], top_n=2, lookback=60)
    prices = _prices(["A", "B", "C"], slope_by_ticker={"A": 0.001, "B": 0.003, "C": 0.0005})
    out = strategy.signal(date(2026, 5, 1), prices)
    assert isinstance(out, Allocation)
    # Top-2 momentum picks (B and A): both kept since no provider.
    assert set(out.weights.keys()).issubset({"A", "B", "C"})
    assert sum(out.weights.values()) <= 1.0 + 1e-9


def test_strategy_filters_by_analyst_cutoff_with_provider():
    strategy = AnalystMomentumFilter(
        universe=["A", "B", "C"],
        top_n=3,
        lookback=60,
        analyst_cutoff=0.1,
    )
    strategy._external_features_provider = _FakeProvider({"A": 0.5, "B": -0.4, "C": 0.2})
    prices = _prices(["A", "B", "C"], slope_by_ticker={"A": 0.001, "B": 0.003, "C": 0.0005})
    out = strategy.signal(date(2026, 5, 1), prices)
    # B has negative analyst score and must be filtered out.
    assert "B" not in out.weights


def test_strategy_fallback_when_all_below_cutoff():
    strategy = AnalystMomentumFilter(
        universe=["A", "B", "C"],
        top_n=2,
        lookback=60,
        analyst_cutoff=0.99,
    )
    strategy._external_features_provider = _FakeProvider({"A": 0.1, "B": -0.2, "C": 0.0})
    prices = _prices(["A", "B", "C"])
    out = strategy.signal(date(2026, 5, 1), prices)
    # All below cutoff -> keep the candidate set (top-N momentum).
    assert len(out.weights) > 0


def test_strategy_missing_provider_data_neutral():
    strategy = AnalystMomentumFilter(
        universe=["A", "B"], top_n=2, lookback=60, analyst_cutoff=0.0
    )
    strategy._external_features_provider = _FakeProvider({"A": 0.0})  # B missing
    prices = _prices(["A", "B"], slope_by_ticker={"A": 0.001, "B": 0.0005})
    out = strategy.signal(date(2026, 5, 1), prices)
    # B's missing score is treated as 0 (neutral), passes cutoff>=0.
    assert "B" in out.weights
