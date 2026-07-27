"""Phase D SignalGenerator ml suffix tests (T-0226)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from backtest.signals import SignalGenerator


class _Snap:
    def __init__(self, df: pd.DataFrame):
        self.data = df


class _FakeProvider:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def snapshot_dataset(self, dataset, *, as_of, tickers):
        return _Snap(self._df)


def _bull_neutral_bear_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "AAA", "feature_name": "ml_forecast_score",
             "feature_value": 0.5},
            {"ticker": "BBB", "feature_name": "ml_forecast_score",
             "feature_value": 0.1},
            {"ticker": "CCC", "feature_name": "ml_forecast_score",
             "feature_value": -0.5},
        ]
    )


def test_collect_ml_suffixes_returns_mapping():
    class _Strat:
        assets = ["AAA"]
        rebalance_frequency = "monthly"
        display_name = "fake"

    gen = SignalGenerator(
        _Strat(),
        external_features_provider=_FakeProvider(_bull_neutral_bear_frame()),
        ml_dataset="synthetic_ml_forecast",
    )
    suffixes = gen._collect_ml_suffixes(date(2026, 5, 13), ["AAA", "BBB", "CCC"])
    assert suffixes["AAA"].startswith("ml: BULLISH")
    assert suffixes["BBB"].startswith("ml: NEUTRAL")
    assert suffixes["CCC"].startswith("ml: BEARISH")


def test_collect_ml_suffixes_empty_without_dataset():
    class _Strat:
        assets = ["AAA"]
        rebalance_frequency = "monthly"
        display_name = "fake"

    gen = SignalGenerator(
        _Strat(),
        external_features_provider=_FakeProvider(_bull_neutral_bear_frame()),
        ml_dataset=None,
    )
    suffixes = gen._collect_ml_suffixes(date(2026, 5, 13), ["AAA"])
    assert suffixes == {}
