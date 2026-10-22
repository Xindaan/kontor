"""Tests for FinnhubAnalystTrendsAdapter (Phase B+ T-0080).

The trends adapter emits one snapshot row per (ticker, period) using the
period field itself as release_date. Future periods (period > as_of) are
filtered out so PIT semantics hold.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from backtest.external_features.adapters.finnhub_analyst_trends import (
    DATASET_ID,
    FinnhubAnalystTrendsAdapter,
)
from backtest.external_features.analyst_schema import validate_analyst_snapshot
from backtest.provenance import ManualDataProvenanceRegistry

pytestmark = pytest.mark.no_network


class _FakeClient:
    def __init__(self, trends_by_ticker):
        self._trends = trends_by_ticker

    def recommendation_trends(self, symbol):
        return self._trends.get(symbol, [])


def _trend_entry(period: str, strong_buy=0, buy=0, hold=0, sell=0, strong_sell=0):
    return {
        "period": period,
        "strongBuy": strong_buy,
        "buy": buy,
        "hold": hold,
        "sell": sell,
        "strongSell": strong_sell,
        "symbol": "AAPL",
    }


def test_emits_one_score_row_per_period():
    trends = {
        "AAPL": [
            _trend_entry("2026-04-01", strong_buy=5, buy=10, hold=3),
            _trend_entry("2026-03-01", strong_buy=4, buy=8, hold=4),
            _trend_entry("2026-02-01", strong_buy=3, buy=6, hold=5),
        ]
    }
    adapter = FinnhubAnalystTrendsAdapter(client=_FakeClient(trends))
    out = adapter.fetch_remote(["AAPL"], date(2026, 5, 1))
    score_rows = out.loc[out["feature_name"] == "analyst_score"]
    periods = sorted(pd.to_datetime(score_rows["release_date"]).dt.date.unique())
    assert periods == [date(2026, 2, 1), date(2026, 3, 1), date(2026, 4, 1)]


def test_filters_periods_after_as_of():
    trends = {
        "AAPL": [
            _trend_entry("2026-05-01", strong_buy=5, buy=10),  # future vs as_of=2026-03-15
            _trend_entry("2026-04-01", strong_buy=4, buy=8),  # future vs as_of=2026-03-15
            _trend_entry("2026-03-01", strong_buy=3, buy=6),
            _trend_entry("2026-02-01", strong_buy=2, buy=5),
        ]
    }
    adapter = FinnhubAnalystTrendsAdapter(client=_FakeClient(trends))
    out = adapter.fetch_remote(["AAPL"], date(2026, 3, 15))
    periods = sorted(
        pd.to_datetime(out.loc[out["feature_name"] == "analyst_score", "release_date"]).dt.date.unique()
    )
    assert periods == [date(2026, 2, 1), date(2026, 3, 1)]


def test_release_date_equals_period_not_today():
    trends = {"AAPL": [_trend_entry("2026-03-01", strong_buy=4, buy=4, hold=2)]}
    adapter = FinnhubAnalystTrendsAdapter(client=_FakeClient(trends))
    out = adapter.fetch_remote(["AAPL"], date(2026, 5, 1))
    score_rows = out.loc[out["feature_name"] == "analyst_score"]
    assert len(score_rows) == 1
    release = pd.to_datetime(score_rows.iloc[0]["release_date"]).date()
    assert release == date(2026, 3, 1)


def test_score_validates_against_analyst_schema():
    trends = {"AAPL": [_trend_entry("2026-03-01", strong_buy=4, buy=4, hold=2)]}
    adapter = FinnhubAnalystTrendsAdapter(client=_FakeClient(trends))
    out = adapter.fetch_remote(["AAPL"], date(2026, 5, 1))
    validate_analyst_snapshot(out)


def test_empty_when_all_periods_too_old():
    trends = {"AAPL": [_trend_entry("2026-04-01", strong_buy=5)]}
    adapter = FinnhubAnalystTrendsAdapter(client=_FakeClient(trends))
    out = adapter.fetch_remote(["AAPL"], date(2024, 1, 1))
    assert out.empty


def test_score_in_range_minus_one_to_one():
    trends = {
        "AAPL": [
            _trend_entry("2026-04-01", strong_sell=10),  # all sells
            _trend_entry("2026-03-01", strong_buy=10),   # all strong buys
        ]
    }
    adapter = FinnhubAnalystTrendsAdapter(client=_FakeClient(trends))
    out = adapter.fetch_remote(["AAPL"], date(2026, 5, 1))
    scores = pd.to_numeric(
        out.loc[out["feature_name"] == "analyst_score", "feature_value"]
    )
    assert scores.min() >= -1.0
    assert scores.max() <= 1.0
    # The April entry (all strongSell) yields -1.
    april_score = float(
        out.loc[
            (out["feature_name"] == "analyst_score")
            & (pd.to_datetime(out["release_date"]).dt.date == date(2026, 4, 1)),
            "feature_value",
        ].iloc[0]
    )
    assert april_score == pytest.approx(-1.0)
    # The March entry (all strongBuy) yields +1.
    march_score = float(
        out.loc[
            (out["feature_name"] == "analyst_score")
            & (pd.to_datetime(out["release_date"]).dt.date == date(2026, 3, 1)),
            "feature_value",
        ].iloc[0]
    )
    assert march_score == pytest.approx(1.0)


def test_pull_snapshot_writes_and_registers(tmp_path: Path):
    trends = {"AAPL": [_trend_entry("2026-03-01", strong_buy=4, buy=4, hold=2)]}
    adapter = FinnhubAnalystTrendsAdapter(client=_FakeClient(trends))
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    path = adapter.pull_snapshot(
        ["AAPL"], date(2026, 5, 1), registry=registry, root=tmp_path / "snap"
    )
    assert path.exists()
    entries = registry.list_entries(dataset=DATASET_ID)
    assert len(entries) == 1


def test_double_pull_is_idempotent(tmp_path: Path):
    trends = {"AAPL": [_trend_entry("2026-03-01", strong_buy=4, buy=4, hold=2)]}
    adapter = FinnhubAnalystTrendsAdapter(client=_FakeClient(trends))
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    adapter.pull_snapshot(
        ["AAPL"], date(2026, 5, 1), registry=registry, root=tmp_path / "snap"
    )
    adapter.pull_snapshot(
        ["AAPL"], date(2026, 5, 1), registry=registry, root=tmp_path / "snap"
    )
    entries = registry.list_entries(dataset=DATASET_ID)
    assert len(entries) == 1


def test_adapter_registered_under_expected_id():
    from backtest.external_features.adapters import _REGISTRY, get_adapter

    assert "finnhub_analyst_trends" in _REGISTRY
    adapter = get_adapter("finnhub_analyst_trends")
    assert isinstance(adapter, FinnhubAnalystTrendsAdapter)
