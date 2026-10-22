"""Tests for FinnhubAnalystCurrentAdapter (Phase B T-0056)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from backtest.external_features.adapters.finnhub_analyst_current import (
    DATASET_ID,
    FinnhubAnalystCurrentAdapter,
)
from backtest.external_features.analyst_schema import validate_analyst_snapshot
from backtest.provenance import ManualDataProvenanceRegistry

pytestmark = pytest.mark.no_network


class _FakeFinnhubClient:
    def __init__(self, trends_by_ticker, targets_by_ticker):
        self._trends = trends_by_ticker
        self._targets = targets_by_ticker

    def recommendation_trends(self, symbol):
        return self._trends.get(symbol, [])

    def price_target(self, symbol):
        return self._targets.get(symbol, {})


def test_historic_as_of_raises():
    adapter = FinnhubAnalystCurrentAdapter(client=_FakeFinnhubClient({}, {}))
    with pytest.raises(ValueError, match="as_of"):
        adapter.fetch_remote(["AAPL"], date(2020, 1, 1))


def test_fetch_today_produces_long_form():
    trends = {
        "AAPL": [
            {"period": "2026-05-01", "strongBuy": 4, "buy": 6, "hold": 2, "sell": 1, "strongSell": 0},
        ]
    }
    targets = {
        "AAPL": {"targetMean": 215.0, "targetMedian": 220.0, "lastUpdated": str(date.today())}
    }
    adapter = FinnhubAnalystCurrentAdapter(client=_FakeFinnhubClient(trends, targets))
    out = adapter.fetch_remote(["AAPL"], date.today())
    validate_analyst_snapshot(out)
    score_row = out[(out["ticker"] == "AAPL") & (out["feature_name"] == "analyst_score")]
    assert len(score_row) == 1
    assert float(score_row.iloc[0]["feature_value"]) > 0
    # Target rows present.
    assert "price_target_mean" in out["feature_name"].tolist()
    assert "price_target_median" in out["feature_name"].tolist()


def test_fetch_skips_ticker_without_trend():
    adapter = FinnhubAnalystCurrentAdapter(client=_FakeFinnhubClient({}, {}))
    out = adapter.fetch_remote(["AAPL"], date.today())
    assert out.empty


def test_pull_snapshot_writes_and_registers(tmp_path: Path):
    trends = {"AAPL": [{"period": "2026-05-01", "strongBuy": 5, "buy": 5, "hold": 0, "sell": 0, "strongSell": 0}]}
    targets = {"AAPL": {"targetMean": 200.0, "targetMedian": 199.0, "lastUpdated": str(date.today())}}
    adapter = FinnhubAnalystCurrentAdapter(client=_FakeFinnhubClient(trends, targets))
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    path = adapter.pull_snapshot(
        ["AAPL"], date.today(), registry=registry, root=tmp_path / "snap"
    )
    assert path.exists()
    entries = registry.list_entries(dataset=DATASET_ID)
    assert len(entries) == 1
