"""Tests for YahooAnalystCurrentAdapter (Phase B T-0055)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.external_features.adapters.yahoo_analyst_current import (
    DATASET_ID,
    YahooAnalystCurrentAdapter,
)
from backtest.external_features.analyst_schema import validate_analyst_snapshot
from backtest.provenance import ManualDataProvenanceRegistry

pytestmark = pytest.mark.no_network


class _FakeTicker:
    def __init__(self, recs, targets):
        self.recommendations = recs
        self.analyst_price_targets = targets


def _factory(payload_by_ticker: dict[str, Any]):
    def _build(ticker):
        recs, targets = payload_by_ticker.get(ticker, (None, None))
        return _FakeTicker(recs, targets)

    return _build


def test_historic_as_of_raises():
    adapter = YahooAnalystCurrentAdapter()
    with pytest.raises(ValueError, match="as_of"):
        adapter.fetch_remote(["AAPL"], date(2020, 1, 1))


def test_fetch_today_emits_long_form(monkeypatch):
    recs = pd.DataFrame(
        [{"strongBuy": 5, "buy": 10, "hold": 3, "sell": 2, "strongSell": 0}]
    )
    targets = {"mean": 215.5, "median": 218.0}
    adapter = YahooAnalystCurrentAdapter(
        ticker_factory=_factory({"AAPL": (recs, targets)})
    )
    out = adapter.fetch_remote(["AAPL"], date.today())
    assert not out.empty
    # Required analyst_score row exists.
    score_row = out[(out["ticker"] == "AAPL") & (out["feature_name"] == "analyst_score")]
    assert len(score_row) == 1
    score = float(score_row.iloc[0]["feature_value"])
    assert -1.0 <= score <= 1.0
    # Score is positive because recommendations are bullish.
    assert score > 0
    # Schema-Validation: analyst contract holds.
    validate_analyst_snapshot(out)


def test_fetch_skips_ticker_without_data(monkeypatch):
    adapter = YahooAnalystCurrentAdapter(ticker_factory=_factory({"AAPL": (None, None)}))
    out = adapter.fetch_remote(["AAPL"], date.today())
    assert out.empty


def test_pull_snapshot_writes_and_registers(tmp_path: Path, monkeypatch):
    recs = pd.DataFrame([{"strongBuy": 3, "buy": 4, "hold": 1, "sell": 0, "strongSell": 0}])
    targets = {"mean": 150.0, "median": 148.0}
    adapter = YahooAnalystCurrentAdapter(ticker_factory=_factory({"MSFT": (recs, targets)}))
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    path = adapter.pull_snapshot(
        ["MSFT"], date.today(), registry=registry, root=tmp_path / "snap"
    )
    assert path.exists()
    entries = registry.list_entries(dataset=DATASET_ID)
    assert len(entries) == 1
