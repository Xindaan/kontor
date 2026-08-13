"""Double-pull idempotency tests for Phase B adapters (T-0070).

Each adapter must produce exactly one provenance entry across multiple
pulls of the same snapshot. Phase A's stable_entry_id + dedup helper is
shared; here we just verify the contract holds for every new adapter.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from backtest.external_features.adapters.finnhub_analyst_actions import (
    FinnhubAnalystActionsAdapter,
)
from backtest.external_features.adapters.finnhub_analyst_current import (
    FinnhubAnalystCurrentAdapter,
)
from backtest.external_features.adapters.synthetic_analyst_pit import (
    SyntheticAnalystPITAdapter,
)
from backtest.external_features.adapters.yahoo_analyst_current import (
    YahooAnalystCurrentAdapter,
)
from backtest.provenance import ManualDataProvenanceRegistry

pytestmark = pytest.mark.no_network


class _FakeYfTicker:
    def __init__(self, recs, targets):
        self.recommendations = recs
        self.analyst_price_targets = targets


def test_yahoo_adapter_double_pull_is_idempotent(tmp_path: Path):
    recs = pd.DataFrame([{"strongBuy": 3, "buy": 2, "hold": 1, "sell": 0, "strongSell": 0}])
    targets = {"mean": 100.0, "median": 99.0}
    adapter = YahooAnalystCurrentAdapter(
        ticker_factory=lambda t: _FakeYfTicker(recs, targets)
    )
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    adapter.pull_snapshot(["MSFT"], date.today(), registry=registry, root=tmp_path / "snap")
    adapter.pull_snapshot(["MSFT"], date.today(), registry=registry, root=tmp_path / "snap")
    entries = registry.list_entries(dataset=adapter.dataset_id)
    assert len(entries) == 1


class _FakeFinnhubClient:
    def __init__(self, trends, targets, events):
        self._trends = trends
        self._targets = targets
        self._events = events

    def recommendation_trends(self, symbol):
        return self._trends.get(symbol, [])

    def price_target(self, symbol):
        return self._targets.get(symbol, {})

    def upgrade_downgrade(self, symbol, from_date, to_date):  # noqa: ARG002
        return self._events.get(symbol, [])


def test_finnhub_current_double_pull_is_idempotent(tmp_path: Path):
    trends = {"AAPL": [{"period": "2026-05-01", "strongBuy": 4, "buy": 6, "hold": 2, "sell": 1, "strongSell": 0}]}
    targets = {"AAPL": {"targetMean": 200.0, "targetMedian": 199.0, "lastUpdated": str(date.today())}}
    adapter = FinnhubAnalystCurrentAdapter(
        client=_FakeFinnhubClient(trends, targets, {})
    )
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    adapter.pull_snapshot(["AAPL"], date.today(), registry=registry, root=tmp_path / "snap")
    adapter.pull_snapshot(["AAPL"], date.today(), registry=registry, root=tmp_path / "snap")
    entries = registry.list_entries(dataset=adapter.dataset_id)
    assert len(entries) == 1


def test_finnhub_actions_double_pull_is_idempotent(tmp_path: Path):
    events = {
        "AAPL": [
            {
                "symbol": "AAPL",
                "gradeTime": int(datetime(2024, 5, 15, tzinfo=timezone.utc).timestamp()),
                "fromGrade": "Hold",
                "toGrade": "Buy",
                "company": "X",
                "action": "up",
            }
        ]
    }
    adapter = FinnhubAnalystActionsAdapter(
        client=_FakeFinnhubClient({}, {}, events)
    )
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    adapter.pull_snapshot(["AAPL"], date(2024, 6, 1), registry=registry, root=tmp_path / "snap")
    adapter.pull_snapshot(["AAPL"], date(2024, 6, 1), registry=registry, root=tmp_path / "snap")
    entries = registry.list_entries(dataset=adapter.dataset_id)
    assert len(entries) == 1


def test_synthetic_double_pull_is_idempotent(tmp_path: Path):
    adapter = SyntheticAnalystPITAdapter()
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    adapter.pull_snapshot(["AAA"], date(2015, 5, 1), registry=registry, root=tmp_path / "snap")
    adapter.pull_snapshot(["AAA"], date(2015, 5, 1), registry=registry, root=tmp_path / "snap")
    entries = registry.list_entries(dataset=adapter.dataset_id)
    assert len(entries) == 1
