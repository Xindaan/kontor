"""Tests for FinnhubAnalystActionsAdapter (Phase B T-0057).

PIT-critical: aggregated rows must set release_date = as_of, NEVER the
event date. Otherwise the absence of later events would leak into the
past — see docs/finnhub_api_compat_spike_2026-05-13.md.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from backtest.external_features.adapters.finnhub_analyst_actions import (
    DATASET_ID,
    FinnhubAnalystActionsAdapter,
)
from backtest.external_features.analyst_schema import validate_analyst_snapshot
from backtest.provenance import ManualDataProvenanceRegistry

pytestmark = pytest.mark.no_network


def _ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


class _FakeClient:
    def __init__(self, events_by_ticker, recorded_calls=None):
        self._events = events_by_ticker
        self.calls = recorded_calls if recorded_calls is not None else []

    def upgrade_downgrade(self, symbol, from_date, to_date):
        self.calls.append({"symbol": symbol, "from": from_date, "to": to_date})
        return self._events.get(symbol, [])


def _event(action, ts, from_grade="Hold", to_grade="Buy", company="X"):
    return {
        "symbol": "AAPL",
        "gradeTime": ts,
        "fromGrade": from_grade,
        "toGrade": to_grade,
        "company": company,
        "action": action,
    }


def test_release_date_is_as_of_not_event_date():
    as_of = date(2024, 6, 1)
    events = [
        _event("up", _ts(date(2024, 3, 15))),
        _event("up", _ts(date(2024, 4, 20))),
        _event("down", _ts(date(2024, 5, 10))),
    ]
    adapter = FinnhubAnalystActionsAdapter(client=_FakeClient({"AAPL": events}))
    out = adapter.fetch_remote(["AAPL"], as_of)
    assert not out.empty
    release_dates = pd.to_datetime(out["release_date"]).dt.date.unique()
    assert list(release_dates) == [as_of]
    # PIT-critical assertion: the oldest event date (2024-03-15) must not
    # appear as a release_date — that would leak future events into the
    # past.
    assert date(2024, 3, 15) not in release_dates


def test_future_events_are_filtered():
    as_of = date(2024, 6, 1)
    events = [
        _event("up", _ts(date(2024, 5, 1))),
        # Should never be counted: gradeTime past as_of
        _event("up", _ts(date(2024, 7, 15))),
    ]
    adapter = FinnhubAnalystActionsAdapter(client=_FakeClient({"AAPL": events}))
    out = adapter.fetch_remote(["AAPL"], as_of)
    buys = float(out.loc[out["feature_name"] == "analyst_buy_count", "feature_value"].iloc[0])
    sells = float(out.loc[out["feature_name"] == "analyst_sell_count", "feature_value"].iloc[0])
    assert buys == 1.0
    assert sells == 0.0


def test_score_sign_reflects_balance():
    as_of = date(2024, 6, 1)
    events = [
        _event("down", _ts(date(2024, 4, 1)), from_grade="Buy", to_grade="Hold"),
        _event("down", _ts(date(2024, 4, 5)), from_grade="Hold", to_grade="Sell"),
        _event("up", _ts(date(2024, 5, 1)), from_grade="Hold", to_grade="Buy"),
    ]
    adapter = FinnhubAnalystActionsAdapter(client=_FakeClient({"AAPL": events}))
    out = adapter.fetch_remote(["AAPL"], as_of)
    score_row = out.loc[(out["ticker"] == "AAPL") & (out["feature_name"] == "analyst_score")]
    assert len(score_row) == 1
    score = float(score_row.iloc[0]["feature_value"])
    assert score < 0  # more downgrades than upgrades
    validate_analyst_snapshot(out)


def test_lookback_range_passed_to_client():
    calls: list[dict] = []
    adapter = FinnhubAnalystActionsAdapter(
        client=_FakeClient({"AAPL": []}, recorded_calls=calls),
        lookback_days=30,
    )
    adapter.fetch_remote(["AAPL"], date(2024, 6, 1))
    assert calls[0]["from"] == "2024-05-02"
    assert calls[0]["to"] == "2024-06-01"


def test_pull_snapshot_writes_and_registers(tmp_path: Path):
    adapter = FinnhubAnalystActionsAdapter(
        client=_FakeClient(
            {"AAPL": [_event("up", _ts(date(2024, 5, 15)))]}
        )
    )
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    path = adapter.pull_snapshot(
        ["AAPL"], date(2024, 6, 1), registry=registry, root=tmp_path / "snap"
    )
    assert path.exists()
    entries = registry.list_entries(dataset=DATASET_ID)
    assert len(entries) == 1
