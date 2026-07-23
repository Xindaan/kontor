"""Tests for the four Phase C news adapters (T-0106..T-0109, T-0125)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pandas as pd
import pytest

from backtest.external_features.adapters._news_common import default_cutoff_ts_utc
from backtest.external_features.adapters.finnhub_news import FinnhubNewsAdapter
from backtest.external_features.adapters.newsapi_news import NewsAPIAdapter
from backtest.external_features.adapters.synthetic_news_pit import (
    SyntheticNewsPITAdapter,
)
from backtest.external_features.adapters.yahoo_news import YahooNewsAdapter
from backtest.external_features.news_schema import validate_news_snapshot
from backtest.external_features.sentiment import MockSentimentEngine
from backtest.provenance import ManualDataProvenanceRegistry

pytestmark = pytest.mark.no_network


# ---------------------------------------------------------------------------
# YahooNewsAdapter (current-only)
# ---------------------------------------------------------------------------


def test_yahoo_news_hardfails_for_historical_as_of():
    adapter = YahooNewsAdapter(engine=MockSentimentEngine())
    with pytest.raises(ValueError, match="current-only"):
        adapter.fetch_remote(["AAPL"], date(2024, 1, 1))


def test_yahoo_news_aggregates_via_injected_ticker_factory():
    def _ticker(symbol):
        return SimpleNamespace(
            news=[
                {
                    "providerPublishTime": int(
                        datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc).timestamp()
                    ),
                    "title": f"{symbol} reports strong quarter",
                    "link": "https://example.com",
                    "publisher": "ExamplePub",
                }
            ]
        )

    adapter = YahooNewsAdapter(engine=MockSentimentEngine(), ticker_factory=_ticker)
    result = adapter.fetch_remote(["AAPL"], date.today())
    df = result.frame
    assert not df.empty
    score_rows = df.loc[df["feature_name"] == "news_sentiment_score"]
    assert set(score_rows["ticker"]) == {"AAPL"}
    # release_date == as_of (aggregated row), not the article ts.
    assert score_rows.iloc[0]["release_date"] == pd.Timestamp(date.today())
    validate_news_snapshot(df)


# ---------------------------------------------------------------------------
# FinnhubNewsAdapter (historical) + PIT cutoff
# ---------------------------------------------------------------------------


class _FakeFinnhubClient:
    def __init__(self, articles_by_ticker):
        self._articles = articles_by_ticker

    def company_news(self, symbol, from_date, to_date):
        return self._articles.get(symbol, [])


def _article(ts: datetime, headline: str = "ok") -> dict:
    return {
        "datetime": int(ts.timestamp()),
        "headline": headline,
        "url": "https://example.com",
        "source": "ExampleWire",
    }


def test_finnhub_news_aggregates_release_date_is_as_of():
    as_of = date(2026, 5, 13)
    cutoff = default_cutoff_ts_utc(as_of)
    articles = {
        "AAPL": [
            _article(datetime(2026, 5, 10, 12, tzinfo=timezone.utc), "AAPL upbeat"),
            _article(datetime(2026, 5, 11, 12, tzinfo=timezone.utc), "AAPL beats"),
        ]
    }
    adapter = FinnhubNewsAdapter(
        engine=MockSentimentEngine(),
        client=_FakeFinnhubClient(articles),
    )
    result = adapter.fetch_remote(["AAPL"], as_of)
    df = result.frame
    score_rows = df.loc[df["feature_name"] == "news_sentiment_score"]
    assert len(score_rows) == 1
    assert score_rows.iloc[0]["release_date"] == pd.Timestamp(as_of)
    # The sidecar must keep the article timestamps for audit.
    assert result.sidecars
    sidecar_rows = result.sidecars[0].rows
    assert all(row["ticker"] == "AAPL" for row in sidecar_rows)
    cutoff_naive = cutoff
    if cutoff_naive.tz is None:
        compare_cutoff = cutoff_naive
    else:
        compare_cutoff = cutoff_naive.tz_localize(None)
    for row in sidecar_rows:
        ts = pd.Timestamp(row["release_ts"])
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        assert ts <= compare_cutoff


def test_finnhub_news_filters_articles_after_cutoff():
    as_of = date(2026, 5, 13)
    articles = {
        "AAPL": [
            _article(datetime(2026, 5, 13, 22, tzinfo=timezone.utc), "after cutoff"),
            _article(datetime(2026, 5, 11, 12, tzinfo=timezone.utc), "before cutoff"),
        ]
    }
    adapter = FinnhubNewsAdapter(
        engine=MockSentimentEngine(),
        client=_FakeFinnhubClient(articles),
    )
    result = adapter.fetch_remote(["AAPL"], as_of)
    sidecar_rows = result.sidecars[0].rows
    headlines = [row["headline"] for row in sidecar_rows]
    assert "before cutoff" in headlines
    assert "after cutoff" not in headlines


def test_finnhub_news_skips_unsupported_ticker_silently():
    adapter = FinnhubNewsAdapter(
        engine=MockSentimentEngine(),
        client=_FakeFinnhubClient({}),
    )
    result = adapter.fetch_remote(["AAPL"], date(2026, 5, 13))
    assert result.frame.empty


# ---------------------------------------------------------------------------
# NewsAPIAdapter
# ---------------------------------------------------------------------------


class _FakeNewsAPIClient:
    def __init__(self, payload_by_query):
        self._payload = payload_by_query

    def everything(self, query, from_date, to_date, **kwargs):
        return self._payload.get(query, {"articles": []})


def test_newsapi_news_quota_fail_fast():
    adapter = NewsAPIAdapter(
        engine=MockSentimentEngine(),
        client=_FakeNewsAPIClient({}),
    )
    too_many = [f"T{i}" for i in range(150)]
    with pytest.raises(Exception):
        adapter.fetch_remote(too_many, date(2026, 5, 13))


def test_newsapi_news_per_ticker_default():
    payload = {
        "AAPL": {
            "articles": [
                {
                    "title": "AAPL good",
                    "description": "good",
                    "publishedAt": "2026-05-12T10:00:00Z",
                    "url": "https://example.com",
                    "source": {"name": "Wire"},
                }
            ]
        }
    }
    adapter = NewsAPIAdapter(
        engine=MockSentimentEngine(),
        client=_FakeNewsAPIClient(payload),
    )
    result = adapter.fetch_remote(["AAPL"], date(2026, 5, 13))
    df = result.frame
    assert not df.empty
    assert pd.Timestamp(date(2026, 5, 13)) in pd.to_datetime(df["release_date"]).unique()


# ---------------------------------------------------------------------------
# SyntheticNewsPITAdapter
# ---------------------------------------------------------------------------


def test_synthetic_news_pit_is_deterministic():
    adapter_a = SyntheticNewsPITAdapter()
    adapter_b = SyntheticNewsPITAdapter()
    a = adapter_a.fetch_remote(["AAA", "BBB"], date(2026, 5, 13))
    b = adapter_b.fetch_remote(["AAA", "BBB"], date(2026, 5, 13))
    # snapshot_ts is "now" and intentionally non-deterministic; the
    # numeric feature_values must match across runs though.
    stable_cols = ["ticker", "feature_name", "feature_value", "release_date"]
    pd.testing.assert_frame_equal(
        a.frame[stable_cols].reset_index(drop=True),
        b.frame[stable_cols].reset_index(drop=True),
    )


def test_synthetic_news_pit_writes_sidecar_and_validates():
    adapter = SyntheticNewsPITAdapter()
    result = adapter.fetch_remote(["AAA"], date(2026, 5, 13))
    validate_news_snapshot(result.frame)
    assert result.sidecars
    assert all(row["ticker"] == "AAA" for row in result.sidecars[0].rows)


# ---------------------------------------------------------------------------
# Provenance idempotency for one of the news adapters
# ---------------------------------------------------------------------------


def test_pull_snapshot_idempotent_with_sidecar(tmp_path: Path):
    adapter = SyntheticNewsPITAdapter()
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    snapshot_root = tmp_path / "snap"
    path_a = adapter.pull_snapshot(
        ["AAA"], date(2026, 5, 13), registry=registry, root=snapshot_root
    )
    path_b = adapter.pull_snapshot(
        ["AAA"], date(2026, 5, 13), registry=registry, root=snapshot_root
    )
    assert path_a == path_b
    entries = registry.list_entries(dataset="synthetic_news_pit")
    assert len(entries) == 1
    # The sidecar lives next to the CSV.
    sidecar = path_a.with_name(f"{date(2026, 5, 13).isoformat()}.headlines.ndjson")
    assert sidecar.exists()


def test_pull_snapshot_assigns_shared_raw_payload_hash(tmp_path: Path):
    adapter = SyntheticNewsPITAdapter()
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    snapshot_root = tmp_path / "snap"
    path = adapter.pull_snapshot(
        ["AAA", "BBB"], date(2026, 5, 13), registry=registry, root=snapshot_root
    )
    df = pd.read_csv(path)
    hashes = set(df["raw_payload_hash"].dropna().unique())
    assert len(hashes) == 1


def test_news_adapter_registry_lists_all_four():
    from backtest.external_features.adapters import _REGISTRY

    expected = {"yahoo_news", "finnhub_news", "newsapi_news", "synthetic_news_pit"}
    assert expected <= set(_REGISTRY.keys())
