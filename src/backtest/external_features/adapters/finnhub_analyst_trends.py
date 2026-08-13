"""Finnhub historical recommendation trends adapter (Phase B+ T-0080).

Source: Finnhub ``stock/recommendation`` endpoint via
``FinnhubClient.recommendation_trends(symbol)``. Unlike the
"current" adapter, this one keeps every period that Finnhub returns and
emits one snapshot row per ``(ticker, period)``.

PIT semantics
-------------
Each Finnhub trend entry carries a ``period`` (YYYY-MM-DD, monthly
beginning). The consensus represented by that entry was known to
investors from that date onward, so the adapter sets
``release_date = period``. Rows with ``period > as_of`` are filtered
out, exactly the same cut-off the loader applies.

Limits
------
The free Finnhub tier only returns roughly the last four monthly
periods. The adapter does not invent older periods: when ``as_of`` is
deep in the past it simply returns an empty frame for that ticker. For
genuinely historical analyst PIT prior to that horizon use
``finnhub_analyst_actions`` or ``synthetic_analyst_pit``.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import List, Optional

import pandas as pd

from backtest.external_features.adapters.base import ExternalFeatureAdapter
from backtest.external_features.adapters.finnhub_client import (
    FinnhubAPIError,
    FinnhubClient,
)

DATASET_ID = "finnhub_analyst_trends"
SOURCE_NAME = "Finnhub"


class FinnhubAnalystTrendsAdapter(ExternalFeatureAdapter):
    """Multi-period historical analyst trend rows from Finnhub."""

    def __init__(self, client: Optional[FinnhubClient] = None) -> None:
        self._client_cache: Optional[FinnhubClient] = client
        self._client_factory = lambda: client or FinnhubClient()

    @property
    def dataset_id(self) -> str:
        return DATASET_ID

    @property
    def source_name(self) -> str:
        return SOURCE_NAME

    @property
    def quality_tag(self) -> str:
        return "proxy"

    @property
    def license_tos_note(self) -> str:
        return (
            "Finnhub free tier; attribution required; "
            "see https://finnhub.io/terms-of-service. "
            "Recommendation trends typically limited to ~4 monthly periods on free tier."
        )

    @property
    def source_url(self):
        return "https://finnhub.io/"

    def _client(self) -> FinnhubClient:
        if self._client_cache is None:
            self._client_cache = self._client_factory()
        return self._client_cache

    def fetch_remote(self, tickers: List[str], as_of: date) -> pd.DataFrame:
        client = self._client()
        snapshot_ts = pd.Timestamp(datetime.now(timezone.utc))
        rows: list[dict] = []
        for ticker in tickers:
            try:
                trend = client.recommendation_trends(ticker)
            except FinnhubAPIError:
                continue
            if not isinstance(trend, list):
                continue
            for entry in trend:
                period_date = _parse_period(entry)
                if period_date is None or period_date > as_of:
                    continue
                score, counts = _entry_to_score(entry)
                if score is None:
                    continue
                base = {
                    "ticker": ticker,
                    # release_date is the period itself — the consensus
                    # was knowable from period onwards.
                    "release_date": pd.Timestamp(period_date),
                    "snapshot_ts": snapshot_ts,
                    "source": self.source_name,
                    "dataset": self.dataset_id,
                }
                rows.append(
                    {
                        **base,
                        "feature_name": "analyst_score",
                        "feature_value": float(score),
                        "confidence": _confidence_from_counts(counts),
                    }
                )
                for key, value in counts.items():
                    rows.append({**base, "feature_name": key, "feature_value": float(value)})
        return pd.DataFrame(rows)


def _parse_period(entry: dict) -> Optional[date]:
    raw = entry.get("period") if isinstance(entry, dict) else None
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw), "%Y-%m-%d").date()
    except ValueError:
        try:
            return datetime.fromisoformat(str(raw)).date()
        except ValueError:
            return None


def _entry_to_score(entry: dict) -> tuple[Optional[float], dict]:
    if not isinstance(entry, dict):
        return None, {}
    counts = {
        "analyst_buy_count": float(entry.get("strongBuy", 0)) + float(entry.get("buy", 0)),
        "analyst_hold_count": float(entry.get("hold", 0)),
        "analyst_sell_count": float(entry.get("sell", 0)) + float(entry.get("strongSell", 0)),
    }
    total = counts["analyst_buy_count"] + counts["analyst_hold_count"] + counts["analyst_sell_count"]
    if total <= 0:
        return 0.0, counts
    weighted = (
        entry.get("strongBuy", 0) * 1.0
        + entry.get("buy", 0) * 0.5
        + entry.get("hold", 0) * 0.0
        + entry.get("sell", 0) * -0.5
        + entry.get("strongSell", 0) * -1.0
    )
    raw = weighted / total
    return max(-1.0, min(1.0, raw)), counts


def _confidence_from_counts(counts: dict) -> float:
    total = sum(counts.values())
    return min(1.0, total / 30.0)
