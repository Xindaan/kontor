"""Finnhub upgrade/downgrade event aggregator (Phase B T-0057).

Source: Finnhub ``upgrade_downgrade(symbol, _from, to)``.

**PIT trap avoidance**: aggregated snapshot rows always set
``release_date = as_of`` (the aggregation date), NEVER ``_from`` of the
oldest event in the window. Otherwise the absence of later events would
leak into the past — see docs/finnhub_api_compat_spike_2026-05-13.md.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd

from backtest.external_features.adapters.base import ExternalFeatureAdapter
from backtest.external_features.adapters.finnhub_client import (
    FinnhubAPIError,
    FinnhubClient,
)

DATASET_ID = "finnhub_analyst_actions"
SOURCE_NAME = "Finnhub"

DEFAULT_LOOKBACK_DAYS = 90

_UP_ACTIONS = {"up", "upgrade", "init", "buy", "outperform", "overweight", "positive"}
_DOWN_ACTIONS = {"down", "downgrade", "sell", "underperform", "underweight", "negative"}


class FinnhubAnalystActionsAdapter(ExternalFeatureAdapter):
    def __init__(
        self,
        client: Optional[FinnhubClient] = None,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        self._client_cache: Optional[FinnhubClient] = client
        self._client_factory = lambda: client or FinnhubClient()
        self.lookback_days = int(lookback_days)

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
            "see https://finnhub.io/terms-of-service."
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
        from_date = (as_of - timedelta(days=self.lookback_days)).isoformat()
        to_date = as_of.isoformat()
        rows: list[dict] = []

        for ticker in tickers:
            try:
                events = client.upgrade_downgrade(ticker, from_date, to_date)
            except FinnhubAPIError:
                continue
            if not isinstance(events, list):
                continue

            ups = 0
            downs = 0
            for event in events:
                if not isinstance(event, dict):
                    continue
                ts = _event_ts(event)
                if ts is not None and ts.date() > as_of:
                    # PIT guard: never count events from after as_of.
                    continue
                action = str(event.get("action", "")).lower()
                from_grade = str(event.get("fromGrade", "")).lower()
                to_grade = str(event.get("toGrade", "")).lower()
                if _is_upgrade(action, from_grade, to_grade):
                    ups += 1
                elif _is_downgrade(action, from_grade, to_grade):
                    downs += 1

            total = ups + downs
            if total <= 0:
                analyst_score = 0.0
            else:
                analyst_score = max(-1.0, min(1.0, (ups - downs) / total))

            base = {
                "ticker": ticker,
                # PIT-correct: aggregate is known on the aggregation day,
                # NOT on the date of the first event. Setting release_date
                # to _from would leak the absence of later events back
                # in time.
                "release_date": pd.Timestamp(as_of),
                "snapshot_ts": snapshot_ts,
                "source": self.source_name,
                "dataset": self.dataset_id,
            }
            rows.append(
                {
                    **base,
                    "feature_name": "analyst_score",
                    "feature_value": float(analyst_score),
                    "confidence": min(1.0, total / 10.0),
                }
            )
            rows.append({**base, "feature_name": "analyst_buy_count", "feature_value": float(ups)})
            rows.append({**base, "feature_name": "analyst_sell_count", "feature_value": float(downs)})
        return pd.DataFrame(rows)


def _event_ts(event: dict) -> Optional[pd.Timestamp]:
    raw = event.get("gradeTime")
    if raw is None:
        return None
    try:
        return pd.Timestamp(int(raw), unit="s", tz="UTC")
    except (TypeError, ValueError):
        try:
            return pd.Timestamp(raw, tz="UTC")
        except (TypeError, ValueError):
            return None


def _is_upgrade(action: str, from_grade: str, to_grade: str) -> bool:
    if action in _UP_ACTIONS:
        return True
    return _grade_value(to_grade) > _grade_value(from_grade)


def _is_downgrade(action: str, from_grade: str, to_grade: str) -> bool:
    if action in _DOWN_ACTIONS:
        return True
    return _grade_value(to_grade) < _grade_value(from_grade)


_GRADE_RANK = {
    "strong sell": -2,
    "sell": -1,
    "underperform": -1,
    "underweight": -1,
    "negative": -1,
    "hold": 0,
    "neutral": 0,
    "market perform": 0,
    "in line": 0,
    "buy": 1,
    "outperform": 1,
    "overweight": 1,
    "positive": 1,
    "strong buy": 2,
}


def _grade_value(grade: str) -> int:
    g = grade.strip().lower()
    if not g:
        return 0
    return _GRADE_RANK.get(g, 0)
