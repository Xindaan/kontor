"""Finnhub current analyst adapter (Phase B T-0056).

Source: Finnhub ``price_target`` endpoint. The API has no date
parameter; for historical analyst PIT use ``finnhub_analyst_actions``
(``upgrade_downgrade``) or ``synthetic_analyst_pit``.
See: docs/finnhub_api_compat_spike_2026-05-13.md.
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

DATASET_ID = "finnhub_analyst_current"
SOURCE_NAME = "Finnhub"


class FinnhubAnalystCurrentAdapter(ExternalFeatureAdapter):
    def __init__(
        self,
        client: Optional[FinnhubClient] = None,
    ) -> None:
        self._client_factory = lambda: client or FinnhubClient()
        self._client_cache: Optional[FinnhubClient] = client

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
        if as_of != date.today():
            raise ValueError(
                "FinnhubAnalystCurrentAdapter only supports as_of == today "
                "(price_target endpoint is current-only). Use "
                "'finnhub_analyst_actions' or 'synthetic_analyst_pit' for "
                "historical PIT."
            )

        client = self._client()
        snapshot_ts = pd.Timestamp(datetime.now(timezone.utc))
        rows: list[dict] = []
        for ticker in tickers:
            try:
                target = client.price_target(ticker)
                trend = client.recommendation_trends(ticker)
            except FinnhubAPIError:
                continue

            score, counts = _trend_to_score(trend)
            if score is None:
                continue

            base = {
                "ticker": ticker,
                "release_date": pd.Timestamp(as_of),
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

            target_mean = _safe_float(target.get("targetMean") if isinstance(target, dict) else None)
            target_median = _safe_float(target.get("targetMedian") if isinstance(target, dict) else None)
            if target_mean is not None:
                rows.append({**base, "feature_name": "price_target_mean", "feature_value": target_mean})
            if target_median is not None:
                rows.append({**base, "feature_name": "price_target_median", "feature_value": target_median})

            age = _age_days(target, as_of) if isinstance(target, dict) else None
            if age is not None:
                rows.append({**base, "feature_name": "price_target_age_days", "feature_value": float(age)})

        return pd.DataFrame(rows)


def _trend_to_score(trend) -> tuple[Optional[float], dict]:
    """Convert the most recent recommendation_trends entry to score+counts."""
    if not trend or not isinstance(trend, list):
        return None, {}
    latest = trend[0]
    if not isinstance(latest, dict):
        return None, {}
    counts = {
        "analyst_buy_count": float(latest.get("strongBuy", 0)) + float(latest.get("buy", 0)),
        "analyst_hold_count": float(latest.get("hold", 0)),
        "analyst_sell_count": float(latest.get("sell", 0)) + float(latest.get("strongSell", 0)),
    }
    total = counts["analyst_buy_count"] + counts["analyst_hold_count"] + counts["analyst_sell_count"]
    if total <= 0:
        return 0.0, counts
    weighted = (
        latest.get("strongBuy", 0) * 1.0
        + latest.get("buy", 0) * 0.5
        + latest.get("hold", 0) * 0.0
        + latest.get("sell", 0) * -0.5
        + latest.get("strongSell", 0) * -1.0
    )
    raw = weighted / total
    return max(-1.0, min(1.0, raw)), counts


def _confidence_from_counts(counts: dict) -> float:
    total = sum(counts.values())
    return min(1.0, total / 30.0)


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _age_days(target: dict, as_of: date) -> Optional[int]:
    updated = target.get("lastUpdated")
    if not updated:
        return None
    try:
        d = datetime.fromisoformat(str(updated).replace("Z", "+00:00")).date()
    except ValueError:
        try:
            d = datetime.strptime(str(updated), "%Y-%m-%d").date()
        except ValueError:
            return None
    return max(0, (as_of - d).days)
