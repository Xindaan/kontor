"""Yahoo Finance analyst adapter (Phase B T-0055).

Source: ``yfinance.Ticker(ticker).recommendations`` and
``.analyst_price_targets``.

**Current-only**: yfinance returns the present analyst view; there is no
date parameter. To prevent silent look-ahead the adapter raises on any
``as_of != date.today()``.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import List

import pandas as pd
import yfinance as yf

from backtest.external_features.adapters.base import ExternalFeatureAdapter

DATASET_ID = "yahoo_analyst_current"
SOURCE_NAME = "YahooFinance"

# Recommendation grade mapping → analyst_score in [-1, 1].
# Yahoo returns numeric counts per bucket (strongBuy, buy, hold, sell,
# strongSell) when ``recommendations`` is queried.
_BUCKET_WEIGHTS = {
    "strongBuy": 1.0,
    "buy": 0.5,
    "hold": 0.0,
    "sell": -0.5,
    "strongSell": -1.0,
}


class YahooAnalystCurrentAdapter(ExternalFeatureAdapter):
    """Pull current analyst recommendations and price targets from Yahoo."""

    def __init__(
        self,
        ticker_factory=None,
        retry_attempts: int = 3,
        retry_base_delay_seconds: float = 0.25,
    ) -> None:
        self._ticker_factory = ticker_factory or yf.Ticker
        self._retry_attempts = max(1, int(retry_attempts))
        self._retry_base_delay_seconds = float(retry_base_delay_seconds)

    @property
    def dataset_id(self) -> str:
        return DATASET_ID

    @property
    def source_name(self) -> str:
        return SOURCE_NAME

    @property
    def quality_tag(self) -> str:
        return "community"

    @property
    def license_tos_note(self) -> str:
        return (
            "Yahoo Finance data via yfinance; non-commercial research only; "
            "see Yahoo TOS at https://policies.yahoo.com/."
        )

    @property
    def source_url(self):
        return "https://finance.yahoo.com/"

    def fetch_remote(self, tickers: List[str], as_of: date) -> pd.DataFrame:
        if as_of != date.today():
            raise ValueError(
                "YahooAnalystCurrentAdapter only supports as_of == today "
                "(Yahoo returns current-only views). For historical "
                "analyst PIT use 'finnhub_analyst_actions' or "
                "'synthetic_analyst_pit'."
            )

        snapshot_ts = pd.Timestamp(datetime.now(timezone.utc))
        rows: list[dict] = []
        for ticker in tickers:
            try:
                payload = self._fetch_one(ticker)
            except Exception:
                continue
            if not payload:
                continue
            score = payload["analyst_score"]
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
                    "confidence": float(payload.get("score_confidence", 0.5)),
                }
            )
            for key in ("analyst_buy_count", "analyst_hold_count", "analyst_sell_count"):
                value = payload.get(key)
                if value is None:
                    continue
                rows.append(
                    {**base, "feature_name": key, "feature_value": float(value)}
                )
            for key in ("price_target_mean", "price_target_median"):
                value = payload.get(key)
                if value is None:
                    continue
                rows.append(
                    {**base, "feature_name": key, "feature_value": float(value)}
                )
        return pd.DataFrame(rows)

    def _fetch_one(self, ticker: str) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self._retry_attempts):
            try:
                tick = self._ticker_factory(ticker)
                recs = getattr(tick, "recommendations", None)
                targets = getattr(tick, "analyst_price_targets", None)
                return _normalize_yahoo_payload(recs, targets)
            except Exception as exc:
                last_exc = exc
                time.sleep(self._retry_base_delay_seconds * (attempt + 1))
        raise RuntimeError(f"yahoo analyst fetch failed for {ticker}: {last_exc}")


def _normalize_yahoo_payload(recommendations, price_targets) -> dict:
    """Extract the score plus counts plus targets from yfinance objects.

    yfinance is somewhat fluid: ``recommendations`` may be a DataFrame
    keyed by date with grade-count columns, or sometimes a structure
    with the latest row. ``analyst_price_targets`` is typically a dict.
    We accept both.
    """

    result: dict = {}
    rec_counts: dict = {}
    if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
        latest = recommendations.iloc[-1]
        for col, weight in _BUCKET_WEIGHTS.items():
            if col in latest.index:
                rec_counts[col] = float(latest[col])
    elif isinstance(recommendations, dict):
        for col, weight in _BUCKET_WEIGHTS.items():
            if col in recommendations:
                rec_counts[col] = float(recommendations[col])

    if rec_counts:
        total = sum(rec_counts.values()) or 0.0
        if total > 0:
            score = sum(
                rec_counts.get(col, 0.0) * weight
                for col, weight in _BUCKET_WEIGHTS.items()
            ) / total
            result["analyst_score"] = max(-1.0, min(1.0, score))
            result["score_confidence"] = min(1.0, total / 30.0)
            result["analyst_buy_count"] = rec_counts.get("strongBuy", 0.0) + rec_counts.get("buy", 0.0)
            result["analyst_hold_count"] = rec_counts.get("hold", 0.0)
            result["analyst_sell_count"] = rec_counts.get("sell", 0.0) + rec_counts.get("strongSell", 0.0)
        else:
            result["analyst_score"] = 0.0
            result["score_confidence"] = 0.0

    if isinstance(price_targets, dict):
        mean_target = price_targets.get("mean") or price_targets.get("targetMean")
        median_target = price_targets.get("median") or price_targets.get("targetMedian")
        if mean_target is not None:
            try:
                result["price_target_mean"] = float(mean_target)
            except (TypeError, ValueError):
                pass
        if median_target is not None:
            try:
                result["price_target_median"] = float(median_target)
            except (TypeError, ValueError):
                pass

    if "analyst_score" not in result:
        return {}
    return result
