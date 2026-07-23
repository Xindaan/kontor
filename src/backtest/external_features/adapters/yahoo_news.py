"""Yahoo Finance news adapter (Phase C T-0106).

Pulls the current set of headlines via ``yfinance.Ticker(symbol).news``,
scores each with the configured :class:`SentimentEngine` and aggregates
into one snapshot row per ticker. **Current-only**: a non-``today``
``as_of`` raises ``ValueError`` with a hint to use the synthetic or
Finnhub adapter for historical PIT.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

import pandas as pd

from backtest.external_features.adapters._news_common import (
    aggregate_articles,
    build_fetch_result,
    parse_cutoff_override,
)
from backtest.external_features.adapters.base import (
    ExternalFeatureAdapter,
    ExternalFeatureFetchResult,
)
from backtest.external_features.sentiment import (
    MockSentimentEngine,
    SentimentEngine,
)

DATASET_ID = "yahoo_news"
SOURCE_NAME = "YahooFinance"


class YahooNewsAdapter(ExternalFeatureAdapter):
    def __init__(
        self,
        engine: Optional[SentimentEngine] = None,
        *,
        cutoff_override: Optional[str] = None,
        ticker_factory=None,
    ) -> None:
        self._engine = engine or MockSentimentEngine()
        self._cutoff_override = cutoff_override
        self._ticker_factory = ticker_factory  # injectable for tests

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
            "Yahoo Finance news headlines via yfinance; attribution required; "
            "no explicit retention limit; current-only."
        )

    @property
    def plan_policy(self):
        return "no_explicit_retention"

    @property
    def source_url(self):
        return "https://finance.yahoo.com/"

    def _resolve_ticker(self, symbol: str):
        if self._ticker_factory is not None:
            return self._ticker_factory(symbol)
        try:
            import yfinance as yf  # pragma: no cover - external
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("yfinance is required for YahooNewsAdapter") from exc
        return yf.Ticker(symbol)

    def fetch_remote(self, tickers: List[str], as_of: date) -> ExternalFeatureFetchResult:
        if as_of != date.today():
            raise ValueError(
                "YahooNewsAdapter is current-only; "
                f"as_of={as_of.isoformat()} != today. Use synthetic_news_pit or "
                "finnhub_news for historical PIT."
            )
        cutoff_ts = parse_cutoff_override(self._cutoff_override, as_of)
        snapshot_ts = pd.Timestamp.utcnow().tz_localize(None)
        csv_rows: List[dict] = []
        sidecar_rows: List[dict] = []
        for ticker in tickers:
            try:
                ticker_obj = self._resolve_ticker(ticker)
                raw_news = getattr(ticker_obj, "news", []) or []
            except Exception:
                raw_news = []
            articles = []
            for entry in raw_news:
                if not isinstance(entry, dict):
                    continue
                articles.append(
                    {
                        "datetime": entry.get("providerPublishTime"),
                        "headline": entry.get("title"),
                        "url": entry.get("link") or entry.get("url"),
                        "source": entry.get("publisher") or "YahooFinance",
                    }
                )
            csv_part, sidecar_part = aggregate_articles(
                ticker=ticker,
                articles=articles,
                engine=self._engine,
                cutoff_ts=cutoff_ts,
                as_of=as_of,
                dataset_id=self.dataset_id,
                source_name=self.source_name,
                snapshot_ts=snapshot_ts,
            )
            # Yahoo can return empty lists for unsupported tickers — Codex
            # C20: tickers we keep present must have a score row; skip
            # tickers with no articles entirely.
            if not sidecar_part:
                continue
            csv_rows.extend(csv_part)
            sidecar_rows.extend(sidecar_part)
        return build_fetch_result(
            csv_rows=csv_rows,
            sidecar_rows=sidecar_rows,
            as_of=as_of,
        )
