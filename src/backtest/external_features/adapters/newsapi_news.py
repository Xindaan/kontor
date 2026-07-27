"""NewsAPI adapter (Phase C T-0108).

Pulls headlines via :class:`NewsAPIClient` (defaults to per-ticker, with
an opt-in OR-batch mode). NewsAPI's developer plan delivers data with a
24h delay and limits history to ~1 month; the adapter sets
``fresh_until`` and ``max_age_hours=24`` as live-freshness hints (Codex
C17) without applying them as compliance limits — backtests can still
load expired snapshots through ``ExternalFeaturesLoader.load(enforce_expiry=False)``.
"""

from __future__ import annotations

from datetime import date, timedelta
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
from backtest.external_features.adapters.newsapi_client import (
    NewsAPIClient,
    NewsAPIError,
    build_or_query,
    enforce_quota,
)
from backtest.external_features.sentiment import (
    MockSentimentEngine,
    SentimentEngine,
)

DATASET_ID = "newsapi_news"
SOURCE_NAME = "NewsAPI"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_FRESHNESS_HOURS = 24


class NewsAPIAdapter(ExternalFeatureAdapter):
    def __init__(
        self,
        engine: Optional[SentimentEngine] = None,
        client: Optional[NewsAPIClient] = None,
        *,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        cutoff_override: Optional[str] = None,
        use_batch_or_query: bool = False,
    ) -> None:
        self._engine = engine or MockSentimentEngine()
        self._lookback_days = int(lookback_days)
        self._cutoff_override = cutoff_override
        self._use_batch = bool(use_batch_or_query)
        self._client_cache: Optional[NewsAPIClient] = client
        self._client_factory = lambda: client or NewsAPIClient()

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
            "NewsAPI Developer tier; 100 req/day; 24h delay on new articles; "
            "~1 month history; attribution required."
        )

    @property
    def plan_policy(self):
        return "developer_tier_24h_delay"

    @property
    def max_age_hours(self):
        return DEFAULT_FRESHNESS_HOURS

    @property
    def source_url(self):
        return "https://newsapi.org/"

    def _client(self) -> NewsAPIClient:
        if self._client_cache is None:
            self._client_cache = self._client_factory()
        return self._client_cache

    def _articles_per_ticker(
        self,
        tickers: List[str],
        from_date: str,
        to_date: str,
    ) -> dict:
        client = self._client()
        per_ticker: dict[str, list] = {}
        for ticker in tickers:
            try:
                payload = client.everything(ticker, from_date, to_date)
            except NewsAPIError:
                continue
            articles_raw = payload.get("articles") if isinstance(payload, dict) else None
            if not isinstance(articles_raw, list):
                continue
            mapped: List[dict] = []
            for article in articles_raw:
                if not isinstance(article, dict):
                    continue
                mapped.append(
                    {
                        "publishedAt": article.get("publishedAt"),
                        "headline": article.get("title"),
                        "summary": article.get("description"),
                        "url": article.get("url"),
                        "source": (article.get("source") or {}).get("name"),
                        "matched_tickers": [ticker],
                    }
                )
            if mapped:
                per_ticker[ticker] = mapped
        return per_ticker

    def _articles_via_batch(
        self,
        tickers: List[str],
        from_date: str,
        to_date: str,
    ) -> dict:
        client = self._client()
        chunks = build_or_query(tickers)
        per_ticker: dict[str, list] = {t: [] for t in tickers}
        ticker_set = {t.lower(): t for t in tickers}
        for chunk in chunks:
            try:
                payload = client.everything(chunk, from_date, to_date)
            except NewsAPIError:
                continue
            articles_raw = payload.get("articles") if isinstance(payload, dict) else None
            if not isinstance(articles_raw, list):
                continue
            for article in articles_raw:
                if not isinstance(article, dict):
                    continue
                text_blob = " ".join(
                    str(article.get(key) or "")
                    for key in ("title", "description")
                ).lower()
                matched: list[str] = []
                for needle_lower, original in ticker_set.items():
                    if needle_lower and needle_lower in text_blob.split():
                        matched.append(original)
                if not matched:
                    continue
                mapped_article = {
                    "publishedAt": article.get("publishedAt"),
                    "headline": article.get("title"),
                    "summary": article.get("description"),
                    "url": article.get("url"),
                    "source": (article.get("source") or {}).get("name"),
                    "matched_tickers": matched,
                }
                for ticker in matched:
                    per_ticker[ticker].append(mapped_article)
        return {k: v for k, v in per_ticker.items() if v}

    def fetch_remote(self, tickers: List[str], as_of: date) -> ExternalFeatureFetchResult:
        normalized = enforce_quota(tickers)
        cutoff_ts = parse_cutoff_override(self._cutoff_override, as_of)
        snapshot_ts = pd.Timestamp.utcnow().tz_localize(None)
        from_date = (as_of - timedelta(days=self._lookback_days)).isoformat()
        to_date = as_of.isoformat()
        if self._use_batch:
            articles_by_ticker = self._articles_via_batch(normalized, from_date, to_date)
        else:
            articles_by_ticker = self._articles_per_ticker(normalized, from_date, to_date)
        csv_rows: List[dict] = []
        sidecar_rows: List[dict] = []
        for ticker in normalized:
            articles = articles_by_ticker.get(ticker)
            if not articles:
                continue
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
            if not sidecar_part:
                continue
            csv_rows.extend(csv_part)
            sidecar_rows.extend(sidecar_part)
        return build_fetch_result(
            csv_rows=csv_rows,
            sidecar_rows=sidecar_rows,
            as_of=as_of,
        )
