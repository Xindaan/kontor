"""Finnhub company-news adapter (Phase C T-0107).

Pulls historical headlines via :meth:`FinnhubClient.company_news`,
scores each article with the configured sentiment engine and emits one
aggregated snapshot row per ticker with ``release_date = as_of`` (Codex
C4 — never the article publish date). Non-supported tickers
(typically non-US) yield empty lists; the adapter simply omits them
from the snapshot.
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
from backtest.external_features.adapters.finnhub_client import (
    FinnhubAPIError,
    FinnhubClient,
)
from backtest.external_features.sentiment import (
    MockSentimentEngine,
    SentimentEngine,
)

DATASET_ID = "finnhub_news"
SOURCE_NAME = "Finnhub"
DEFAULT_LOOKBACK_DAYS = 365


class FinnhubNewsAdapter(ExternalFeatureAdapter):
    def __init__(
        self,
        engine: Optional[SentimentEngine] = None,
        client: Optional[FinnhubClient] = None,
        *,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        cutoff_override: Optional[str] = None,
    ) -> None:
        self._engine = engine or MockSentimentEngine()
        self._lookback_days = int(lookback_days)
        self._cutoff_override = cutoff_override
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
            "Finnhub company-news; free tier non-commercial use; "
            "attribution required; US/NA primary coverage."
        )

    @property
    def plan_policy(self):
        return "non_commercial_unlimited"

    @property
    def source_url(self):
        return "https://finnhub.io/"

    def _client(self) -> FinnhubClient:
        if self._client_cache is None:
            self._client_cache = self._client_factory()
        return self._client_cache

    def fetch_remote(self, tickers: List[str], as_of: date) -> ExternalFeatureFetchResult:
        client = self._client()
        cutoff_ts = parse_cutoff_override(self._cutoff_override, as_of)
        snapshot_ts = pd.Timestamp.utcnow().tz_localize(None)
        from_date = (as_of - timedelta(days=self._lookback_days)).isoformat()
        to_date = as_of.isoformat()
        csv_rows: List[dict] = []
        sidecar_rows: List[dict] = []
        for ticker in tickers:
            try:
                articles_raw = client.company_news(ticker, from_date, to_date)
            except FinnhubAPIError:
                continue
            if not isinstance(articles_raw, list):
                continue
            articles = [a for a in articles_raw if isinstance(a, dict)]
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
