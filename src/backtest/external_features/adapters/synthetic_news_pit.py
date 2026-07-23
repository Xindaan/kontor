"""Synthetic point-in-time news sentiment adapter (Phase C T-0109).

Generates deterministic sentiment scores from a price-history seed so
that historical backtests have *something* to consume without exposing
real news data. The synthetic adapter has no headlines (the sidecar is
empty by design — Codex C7: sidecar still required for hash anchoring).

Plan: ``score = clip(tanh(rolling_ret_21 * 10) + noise, -1, 1)`` where
``noise`` is a deterministic function of ``(ticker, as_of)``. The
output is reproducible across runs.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import List, Optional

import pandas as pd

from backtest.external_features.adapters._news_common import build_fetch_result
from backtest.external_features.adapters.base import (
    ExternalFeatureAdapter,
    ExternalFeatureFetchResult,
)
from backtest.external_features.sentiment import (
    MockSentimentEngine,
    SentimentEngine,
)

DATASET_ID = "synthetic_news_pit"
SOURCE_NAME = "SyntheticNewsPIT"


class SyntheticNewsPITAdapter(ExternalFeatureAdapter):
    def __init__(
        self,
        engine: Optional[SentimentEngine] = None,
        *,
        articles_per_ticker: int = 3,
    ) -> None:
        # The "engine" only contributes the ``engine_code`` / version so
        # provenance remains auditable; the sentiment value itself is
        # derived deterministically from (ticker, as_of).
        self._engine = engine or MockSentimentEngine()
        self._articles_per_ticker = max(1, int(articles_per_ticker))

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
        return "synthetic sentiment derived from price seeds; no real source."

    @property
    def plan_policy(self):
        return "synthetic_proxy"

    @property
    def source_url(self):
        return None

    def fetch_remote(self, tickers: List[str], as_of: date) -> ExternalFeatureFetchResult:
        if not tickers:
            return ExternalFeatureFetchResult(frame=pd.DataFrame(), sidecars=[])
        snapshot_ts = pd.Timestamp.utcnow().tz_localize(None)
        csv_rows: List[dict] = []
        sidecar_rows: List[dict] = []
        as_of_iso = as_of.isoformat()
        for ticker in tickers:
            score = _deterministic_score(ticker, as_of_iso)
            count = self._articles_per_ticker
            # Article-level scores fan out around the mean so dispersion
            # remains reproducible and non-zero by default.
            article_scores = _spread_scores(score, count)
            dispersion = float(pd.Series(article_scores).std(ddof=0))
            base_row = {
                "ticker": ticker,
                "release_date": pd.Timestamp(as_of),
                "snapshot_ts": snapshot_ts,
                "source": self.source_name,
                "dataset": self.dataset_id,
            }
            csv_rows.extend(
                [
                    {
                        **base_row,
                        "feature_name": "news_sentiment_score",
                        "feature_value": score,
                    },
                    {
                        **base_row,
                        "feature_name": "news_article_count",
                        "feature_value": float(count),
                    },
                    {
                        **base_row,
                        "feature_name": "news_sentiment_dispersion",
                        "feature_value": dispersion,
                    },
                    {
                        **base_row,
                        "feature_name": "news_engine_version",
                        "feature_value": float(self._engine.engine_version),
                    },
                ]
            )
            for idx, article_score in enumerate(article_scores):
                sidecar_rows.append(
                    {
                        "ticker": ticker,
                        "release_ts": (
                            pd.Timestamp(as_of) - pd.Timedelta(days=idx)
                        ).strftime("%Y-%m-%dT00:00:00Z"),
                        "source": self.source_name,
                        "headline": f"synthetic-{ticker}-{idx}",
                        "url": "",
                        "engine_score": article_score,
                        "engine_code": self._engine.engine_code,
                    }
                )
        return build_fetch_result(
            csv_rows=csv_rows,
            sidecar_rows=sidecar_rows,
            as_of=as_of,
        )


def _deterministic_score(ticker: str, as_of_iso: str) -> float:
    seed = f"{ticker.upper()}|{as_of_iso}".encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    raw = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return round(raw * 2.0 - 1.0, 6)


def _spread_scores(mean: float, count: int) -> List[float]:
    if count <= 0:
        return []
    if count == 1:
        return [mean]
    spread = [mean - 0.05, mean, mean + 0.05]
    while len(spread) < count:
        spread.append(mean)
    return [max(-1.0, min(1.0, round(s, 6))) for s in spread[:count]]
