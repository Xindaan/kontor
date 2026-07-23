"""Helpers shared by Phase C news adapters.

Centralises:
- aggregation of per-article sentiment scores into the long-form CSV
  rows (T-0125 formula: mean / std(ddof=0); count=0 -> neutral row);
- assembly of the sidecar NDJSON payload;
- intraday cutoff handling.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import math

import numpy as np
import pandas as pd

from backtest.external_features.adapters.base import (
    ExternalFeatureFetchResult,
    SidecarBlob,
)
from backtest.external_features.sentiment import SentimentEngine


def default_cutoff_ts_utc(as_of: date) -> pd.Timestamp:
    """Default intraday cutoff: previous-day EOD UTC.

    Returned as a *naive* UTC timestamp so it can be compared directly
    to the naive-UTC timestamps produced by :func:`_article_release_ts`.
    """

    base = datetime.combine(as_of, time.min)
    return pd.Timestamp(base) - pd.Timedelta(seconds=1)


def parse_cutoff_override(override: Optional[str], as_of: date) -> pd.Timestamp:
    """Parse a ``HH:MM`` cutoff override into a naive UTC ``Timestamp``."""

    if override is None or not str(override).strip():
        return default_cutoff_ts_utc(as_of)
    raw = str(override).strip()
    try:
        parts = raw.split(":")
        hh = int(parts[0])
        mm = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"invalid --news-intraday-cutoff value '{override}', expected HH:MM"
        ) from exc
    base = datetime.combine(as_of, time(hour=hh, minute=mm))
    return pd.Timestamp(base)


def aggregate_articles(
    *,
    ticker: str,
    articles: Sequence[Mapping],
    engine: SentimentEngine,
    cutoff_ts: pd.Timestamp,
    as_of: date,
    dataset_id: str,
    source_name: str,
    snapshot_ts: pd.Timestamp,
) -> Tuple[List[dict], List[dict]]:
    """Compute long-form CSV rows + sidecar rows for one ticker.

    Returns ``(csv_rows, sidecar_rows)``. ``csv_rows`` always includes a
    ``news_sentiment_score`` row (neutral when no article survives the
    cutoff) plus ``news_article_count`` and ``news_sentiment_dispersion``.
    """

    scored: List[Tuple[float, dict]] = []
    sidecar_rows: List[dict] = []
    for article in articles:
        ts = _article_release_ts(article)
        if ts is None:
            continue
        if ts > cutoff_ts:
            continue
        text = _article_text(article)
        score = float(engine.score(text)) if text else 0.0
        if not math.isfinite(score):
            score = 0.0
        score = max(-1.0, min(1.0, score))
        scored.append((score, article))
        sidecar_rows.append(
            {
                "ticker": ticker,
                "release_ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": str(article.get("source") or article.get("publisher") or source_name),
                "headline": text or "",
                "url": str(article.get("url") or article.get("link") or ""),
                "engine_score": score,
                "engine_code": engine.engine_code,
            }
        )
    if scored:
        scores = np.array([s for s, _ in scored], dtype=float)
        mean_score = float(np.mean(scores))
        dispersion = float(np.std(scores, ddof=0))
        count = int(scores.size)
    else:
        mean_score = 0.0
        dispersion = 0.0
        count = 0
    base_row = {
        "ticker": ticker,
        "release_date": pd.Timestamp(as_of),
        "snapshot_ts": snapshot_ts,
        "source": source_name,
        "dataset": dataset_id,
    }
    csv_rows: List[dict] = [
        {**base_row, "feature_name": "news_sentiment_score", "feature_value": mean_score},
        {**base_row, "feature_name": "news_article_count", "feature_value": float(count)},
        {**base_row, "feature_name": "news_sentiment_dispersion", "feature_value": dispersion},
        {
            **base_row,
            "feature_name": "news_engine_version",
            "feature_value": float(engine.engine_version),
        },
    ]
    return csv_rows, sidecar_rows


def build_fetch_result(
    *,
    csv_rows: Iterable[dict],
    sidecar_rows: Iterable[dict],
    as_of: date,
) -> ExternalFeatureFetchResult:
    """Pack accumulated rows into the adapter return value."""

    frame = pd.DataFrame(list(csv_rows))
    if frame.empty:
        return ExternalFeatureFetchResult(frame=frame, sidecars=[])
    sidecar = SidecarBlob(
        relative_name=f"{as_of.isoformat()}.headlines.ndjson",
        rows=list(sidecar_rows),
        kind="headlines_ndjson",
    )
    return ExternalFeatureFetchResult(frame=frame, sidecars=[sidecar])


def _article_release_ts(article: Mapping) -> Optional[pd.Timestamp]:
    for key in ("datetime", "providerPublishTime", "publishedAt", "release_ts"):
        if key not in article:
            continue
        raw = article[key]
        if raw is None:
            continue
        try:
            if isinstance(raw, (int, float)):
                ts = pd.to_datetime(int(raw), unit="s", utc=True)
            else:
                ts = pd.to_datetime(str(raw), utc=True)
        except (ValueError, TypeError):
            continue
        if pd.isna(ts):
            continue
        # Normalise to naive UTC so all comparisons happen in one tz.
        return ts.tz_convert("UTC").tz_localize(None)
    return None


def _article_text(article: Mapping) -> str:
    for key in ("headline", "title", "summary", "description"):
        value = article.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
