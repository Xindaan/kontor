"""NewsAPI HTTP client (Phase C T-0105).

Wraps the NewsAPI v2 ``/everything`` endpoint with retry, throttle,
and a daily-quota awareness check. Mirrors the
:class:`FinnhubClient`-style API used by other phase-B/C adapters so
the same monkeypatch test patterns apply.

NewsAPI Developer plan constraints (per docs as of 2026-05-13):
- 100 requests / day
- ~1 month history window
- 24h delay on new articles
- 5min server-side cache, no client-side hard retention
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

NEWSAPI_BASE_URL = "https://newsapi.org/v2"
NEWSAPI_API_KEY_ENV = "NEWSAPI_API_KEY"

DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_THROTTLE_SECONDS = 0.9
MAX_TICKERS_PER_PULL = 100


class NewsAPIError(RuntimeError):
    """Raised for NewsAPI client errors."""


class NewsAPIClient:
    """Thin HTTP wrapper around NewsAPI v2 with retry + throttle.

    Construction-time configuration:

    - ``api_key``: explicit key; falls back to ``NEWSAPI_API_KEY`` env.
    - ``throttle_seconds``: minimum gap between successive HTTP calls.
    - ``timeout_seconds``: per-request timeout.
    - ``max_retries``: total attempts including the first call.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = NEWSAPI_BASE_URL,
        throttle_seconds: float = DEFAULT_THROTTLE_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        resolved = api_key or os.getenv(NEWSAPI_API_KEY_ENV)
        if not resolved:
            raise NewsAPIError(
                "NewsAPI key not configured. Set NEWSAPI_API_KEY or pass api_key=..."
            )
        self._api_key = str(resolved)
        self.base_url = base_url.rstrip("/")
        self.throttle_seconds = max(0.0, float(throttle_seconds))
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(1, int(max_retries))
        self._last_call_ts: float = 0.0

    def _wait_for_slot(self) -> None:
        if self.throttle_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_call_ts
        wait = self.throttle_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        full_params = dict(params or {})
        full_params["apiKey"] = self._api_key
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_slot()
            try:
                response = requests.get(url, params=full_params, timeout=self.timeout_seconds)
            except requests.RequestException as exc:
                last_error = exc
                self._last_call_ts = time.monotonic()
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))
                continue
            self._last_call_ts = time.monotonic()
            if response.status_code == 429 and attempt < self.max_retries:
                time.sleep(2 ** (attempt - 1))
                continue
            if response.status_code >= 500 and attempt < self.max_retries:
                time.sleep(2 ** (attempt - 1))
                continue
            if response.status_code >= 400:
                raise NewsAPIError(
                    f"NewsAPI {path} returned HTTP {response.status_code}: {response.text[:200]}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise NewsAPIError(f"NewsAPI returned non-JSON: {exc}") from exc
            if isinstance(payload, dict) and payload.get("status") == "error":
                code = payload.get("code", "")
                msg = payload.get("message", "")
                raise NewsAPIError(f"NewsAPI error code={code}: {msg}")
            return payload
        raise NewsAPIError(
            f"NewsAPI {path} failed after {self.max_retries} attempts: {last_error}"
        )

    def everything(
        self,
        query: str,
        from_date: str,
        to_date: str,
        *,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 100,
    ) -> Any:
        """Wrap the ``/everything`` endpoint.

        Returns the parsed JSON dict (``{status, totalResults, articles}``).
        """

        return self.get(
            "everything",
            {
                "q": query,
                "from": from_date,
                "to": to_date,
                "language": language,
                "sortBy": sort_by,
                "pageSize": int(page_size),
            },
        )


def build_or_query(
    tickers: Sequence[str],
    *,
    max_chars: int = 480,
    quote_each: bool = True,
) -> List[str]:
    """Pack tickers into one or more NewsAPI ``q`` strings.

    Each returned string respects ``max_chars`` (NewsAPI rejects queries
    that grow beyond ~500 chars). Tickers are quoted by default so they
    match as exact tokens. Caller is responsible for spawning one NewsAPI
    request per returned chunk.
    """

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for ticker in tickers:
        token = f'"{ticker}"' if quote_each else str(ticker)
        added = len(token) + (4 if current else 0)  # account for ' OR ' join
        if current and current_len + added > max_chars:
            chunks.append(" OR ".join(current))
            current = [token]
            current_len = len(token)
        else:
            current.append(token)
            current_len += added
    if current:
        chunks.append(" OR ".join(current))
    return chunks


def enforce_quota(tickers: Iterable[str]) -> List[str]:
    """Fail-fast when too many tickers would exhaust the free quota."""

    ticker_list = [str(t).upper() for t in tickers]
    if len(ticker_list) > MAX_TICKERS_PER_PULL:
        raise NewsAPIError(
            f"NewsAPI free tier limited to {MAX_TICKERS_PER_PULL} tickers per pull, "
            f"got {len(ticker_list)}"
        )
    return ticker_list


__all__ = [
    "MAX_TICKERS_PER_PULL",
    "NEWSAPI_API_KEY_ENV",
    "NEWSAPI_BASE_URL",
    "NewsAPIClient",
    "NewsAPIError",
    "build_or_query",
    "enforce_quota",
]
