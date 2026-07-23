"""HTTP client for the Finnhub REST API.

Synchronous on purpose (project style — no aiohttp/tenacity in Phase B).
Pattern is copied from ``backtest.constituents.nport`` (SEC client):
- explicit retries with exponential backoff,
- baseline throttle between calls to respect rate limits,
- API key resolved from constructor or FINNHUB_API_KEY env var.

The free Finnhub tier is limited to 60 requests/minute, so the default
throttle is 1.05 seconds. Adapters may override it for higher tiers.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests

FINNHUB_API_BASE = "https://finnhub.io/api/v1"
FINNHUB_API_KEY_ENV = "FINNHUB_API_KEY"
DEFAULT_THROTTLE_SECONDS = 1.05
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_MAX_RETRIES = 3


class FinnhubAPIError(RuntimeError):
    """Raised when the Finnhub API returns a non-recoverable error."""


class FinnhubClient:
    """Minimal Finnhub REST client.

    Parameters
    ----------
    api_key
        Optional API key. Falls back to ``FINNHUB_API_KEY`` env var. A
        missing key raises ``RuntimeError`` to fail fast at adapter init.
    throttle_seconds
        Seconds to sleep between calls. Default 1.05 (60 req/min).
    timeout_seconds
        Per-request timeout.
    max_retries
        Maximum retry attempts on transport errors / HTTP 5xx / 429.
    base_url
        Overrideable for tests.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        throttle_seconds: float = DEFAULT_THROTTLE_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_url: str = FINNHUB_API_BASE,
    ) -> None:
        resolved = api_key or os.environ.get(FINNHUB_API_KEY_ENV, "")
        if not resolved or not resolved.strip():
            raise RuntimeError(
                "FINNHUB_API_KEY is required. Pass api_key=... or set the env var."
            )
        self._api_key = resolved.strip()
        self.throttle_seconds = float(throttle_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.base_url = base_url.rstrip("/")
        self._last_call_ts: float = 0.0

    def _wait_for_slot(self) -> None:
        if self.throttle_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_call_ts
        wait = self.throttle_seconds - elapsed
        if wait > 0:
            time.sleep(wait)

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Perform GET against the Finnhub API.

        Returns the parsed JSON body on success. Raises ``FinnhubAPIError``
        when the API call cannot be completed after ``max_retries``.
        """

        url = f"{self.base_url}/{path.lstrip('/')}"
        full_params: Dict[str, Any] = dict(params or {})
        full_params["token"] = self._api_key

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._wait_for_slot()
            try:
                resp = requests.get(
                    url, params=full_params, timeout=self.timeout_seconds
                )
            except requests.RequestException as exc:
                last_error = exc
                self._last_call_ts = time.monotonic()
                if attempt >= self.max_retries:
                    break
                time.sleep(2 ** (attempt - 1))
                continue

            self._last_call_ts = time.monotonic()
            status = resp.status_code
            if status == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise FinnhubAPIError(f"invalid JSON from {url}") from exc
            if status in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                time.sleep(2 ** (attempt - 1))
                continue
            raise FinnhubAPIError(
                f"finnhub {path} returned HTTP {status}: {resp.text[:200]}"
            )

        raise FinnhubAPIError(
            f"finnhub {path} failed after {self.max_retries} attempts: {last_error}"
        )

    def recommendation_trends(self, symbol: str) -> Any:
        return self.get("stock/recommendation", {"symbol": symbol})

    def price_target(self, symbol: str) -> Any:
        return self.get("stock/price-target", {"symbol": symbol})

    def upgrade_downgrade(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
    ) -> Any:
        return self.get(
            "stock/upgrade-downgrade",
            {"symbol": symbol, "from": from_date, "to": to_date},
        )

    def company_news(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
    ) -> Any:
        """Fetch a Finnhub ``company-news`` window (Phase C T-0104).

        Returns the raw JSON list. Each article is a dict with
        ``category, datetime, headline, image, related, source,
        summary, url, id``. Empty list for non-supported tickers.
        """

        return self.get(
            "company-news",
            {"symbol": symbol, "from": from_date, "to": to_date},
        )
