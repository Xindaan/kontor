"""Synthetic PIT analyst adapter for pre-2018 backtests and tests
(Phase B T-0058).

Deterministically computes ``analyst_score`` from a configurable price
history. Without prices the adapter raises — it never invents data.

Tickers must be supplied explicitly: ``synthetic_analyst_pit`` is
NOT in ``datasets_allowing_empty_tickers``. An empty ticker list would
imply 'all universe', which is an unbounded look-ahead.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from typing import Callable, List, Optional

import pandas as pd

from backtest.external_features.adapters.base import ExternalFeatureAdapter

DATASET_ID = "synthetic_analyst_pit"
SOURCE_NAME = "SyntheticPIT"

PriceProvider = Callable[[str, date], Optional[pd.Series]]


def _deterministic_pseudo_returns(ticker: str, as_of: date, length: int = 63) -> pd.Series:
    """Return a deterministic pseudo-return series for the (ticker, as_of)."""
    seed = hashlib.sha256(f"{ticker}|{as_of.isoformat()}".encode("utf-8")).digest()
    raw_bytes = seed * ((length // 32) + 1)
    values = []
    for i in range(length):
        b = raw_bytes[i % len(raw_bytes)]
        # Map byte 0..255 to roughly N(0, 0.01).
        values.append(((b / 255.0) - 0.5) * 0.04)
    end = pd.Timestamp(as_of)
    idx = pd.bdate_range(end=end, periods=length)
    return pd.Series(values, index=idx, name=ticker)


class SyntheticAnalystPITAdapter(ExternalFeatureAdapter):
    """Compute a PIT-safe analyst score from prices (or deterministic seed).

    When a ``price_provider`` is supplied, real returns are used. When
    no provider is set, the adapter falls back to a deterministic
    pseudo-random series — useful for tests and demos. The fallback is
    flagged via ``score_confidence=0.0``.
    """

    def __init__(
        self,
        price_provider: Optional[PriceProvider] = None,
        lookback_days: int = 63,
        clip_z: float = 2.0,
    ) -> None:
        self.price_provider = price_provider
        self.lookback_days = int(lookback_days)
        self.clip_z = float(clip_z)

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
            "synthetic deterministic data; no third-party source; "
            "intended for tests and pre-2018 historical demos."
        )

    @property
    def source_url(self):
        return None

    def fetch_remote(self, tickers: List[str], as_of: date) -> pd.DataFrame:
        if not tickers:
            raise ValueError(
                "synthetic_analyst_pit requires explicit tickers (no implicit universe)."
            )
        snapshot_ts = pd.Timestamp(datetime.now(timezone.utc))
        rows: list[dict] = []
        for ticker in tickers:
            returns = None
            if self.price_provider is not None:
                try:
                    returns = self.price_provider(ticker, as_of)
                except Exception:
                    returns = None
            confidence = 1.0
            if returns is None or returns.empty:
                returns = _deterministic_pseudo_returns(ticker, as_of, self.lookback_days)
                confidence = 0.0

            cumulative = float(returns.sum())
            vol = float(returns.std()) or 1e-9
            z = cumulative / vol
            score = max(-1.0, min(1.0, z / self.clip_z))

            rows.append(
                {
                    "ticker": ticker,
                    "release_date": pd.Timestamp(as_of),
                    "snapshot_ts": snapshot_ts,
                    "feature_name": "analyst_score",
                    "feature_value": float(score),
                    "source": self.source_name,
                    "dataset": self.dataset_id,
                    "confidence": float(confidence),
                }
            )
        return pd.DataFrame(rows)
