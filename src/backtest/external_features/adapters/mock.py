"""Deterministic mock adapter for tests, smoke checks and CLI bootstrap.

This adapter is intentionally part of the production package — without
it, ``backtest features pull`` cannot be exercised end-to-end before any
real upstream adapter is implemented in phase B.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timezone
from typing import List

import pandas as pd

from backtest.external_features.adapters.base import ExternalFeatureAdapter

DEFAULT_MOCK_TICKERS = ("AAA", "BBB", "CCC")
FEATURE_NAMES = ("score", "confidence_rating")


def _deterministic_value(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:4], "big")
    return round((raw / 0xFFFFFFFF) * 2.0 - 1.0, 6)


class MockAnalystAdapter(ExternalFeatureAdapter):
    """Synthetic adapter producing reproducible analyst-like features."""

    fetch_call_count: int = 0

    @property
    def dataset_id(self) -> str:
        return "mock_analyst"

    @property
    def source_name(self) -> str:
        return "MockAnalyst"

    @property
    def quality_tag(self) -> str:
        return "proxy"

    @property
    def license_tos_note(self) -> str:
        return "synthetic test data; no real source"

    @property
    def source_url(self):
        return None

    def fetch_remote(self, tickers: List[str], as_of: date) -> pd.DataFrame:
        type(self).fetch_call_count += 1
        effective = [str(t).upper() for t in tickers] or list(DEFAULT_MOCK_TICKERS)
        as_of_iso = as_of.isoformat()
        # Deterministic snapshot_ts derived from as_of (noon UTC). Wall-clock
        # time would break the "same snapshot identity → same checksum"
        # contract that pull_snapshot's dedup relies on when force=True is
        # used in tests (two refetches across a second boundary would
        # otherwise produce different CSV bytes and register twice).
        snapshot_ts = datetime.combine(as_of, time(12, 0), tzinfo=timezone.utc)
        rows = []
        for ticker in effective:
            for feature in FEATURE_NAMES:
                value = _deterministic_value(f"{ticker}|{feature}|{as_of_iso}")
                rows.append(
                    {
                        "ticker": ticker,
                        "release_date": pd.Timestamp(as_of),
                        "snapshot_ts": pd.Timestamp(snapshot_ts),
                        "feature_name": feature,
                        "feature_value": value,
                        "source": self.source_name,
                        "dataset": self.dataset_id,
                        "confidence": round((value + 1.0) / 2.0, 6),
                    }
                )
        return pd.DataFrame(rows)
