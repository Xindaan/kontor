"""Tests for conditioned analyst evidence per regime bucket (T-0099)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import pytest

from backtest.external_features.loader import ExternalFeatureSnapshot
from backtest.meta_evidence import (
    assess_analyst_evidence,
    assess_conditioned_analyst_evidence_status,
)

pytestmark = pytest.mark.no_network


@dataclass
class _Provider:
    score_map: dict  # month_end_iso -> {ticker: score}

    def snapshot(self, as_of, tickers=None):
        scores = self.score_map.get(as_of.isoformat(), {})
        if not scores:
            return ExternalFeatureSnapshot(
                as_of=as_of, dataset="test", data=pd.DataFrame()
            )
        rows = []
        for ticker, score in scores.items():
            rows.append(
                {
                    "ticker": ticker,
                    "release_date": pd.Timestamp(as_of),
                    "snapshot_ts": pd.Timestamp(as_of),
                    "feature_name": "analyst_score",
                    "feature_value": float(score),
                    "source": "MockTest",
                    "dataset": "test",
                }
            )
        return ExternalFeatureSnapshot(
            as_of=as_of, dataset="test", data=pd.DataFrame(rows)
        )


def _make_price_loader(prices: pd.DataFrame):
    def _loader(tickers, start, end):
        cols = [c for c in tickers if c in prices.columns]
        return prices[cols].loc[start:end]

    return _loader


def _build_price_frame(end: date, months: int = 30, tickers: Iterable[str] = ("AAA", "BBB", "CCC")):
    idx = pd.date_range(end=end, periods=months * 22, freq="B")
    rng = np.random.default_rng(seed=42)
    base = np.cumprod(1 + rng.normal(0.0005, 0.01, size=len(idx)))
    cols = {}
    for i, t in enumerate(tickers):
        drift = 1 + rng.normal(0.0002 * (i + 1), 0.012, size=len(idx))
        cols[t] = base * np.cumprod(drift)
    return pd.DataFrame(cols, index=idx)


def test_assess_analyst_evidence_produces_conditioned_block_per_bucket():
    end = date(2026, 5, 1)
    prices = _build_price_frame(end)
    # synthetic provider always sees ticker AAA as bullish.
    score_map = {}
    for month_end in pd.date_range(end=end, periods=25, freq="ME"):
        score_map[month_end.date().isoformat()] = {"AAA": 0.8, "BBB": -0.2, "CCC": 0.1}
    status, _reasons, summary = assess_analyst_evidence(
        external_provider=_Provider(score_map),
        current_assets=["AAA", "BBB"],
        candidate_assets=["BBB", "CCC"],
        as_of=end,
        analyst_dataset=None,
        lookback_months=24,
        price_loader=_make_price_loader(prices),
        profile="ausgewogen",
    )
    assert status in {"pass", "fail"}
    assert summary is not None
    assert "conditioned" in summary
    cond = summary["conditioned"]
    # Should produce at least one bucket; insufficient_history is allowed
    # for the early windows when very short price history is available.
    assert isinstance(cond, dict) and cond
    for bucket, payload in cond.items():
        assert "num_windows" in payload and payload["num_windows"] > 0
        assert "hit_rate" in payload
        assert "cagr_edge_pp" in payload


def test_conditioned_status_pass_when_bucket_has_strong_edge():
    # Hand-rolled summary so we can deterministically drive the gate.
    summary = {
        "profile": "aggressiv",
        "thresholds": {"min_windows": 6, "min_cagr_edge_pp": 0.25, "min_hit_rate": 0.50},
        "conditioned": {
            "normal": {
                "num_windows": 12,
                "hit_rate": 0.66,
                "mean_edge": 0.01,
                "cagr_edge_pp": 12.0,
            }
        },
    }
    status, _reasons, windows, bucket = assess_conditioned_analyst_evidence_status(
        summary,
        current_bucket="normal",
        profile="aggressiv",
    )
    assert status == "pass"
    assert windows == 12
    assert bucket is not None and bucket["checks"]["min_hit_rate"] is True


def test_conditioned_status_fail_when_bucket_falls_below_thresholds():
    summary = {
        "profile": "ausgewogen",
        "thresholds": {"min_windows": 8, "min_cagr_edge_pp": 0.75, "min_hit_rate": 0.54},
        "conditioned": {
            "fragile": {
                "num_windows": 5,
                "hit_rate": 0.40,
                "mean_edge": -0.001,
                "cagr_edge_pp": -1.2,
            }
        },
    }
    status, reasons, _windows, _bucket = assess_conditioned_analyst_evidence_status(
        summary,
        current_bucket="fragile",
        profile="ausgewogen",
    )
    assert status == "fail"
    assert any("Failed conditioned analyst gates" in r for r in reasons)


def test_conditioned_status_missing_when_bucket_absent():
    summary = {"profile": "ausgewogen", "conditioned": {"normal": {"num_windows": 10}}}
    status, _reasons, windows, _bucket = assess_conditioned_analyst_evidence_status(
        summary,
        current_bucket="stressed",
        profile="ausgewogen",
    )
    assert status == "missing"
    assert windows == 0


def test_conditioned_status_missing_when_summary_none():
    status, _reasons, _w, _b = assess_conditioned_analyst_evidence_status(
        None,
        current_bucket="normal",
        profile="ausgewogen",
    )
    assert status == "missing"


def test_conditioned_min_windows_override():
    summary = {
        "profile": "aggressiv",
        "thresholds": {"min_windows": 6, "min_cagr_edge_pp": 0.25, "min_hit_rate": 0.50},
        "conditioned": {
            "normal": {
                "num_windows": 4,
                "hit_rate": 0.75,
                "mean_edge": 0.02,
                "cagr_edge_pp": 24.0,
            }
        },
    }
    # 4 windows < default conditioned_min_windows (=3, half of 6) -> still pass.
    status, _r, _w, _b = assess_conditioned_analyst_evidence_status(
        summary,
        current_bucket="normal",
        profile="aggressiv",
    )
    assert status == "pass"
    # When the caller bumps the requirement to 10, status flips to fail.
    status_strict, _r, _w, _b = assess_conditioned_analyst_evidence_status(
        summary,
        current_bucket="normal",
        profile="aggressiv",
        conditioned_min_windows=10,
    )
    assert status_strict == "fail"
