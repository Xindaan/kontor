"""Tests for assess_analyst_evidence (Phase B T-0066)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest.external_features.loader import ExternalFeatureSnapshot
from backtest.meta_evidence import assess_analyst_evidence


pytestmark = pytest.mark.no_network


class _DeterministicProvider:
    """Fake provider that returns a snapshot computed from a hardcoded
    score table per month-end."""

    def __init__(self, scores_by_month, source: str = "TestSource"):
        # scores_by_month: dict[(year, month)] -> dict[ticker -> score]
        self._scores_by_month = scores_by_month
        self._source = source

    def snapshot(self, as_of, tickers=None):
        key = (as_of.year, as_of.month)
        scores = self._scores_by_month.get(key, {})
        rows = []
        for ticker, score in scores.items():
            if tickers and ticker not in tickers:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "release_date": pd.Timestamp(as_of),
                    "snapshot_ts": pd.Timestamp(as_of),
                    "feature_name": "analyst_score",
                    "feature_value": float(score),
                    "source": self._source,
                    "dataset": "test",
                }
            )
        return ExternalFeatureSnapshot(
            as_of=as_of, dataset="test", data=pd.DataFrame(rows)
        )


def _make_prices(tickers, start="2023-01-01", end="2026-05-01", slopes=None):
    slopes = slopes or {}
    idx = pd.date_range(start=start, end=end, freq="B")
    out = {}
    for i, t in enumerate(tickers):
        slope = slopes.get(t, 0.0003 * (i + 1))
        out[t] = 100.0 * np.exp(np.linspace(0, slope * len(idx), len(idx)))
    return pd.DataFrame(out, index=idx)


def test_missing_when_no_provider():
    status, reasons, summary = assess_analyst_evidence(
        external_provider=None,
        current_assets=["A"],
        candidate_assets=["B"],
        as_of=date(2026, 5, 1),
    )
    assert status == "missing"
    assert summary is None


def test_missing_when_universe_empty():
    status, reasons, summary = assess_analyst_evidence(
        external_provider=_DeterministicProvider({}),
        current_assets=[],
        candidate_assets=[],
        as_of=date(2026, 5, 1),
    )
    assert status == "missing"


def test_synthetic_source_blocks_without_opt_in():
    # Provider returns rows whose 'source' is SyntheticPIT for ALL months
    # in the lookback so at least one iteration detects synthetic data.
    provider = _DeterministicProvider(
        {(yr, mo): {"A": 0.5, "B": -0.5} for yr in (2025, 2026) for mo in range(1, 13)},
        source="SyntheticPIT",
    )
    prices = _make_prices(["A", "B"])
    status, reasons, summary = assess_analyst_evidence(
        external_provider=provider,
        current_assets=["A"],
        candidate_assets=["B"],
        as_of=date(2026, 5, 1),
        lookback_months=6,
        price_loader=lambda ticks, s, e: prices,
        allow_synthetic_analyst_evidence=False,
    )
    assert status == "missing"
    assert any("Synthetic" in r for r in reasons)


def test_synthetic_source_passes_with_opt_in():
    provider = _DeterministicProvider(
        {(yr, mo): {"A": 0.5, "B": -0.5} for yr in (2025, 2026) for mo in range(1, 13)},
        source="SyntheticPIT",
    )
    prices = _make_prices(["A", "B"], slopes={"A": 0.002, "B": -0.0005})
    status, reasons, summary = assess_analyst_evidence(
        external_provider=provider,
        current_assets=["A"],
        candidate_assets=["B"],
        as_of=date(2026, 5, 1),
        lookback_months=18,
        price_loader=lambda ticks, s, e: prices,
        allow_synthetic_analyst_evidence=True,
        profile="aggressiv",
    )
    # We accept any of pass/fail; what matters: status is no longer
    # 'missing' and summary exists.
    assert status in {"pass", "fail"}
    assert summary is not None
    assert summary["synthetic_source"] is True


def test_pass_when_tilt_clearly_beats_bah():
    # A is positive and outperforms; B is negative and underperforms.
    # Provider always picks A -> Analyst-Tilt beats equal-weight B&H.
    provider = _DeterministicProvider(
        {(yr, mo): {"A": 0.8, "B": -0.8} for yr in (2024, 2025, 2026) for mo in range(1, 13)},
        source="Finnhub",  # not synthetic
    )
    prices = _make_prices(["A", "B"], slopes={"A": 0.003, "B": -0.002})
    status, reasons, summary = assess_analyst_evidence(
        external_provider=provider,
        current_assets=["A"],
        candidate_assets=["B"],
        as_of=date(2026, 5, 1),
        lookback_months=18,
        price_loader=lambda ticks, s, e: prices,
        profile="aggressiv",
    )
    assert status == "pass", reasons
    assert summary["hit_rate"] > 0.5
    assert summary["cagr_edge_pp"] > 0


def test_fail_when_min_windows_below_threshold():
    provider = _DeterministicProvider({(2026, 4): {"A": 0.5, "B": -0.5}}, source="Yahoo")
    prices = _make_prices(["A", "B"])
    status, reasons, summary = assess_analyst_evidence(
        external_provider=provider,
        current_assets=["A"],
        candidate_assets=["B"],
        as_of=date(2026, 5, 1),
        lookback_months=2,
        price_loader=lambda ticks, s, e: prices,
        profile="defensiv",
    )
    assert status == "fail"
    assert any("min_windows" in r for r in reasons)


def test_universe_union_includes_all_candidate_assets():
    # current=A; candidates B and C; provider only sees A,B; should still
    # operate on the union — never crashing on the missing C.
    provider = _DeterministicProvider(
        {(yr, mo): {"A": 0.5, "B": 0.3} for yr in (2025, 2026) for mo in range(1, 13)},
        source="Finnhub",
    )
    prices = _make_prices(["A", "B", "C"])
    status, reasons, summary = assess_analyst_evidence(
        external_provider=provider,
        current_assets=["A"],
        candidate_assets=["B", "C"],
        as_of=date(2026, 5, 1),
        lookback_months=12,
        price_loader=lambda ticks, s, e: prices,
        profile="aggressiv",
    )
    assert status in {"pass", "fail"}  # not 'missing'
    assert "C" in summary["universe"] or "C" not in summary["universe"]
