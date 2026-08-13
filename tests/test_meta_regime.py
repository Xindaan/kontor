from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datetime import date

from backtest.external_features.loader import ExternalFeatureSnapshot
from backtest.meta_regime import (
    RegimeMeasurements,
    _percentile_rank,
    assess_regime_from_equity_curve,
    build_regime_measurements,
    classify_regime_measurements,
)


def test_regime_engine_marks_stable_trend_as_normal():
    idx = pd.bdate_range("2020-01-01", periods=900)
    equity = pd.Series(100.0 * np.exp(np.linspace(0.0, 0.9, len(idx))), index=idx)

    snapshot = assess_regime_from_equity_curve(equity, profile="ausgewogen")

    assert snapshot.status == "ok"
    # A smooth trend has a degenerate (near-constant) reference distribution for
    # every metric; _percentile_rank ranks each at the median (0.5), so no
    # fragility flag fires. Deterministic across FP/SIMD paths (regression:
    # exact-equality tie detection at ~1e-15 noise flipped this normal/fragile
    # per CI runner).
    assert snapshot.bucket == "normal"
    assert all(abs(p - 0.5) < 1e-9 for p in snapshot.percentiles.values())


def test_percentile_rank_degenerate_reference_is_stable():
    # Degenerate / near-constant reference -> median rank, independent of the
    # float noise in the series (this is what keeps the regime buckets stable).
    assert _percentile_rank(pd.Series([1e-15, 2e-15, 1.5e-15] * 50), 1.5e-15) == 0.5
    assert _percentile_rank(pd.Series([0.0] * 40), 0.0) == 0.5
    # Non-degenerate reference is unchanged.
    assert _percentile_rank(pd.Series(np.linspace(0.0, 1.0, 101)), 0.25) == pytest.approx(0.25, abs=0.01)


def test_regime_engine_marks_recent_crash_as_stressed():
    idx = pd.bdate_range("2020-01-01", periods=900)
    base = np.exp(np.linspace(0.0, 0.8, len(idx)))
    base[-90:] *= np.linspace(1.0, 0.55, 90)
    equity = pd.Series(100.0 * base, index=idx)

    snapshot = assess_regime_from_equity_curve(equity, profile="ausgewogen")

    assert snapshot.status == "ok"
    assert snapshot.bucket in {"fragile", "stressed"}
    assert len(snapshot.reasons) >= 1


def test_regime_engine_reports_insufficient_history():
    idx = pd.bdate_range("2024-01-01", periods=180)
    equity = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)

    snapshot = assess_regime_from_equity_curve(equity, profile="ausgewogen")

    assert snapshot.status == "insufficient_history"
    assert snapshot.bucket == "insufficient_history"


def test_regime_profiles_change_bucket_thresholds():
    measurements = RegimeMeasurements(
        status="ok",
        metrics={
            "ret_21": -0.02,
            "ret_63": 0.01,
            "vol_63": 0.25,
            "maxdd_126": -0.08,
            "trend_spread": -0.03,
        },
        percentiles={
            "ret_21": 0.35,
            "ret_63": 0.50,
            "vol_63": 0.50,
            "maxdd_126": 0.50,
            "trend_spread": 0.50,
        },
        reference_days=600,
        available_feature_days=600,
    )

    defensive = classify_regime_measurements(measurements, profile="defensiv")
    aggressive = classify_regime_measurements(measurements, profile="aggressiv")

    assert defensive.bucket == "fragile"
    assert aggressive.bucket == "normal"


# ---------------------------------------------------------------------------
# Phase B: analyst_dispersion field.
# ---------------------------------------------------------------------------


class _StubProvider:
    def __init__(self, scores):
        self._scores = scores

    def snapshot(self, as_of, tickers=None):
        rows = []
        for ticker, score in self._scores.items():
            if tickers and ticker not in tickers:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "release_date": pd.Timestamp(as_of),
                    "snapshot_ts": pd.Timestamp(as_of),
                    "feature_name": "analyst_score",
                    "feature_value": float(score),
                    "source": "Test",
                    "dataset": "test_ds",
                }
            )
        return ExternalFeatureSnapshot(
            as_of=as_of, dataset="test_ds", data=pd.DataFrame(rows)
        )


def test_analyst_dispersion_none_without_provider():
    idx = pd.bdate_range("2020-01-01", periods=400)
    equity = pd.Series(100.0 * np.exp(np.linspace(0, 0.4, len(idx))), index=idx)
    m = build_regime_measurements(equity)
    assert m.analyst_dispersion is None
    assert "analyst_dispersion" not in m.to_dict()


def test_analyst_dispersion_uses_provider_universe():
    idx = pd.bdate_range("2020-01-01", periods=400)
    equity = pd.Series(100.0 * np.exp(np.linspace(0, 0.4, len(idx))), index=idx)
    provider = _StubProvider({"A": 0.8, "B": -0.4, "C": 0.1})
    m = build_regime_measurements(
        equity,
        external_provider=provider,
        as_of=date(2026, 5, 1),
        universe=["A", "B", "C"],
    )
    assert m.analyst_dispersion is not None
    # Manually computed std (population) for [0.8, -0.4, 0.1]:
    values = pd.Series([0.8, -0.4, 0.1])
    expected = float(values.std(ddof=0))
    assert m.analyst_dispersion == expected
    assert m.to_dict()["analyst_dispersion"] == expected


def test_analyst_dispersion_none_when_no_scores():
    idx = pd.bdate_range("2020-01-01", periods=400)
    equity = pd.Series(100.0 * np.exp(np.linspace(0, 0.4, len(idx))), index=idx)
    provider = _StubProvider({})  # empty
    m = build_regime_measurements(
        equity,
        external_provider=provider,
        as_of=date(2026, 5, 1),
        universe=["A", "B"],
    )
    assert m.analyst_dispersion is None
