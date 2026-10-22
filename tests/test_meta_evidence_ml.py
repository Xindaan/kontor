"""Phase D meta_evidence ml-forecast tests (T-0223/T-0224)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from backtest.meta_evidence import (
    _ml_snapshot_filter_lookahead,
    _ml_snapshot_is_synthetic,
    _ml_tilt_weights,
    assess_conditioned_ml_evidence_status,
    assess_ml_evidence,
)


def _snapshot(rows):
    return pd.DataFrame(rows)


def test_lookahead_filter_drops_future_models():
    """Codex D14/D18: filter rows whose available_from > release_date."""

    future_ordinal = date(2027, 1, 1).toordinal()
    past_ordinal = date(2024, 1, 1).toordinal()
    rows = [
        {"ticker": "AAA", "feature_name": "ml_forecast_score", "feature_value": 0.5,
         "source": "X"},
        {"ticker": "AAA", "feature_name": "ml_available_from_ordinal",
         "feature_value": float(future_ordinal), "source": "X"},
        {"ticker": "BBB", "feature_name": "ml_forecast_score", "feature_value": 0.5,
         "source": "X"},
        {"ticker": "BBB", "feature_name": "ml_available_from_ordinal",
         "feature_value": float(past_ordinal), "source": "X"},
    ]
    df = _snapshot(rows)
    filtered = _ml_snapshot_filter_lookahead(df, release_date=date(2025, 1, 1))
    remaining = set(filtered["ticker"].unique())
    assert remaining == {"BBB"}


def test_synthetic_detection():
    df_synth = _snapshot(
        [{"ticker": "AAA", "feature_name": "ml_forecast_score",
          "feature_value": 0.0, "source": "SyntheticMLForecast"}]
    )
    df_real = _snapshot(
        [{"ticker": "AAA", "feature_name": "ml_forecast_score",
          "feature_value": 0.0, "source": "LightGBMForecast"}]
    )
    assert _ml_snapshot_is_synthetic(df_synth) is True
    assert _ml_snapshot_is_synthetic(df_real) is False


def test_ml_tilt_top_quartile_equal_weight():
    rows = [
        {"ticker": t, "feature_name": "ml_forecast_score", "feature_value": v,
         "source": "X"}
        for t, v in [
            ("AAA", 0.9),
            ("BBB", 0.5),
            ("CCC", 0.2),
            ("DDD", -0.3),
        ]
    ]
    weights = _ml_tilt_weights(_snapshot(rows), ["AAA", "BBB", "CCC", "DDD"])
    # Top-quartile cutoff at 1 element => AAA only.
    assert weights == {"AAA": 1.0}


def test_assess_ml_evidence_no_provider_returns_missing():
    status, reasons, summary = assess_ml_evidence(
        external_provider=None,
        current_assets=["AAA"],
        candidate_assets=["BBB"],
        as_of=date(2026, 5, 13),
    )
    assert status == "missing"
    assert summary is None


def test_assess_conditioned_ml_handles_missing_summary():
    status, reasons, num_windows, bucket = assess_conditioned_ml_evidence_status(
        summary=None, current_bucket="normal"
    )
    assert status == "missing"
    assert num_windows == 0
