"""Tests for analyst_schema.validate_analyst_snapshot (Phase B T-0054)."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.external_features.analyst_schema import validate_analyst_snapshot
from backtest.external_features.schema import validate_schema


def _row(ticker, feature_name, feature_value):
    return {
        "ticker": ticker,
        "release_date": pd.Timestamp("2026-05-01"),
        "snapshot_ts": pd.Timestamp("2026-05-13T12:00:00Z"),
        "feature_name": feature_name,
        "feature_value": feature_value,
        "source": "TestSource",
        "dataset": "test_ds",
    }


def test_empty_frame_is_ok():
    validate_analyst_snapshot(pd.DataFrame())


def test_minimal_required_row_passes():
    df = pd.DataFrame([_row("AAA", "analyst_score", 0.5)])
    validate_analyst_snapshot(df)


def test_missing_required_score_row_for_a_ticker_raises():
    df = pd.DataFrame(
        [
            _row("AAA", "analyst_score", 0.5),
            _row("BBB", "analyst_buy_count", 3),  # no analyst_score row for BBB
        ]
    )
    with pytest.raises(ValueError, match="missing 'analyst_score'"):
        validate_analyst_snapshot(df)


def test_score_out_of_range_raises():
    df = pd.DataFrame([_row("AAA", "analyst_score", 1.5)])
    with pytest.raises(ValueError, match="within"):
        validate_analyst_snapshot(df)


def test_score_below_range_raises():
    df = pd.DataFrame([_row("AAA", "analyst_score", -2.0)])
    with pytest.raises(ValueError, match="within"):
        validate_analyst_snapshot(df)


def test_negative_count_raises():
    df = pd.DataFrame(
        [
            _row("AAA", "analyst_score", 0.1),
            _row("AAA", "analyst_buy_count", -1),
        ]
    )
    with pytest.raises(ValueError, match="analyst_buy_count"):
        validate_analyst_snapshot(df)


def test_fractional_count_raises():
    df = pd.DataFrame(
        [
            _row("AAA", "analyst_score", 0.1),
            _row("AAA", "analyst_hold_count", 2.5),
        ]
    )
    with pytest.raises(ValueError, match="integer-like"):
        validate_analyst_snapshot(df)


def test_negative_price_target_raises():
    df = pd.DataFrame(
        [
            _row("AAA", "analyst_score", 0.0),
            _row("AAA", "price_target_mean", -10.0),
        ]
    )
    with pytest.raises(ValueError, match="price_target_mean"):
        validate_analyst_snapshot(df)


def test_missing_required_column_raises():
    df = pd.DataFrame([{"ticker": "AAA", "feature_value": 0.1}])
    with pytest.raises(ValueError, match="feature_name"):
        validate_analyst_snapshot(df)


def test_generic_schema_validate_still_works_without_analyst_rules():
    # Mock-style snapshot (no analyst_score) must remain valid against
    # the generic schema. Analyst-specific validator is intentionally
    # decoupled.
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "release_date": pd.Timestamp("2026-05-01"),
                "snapshot_ts": pd.Timestamp("2026-05-13T12:00:00Z"),
                "feature_name": "score",
                "feature_value": 0.42,
                "source": "MockSource",
                "dataset": "mock_analyst",
            }
        ]
    )
    validate_schema(df)
