"""Phase D ml_schema validation tests (T-0213)."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.external_features.ml.ml_schema import (
    REQUIRED_FEATURES,
    validate_ml_snapshot,
)


def _row(ticker: str, feature: str, value: float) -> dict:
    return {
        "ticker": ticker,
        "feature_name": feature,
        "feature_value": value,
    }


def _required_rows(ticker: str, score: float = 0.0, available_from: float = 0.0):
    return [
        _row(ticker, "ml_forecast_score", score),
        _row(ticker, "ml_available_from_ordinal", available_from),
        _row(ticker, "ml_feature_trained_through_ordinal", available_from),
    ]


def test_validate_passes_on_complete_snapshot():
    rows = []
    for ticker in ("AAA", "BBB"):
        rows.extend(_required_rows(ticker, score=0.42))
    validate_ml_snapshot(pd.DataFrame(rows))


def test_missing_required_feature_raises():
    rows = [_row("AAA", "ml_forecast_score", 0.0)]
    with pytest.raises(ValueError):
        validate_ml_snapshot(pd.DataFrame(rows))


def test_out_of_range_score_raises():
    rows = _required_rows("AAA", score=1.5)
    with pytest.raises(ValueError):
        validate_ml_snapshot(pd.DataFrame(rows))


def test_required_feature_names_complete():
    assert "ml_forecast_score" in REQUIRED_FEATURES
    assert "ml_available_from_ordinal" in REQUIRED_FEATURES
    assert "ml_feature_trained_through_ordinal" in REQUIRED_FEATURES
