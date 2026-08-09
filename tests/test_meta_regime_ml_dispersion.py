"""Phase D meta_regime.ml_dispersion smoke (T-0219)."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

from backtest.meta_regime import RegimeMeasurements, build_regime_measurements


def test_ml_dispersion_field_optional():
    rm = RegimeMeasurements(status="ok")
    assert rm.ml_dispersion is None
    payload = rm.to_dict()
    # Default-None ml_dispersion should not pollute dict output.
    assert "ml_dispersion" not in payload


def test_ml_dispersion_serialises_when_set():
    rm = RegimeMeasurements(status="ok", ml_dispersion=0.42)
    assert rm.to_dict()["ml_dispersion"] == 0.42


def test_build_measurements_accepts_ml_datasets_kwarg():
    # Provider is None — function must still accept the kwarg gracefully.
    rm = build_regime_measurements(
        pd.Series([100.0] * 5),
        ml_datasets=("synthetic_ml_forecast",),
    )
    assert rm.ml_dispersion is None


def test_build_measurements_computes_dispersion_from_provider():
    class _FakeSnap:
        def __init__(self, df):
            self.data = df

    class _FakeProvider:
        def snapshot_dataset(self, dataset, *, as_of, tickers):
            df = pd.DataFrame(
                [
                    {"ticker": "AAA", "feature_name": "ml_forecast_score",
                     "feature_value": 0.6},
                    {"ticker": "BBB", "feature_name": "ml_forecast_score",
                     "feature_value": -0.4},
                ]
            )
            return _FakeSnap(df)

    rm = build_regime_measurements(
        pd.Series([100.0] * 5),
        external_provider=_FakeProvider(),
        as_of=date(2026, 5, 13),
        universe=("AAA", "BBB"),
        ml_datasets=("synthetic_ml_forecast",),
    )
    assert rm.ml_dispersion is not None
    assert rm.ml_dispersion > 0
