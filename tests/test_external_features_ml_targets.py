"""Phase D: forward-return target computation (T-0202)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.external_features.ml.targets import compute_forward_returns


def _make_prices(start: str, days: int, tickers: list[str]) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=days)
    data = {}
    for t in tickers:
        # monotonically increasing prices to keep forward returns positive.
        data[t] = np.linspace(100.0, 150.0, days)
    return pd.DataFrame(data, index=idx)


def test_forward_returns_emits_label_end_columns():
    prices = _make_prices("2024-01-01", 400, ["AAA", "BBB"])
    out = compute_forward_returns(prices, horizons_days=[21, 63, 252])
    expected_cols = {
        "as_of",
        "ticker",
        "horizon_21d",
        "horizon_63d",
        "horizon_252d",
        "label_end_21d",
        "label_end_63d",
        "label_end_252d",
    }
    assert expected_cols.issubset(set(out.columns))


def test_forward_returns_nan_when_window_unavailable():
    prices = _make_prices("2024-01-01", 30, ["AAA"])
    out = compute_forward_returns(prices, horizons_days=[21, 63])
    # 30 BDays only — 63d forward returns must all be NaN.
    assert out["horizon_63d"].isna().all()


def test_label_end_is_business_day():
    prices = _make_prices("2024-01-01", 400, ["AAA"])
    out = compute_forward_returns(prices, horizons_days=[21])
    sample = out.dropna(subset=["label_end_21d"]).iloc[0]
    label_end = pd.Timestamp(sample["label_end_21d"])
    # Label-end must be a business day (Codex D16).
    assert label_end.weekday() < 5
