"""Phase E3 — sequence tensor tests (T-0343, Codex R3.12).

framework-free (no torch needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.external_features.ml.models._sequence import build_sequence_tensor


def _toy_frame(n_per_ticker: int = 80) -> pd.DataFrame:
    """Long-form feature frame with two tickers."""
    rows = []
    days = pd.bdate_range("2024-01-01", periods=n_per_ticker)
    for ticker in ["AAA", "BBB"]:
        for i, d in enumerate(days):
            rows.append(
                {
                    "as_of": d,
                    "ticker": ticker,
                    "f1": float(i),
                    "f2": float(i * 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_build_sequence_tensor_shape():
    frame = _toy_frame(80)
    tensor, meta = build_sequence_tensor(frame, seq_len=21)
    # Per ticker: 80 - 21 + 1 = 60 windows. 2 tickers -> 120 rows.
    assert tensor.shape == (120, 21, 2)
    assert len(meta) == 120


def test_build_sequence_tensor_excludes_future():
    """Codex R3.12: sub-window contains ONLY values up to and including as_of."""

    frame = _toy_frame(80)
    tensor, meta = build_sequence_tensor(frame, seq_len=21)
    # For a specific row: check that the last element of the sub-window
    # is the value at `as_of` (not a future one).
    for idx, (as_of, ticker) in enumerate(meta):
        # `f1` from the frame for `(ticker, as_of)`:
        row = frame.loc[
            (frame["ticker"] == ticker) & (frame["as_of"] == as_of)
        ].iloc[0]
        expected_last_f1 = float(row["f1"])
        actual_last_f1 = float(tensor[idx, -1, 0])  # f1 is the 1st feature
        assert actual_last_f1 == expected_last_f1, (idx, as_of, ticker)


def test_build_sequence_tensor_rejects_short_history():
    """seq_len > number of days per ticker -> empty tensor."""

    frame = _toy_frame(10)
    tensor, meta = build_sequence_tensor(frame, seq_len=21)
    assert tensor.shape[0] == 0
    assert meta == []


def test_build_sequence_tensor_leaky_feature_detection():
    """If someone puts `shift(-1)` into the frame (future leak), the
    last value of the sub-window is the future. Test for an
    explicit leaky value.
    """

    frame = _toy_frame(80)
    # Inject `leaky_feature = next-day-f1`. With a correct builder
    # implementation, the sub-window contains this value too
    # (the builder cannot detect that the value is leaky — that
    # is the job of the feature engineering layer).
    # This test shows that the builder copies feature values 1:1, which
    # means: leaky features in the input = leaky features in the tensor.
    # So the test is not a builder protection check, but rather
    # documentation of the contract.
    frame["leaky_feature"] = (
        frame.groupby("ticker")["f1"].shift(-1).fillna(method="ffill")
    )
    tensor, meta = build_sequence_tensor(
        frame, seq_len=21, feature_columns=["f1", "leaky_feature"]
    )
    # For a `(ticker, as_of=T)` row, the sub-window at pos
    # (seq_len-1, 1) contains the `leaky_feature` value at T, which
    # equals `f1` at T+1 (leak). The test documents the behavior — the
    # builder does not prevent this, that's the job of feature engineering.
    assert tensor.shape == (120, 21, 2)
    # Since the `leaky_feature` value at T is `f1`@(T+1) and our
    # toy frame has `f1 = i`, `leaky_feature`@T = i+1 if T+1 is
    # in the series, otherwise ffill = last i. We simply require
    # finiteness + shape.
    assert np.isfinite(tensor).all()


def test_explicit_feature_columns_subset():
    frame = _toy_frame(40)
    tensor, _ = build_sequence_tensor(
        frame, seq_len=10, feature_columns=["f2"]
    )
    # 40 - 10 + 1 = 31 windows per ticker * 2 tickers = 62 rows.
    assert tensor.shape == (62, 10, 1)


def test_per_ticker_isolation():
    """Sub-windows must be isolated per ticker — no cross-
    ticker leak."""
    frame = _toy_frame(40)
    tensor, meta = build_sequence_tensor(
        frame, seq_len=5, feature_columns=["f1"]
    )
    # For row (ticker=AAA, as_of=last_day_AAA) the last 5
    # `f1` values must come ONLY from AAA. The toy has `f1=i` for both
    # tickers in parallel, so we check the exact sequence: [n-5..n-1].
    aaa_indices = [i for i, (_, t) in enumerate(meta) if t == "AAA"]
    last_aaa = aaa_indices[-1]
    expected = np.array([35.0, 36.0, 37.0, 38.0, 39.0])
    np.testing.assert_array_equal(tensor[last_aaa, :, 0], expected)
