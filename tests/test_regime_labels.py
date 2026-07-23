"""Tests for regime_labels.compute_regime_labels — schema, classification, PIT."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.external_features.ml.regime_labels import (
    DEFAULT_LABEL_HORIZON,
    DEFAULT_VOL_PERCENTILE_WINDOW,
    MIN_TRAILING_HISTORY,
    REGIME_NAMES,
    compute_regime_labels,
)


def _gbm_prices(n_days: int, ann_drift=0.0, ann_vol=0.18, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n_days)
    dt = 1 / 252.0
    rets = rng.normal(ann_drift * dt, ann_vol * np.sqrt(dt), n_days)
    rets[0] = 0.0
    return pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx)


def test_empty_input_returns_empty_frame_with_expected_columns():
    out = compute_regime_labels(pd.DataFrame())
    assert out.empty
    expected = {
        "as_of",
        "ticker",
        f"label_end_{DEFAULT_LABEL_HORIZON}d",
        f"next_vol_{DEFAULT_LABEL_HORIZON}d",
        f"next_max_dd_{DEFAULT_LABEL_HORIZON}d",
        "vol_threshold_fragile",
        "vol_threshold_stressed",
        "regime_label",
        "regime_name",
    }
    assert set(out.columns) == expected


def test_validation_rejects_bad_args():
    prices = pd.DataFrame({"A": _gbm_prices(50)})
    with pytest.raises(ValueError):
        compute_regime_labels(prices, label_horizon_days=0)
    with pytest.raises(ValueError):
        compute_regime_labels(prices, vol_percentile_fragile=1.5)
    with pytest.raises(ValueError):
        compute_regime_labels(
            prices, vol_percentile_fragile=0.95, vol_percentile_stressed=0.90
        )
    with pytest.raises(ValueError):
        compute_regime_labels(
            prices, dd_threshold_fragile=-0.20, dd_threshold_stressed=-0.10
        )


def test_calm_prices_label_normal():
    """Low vol + no drawdown -> mostly 'normal'."""
    n = 400
    prices = pd.DataFrame({"CALM": _gbm_prices(n, ann_drift=0.05, ann_vol=0.10, seed=1)})
    out = compute_regime_labels(prices, label_horizon_days=21)
    assert not out.empty
    counts = out["regime_name"].value_counts()
    assert counts.get("normal", 0) > 0.5 * len(out)


def test_stressed_event_produces_stressed_label():
    """Constructed crash in the forward window -> 'stressed' label."""
    idx = pd.bdate_range("2010-01-04", periods=400)
    prices = pd.Series(100.0, index=idx)
    # First 300 days stable, then crash to 50% over 30 days.
    crash_start = 300
    for i in range(crash_start, len(prices)):
        prices.iloc[i] = 100.0 * max(0.45, 1.0 - 0.025 * (i - crash_start))
    out = compute_regime_labels(
        pd.DataFrame({"CRASH": prices}), label_horizon_days=21
    )
    crash_window_start_ts = idx[crash_start - 21]
    crash_window_end_ts = idx[crash_start]
    crash_rows = out[
        (out["as_of"] >= crash_window_start_ts)
        & (out["as_of"] <= crash_window_end_ts)
    ]
    assert len(crash_rows) > 0
    # In this window AT LEAST one 'stressed' label must occur.
    assert (crash_rows["regime_label"] == 2).any()


def test_forward_window_past_data_is_dropped():
    """Last rows where a full forward window no longer fits -> dropped."""
    prices = pd.DataFrame({"A": _gbm_prices(400, seed=2)})
    out = compute_regime_labels(prices, label_horizon_days=21)
    last_as_of = out["as_of"].max()
    last_label_end = out[f"label_end_21d"].max()
    # label_end must never fall after prices.index[-1].
    assert last_label_end <= prices.index[-1]
    # There must be at least 21 trading days between the last as_of and the end of the data.
    days_remaining = (prices.index[-1] - last_as_of).days
    assert days_remaining >= 21


def test_short_trailing_history_is_dropped():
    """If trailing window < MIN_TRAILING_HISTORY -> row dropped."""
    n = 80  # too short: needs > MIN_TRAILING_HISTORY (63) + label_horizon (21)
    prices = pd.DataFrame({"A": _gbm_prices(n, seed=3)})
    out = compute_regime_labels(prices, label_horizon_days=21)
    assert out.empty or len(out) < n - MIN_TRAILING_HISTORY - 21


def test_regime_name_matches_label():
    prices = pd.DataFrame({"A": _gbm_prices(400, seed=4)})
    out = compute_regime_labels(prices, label_horizon_days=21)
    assert not out.empty
    for _, row in out.iterrows():
        assert row["regime_name"] == REGIME_NAMES[int(row["regime_label"])]


def test_pit_threshold_unaffected_by_future_data():
    """CORE PIT TEST: threshold at as_of must NOT depend on data AFTER as_of.

    Strategy: two datasets D_orig and D_poisoned that are exactly
    identical up to position X, but diverge dramatically AFTER X.
    For all as_of <= ts_X, vol_threshold_fragile/_stressed must
    be bit-identical.
    """
    n = 600
    prices_orig = pd.DataFrame({"A": _gbm_prices(n, ann_vol=0.15, seed=5)})
    prices_poisoned = prices_orig.copy()
    X = 400
    ts_X = prices_orig.index[X]
    # Poison everything AFTER X: multiply by a random factor (10x..0.1x).
    rng = np.random.default_rng(99)
    factors = rng.uniform(0.1, 10.0, n - X - 1)
    prices_poisoned.iloc[X + 1 :, 0] = (
        prices_orig.iloc[X + 1 :, 0].values * factors
    )

    out_orig = compute_regime_labels(prices_orig, label_horizon_days=21)
    out_pois = compute_regime_labels(prices_poisoned, label_horizon_days=21)

    # Comparison only for rows with as_of <= ts_X.
    mask_orig = out_orig["as_of"] <= ts_X
    mask_pois = out_pois["as_of"] <= ts_X
    rows_orig = out_orig.loc[mask_orig].sort_values(["as_of", "ticker"]).reset_index(drop=True)
    rows_pois = out_pois.loc[mask_pois].sort_values(["as_of", "ticker"]).reset_index(drop=True)
    assert len(rows_orig) == len(rows_pois) > 0

    np.testing.assert_allclose(
        rows_orig["vol_threshold_fragile"].values,
        rows_pois["vol_threshold_fragile"].values,
        rtol=0,
        atol=1e-12,
        err_msg="vol_threshold_fragile leaks future data",
    )
    np.testing.assert_allclose(
        rows_orig["vol_threshold_stressed"].values,
        rows_pois["vol_threshold_stressed"].values,
        rtol=0,
        atol=1e-12,
        err_msg="vol_threshold_stressed leaks future data",
    )


def test_pit_label_does_depend_on_future_data():
    """Sanity check: the LABEL itself is forward-looking, so it MUST
    change if forward data is manipulated. Otherwise the label would
    NOT be informative.
    """
    n = 400
    prices_orig = pd.DataFrame({"A": _gbm_prices(n, ann_vol=0.10, seed=6)})
    prices_poisoned = prices_orig.copy()
    # Severe crash AFTER position 200.
    X = 200
    for i in range(X + 1, n):
        prices_poisoned.iloc[i, 0] = prices_orig.iloc[X, 0] * max(
            0.3, 1.0 - 0.03 * (i - X)
        )

    out_orig = compute_regime_labels(prices_orig, label_horizon_days=21)
    out_pois = compute_regime_labels(prices_poisoned, label_horizon_days=21)
    ts_X = prices_orig.index[X]
    # For as_of near X (forward window falls into crash), the
    # label should change (more stressed labels after poisoning).
    near_X = (out_orig["as_of"] >= prices_orig.index[X - 20]) & (
        out_orig["as_of"] <= ts_X
    )
    n_stressed_orig = (out_orig.loc[near_X, "regime_label"] == 2).sum()
    near_X_pois = (out_pois["as_of"] >= prices_orig.index[X - 20]) & (
        out_pois["as_of"] <= ts_X
    )
    n_stressed_pois = (out_pois.loc[near_X_pois, "regime_label"] == 2).sum()
    assert n_stressed_pois > n_stressed_orig


def test_as_of_range_restriction():
    prices = pd.DataFrame({"A": _gbm_prices(400, seed=7)})
    pick = [prices.index[100], prices.index[200], prices.index[300]]
    out = compute_regime_labels(prices, as_of_range=pick)
    # At most len(pick) entries (possibly fewer if a pick is dropped).
    assert len(out) <= len(pick)
    assert set(out["as_of"]).issubset(set(pick))


def test_multi_ticker_independent_labels():
    """Different tickers are classified independently (own thresholds)."""
    n = 400
    prices = pd.DataFrame(
        {
            "CALM": _gbm_prices(n, ann_vol=0.10, seed=10),
            "WILD": _gbm_prices(n, ann_vol=0.50, seed=11),
        }
    )
    out = compute_regime_labels(prices, label_horizon_days=21)
    calm_thresh = out[out["ticker"] == "CALM"]["vol_threshold_fragile"]
    wild_thresh = out[out["ticker"] == "WILD"]["vol_threshold_fragile"]
    # WILD has higher trailing vol -> higher threshold values.
    assert wild_thresh.median() > calm_thresh.median()
