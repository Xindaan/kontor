"""Tests for StickyLeveredVolTargeted (vol-targeting overlay on Sticky Levered)."""

from datetime import date

import numpy as np
import pandas as pd

from backtest.backtester import Backtester, BacktestConfig
from backtest.data import PriceData
from strategies import StickyLeveredVolTargeted as ExportedVolTgt
from strategies.sticky_levered_vol_targeted import StickyLeveredVolTargeted, _realized_vol


def _price_frame(days: int, tickers, *, ann_vol=0.30, drift=0.0004, seed=0):
    """Synthetic price frame with a defined annualized volatility."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=days)
    daily_sigma = ann_vol / np.sqrt(252.0)
    frame = {}
    for i, t in enumerate(tickers):
        rets = rng.normal(drift, daily_sigma, days)
        frame[t] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(frame, index=idx)


def test_exported_in_strategy_package():
    assert ExportedVolTgt is StickyLeveredVolTargeted


def test_defaults_use_liquid_us_3x_proxies():
    s = StickyLeveredVolTargeted()
    # Liquid US 3x ETFs as the default universe (long Yahoo history).
    assert s.params["candidates"] == ["TQQQ", "UPRO", "SOXL"]
    assert s.params["safe_asset"] == "SPY"
    assert s.params["vol_target"] == 0.40
    assert s.params["vol_lookback_days"] == 20
    assert s.params["rebal_band"] == 0.20
    assert s.rebalance_frequency == "weekly"
    assert "SPY" in s.assets


def test_realized_vol_matches_known_sigma():
    # Series with ~40% annualized vol -> _realized_vol close to 0.40.
    frame = _price_frame(120, ["X"], ann_vol=0.40, drift=0.0, seed=42)
    rv = _realized_vol(frame["X"], 60)
    assert rv is not None
    assert 0.30 < rv < 0.52


def test_realized_vol_insufficient_data_returns_none():
    frame = _price_frame(3, ["X"])
    assert _realized_vol(frame["X"], 20) is None


def test_empty_data_goes_safe():
    s = StickyLeveredVolTargeted()
    alloc = s.signal(date(2020, 6, 1), pd.DataFrame())
    weights = dict(alloc.weights)
    assert weights == {"SPY": 1.0}


def test_low_vol_pick_stays_near_full_weight():
    # Low vol (15%) << vol_target (40%) -> w_risky clipped to 1.0.
    tickers = ["TQQQ", "UPRO", "SOXL", "SPY"]
    frame = _price_frame(200, tickers, ann_vol=0.15, drift=0.001, seed=1)
    # Make TQQQ the clear momentum winner.
    frame["TQQQ"] *= np.linspace(1.0, 2.5, len(frame))
    s = StickyLeveredVolTargeted()
    alloc = s.signal(frame.index[-1].date(), frame)
    weights = dict(alloc.weights)
    # With vol << target the risk weight is 1.0 (no safe sleeve).
    assert weights.get("TQQQ", 0.0) == 1.0


def test_high_vol_pick_gets_de_levered():
    # SOXL: clear upward momentum winner, but HIGH vol (>> 40%).
    # -> gets picked, but the overlay de-levers it (w_risky < 1.0).
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(7)
    # SOXL: strong trend + high vol (~80% annualized).
    soxl_rets = rng.normal(0.004, 0.80 / np.sqrt(252.0), 200)
    # Other 3x: slightly falling, so SOXL is the unambiguous leader.
    tqqq = 100.0 * np.exp(np.cumsum(rng.normal(-0.001, 0.20 / np.sqrt(252.0), 200)))
    upro = 100.0 * np.exp(np.cumsum(rng.normal(-0.001, 0.20 / np.sqrt(252.0), 200)))
    frame = pd.DataFrame(
        {
            "TQQQ": tqqq,
            "UPRO": upro,
            "SOXL": 100.0 * np.exp(np.cumsum(soxl_rets)),
            "SPY": 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.12 / np.sqrt(252.0), 200))),
        },
        index=idx,
    )
    s = StickyLeveredVolTargeted()
    alloc = s.signal(frame.index[-1].date(), frame)
    weights = dict(alloc.weights)
    risky = weights.get("SOXL", 0.0)
    safe = weights.get("SPY", 0.0)
    assert 0.0 < risky < 1.0
    assert safe > 0.0
    assert abs(risky + safe - 1.0) < 1e-9


def test_rebal_band_suppresses_small_weight_changes():
    # Once a risk weight is established, a small vol change must NOT move
    # the weight (turnover damper).
    tickers = ["TQQQ", "UPRO", "SOXL", "SPY"]
    frame = _price_frame(200, tickers, ann_vol=0.60, drift=0.0015, seed=3)
    frame["TQQQ"] *= np.linspace(1.0, 2.8, len(frame))
    s = StickyLeveredVolTargeted(rebal_band=0.99)  # extremely wide band
    s.signal(frame.index[-1].date(), frame)
    w1 = s._cur_risky_weight
    # Next call in the same month -> pick unchanged, band suppresses the change.
    s.signal(frame.index[-1].date(), frame)
    assert s._cur_risky_weight == w1


def test_backtester_respects_strategy_weekly_frequency_when_config_is_none():
    tickers = ["TQQQ", "UPRO", "SOXL", "SPY"]
    frame = _price_frame(180, tickers, ann_vol=0.30, drift=0.001, seed=5)
    strategy = StickyLeveredVolTargeted()
    config = BacktestConfig(
        initial_capital=10_000.0,
        rebalance_frequency=None,
        tax_enabled=False,
        validate=False,
    )

    result = Backtester(
        strategy=strategy,
        data=PriceData(prices=frame, currency={ticker: "USD" for ticker in tickers}),
        config=config,
    ).run()

    assert result.config.rebalance_frequency == "weekly"


def test_monthly_pick_refresh_once_per_month():
    tickers = ["TQQQ", "UPRO", "SOXL", "SPY"]
    frame = _price_frame(300, tickers, ann_vol=0.30, drift=0.001, seed=4)
    s = StickyLeveredVolTargeted()
    d1 = frame.index[200].date()
    s.signal(d1, frame.loc[:frame.index[200]])
    month_after_first = s._last_pick_month
    # Same month -> _last_pick_month unchanged.
    s.signal(d1, frame.loc[:frame.index[200]])
    assert s._last_pick_month == month_after_first
