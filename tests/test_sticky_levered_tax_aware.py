"""Tests for StickyLeveredTaxAware (tax-lot-aware switch buffer, T-0411)."""

import numpy as np
import pandas as pd

from strategies import StickyLeveredTaxAware as ExportedTaxAware
from strategies.sticky_levered_tax_aware import StickyLeveredTaxAware
from strategies.sticky_levered_vol_targeted import StickyLeveredVolTargeted


def _price_frame(days: int, tickers, *, ann_vol=0.30, drift=0.0004, seed=0):
    """Synthetic price frame with a defined annualized vol."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=days)
    daily_sigma = ann_vol / np.sqrt(252.0)
    frame = {}
    for t in tickers:
        rets = rng.normal(drift, daily_sigma, days)
        frame[t] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(frame, index=idx)


def test_exported_in_strategy_package():
    assert ExportedTaxAware is StickyLeveredTaxAware


def test_defaults_and_param_grid():
    s = StickyLeveredTaxAware()
    assert s.params["tax_buffer_factor"] == 0.10
    assert isinstance(s, StickyLeveredVolTargeted)
    # tax_buffer_factor is an additional sweep knob.
    assert "tax_buffer_factor" in s.get_param_grid()
    assert "rebal_band" in s.get_param_grid()


def test_factor_zero_is_identical_to_vol_targeted():
    # With tax_buffer_factor=0, NO behavioral change vs.
    # StickyLeveredVolTargeted may occur -- same allocation every day.
    tickers = ["TQQQ", "UPRO", "SOXL", "SPY"]
    frame = _price_frame(400, tickers, ann_vol=0.35, drift=0.001, seed=11)
    frame["SOXL"] *= np.linspace(1.0, 3.0, len(frame))  # SOXL clear winner
    base = StickyLeveredVolTargeted()
    taxed = StickyLeveredTaxAware(tax_buffer_factor=0.0)
    for i in range(120, len(frame), 5):
        d = frame.index[i].date()
        sub = frame.iloc[: i + 1]
        assert dict(base.signal(d, sub).weights) == dict(taxed.signal(d, sub).weights)


def test_unrealized_gain_zero_without_entry_or_in_safe():
    frame = _price_frame(120, ["SOXL", "SPY"], seed=8)
    s = StickyLeveredTaxAware()
    # No entry tracked -> 0.
    assert s._unrealized_gain(frame) == 0.0
    # Safe asset held -> no taxable switch cost.
    s._entry_ticker = "SPY"
    s._entry_price = 100.0
    assert s._unrealized_gain(frame) == 0.0


def test_unrealized_gain_tracks_price_move():
    frame = _price_frame(120, ["SOXL", "SPY"], seed=8)
    s = StickyLeveredTaxAware()
    last = float(frame["SOXL"].iloc[-1])
    s._entry_ticker = "SOXL"
    s._entry_price = last / 2.0  # position has doubled
    assert s._unrealized_gain(frame) > 0.9  # ~ +100%
    s._entry_price = last * 2.0  # position halved
    assert s._unrealized_gain(frame) < -0.4


def test_buffer_rises_when_holding_a_winner():
    # If the strategy is sitting on a large unrealized gain, the
    # effective switch_buffer must exceed the base value.
    tickers = ["TQQQ", "UPRO", "SOXL", "SPY"]
    frame = _price_frame(500, tickers, ann_vol=0.30, drift=0.0005, seed=4)
    frame["SOXL"] *= np.linspace(1.0, 4.0, len(frame))  # SOXL gains strongly
    s = StickyLeveredTaxAware(tax_buffer_factor=0.5)
    for i in range(120, len(frame), 5):
        s.signal(frame.index[i].date(), frame.iloc[: i + 1])
    assert s._entry_ticker == "SOXL"
    assert s._unrealized_gain(frame) > 0.0
    assert s.switch_buffer > s._base_switch_buffer
