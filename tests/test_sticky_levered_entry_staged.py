"""Tests for StickyLeveredEntryStaged (staged entry, T-0412)."""

import numpy as np
import pandas as pd

from strategies import StickyLeveredEntryStaged as ExportedStaged
from strategies.sticky_levered_entry_staged import StickyLeveredEntryStaged
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
    assert ExportedStaged is StickyLeveredEntryStaged


def test_defaults_and_param_grid():
    s = StickyLeveredEntryStaged()
    assert s.params["entry_stage_periods"] == 4
    assert isinstance(s, StickyLeveredVolTargeted)
    assert "entry_stage_periods" in s.get_param_grid()
    assert "rebal_band" in s.get_param_grid()


def test_one_stage_is_identical_to_vol_targeted():
    # entry_stage_periods=1 must NOT produce any behavioral change.
    tickers = ["TQQQ", "UPRO", "SOXL", "SPY"]
    frame = _price_frame(400, tickers, ann_vol=0.35, drift=0.001, seed=11)
    frame["SOXL"] *= np.linspace(1.0, 3.0, len(frame))
    base = StickyLeveredVolTargeted()
    staged = StickyLeveredEntryStaged(entry_stage_periods=1)
    for i in range(120, len(frame), 5):
        d = frame.index[i].date()
        sub = frame.iloc[: i + 1]
        assert dict(base.signal(d, sub).weights) == dict(staged.signal(d, sub).weights)


def test_first_entry_is_under_full_weight():
    # On the first entry into a fresh pick, the risk weight must
    # NOT immediately match the unstaged full weight.
    tickers = ["TQQQ", "UPRO", "SOXL", "SPY"]
    frame = _price_frame(200, tickers, ann_vol=0.20, drift=0.001, seed=1)
    frame["SOXL"] *= np.linspace(1.0, 3.0, len(frame))  # SOXL clear winner
    d = frame.index[-1].date()
    base = StickyLeveredVolTargeted()
    staged = StickyLeveredEntryStaged(entry_stage_periods=4)
    w_base = dict(base.signal(d, frame).weights)
    w_staged = dict(staged.signal(d, frame).weights)
    picked = staged._picked
    # First ramp step -> only 1/4 of the full weight is invested.
    assert w_staged.get(picked, 0.0) < w_base.get(picked, 0.0)
    assert w_staged.get("SPY", 0.0) > 0.0
    assert abs(sum(w_staged.values()) - 1.0) < 1e-9


def test_ramp_reaches_full_weight_after_stage_periods():
    # After entry_stage_periods calls on the same pick, the ramp is complete.
    tickers = ["TQQQ", "UPRO", "SOXL", "SPY"]
    frame = _price_frame(300, tickers, ann_vol=0.18, drift=0.001, seed=2)
    frame["SOXL"] *= np.linspace(1.0, 3.5, len(frame))
    staged = StickyLeveredEntryStaged(entry_stage_periods=3)
    base = StickyLeveredVolTargeted()
    # Enough calls in the same calendar month for the ramp to complete.
    idx = frame.index[-1]
    for _ in range(5):
        staged.signal(idx.date(), frame)
        w_full = dict(base.signal(idx.date(), frame).weights)
    w_staged = dict(staged.signal(idx.date(), frame).weights)
    assert w_staged == w_full


def test_safe_phase_resets_staging_state():
    # No staging state may persist during the safe phase.
    frame = _price_frame(120, ["TQQQ", "UPRO", "SOXL", "SPY"], seed=3)
    s = StickyLeveredEntryStaged(entry_stage_periods=4)
    s._staged_ticker = "SOXL"
    s._periods_held = 2
    # Empty frame -> safe phase.
    s.signal(frame.index[-1].date(), pd.DataFrame())
    assert s._staged_ticker is None
    assert s._periods_held == 0
