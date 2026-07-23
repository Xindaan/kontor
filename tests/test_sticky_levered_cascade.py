"""Tests for StickyLeveredCascade (multi-3x allocation: single_pick/cascade/inverse_vol)."""

from datetime import date

import numpy as np
import pandas as pd

from strategies import StickyLeveredCascade as ExportedCascade
from strategies.sticky_levered_cascade import StickyLeveredCascade
from strategies.sticky_levered_vol_targeted import StickyLeveredVolTargeted


def _frame(days=200, seed=0, *, soxl_trend=2.5, soxl_vol=0.80):
    """Synthetic frame: SOXL = clear momentum winner with high vol."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=days)
    qqq3 = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.40 / np.sqrt(252.0), days)))
    lus = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.25 / np.sqrt(252.0), days)))
    soxl = 100.0 * np.exp(np.cumsum(rng.normal(0.001, soxl_vol / np.sqrt(252.0), days)))
    soxl *= np.linspace(1.0, soxl_trend, days)  # clear uptrend
    sxr8 = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.12 / np.sqrt(252.0), days)))
    return pd.DataFrame(
        {"TQQQ": qqq3, "UPRO": lus, "SOXL": soxl, "SPY": sxr8}, index=idx
    )


def test_exported_in_strategy_package():
    assert ExportedCascade is StickyLeveredCascade
    assert isinstance(StickyLeveredCascade(), StickyLeveredVolTargeted)


def test_invalid_modes_raise():
    import pytest

    with pytest.raises(ValueError):
        StickyLeveredCascade(allocation_mode="bogus")
    with pytest.raises(ValueError):
        StickyLeveredCascade(cascade_order="bogus")


def test_single_pick_mode_matches_vol_targeted_base():
    """allocation_mode='single_pick' must be bit-identical to StickyLeveredVolTargeted."""
    frame = _frame(seed=1)
    cands = ["TQQQ", "UPRO", "SOXL"]
    casc = StickyLeveredCascade(allocation_mode="single_pick", candidates=cands)
    base = StickyLeveredVolTargeted(candidates=cands)
    d = frame.index[-1].date()
    a_casc = dict(casc.signal(d, frame).weights)
    a_base = dict(base.signal(d, frame).weights)
    assert a_casc.keys() == a_base.keys()
    for k in a_casc:
        assert abs(a_casc[k] - a_base[k]) < 1e-9


def test_cascade_top_pick_weight_equals_single_pick():
    """CORE INVARIANT (user requirement): the cascade's top pick has
    EXACTLY the weight single_pick would give it."""
    frame = _frame(seed=2)
    cands = ["TQQQ", "UPRO", "SOXL"]
    d = frame.index[-1].date()
    single = StickyLeveredVolTargeted(candidates=cands)
    a_single = dict(single.signal(d, frame).weights)
    # Top pick = the non-safe ticker in the single_picks allocation.
    pick = next((t for t in a_single if t != "SPY"), None)
    assert pick is not None  # in this frame, SOXL is the momentum winner
    w_single = a_single[pick]
    casc = StickyLeveredCascade(
        allocation_mode="cascade", cascade_order="vol", candidates=cands
    )
    a_casc = dict(casc.signal(d, frame).weights)
    assert pick in a_casc
    assert abs(a_casc[pick] - w_single) < 1e-9


def test_cascade_fills_remainder_into_other_3x():
    """The cascade pushes the remainder into further 3x assets instead of fully into SPY."""
    frame = _frame(seed=3)
    cands = ["TQQQ", "UPRO", "SOXL"]
    d = frame.index[-1].date()
    casc = StickyLeveredCascade(
        allocation_mode="cascade", cascade_order="vol",
        max_total_leverage=1.0, candidates=cands,
    )
    w = dict(casc.signal(d, frame).weights)
    # At least one further 3x asset besides the top pick is allocated.
    three_x = [t for t in w if t in ("TQQQ", "UPRO", "SOXL")]
    assert len(three_x) >= 2
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_max_total_leverage_caps_3x_sum():
    """max_total_leverage caps the sum of all 3x weights."""
    frame = _frame(seed=4)
    cands = ["TQQQ", "UPRO", "SOXL"]
    d = frame.index[-1].date()
    casc = StickyLeveredCascade(
        allocation_mode="cascade", cascade_order="vol",
        max_total_leverage=0.75, candidates=cands,
    )
    w = dict(casc.signal(d, frame).weights)
    three_x_sum = sum(v for t, v in w.items() if t in ("TQQQ", "UPRO", "SOXL"))
    assert three_x_sum <= 0.75 + 1e-9
    assert w.get("SPY", 0.0) >= 0.25 - 1e-9


def test_inverse_vol_mode_allocates_all_positive_momentum_3x():
    frame = _frame(seed=5)
    cands = ["TQQQ", "UPRO", "SOXL"]
    d = frame.index[-1].date()
    casc = StickyLeveredCascade(allocation_mode="inverse_vol", candidates=cands)
    w = dict(casc.signal(d, frame).weights)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    # Low-vol asset gets more weight than high-vol asset.
    if "UPRO" in w and "SOXL" in w:
        assert w["UPRO"] > w["SOXL"]


def test_empty_data_goes_safe():
    casc = StickyLeveredCascade(allocation_mode="cascade")
    w = dict(casc.signal(date(2020, 6, 1), pd.DataFrame()).weights)
    assert w == {"SPY": 1.0}
