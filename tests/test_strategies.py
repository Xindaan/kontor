"""Smoke tests for the curated root strategy set."""

from datetime import date

import pandas as pd

from strategies import (
    BuyAndHold,
    Levered5xMomentumGuard,
    LeveredETFMomentumSticky,
    LeveredETFMomentumStickyAdaptiveV2,
    LeveredETFMomentumStickyAdaptiveV2,
    LeveredETFMomentumStickyAdaptiveV2,
    LeveredETFMomentumSticky,
    LeveredETFMomentumSticky,
    LeveredMomentumCrashGuard,
    LeveredMomentumCrashGuardLBHChallenger,
)


def test_curated_strategy_exports_are_importable():
    strategies = [
        BuyAndHold({"SPY": 1.0}),
        Levered5xMomentumGuard(),
        LeveredETFMomentumSticky(),
        LeveredETFMomentumStickyAdaptiveV2(),
        LeveredETFMomentumStickyAdaptiveV2(),
        LeveredETFMomentumStickyAdaptiveV2(),
        LeveredETFMomentumSticky(),
        LeveredETFMomentumSticky(),
        LeveredMomentumCrashGuard(),
        LeveredMomentumCrashGuardLBHChallenger(),
    ]

    assert all(strategy.name for strategy in strategies)
    assert all(strategy.assets for strategy in strategies)


def test_buy_and_hold_signal_is_constant():
    strategy = BuyAndHold({"SPY": 0.6, "BND": 0.4})

    signal1 = strategy.signal(date(2020, 1, 1), pd.DataFrame())
    signal2 = strategy.signal(date(2021, 1, 1), pd.DataFrame())

    assert signal1.weights == {"SPY": 0.6, "BND": 0.4}
    assert signal2.weights == signal1.weights


def test_us_sticky_defaults_match_robust_core_params():
    strategy = LeveredETFMomentumSticky()

    assert strategy.params["lookback_days"] == 63
    assert strategy.params["switch_buffer"] == 0.04
    assert strategy.params["min_hold_periods"] == 2
    assert strategy.params["momentum_floor"] == 0.0
    assert strategy.params["baseline_drawdown_floor"] == -1.0


def test_sticky_selects_best_momentum_asset():
    strategy = LeveredETFMomentumSticky(
        candidates=["TQQQ", "UPRO"],
        lookback_days=2,
        switch_buffer=0.05,
        min_hold_periods=1,
    )
    data = pd.DataFrame(
        {
            "TQQQ": [100, 110, 130],
            "UPRO": [100, 104, 108],
            "SPY": [100, 101, 102],
        }
    )

    signal = strategy.signal(date(2026, 4, 13), data)

    assert signal.weights == {"TQQQ": 1.0}
