from datetime import date

import pandas as pd

from strategies import LeveredETFMomentumStickyAdaptiveV2 as ExportedAdaptiveStickyV2
from strategies.levered_etf_momentum_sticky import LeveredETFMomentumSticky
from strategies.levered_etf_momentum_sticky_adaptive_v2 import LeveredETFMomentumStickyAdaptiveV2


def test_adaptive_sticky_v2_is_exported_in_strategy_package():
    assert ExportedAdaptiveStickyV2 is LeveredETFMomentumStickyAdaptiveV2


def test_adaptive_sticky_v2_matches_original_when_no_challengers_are_enabled():
    original = LeveredETFMomentumSticky(
        lookback_days=2,
        switch_buffer=0.05,
        min_hold_periods=1,
        momentum_floor=0.0,
    )
    adaptive = LeveredETFMomentumStickyAdaptiveV2(
        challenger_map={},
        lookback_days=2,
        switch_buffer=0.05,
        min_hold_periods=1,
        momentum_floor=0.0,
    )

    frames = [
        pd.DataFrame(
            {
                "TQQQ": [100, 105, 110, 120, 130],
                "UPRO": [100, 103, 106, 108, 110],
                "SOXL": [100, 104, 108, 112, 116],
                "TECL": [100, 102, 104, 107, 109],
                "SPY": [100, 101, 102, 103, 104],
            }
        ),
        pd.DataFrame(
            {
                "TQQQ": [100, 105, 110, 120, 130, 134],
                "UPRO": [100, 103, 106, 108, 110, 112],
                "SOXL": [100, 104, 108, 112, 116, 118],
                "TECL": [100, 102, 104, 107, 109, 113],
                "SPY": [100, 101, 102, 103, 104, 105],
            }
        ),
        pd.DataFrame(
            {
                "TQQQ": [100, 105, 110, 120, 130, 134, 136],
                "UPRO": [100, 103, 106, 108, 110, 112, 113],
                "SOXL": [100, 104, 108, 112, 116, 118, 119],
                "TECL": [100, 102, 104, 107, 109, 113, 128],
                "SPY": [100, 101, 102, 103, 104, 105, 106],
            }
        ),
    ]

    for idx, frame in enumerate(frames, start=3):
        current_date = date(2026, idx, 25)
        assert adaptive.signal(current_date, frame).weights == original.signal(current_date, frame).weights


def test_adaptive_sticky_v2_ignores_redundant_challengers_by_default():
    strategy = LeveredETFMomentumStickyAdaptiveV2(
        challenger_map={"XLK": "TECL"},
        lookback_days=2,
        challenge_signal_short_days=2,
        challenge_signal_long_days=4,
        challenge_execution_short_days=2,
        challenge_execution_long_days=4,
        carrier_momentum_days=6,
        entry_confirmation_periods=2,
        min_hold_periods=1,
    )

    first_data = pd.DataFrame(
        {
            "TQQQ": [100, 104, 108, 112, 116, 120, 124, 129],
            "UPRO": [100, 101, 102, 103, 104, 105, 106, 107],
            "SOXL": [100, 101, 102, 104, 106, 108, 110, 112],
            "TECL": [100, 101, 103, 105, 108, 111, 114, 118],
            "XLK": [100, 102, 104, 108, 112, 118, 124, 132],
            "SPY": [100, 101, 102, 103, 104, 105, 106, 107],
        }
    )
    second_data = pd.DataFrame(
        {
            "TQQQ": [100, 104, 108, 112, 116, 120, 124, 129, 133],
            "UPRO": [100, 101, 102, 103, 104, 105, 106, 107, 108],
            "SOXL": [100, 101, 102, 104, 106, 108, 110, 112, 114],
            "TECL": [100, 101, 103, 105, 108, 111, 114, 118, 123],
            "XLK": [100, 102, 104, 108, 112, 118, 124, 132, 138],
            "SPY": [100, 101, 102, 103, 104, 105, 106, 107, 108],
        }
    )

    first_signal = strategy.signal(date(2026, 3, 25), first_data)
    second_signal = strategy.signal(date(2026, 4, 25), second_data)

    assert first_signal.weights == {"TQQQ": 1.0}
    assert second_signal.weights == {"TQQQ": 1.0}


def test_adaptive_sticky_v2_requires_two_confirmations_before_challenger_entry():
    strategy = LeveredETFMomentumStickyAdaptiveV2(
        challenger_map={"XLF": "FAS"},
        lookback_days=2,
        challenge_signal_short_days=2,
        challenge_signal_long_days=4,
        challenge_execution_short_days=2,
        challenge_execution_long_days=4,
        challenge_short_buffer=0.10,
        challenge_long_buffer=0.05,
        carrier_momentum_days=6,
        carrier_drawdown_lookback_days=4,
        entry_confirmation_periods=2,
        min_hold_periods=1,
    )

    first_data = pd.DataFrame(
        {
            "TQQQ": [100, 102, 104, 106, 108, 110, 112, 114],
            "UPRO": [100, 101, 102, 103, 104, 105, 106, 107],
            "SOXL": [100, 101, 103, 104, 105, 107, 108, 109],
            "TECL": [100, 102, 103, 104, 105, 106, 107, 108],
            "XLF": [100, 101, 103, 106, 110, 115, 120, 126],
            "FAS": [100, 101, 104, 108, 120, 135, 150, 170],
            "SPY": [100, 100, 101, 102, 103, 104, 105, 106],
        }
    )
    second_data = pd.DataFrame(
        {
            "TQQQ": [100, 102, 104, 106, 108, 110, 112, 114, 116],
            "UPRO": [100, 101, 102, 103, 104, 105, 106, 107, 108],
            "SOXL": [100, 101, 103, 104, 105, 107, 108, 109, 110],
            "TECL": [100, 102, 103, 104, 105, 106, 107, 108, 109],
            "XLF": [100, 101, 103, 106, 110, 115, 120, 126, 132],
            "FAS": [100, 101, 104, 108, 120, 135, 150, 170, 195],
            "SPY": [100, 100, 101, 102, 103, 104, 105, 106, 107],
        }
    )

    first_signal = strategy.signal(date(2026, 3, 25), first_data)
    second_signal = strategy.signal(date(2026, 4, 25), second_data)

    assert first_signal.weights == {"TQQQ": 1.0}
    assert second_signal.weights == {"FAS": 1.0}


def test_adaptive_sticky_v2_keeps_baseline_when_challenger_carrier_fails():
    strategy = LeveredETFMomentumStickyAdaptiveV2(
        challenger_map={"XLE": "ERX"},
        lookback_days=2,
        challenge_signal_short_days=2,
        challenge_signal_long_days=4,
        challenge_execution_short_days=2,
        challenge_execution_long_days=4,
        challenge_short_buffer=0.10,
        challenge_long_buffer=0.05,
        carrier_momentum_days=6,
        carrier_drawdown_lookback_days=4,
        carrier_drawdown_floor=-0.20,
        entry_confirmation_periods=2,
        min_hold_periods=1,
    )

    first_data = pd.DataFrame(
        {
            "TQQQ": [100, 102, 104, 106, 108, 110, 112, 114],
            "UPRO": [100, 101, 102, 103, 104, 105, 106, 107],
            "SOXL": [100, 101, 103, 104, 105, 107, 108, 109],
            "TECL": [100, 102, 103, 104, 105, 106, 107, 108],
            "XLE": [100, 102, 104, 106, 108, 112, 120, 128],
            "ERX": [100, 120, 140, 150, 240, 160, 180, 190],
            "SPY": [100, 100, 101, 102, 103, 104, 105, 106],
        }
    )
    second_data = pd.DataFrame(
        {
            "TQQQ": [100, 102, 104, 106, 108, 110, 112, 114, 116],
            "UPRO": [100, 101, 102, 103, 104, 105, 106, 107, 108],
            "SOXL": [100, 101, 103, 104, 105, 107, 108, 109, 110],
            "TECL": [100, 102, 103, 104, 105, 106, 107, 108, 109],
            "XLE": [100, 102, 104, 106, 108, 112, 120, 128, 136],
            "ERX": [100, 120, 140, 150, 240, 160, 180, 190, 200],
            "SPY": [100, 100, 101, 102, 103, 104, 105, 106, 107],
        }
    )

    first_signal = strategy.signal(date(2026, 3, 25), first_data)
    second_signal = strategy.signal(date(2026, 4, 25), second_data)

    assert first_signal.weights == {"TQQQ": 1.0}
    assert second_signal.weights == {"TQQQ": 1.0}


def test_adaptive_sticky_v2_exits_override_back_to_live_shadow_baseline():
    strategy = LeveredETFMomentumStickyAdaptiveV2(
        challenger_map={"XLF": "FAS"},
        lookback_days=2,
        challenge_signal_short_days=2,
        challenge_signal_long_days=4,
        challenge_execution_short_days=2,
        challenge_execution_long_days=4,
        challenge_short_buffer=0.10,
        challenge_long_buffer=0.05,
        carrier_momentum_days=6,
        carrier_drawdown_lookback_days=4,
        entry_confirmation_periods=2,
        min_hold_periods=1,
    )

    first_data = pd.DataFrame(
        {
            "TQQQ": [100, 102, 104, 106, 108, 110, 112, 114],
            "UPRO": [100, 101, 102, 103, 104, 105, 106, 107],
            "SOXL": [100, 101, 102, 103, 104, 106, 108, 110],
            "TECL": [100, 101, 102, 103, 104, 105, 106, 107],
            "XLF": [100, 101, 103, 106, 110, 115, 120, 125],
            "FAS": [100, 101, 104, 108, 120, 135, 150, 170],
            "SPY": [100, 100, 101, 102, 103, 104, 105, 106],
        }
    )
    second_data = pd.DataFrame(
        {
            "TQQQ": [100, 102, 104, 106, 108, 110, 112, 114, 116],
            "UPRO": [100, 101, 102, 103, 104, 105, 106, 107, 108],
            "SOXL": [100, 101, 102, 103, 104, 106, 108, 110, 111],
            "TECL": [100, 101, 102, 103, 104, 105, 106, 107, 108],
            "XLF": [100, 101, 103, 106, 110, 115, 120, 125, 132],
            "FAS": [100, 101, 104, 108, 120, 135, 150, 170, 195],
            "SPY": [100, 100, 101, 102, 103, 104, 105, 106, 107],
        }
    )
    third_data = pd.DataFrame(
        {
            "TQQQ": [100, 102, 104, 106, 108, 110, 112, 114, 116, 117],
            "UPRO": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "SOXL": [100, 101, 102, 103, 104, 106, 108, 110, 111, 140],
            "TECL": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "XLF": [100, 101, 103, 106, 110, 115, 120, 125, 132, 130],
            "FAS": [100, 101, 104, 108, 120, 135, 150, 170, 195, 175],
            "SPY": [100, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        }
    )

    first_signal = strategy.signal(date(2026, 3, 25), first_data)
    second_signal = strategy.signal(date(2026, 4, 25), second_data)
    third_signal = strategy.signal(date(2026, 5, 25), third_data)

    assert first_signal.weights == {"SOXL": 1.0}
    assert second_signal.weights == {"FAS": 1.0}
    assert third_signal.weights == {"SOXL": 1.0}
