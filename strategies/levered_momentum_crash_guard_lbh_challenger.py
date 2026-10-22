"""
LBH-challenger preset for LeveredMomentumCrashGuard.

Tuned for high CAGR versus Levered Buy & Hold while keeping
drawdown in a similar ballpark.
"""

from __future__ import annotations

from strategies.levered_momentum_crash_guard import LeveredMomentumCrashGuard


class LeveredMomentumCrashGuardLBHChallenger(LeveredMomentumCrashGuard):
    name = "[Research] Levered Momentum + Crash Guard (LBH Challenger, US)"

    def __init__(
        self,
        candidates_3x: list[str] | None = None,
        fallback_1x: dict[str, str] | None = None,
        safe_asset: str = "SPY",
        trend_asset: str | None = None,
        trend_window_days: int = 150,
        momentum_lookback_days: int = 84,
        momentum_floor: float = -0.02,
        vol_lookback_days: int = 21,
        max_vol_for_3x: float = 0.9,
        drawdown_lookback_days: int = 63,
        max_drawdown_for_3x: float = -0.55,
    ) -> None:
        super().__init__(
            candidates_3x=candidates_3x,
            fallback_1x=fallback_1x,
            safe_asset=safe_asset,
            trend_asset=trend_asset,
            trend_window_days=trend_window_days,
            momentum_lookback_days=momentum_lookback_days,
            momentum_floor=momentum_floor,
            vol_lookback_days=vol_lookback_days,
            max_vol_for_3x=max_vol_for_3x,
            drawdown_lookback_days=drawdown_lookback_days,
            max_drawdown_for_3x=max_drawdown_for_3x,
        )

    @classmethod
    def get_param_grid(cls):
        return {
            "momentum_lookback_days": [63, 84, 126],
            "max_vol_for_3x": [0.7, 0.9, 1.1],
            "max_drawdown_for_3x": [-0.55, -0.45, -0.35],
        }


strategy = LeveredMomentumCrashGuardLBHChallenger()
