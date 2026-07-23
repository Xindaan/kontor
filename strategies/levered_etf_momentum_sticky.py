"""
Sticky levered ETF momentum rotation.

Goal:
- Keep much of winner-takes-all upside.
- Reduce turnover/tax drag via switch hysteresis and minimum holding periods.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.strategy import Allocation, Strategy


def _momentum(prices: pd.Series, lookback_days: int) -> Optional[float]:
    clean = prices.dropna()
    if len(clean) <= lookback_days:
        return None
    p0 = float(clean.iloc[-lookback_days - 1])
    p1 = float(clean.iloc[-1])
    if p0 <= 0.0:
        return None
    return (p1 / p0) - 1.0


def _recent_drawdown(prices: pd.Series, lookback_days: int) -> Optional[float]:
    """Drawdown from the peak within the lookback window."""
    clean = prices.dropna()
    if len(clean) <= lookback_days:
        return None
    window = clean.iloc[-lookback_days:]
    peak = float(window.max())
    if peak <= 0.0:
        return None
    return (float(clean.iloc[-1]) / peak) - 1.0


class LeveredETFMomentumSticky(Strategy):
    name = "[Production] Levered ETF Momentum (Sticky Winner, US)"
    rebalance_frequency = "monthly"

    def __init__(
        self,
        candidates: List[str] | None = None,
        lookback_days: int = 63,
        safe_asset: str = "SPY",
        momentum_floor: float = 0.0,
        switch_buffer: float = 0.04,
        min_hold_periods: int = 2,
        force_switch_floor: float = -0.20,
        baseline_drawdown_floor: float = -1.0,
        baseline_drawdown_lookback_days: int = 126,
        baseline_drawdown_target: Optional[str] = None,
        reentry_lookback_days: Optional[int] = None,
    ) -> None:
        if candidates is None:
            candidates = ["TQQQ", "UPRO", "SOXL", "TECL"]

        self.candidates = list(candidates)
        self.lookback_days = int(lookback_days)
        self.safe_asset = safe_asset
        self.momentum_floor = float(momentum_floor)
        self.switch_buffer = float(switch_buffer)
        self.min_hold_periods = int(min_hold_periods)
        self.force_switch_floor = float(force_switch_floor)
        self.baseline_drawdown_floor = float(baseline_drawdown_floor)
        self.baseline_drawdown_lookback_days = int(baseline_drawdown_lookback_days)
        self.baseline_drawdown_target = baseline_drawdown_target
        self.reentry_lookback_days = (
            int(reentry_lookback_days) if reentry_lookback_days is not None else None
        )
        self._exited_via_drawdown = False

        extra_assets: List[str] = []
        if baseline_drawdown_target is not None:
            extra_assets.append(baseline_drawdown_target)
        self.assets = list(dict.fromkeys(self.candidates + [safe_asset] + extra_assets))
        self.params = {
            "candidates": self.candidates,
            "lookback_days": self.lookback_days,
            "safe_asset": safe_asset,
            "momentum_floor": momentum_floor,
            "switch_buffer": switch_buffer,
            "min_hold_periods": min_hold_periods,
            "force_switch_floor": force_switch_floor,
            "baseline_drawdown_floor": baseline_drawdown_floor,
            "baseline_drawdown_lookback_days": baseline_drawdown_lookback_days,
            "baseline_drawdown_target": baseline_drawdown_target,
            "reentry_lookback_days": reentry_lookback_days,
        }

        self._current: Optional[str] = None
        self._hold_periods = 0

    @classmethod
    def get_param_grid(cls):
        return {
            "lookback_days": [42, 63, 84],
            "switch_buffer": [0.02, 0.04, 0.06],
            "min_hold_periods": [1, 2, 3],
            "momentum_floor": [-0.05, 0.0, 0.05],
            "baseline_drawdown_floor": [-0.20, -0.25, -0.30],
        }

    def _effective_lookback(self) -> int:
        """Lookback days: shorter after a drawdown exit for a faster re-entry."""
        if self._exited_via_drawdown and self.reentry_lookback_days is not None:
            return self.reentry_lookback_days
        return self.lookback_days

    def _leader(self, data: pd.DataFrame) -> tuple[Optional[str], float, Dict[str, float]]:
        lookback = self._effective_lookback()
        scores: Dict[str, float] = {}
        for ticker in self.candidates:
            if ticker not in data.columns:
                continue
            mom = _momentum(data[ticker], lookback)
            if mom is None:
                continue
            scores[ticker] = mom

        if not scores:
            return None, -np.inf, scores

        leader = max(scores, key=scores.get)
        return leader, float(scores[leader]), scores

    def _check_baseline_drawdown(self, data: pd.DataFrame) -> Optional[str]:
        """Check whether the active baseline position breaches the drawdown floor.

        Returns the exit target (ticker) or None if no exit is needed.
        """
        if self._current is None or self._current == self.safe_asset:
            return None
        if self.baseline_drawdown_floor >= 0.0 or self.baseline_drawdown_floor <= -1.0:
            return None
        if self._current not in data.columns:
            return None
        dd = _recent_drawdown(data[self._current], self.baseline_drawdown_lookback_days)
        if dd is not None and dd < self.baseline_drawdown_floor:
            self._exited_via_drawdown = True
            return self.baseline_drawdown_target or self.safe_asset
        return None

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        if data is None or data.empty:
            self._current = self.safe_asset
            self._hold_periods = 0
            return Allocation({self.safe_asset: 1.0})

        # Baseline drawdown check first (before the momentum gate)
        dd_target = self._check_baseline_drawdown(data)
        if dd_target is not None:
            self._current = dd_target
            self._hold_periods = 0
            return Allocation({dd_target: 1.0})

        leader, leader_mom, scores = self._leader(data)
        if leader is None or leader_mom <= self.momentum_floor:
            self._current = self.safe_asset
            self._hold_periods = 0
            return Allocation({self.safe_asset: 1.0})

        if self._current is None or self._current == self.safe_asset:
            self._current = leader
            self._hold_periods = 1
            self._exited_via_drawdown = False
            return Allocation({leader: 1.0})

        curr_mom = scores.get(self._current, -np.inf)
        if curr_mom <= self.force_switch_floor:
            self._current = leader
            self._hold_periods = 1
            self._exited_via_drawdown = False
            return Allocation({leader: 1.0})

        if self._hold_periods < self.min_hold_periods:
            self._hold_periods += 1
            return Allocation({self._current: 1.0})

        if leader != self._current and (leader_mom - curr_mom) >= self.switch_buffer:
            self._current = leader
            self._hold_periods = 1
            return Allocation({leader: 1.0})

        self._hold_periods += 1
        return Allocation({self._current: 1.0})


strategy = LeveredETFMomentumSticky()
