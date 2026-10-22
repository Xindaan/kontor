"""
Baseline-preserving adaptive sticky rotation for levered ETFs.

Design:
- Run the original sticky winner logic as a shadow baseline path.
- Allow only non-redundant sector challengers to override the baseline.
- Challengers enter only after repeated, baseline-relative evidence.
- If a challenger weakens, immediately fall back to the live baseline path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from backtest.strategy import Allocation, Strategy
from strategies.levered_etf_momentum_sticky import LeveredETFMomentumSticky


DEFAULT_BASE_CANDIDATES: List[str] = ["TQQQ", "UPRO", "SOXL", "TECL"]
DEFAULT_CHALLENGER_MAP: Dict[str, str] = {
    "XLF": "FAS",
    "XLI": "DUSL",
    "XLE": "ERX",
}
OPTIONAL_CHALLENGER_MAP: Dict[str, str] = {
    "XLV": "CURE",
    "XLB": "UYM",
    "XLU": "UTSL",
    "XLRE": "DRN",
}


@dataclass(frozen=True)
class _ChallengerSnapshot:
    signal_ticker: str
    execution_ticker: str
    signal_short: float
    signal_long: float
    execution_short: float
    execution_long: float
    carrier_long: float
    carrier_drawdown: float
    carrier_vol: Optional[float]
    entry_ready: bool


def _momentum(prices: pd.Series, lookback_days: int) -> Optional[float]:
    clean = prices.dropna()
    if len(clean) <= lookback_days:
        return None
    start_price = float(clean.iloc[-lookback_days - 1])
    end_price = float(clean.iloc[-1])
    if start_price <= 0.0:
        return None
    return (end_price / start_price) - 1.0


def _recent_drawdown(prices: pd.Series, lookback_days: int) -> Optional[float]:
    clean = prices.dropna()
    if len(clean) < lookback_days:
        return None
    window = clean.iloc[-lookback_days:]
    peak = float(window.max())
    if peak <= 0.0:
        return None
    drawdown = float(window.iloc[-1] / peak - 1.0)
    return drawdown if np.isfinite(drawdown) else None


def _annualized_vol(prices: pd.Series, lookback_days: int) -> Optional[float]:
    clean = prices.dropna()
    if len(clean) < lookback_days + 1:
        return None
    rets = clean.pct_change(fill_method=None).dropna().iloc[-lookback_days:]
    if len(rets) < lookback_days:
        return None
    vol = float(rets.std(ddof=0) * np.sqrt(252))
    return vol if np.isfinite(vol) else None


class LeveredETFMomentumStickyAdaptiveV2(Strategy):
    name = "[Research] Levered ETF Momentum (Sticky Adaptive V2, US)"
    rebalance_frequency = "monthly"

    def __init__(
        self,
        base_candidates: List[str] | None = None,
        challenger_map: Mapping[str, str] | None = None,
        lookback_days: int = 63,
        safe_asset: str = "SPY",
        momentum_floor: float = 0.0,
        switch_buffer: float = 0.04,
        min_hold_periods: int = 2,
        force_switch_floor: float = -0.20,
        challenge_signal_short_days: int = 63,
        challenge_signal_long_days: int = 126,
        challenge_execution_short_days: int = 63,
        challenge_execution_long_days: int = 126,
        challenge_short_buffer: float = 0.08,
        challenge_long_buffer: float = 0.05,
        carrier_momentum_days: int = 252,
        carrier_momentum_floor: float = 0.05,
        carrier_drawdown_lookback_days: int = 126,
        carrier_drawdown_floor: float = -0.40,
        carrier_vol_lookback_days: int = 21,
        max_carrier_vol: float | None = None,
        entry_confirmation_periods: int = 1,
        baseline_reclaim_buffer: float = 0.02,
        override_force_exit_floor: float = -0.20,
        allow_redundant_challengers: bool = False,
        baseline_drawdown_floor: float = -1.0,
        baseline_drawdown_lookback_days: int = 126,
        baseline_drawdown_target: Optional[str] = None,
        challenger_max_weight: float = 1.0,
        reentry_lookback_days: Optional[int] = None,
    ) -> None:
        if base_candidates is None:
            base_candidates = DEFAULT_BASE_CANDIDATES
        if challenger_map is None:
            challenger_map = DEFAULT_CHALLENGER_MAP

        self.baseline_drawdown_floor = float(baseline_drawdown_floor)
        self.baseline_drawdown_lookback_days = int(baseline_drawdown_lookback_days)
        self.baseline_drawdown_target = baseline_drawdown_target
        self.challenger_max_weight = max(0.0, min(1.0, float(challenger_max_weight)))
        self.base_candidates = list(base_candidates)
        self.challenger_map = dict(challenger_map)
        self.lookback_days = int(lookback_days)
        self.safe_asset = safe_asset
        self.momentum_floor = float(momentum_floor)
        self.switch_buffer = float(switch_buffer)
        self.min_hold_periods = int(min_hold_periods)
        self.force_switch_floor = float(force_switch_floor)
        self.challenge_signal_short_days = int(challenge_signal_short_days)
        self.challenge_signal_long_days = int(challenge_signal_long_days)
        self.challenge_execution_short_days = int(challenge_execution_short_days)
        self.challenge_execution_long_days = int(challenge_execution_long_days)
        self.challenge_short_buffer = float(challenge_short_buffer)
        self.challenge_long_buffer = float(challenge_long_buffer)
        self.carrier_momentum_days = int(carrier_momentum_days)
        self.carrier_momentum_floor = float(carrier_momentum_floor)
        self.carrier_drawdown_lookback_days = int(carrier_drawdown_lookback_days)
        self.carrier_drawdown_floor = float(carrier_drawdown_floor)
        self.carrier_vol_lookback_days = int(carrier_vol_lookback_days)
        self.max_carrier_vol = None if max_carrier_vol is None else float(max_carrier_vol)
        self.entry_confirmation_periods = max(1, int(entry_confirmation_periods))
        self.baseline_reclaim_buffer = float(baseline_reclaim_buffer)
        self.override_force_exit_floor = float(override_force_exit_floor)
        self.allow_redundant_challengers = bool(allow_redundant_challengers)

        self.signal_assets = list(self.challenger_map.keys())
        self.execution_assets = list(self.challenger_map.values())
        self.assets = list(
            dict.fromkeys(
                self.base_candidates
                + self.signal_assets
                + self.execution_assets
                + [self.safe_asset]
            )
        )
        self.params = {
            "base_candidates": self.base_candidates,
            "challenger_map": dict(self.challenger_map),
            "lookback_days": self.lookback_days,
            "safe_asset": self.safe_asset,
            "momentum_floor": self.momentum_floor,
            "switch_buffer": self.switch_buffer,
            "min_hold_periods": self.min_hold_periods,
            "force_switch_floor": self.force_switch_floor,
            "challenge_signal_short_days": self.challenge_signal_short_days,
            "challenge_signal_long_days": self.challenge_signal_long_days,
            "challenge_execution_short_days": self.challenge_execution_short_days,
            "challenge_execution_long_days": self.challenge_execution_long_days,
            "challenge_short_buffer": self.challenge_short_buffer,
            "challenge_long_buffer": self.challenge_long_buffer,
            "carrier_momentum_days": self.carrier_momentum_days,
            "carrier_momentum_floor": self.carrier_momentum_floor,
            "carrier_drawdown_lookback_days": self.carrier_drawdown_lookback_days,
            "carrier_drawdown_floor": self.carrier_drawdown_floor,
            "carrier_vol_lookback_days": self.carrier_vol_lookback_days,
            "max_carrier_vol": self.max_carrier_vol,
            "entry_confirmation_periods": self.entry_confirmation_periods,
            "baseline_reclaim_buffer": self.baseline_reclaim_buffer,
            "override_force_exit_floor": self.override_force_exit_floor,
            "allow_redundant_challengers": self.allow_redundant_challengers,
            "baseline_drawdown_floor": self.baseline_drawdown_floor,
            "baseline_drawdown_lookback_days": self.baseline_drawdown_lookback_days,
            "baseline_drawdown_target": self.baseline_drawdown_target,
            "challenger_max_weight": self.challenger_max_weight,
            "reentry_lookback_days": reentry_lookback_days,
        }

        self._baseline_strategy = LeveredETFMomentumSticky(
            candidates=self.base_candidates,
            lookback_days=self.lookback_days,
            safe_asset=self.safe_asset,
            momentum_floor=self.momentum_floor,
            switch_buffer=self.switch_buffer,
            min_hold_periods=self.min_hold_periods,
            force_switch_floor=self.force_switch_floor,
            baseline_drawdown_floor=self.baseline_drawdown_floor,
            baseline_drawdown_lookback_days=self.baseline_drawdown_lookback_days,
            baseline_drawdown_target=self.baseline_drawdown_target,
            reentry_lookback_days=reentry_lookback_days,
        )
        self._entry_streaks: Dict[str, int] = {ticker: 0 for ticker in self.signal_assets}
        self._active_override_signal: Optional[str] = None

    @classmethod
    def get_param_grid(cls):
        return {
            "switch_buffer": [0.02, 0.04],
            "min_hold_periods": [1, 2, 3],
            "challenge_short_buffer": [0.06, 0.08, 0.10],
            "carrier_momentum_floor": [0.05, 0.10, 0.15],
            "carrier_drawdown_floor": [-0.40, -0.35, -0.30],
            "entry_confirmation_periods": [1, 2],
        }

    def _asset_momentum(self, data: pd.DataFrame, ticker: str, lookback_days: int) -> float:
        if ticker not in data.columns:
            return -np.inf
        value = _momentum(data[ticker], lookback_days)
        return -np.inf if value is None else float(value)

    def _extract_allocation_asset(self, allocation: Allocation) -> str:
        if not allocation.weights:
            return self.safe_asset
        ticker, _ = max(allocation.weights.items(), key=lambda item: item[1])
        return str(ticker)

    def _build_snapshot(
        self,
        data: pd.DataFrame,
        signal_ticker: str,
        execution_ticker: str,
        baseline_short: float,
        baseline_long: float,
    ) -> Optional[_ChallengerSnapshot]:
        if (
            signal_ticker not in data.columns
            or execution_ticker not in data.columns
            or (
                not self.allow_redundant_challengers
                and execution_ticker in self.base_candidates
            )
        ):
            return None

        signal_short = _momentum(data[signal_ticker], self.challenge_signal_short_days)
        signal_long = _momentum(data[signal_ticker], self.challenge_signal_long_days)
        execution_short = _momentum(data[execution_ticker], self.challenge_execution_short_days)
        execution_long = _momentum(data[execution_ticker], self.challenge_execution_long_days)
        carrier_long = _momentum(data[execution_ticker], self.carrier_momentum_days)
        carrier_drawdown = _recent_drawdown(data[execution_ticker], self.carrier_drawdown_lookback_days)
        carrier_vol = None
        if self.max_carrier_vol is not None:
            carrier_vol = _annualized_vol(data[execution_ticker], self.carrier_vol_lookback_days)

        fields = (
            signal_short,
            signal_long,
            execution_short,
            execution_long,
            carrier_long,
            carrier_drawdown,
        )
        if any(value is None for value in fields):
            return None

        entry_pass = (
            signal_short > 0.0
            and signal_long > 0.0
            and execution_short > 0.0
            and execution_long > 0.0
            and execution_short >= (baseline_short + self.challenge_short_buffer)
            and execution_long >= (baseline_long + self.challenge_long_buffer)
            and carrier_long > self.carrier_momentum_floor
            and carrier_drawdown >= self.carrier_drawdown_floor
            and (
                self.max_carrier_vol is None
                or (carrier_vol is not None and carrier_vol <= self.max_carrier_vol)
            )
        )

        streak = self._entry_streaks.get(signal_ticker, 0)
        streak = streak + 1 if entry_pass else 0
        self._entry_streaks[signal_ticker] = streak

        return _ChallengerSnapshot(
            signal_ticker=signal_ticker,
            execution_ticker=execution_ticker,
            signal_short=float(signal_short),
            signal_long=float(signal_long),
            execution_short=float(execution_short),
            execution_long=float(execution_long),
            carrier_long=float(carrier_long),
            carrier_drawdown=float(carrier_drawdown),
            carrier_vol=None if carrier_vol is None else float(carrier_vol),
            entry_ready=streak >= self.entry_confirmation_periods,
        )

    def _snapshot_still_valid(
        self,
        snapshot: _ChallengerSnapshot,
        baseline_short: float,
    ) -> bool:
        if snapshot.execution_short <= self.override_force_exit_floor:
            return False
        if snapshot.signal_short <= 0.0 or snapshot.signal_long <= 0.0:
            return False
        if snapshot.execution_short <= 0.0 or snapshot.execution_long <= 0.0:
            return False
        if snapshot.carrier_long <= self.carrier_momentum_floor:
            return False
        if snapshot.carrier_drawdown < self.carrier_drawdown_floor:
            return False
        if self.max_carrier_vol is not None:
            if snapshot.carrier_vol is None or snapshot.carrier_vol > self.max_carrier_vol:
                return False
        if baseline_short >= (snapshot.execution_short + self.baseline_reclaim_buffer):
            return False
        return True

    def _clear_override_if_missing(self) -> None:
        if (
            self._active_override_signal is not None
            and self._active_override_signal not in self.challenger_map
        ):
            self._active_override_signal = None

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        if data is None or data.empty:
            self._active_override_signal = None
            self._entry_streaks = {ticker: 0 for ticker in self.signal_assets}
            return Allocation({self.safe_asset: 1.0})

        self._clear_override_if_missing()

        baseline_allocation = self._baseline_strategy.signal(current_date, data)
        baseline_asset = self._extract_allocation_asset(baseline_allocation)
        baseline_short = self._asset_momentum(
            data,
            baseline_asset,
            self.challenge_execution_short_days,
        )
        baseline_long = self._asset_momentum(
            data,
            baseline_asset,
            self.challenge_execution_long_days,
        )

        snapshots: Dict[str, _ChallengerSnapshot] = {}
        for signal_ticker, execution_ticker in self.challenger_map.items():
            snapshot = self._build_snapshot(
                data=data,
                signal_ticker=signal_ticker,
                execution_ticker=execution_ticker,
                baseline_short=baseline_short,
                baseline_long=baseline_long,
            )
            if snapshot is not None:
                snapshots[signal_ticker] = snapshot
            else:
                self._entry_streaks[signal_ticker] = 0

        if self._active_override_signal is not None:
            active_snapshot = snapshots.get(self._active_override_signal)
            if active_snapshot is not None and self._snapshot_still_valid(
                active_snapshot,
                baseline_short,
            ):
                return self._build_challenger_allocation(
                    active_snapshot.execution_ticker,
                    baseline_asset,
                    baseline_allocation,
                )
            self._active_override_signal = None

        ready = [snapshot for snapshot in snapshots.values() if snapshot.entry_ready]
        if ready:
            best = max(
                ready,
                key=lambda snapshot: (
                    snapshot.execution_short - baseline_short,
                    snapshot.execution_long - baseline_long,
                    snapshot.execution_short,
                ),
            )
            self._active_override_signal = best.signal_ticker
            return self._build_challenger_allocation(
                best.execution_ticker,
                baseline_asset,
                baseline_allocation,
            )

        return baseline_allocation

    def _build_challenger_allocation(
        self,
        challenger_ticker: str,
        baseline_asset: str,
        baseline_allocation: Allocation,
    ) -> Allocation:
        """Build the allocation for a challenger override, staged if applicable.

        If challenger_max_weight < 1.0, the position is split between
        the challenger and the baseline asset.
        """
        if self.challenger_max_weight >= 1.0:
            return Allocation({challenger_ticker: 1.0})
        cw = self.challenger_max_weight
        bw = 1.0 - cw
        if challenger_ticker == baseline_asset:
            return Allocation({challenger_ticker: 1.0})
        return Allocation({challenger_ticker: cw, baseline_asset: bw})


strategy = LeveredETFMomentumStickyAdaptiveV2()
