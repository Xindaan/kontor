"""Sticky Levered + vol-targeting overlay.

Builds on the Sticky Levered strategy over a 3x leveraged ETF universe and
layers a **vol-targeting overlay** with DECOUPLED rebalance frequencies on
top:

1. **Momentum pick** (monthly): the base class's Sticky Levered logic
   picks the 3x asset (or safe) once per calendar month.
2. **Vol-target overlay** (weekly, depending on `rebalance_frequency`):
   the pick's position size is scaled by realized vol::

       w_risky = clip(vol_target / realized_vol, 0, 1)
       w_safe  = 1 - w_risky   ->  SPY

   Automatically de-levers when vol explodes — and 3x ETFs explode
   exactly when vol explodes.

3. **Turnover damper** (`rebal_band`): the overlay only changes the
   weight if `|w_new - w_old| > rebal_band`. At `0.20` (default) that is
   ~10-14 trades/year — few enough to execute manually — instead of
   ~115 (daily, undamped). Sweep finding: 12-20pp after-tax equivalent
   (noise), 20pp is the best turnover/MaxDD trade-off; the finding
   below was measured at 0.15.

Backtest finding 2016-2024 (real backtester incl. German tax model,
default parameters vol_target=0.40 / rebal_band=0.15 / weekly):
- Baseline Sticky Levered monthly:     CAGR 27.3%, Sharpe 0.70, MaxDD -56.2%
- StickyLeveredVolTargeted weekly:     CAGR 35.1%, Sharpe 0.90, MaxDD -44.5%
  -> +7.8pp CAGR, +29% Sharpe, -11.7pp MaxDD — better on ALL three.

Sub-period robustness (baseline monthly -> VolTgt weekly):
- 2016-2019:  39.2%/-45.7% -> 36.1%/-36.2%  (slight CAGR trade for DD)
- 2020-2021:  57.4%/-27.3% -> 57.2%/-28.7%  (Sharpe 1.15 -> 1.51)
- 2022 crash: -36.7%/-36.5% -> -34.4%/-36.2% (wash)
- 2023-2024:  43.1%/-34.6% -> 58.7%/-30.2%  (clear win)

The full-period win is larger than any sub-period win: a compounding
effect — less loss in the 2022 crash means more capital base for the
2023 recovery. Unlike the ML veto, ML re-entry, and 3x hedge diversifier
overlays — all of which were worse than pure Sticky Levered — vol
targeting is the first robust win.

IMPORTANT: for the overlay to take effect, the strategy MUST be run
with a weekly (or daily) `rebalance_frequency`. With a monthly
rebalance it behaves like pure Sticky Levered (the overlay would only check
monthly and miss fast crashes like COVID-2020).

INSTRUMENT SUBSTITUTION: the sleeves default to the liquid US 3x ETFs
(``TQQQ`` / ``UPRO`` / ``SOXL``), which carry a long Yahoo history and
are therefore the right choice for backtesting. If you actually trade
different instruments (e.g. UCITS ETPs at a European broker), keep these
as the *signal* tickers and map them to your broker's tradable line via
``instrument_mapping`` — or pass ``candidates`` / ``safe_asset``
explicitly to backtest the real tickers directly.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.strategy import Allocation
from strategies.levered_etf_momentum_sticky import (
    LeveredETFMomentumSticky,
)


def _realized_vol(prices: pd.Series, lookback_days: int) -> Optional[float]:
    """Annualized realized vol over the last `lookback_days`."""
    clean = prices.dropna()
    if len(clean) < max(5, lookback_days // 2):
        return None
    window = clean.tail(lookback_days + 1)
    rets = window.pct_change().dropna()
    if rets.empty:
        return None
    return float(rets.std(ddof=0) * np.sqrt(252.0))


class StickyLeveredVolTargeted(LeveredETFMomentumSticky):
    """Sticky Levered with a weekly vol-targeting overlay."""

    name = "[Pilot] Sticky Levered + Vol-Targeting (Backtest, SOXL-Proxy)"
    # Required: weekly (or daily), otherwise the overlay has no effect.
    rebalance_frequency = "weekly"

    def __init__(
        self,
        candidates: Optional[List[str]] = None,
        lookback_days: int = 63,
        safe_asset: str = "SPY",
        momentum_floor: float = -0.05,
        switch_buffer: float = 0.06,
        min_hold_periods: int = 1,
        force_switch_floor: float = -0.20,
        baseline_drawdown_floor: float = -1.0,
        baseline_drawdown_lookback_days: int = 126,
        baseline_drawdown_target: Optional[str] = None,
        reentry_lookback_days: Optional[int] = None,
        *,
        vol_target: float = 0.40,
        vol_lookback_days: int = 20,
        rebal_band: float = 0.20,
    ) -> None:
        # Default universe with SOXL as the semiconductor 3x proxy (long
        # history). Live, SOXL is substituted -> SOXL.
        if candidates is None:
            candidates = ["TQQQ", "UPRO", "SOXL"]
        super().__init__(
            candidates=candidates,
            lookback_days=lookback_days,
            safe_asset=safe_asset,
            momentum_floor=momentum_floor,
            switch_buffer=switch_buffer,
            min_hold_periods=min_hold_periods,
            force_switch_floor=force_switch_floor,
            baseline_drawdown_floor=baseline_drawdown_floor,
            baseline_drawdown_lookback_days=baseline_drawdown_lookback_days,
            baseline_drawdown_target=baseline_drawdown_target,
            reentry_lookback_days=reentry_lookback_days,
        )
        self.vol_target = float(vol_target)
        self.vol_lookback_days = int(vol_lookback_days)
        self.rebal_band = float(rebal_band)
        self.params.update(
            {
                "vol_target": self.vol_target,
                "vol_lookback_days": self.vol_lookback_days,
                "rebal_band": self.rebal_band,
            }
        )
        # Overlay state: monthly-updated pick + current risk weight.
        self._picked: Optional[str] = None
        self._last_pick_month: Optional[tuple[int, int]] = None
        self._cur_risky_weight: float = 0.0

    @classmethod
    def get_param_grid(cls) -> Dict[str, List]:
        return {
            "vol_target": [0.40, 0.50, 0.60],
            "vol_lookback_days": [15, 20, 30],
            "rebal_band": [0.0, 0.10, 0.15, 0.20],
        }

    def _refresh_monthly_pick(
        self, current_date: date, data: pd.DataFrame
    ) -> None:
        """Run the Sticky Levered pick logic once per calendar month."""
        month_key = (current_date.year, current_date.month)
        if month_key == self._last_pick_month:
            return
        self._last_pick_month = month_key
        # The base class's Sticky logic updates its internal state
        # machine (_current / _hold_periods) and returns the pick.
        base_alloc = super().signal(current_date, data)
        weights = dict(base_alloc.weights) if hasattr(base_alloc, "weights") else dict(base_alloc)
        # Sticky always returns a single-ticker allocation.
        self._picked = next(iter(weights), self.safe_asset)

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        if data is None or data.empty:
            self._picked = self.safe_asset
            self._last_pick_month = (current_date.year, current_date.month)
            self._cur_risky_weight = 0.0
            return Allocation({self.safe_asset: 1.0})

        # 1. Monthly momentum pick (base class's Sticky Levered logic).
        self._refresh_monthly_pick(current_date, data)
        picked = self._picked or self.safe_asset

        # 2. Safe phase: no overlay, fully in the safe asset.
        if picked == self.safe_asset:
            self._cur_risky_weight = 0.0
            return Allocation({self.safe_asset: 1.0})

        # 3. Vol-target overlay on the 3x pick.
        series = data[picked] if picked in data.columns else None
        rv = _realized_vol(series, self.vol_lookback_days) if series is not None else None
        if rv is None or rv <= 0.0:
            target_w = 1.0
        else:
            target_w = min(1.0, self.vol_target / rv)

        # 4. Turnover damper: only change the weight if delta > band.
        #    Exception: always set it on the first entry into the pick (cur==0).
        if (
            self._cur_risky_weight > 0.0
            and abs(target_w - self._cur_risky_weight) < self.rebal_band
        ):
            target_w = self._cur_risky_weight
        else:
            self._cur_risky_weight = target_w

        safe_w = 1.0 - target_w
        if safe_w <= 1e-6:
            return Allocation({picked: 1.0})
        return Allocation({picked: target_w, self.safe_asset: safe_w})


strategy = StickyLeveredVolTargeted()
