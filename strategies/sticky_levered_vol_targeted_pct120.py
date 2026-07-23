"""Sticky Levered + vol targeting with the pct120_wide regime percentile gate (RESEARCH candidate).

FOR-ANDRE-REVIEW candidate from the autonomous strategy search loop (docs/strategy_search_log.md, iter 7-19).
Extends the locked master StickyLeveredVolTargeted: instead of w = clip(vt / realized_vol_20d, 0, 1) it uses
a REGIME percentile gate on vol (pct120_wide, reproduced from
docs/dynamic_risk_optimization_2026-06-02.md):

    factor   = clip(pct_hi - (pct_hi - pct_lo) * pctrank(vol_120d; trailing pct_dist), pct_lo, pct_hi)
    eff_risk = realized_vol_20d / factor
    w_risky  = clip(vol_target / eff_risk, 0, 1)        remainder -> safe_asset

Calm vol regime (low 120d percentile in its own trailing 3y distribution) -> factor > 1
-> more risk budget; stressed regime -> factor < 1 -> less. So it uses the RELATIVE vol level
(regime), not just the absolute level.

Backtest finding (matched avg leverage, after-tax net_liquidation, S&P-safe; vs master VolTarget):
- REAL 2012-2024 @vt0.40: +2.8pp CAGR / +0.8pp MaxDD / +0.06 Sharpe (robust in 74-85% of rolling windows)
- DEEP 1994-2024 @vt0.40: CAGR ~wash / ~+4pp flatter MaxDD
- Passed the parameter plateau test (no overfit); cost-robust up to 50 bps/trade; holds on both the TR and DB config.
NUANCE (iter 9): the real-data edge is broad, the deep-data edge is CONCENTRATED (caps the worst-case
crash, but levers up in calm regimes -> lower TYPICAL drawdowns). A trade-off, no free lunch.

IMPORTANT: NOT the live master, NOT to be auto-promoted — the master is locked; promotion is decided by
the quarterly review + Andre (charter docs/strategy_evaluation_charter.md). Pure research/review candidate.
"""
from __future__ import annotations

from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest.strategy import Allocation
from strategies.sticky_levered_vol_targeted import StickyLeveredVolTargeted


class StickyLeveredVolTargetedPct120(StickyLeveredVolTargeted):
    """Master sizing replaced by the pct120_wide regime percentile gate. Pick/safe phase/band identical."""

    name = "[Research] Sticky Levered + pct120_wide Regime-Vol-Gate"

    def __init__(
        self,
        *args,
        pct_lo: float = 0.55,
        pct_hi: float = 1.55,
        pct_slow_days: int = 120,
        pct_dist_days: int = 756,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pct_lo = float(pct_lo)
        self.pct_hi = float(pct_hi)
        self.pct_slow_days = int(pct_slow_days)
        self.pct_dist_days = int(pct_dist_days)
        self.params.update(
            {
                "pct_lo": self.pct_lo,
                "pct_hi": self.pct_hi,
                "pct_slow_days": self.pct_slow_days,
                "pct_dist_days": self.pct_dist_days,
            }
        )

    @classmethod
    def get_param_grid(cls) -> Dict[str, List]:
        # Plateau-tested (iter 8): band/slow/dist robust around the defaults.
        return {
            "vol_target": [0.30, 0.40, 0.50],
            "pct_lo": [0.55, 0.65, 0.70],
            "pct_hi": [1.30, 1.45, 1.55],
        }

    def _effective_risk(self, series: pd.Series):
        """eff_risk = realized_vol_20d / regime_factor (pct120_wide)."""
        clean = series.dropna()
        if len(clean) < max(5, self.vol_lookback_days // 2):
            return None
        r20 = clean.tail(self.vol_lookback_days + 1).pct_change().dropna()
        if r20.empty:
            return None
        v20 = float(r20.std(ddof=0) * np.sqrt(252.0))
        if v20 <= 0.0:
            return None
        vols = (
            clean.pct_change()
            .rolling(self.pct_slow_days, min_periods=max(5, self.pct_slow_days // 2))
            .std(ddof=0)
            * np.sqrt(252.0)
        ).dropna()
        if vols.empty:
            return v20
        rank = float((vols.tail(self.pct_dist_days) <= float(vols.iloc[-1])).mean())
        factor = float(np.clip(self.pct_hi - (self.pct_hi - self.pct_lo) * rank, self.pct_lo, self.pct_hi))
        return v20 / factor

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        if data is None or data.empty:
            self._picked = self.safe_asset
            self._last_pick_month = (current_date.year, current_date.month)
            self._cur_risky_weight = 0.0
            return Allocation({self.safe_asset: 1.0})

        # 1. Monthly momentum pick (master logic, unchanged).
        self._refresh_monthly_pick(current_date, data)
        picked = self._picked or self.safe_asset

        # 2. Safe phase: fully in the safe asset (unchanged).
        if picked == self.safe_asset:
            self._cur_risky_weight = 0.0
            return Allocation({self.safe_asset: 1.0})

        # 3. pct120_wide regime sizing instead of plain vol target.
        series = data[picked] if picked in data.columns else None
        eff = self._effective_risk(series) if series is not None else None
        target_w = 1.0 if (eff is None or eff <= 0.0) else min(1.0, self.vol_target / eff)

        # 4. Turnover damper (master rebal_band, unchanged).
        if self._cur_risky_weight > 0.0 and abs(target_w - self._cur_risky_weight) < self.rebal_band:
            target_w = self._cur_risky_weight
        else:
            self._cur_risky_weight = target_w

        safe_w = 1.0 - target_w
        if safe_w <= 1e-6:
            return Allocation({picked: 1.0})
        return Allocation({picked: target_w, self.safe_asset: safe_w})


strategy = StickyLeveredVolTargetedPct120()
