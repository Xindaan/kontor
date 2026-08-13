"""Sticky Levered — multi-3x allocation variants (strategy research).

Extends the vol-targeting concept with multi-asset allocation: instead
of holding just the one momentum winner and de-levering with the 1x
safe asset, the DE-LEVERED REMAINDER is spread across further 3x
assets.

Three modes (`allocation_mode`):

- ``single_pick``  — reference: identical to StickyLeveredVolTargeted.
  One momentum pick, vol-targeted, remainder in SPY.

- ``cascade``      — STRICT extension of single_pick:
  1. The top pick + weight are EXACTLY those of single_pick (full
     Sticky hysteresis, monthly pick + weekly vol overlay).
  2. The remainder that single_pick would push entirely into SPY
     is instead cascaded into the other candidates — each up to its
     own vol-target cap ``clip(vol_target/vola, 0, 1)``, in
     ``cascade_order`` order.
  3. Only the final remainder goes into SPY.
  ``max_total_leverage`` caps the sum of all 3x weights
  (1.0 = no 1x anchor enforced, 0.75 = at least a 25% anchor).

- ``inverse_vol``  — all candidates with momentum > floor are
  inverse-vol weighted, then scaled to ``vol_target``. Remainder
  in SPY. (Discards the Sticky concentration entirely.)

Background: in a crash the 3x ETFs are NOT diversifiers of each other
(2022: UPRO -49%, TQQQ -80%, while SPY only -11%). ``cascade``
with ``max_total_leverage`` allows a deliberate choice between capital
efficiency (more leverage) and a crash anchor.

Defaults (backtest-validated, real backtester incl. German tax,
2016-2024): ``cascade_order="vol"`` (de-levered remainder first into
the least volatile follow-up candidate — clearly beats momentum
order), ``max_total_leverage=1.0``. Results vs comparison:
- single_pick:        35.1% CAGR, Sharpe 0.90, MaxDD -44.5%
- cascade vol 1.00:   39.3% CAGR, Sharpe 0.90, MaxDD -56.8%
- ALT Sticky Levered pure: 27.3% CAGR, Sharpe 0.70, MaxDD -56.2%
- ALT 3x Buy&Hold:    39.3% CAGR, Sharpe 0.80, MaxDD -85.0%
cascade vol 1.00 delivers the 3x buy&hold return at a significantly
better drawdown and Sharpe; single_pick remains the most
drawdown-protected variant.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.strategy import Allocation
from strategies.levered_etf_momentum_sticky import _momentum
from strategies.sticky_levered_vol_targeted import StickyLeveredVolTargeted, _realized_vol


class StickyLeveredCascade(StickyLeveredVolTargeted):
    """Sticky Levered with multi-3x allocation (single_pick / cascade / inverse_vol)."""

    name = "[Pilot] Sticky Levered + Cascade Vol-Targeting"
    rebalance_frequency = "weekly"

    def __init__(
        self,
        *,
        allocation_mode: str = "cascade",
        max_total_leverage: float = 1.0,
        cascade_order: str = "vol",
        assumed_correlation: float = 0.85,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if allocation_mode not in {"single_pick", "inverse_vol", "cascade"}:
            raise ValueError(f"unknown allocation_mode: {allocation_mode}")
        if cascade_order not in {"momentum", "vol"}:
            raise ValueError(f"unknown cascade_order: {cascade_order}")
        self.allocation_mode = allocation_mode
        self.max_total_leverage = float(max_total_leverage)
        self.cascade_order = cascade_order
        self.assumed_correlation = float(assumed_correlation)
        self.params.update(
            {
                "allocation_mode": allocation_mode,
                "max_total_leverage": self.max_total_leverage,
                "cascade_order": cascade_order,
                "assumed_correlation": self.assumed_correlation,
            }
        )

    def _candidate_mom_vol(
        self, data: pd.DataFrame, exclude: Optional[str] = None
    ) -> Dict[str, Tuple[float, float]]:
        """63d momentum + 20d vol for candidates with momentum > floor."""
        out: Dict[str, Tuple[float, float]] = {}
        for ticker in self.candidates:
            if ticker == exclude:
                continue
            if ticker not in data.columns:
                continue
            mom = _momentum(data[ticker], self.lookback_days)
            vol = _realized_vol(data[ticker], self.vol_lookback_days)
            if mom is None or vol is None or vol <= 0.0:
                continue
            if mom <= self.momentum_floor:
                continue
            out[ticker] = (mom, vol)
        return out

    def _portfolio_vol(
        self, weights: Dict[str, float], cand: Dict[str, Tuple[float, float]]
    ) -> float:
        """Portfolio vol assuming a fixed correlation between the 3x assets."""
        rho = self.assumed_correlation
        tickers = [t for t in weights if t in cand]
        var = 0.0
        for ti in tickers:
            vi = cand[ti][1]
            for tj in tickers:
                vj = cand[tj][1]
                corr = 1.0 if ti == tj else rho
                var += weights[ti] * weights[tj] * vi * vj * corr
        return float(np.sqrt(max(var, 0.0)))

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        if data is None or data.empty:
            return Allocation({self.safe_asset: 1.0})

        # --- inverse_vol: standalone logic, ignores the Sticky pick ---
        if self.allocation_mode == "inverse_vol":
            cand = self._candidate_mom_vol(data)
            if not cand:
                return Allocation({self.safe_asset: 1.0})
            inv = {t: 1.0 / v for t, (_m, v) in cand.items()}
            s = sum(inv.values())
            raw = {t: inv[t] / s for t in inv}
            port_vol = self._portfolio_vol(raw, cand)
            scale = min(1.0, self.vol_target / port_vol) if port_vol > 0.0 else 1.0
            weights = {t: raw[t] * scale for t in raw}
            safe_w = 1.0 - sum(weights.values())
            if safe_w > 1e-6:
                weights[self.safe_asset] = safe_w
            return Allocation(weights)

        # --- single_pick + cascade share the Sticky pick as their base ---
        # Step 1: EXACTLY single_pick's decision (full hysteresis,
        # monthly pick + weekly vol overlay).
        base_alloc = super().signal(current_date, data)
        base_w = dict(base_alloc.weights) if hasattr(base_alloc, "weights") else dict(base_alloc)

        if self.allocation_mode == "single_pick":
            return base_alloc

        # --- cascade: top pick unchanged, only cascade the remainder ---
        # Safe phase (only SPY) -> nothing to cascade.
        non_safe = [t for t in base_w if t != self.safe_asset]
        if not non_safe:
            return base_alloc
        pick = non_safe[0]
        w_pick = float(base_w[pick])
        # Top pick is already at max_total_leverage or above -> done.
        cum = w_pick
        if cum >= self.max_total_leverage - 1e-6:
            return base_alloc

        # Step 2: cascade the remainder into the OTHER candidates.
        others = self._candidate_mom_vol(data, exclude=pick)
        if not others:
            return base_alloc
        if self.cascade_order == "momentum":
            order = sorted(others, key=lambda t: others[t][0], reverse=True)
        else:  # vol: least volatile first
            order = sorted(others, key=lambda t: others[t][1])

        weights: Dict[str, float] = {pick: w_pick}
        for ticker in order:
            _mom, vol = others[ticker]
            cap_i = min(1.0, self.vol_target / vol)
            add = min(cap_i, self.max_total_leverage - cum)
            if add <= 1e-6:
                continue
            weights[ticker] = weights.get(ticker, 0.0) + add
            cum += add

        # Step 3: final remainder -> SPY.
        safe_w = 1.0 - sum(weights.values())
        if safe_w > 1e-6:
            weights[self.safe_asset] = weights.get(self.safe_asset, 0.0) + safe_w
        return Allocation(weights)


strategy = StickyLeveredCascade()
