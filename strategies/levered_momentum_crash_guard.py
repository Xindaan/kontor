"""
Levered momentum rotation with crash guard.

Compared with pure winner-takes-all 3x rotation, this variant adds:
- broad trend filter (risk-off below long SMA),
- 3x vs 1x switch based on short-term realized vol and drawdown.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.strategy import Strategy, Allocation


def _momentum(prices: pd.Series, lookback_days: int) -> Optional[float]:
    clean = prices.dropna()
    if len(clean) <= lookback_days:
        return None
    p0 = float(clean.iloc[-lookback_days - 1])
    p1 = float(clean.iloc[-1])
    if p0 <= 0:
        return None
    return (p1 / p0) - 1.0


def _annualized_vol(prices: pd.Series, lookback_days: int) -> Optional[float]:
    clean = prices.dropna()
    if len(clean) < lookback_days + 1:
        return None
    rets = clean.pct_change(fill_method=None).dropna().iloc[-lookback_days:]
    if len(rets) < lookback_days:
        return None
    vol = float(rets.std(ddof=0) * np.sqrt(252))
    return vol if np.isfinite(vol) else None


def _recent_drawdown(prices: pd.Series, lookback_days: int) -> Optional[float]:
    clean = prices.dropna()
    if len(clean) < lookback_days:
        return None
    w = clean.iloc[-lookback_days:]
    peak = float(w.max())
    if peak <= 0:
        return None
    dd = float(w.iloc[-1] / peak - 1.0)
    return dd if np.isfinite(dd) else None


class LeveredMomentumCrashGuard(Strategy):
    name = "[Research] Levered Momentum + Crash Guard (3x/1x/fallback)"
    rebalance_frequency = "monthly"

    def __init__(
        self,
        candidates_3x: List[str] | None = None,
        fallback_1x: Dict[str, str] | None = None,
        safe_asset: str = "SPY",
        trend_asset: str | None = "SPY",
        trend_window_days: int = 200,
        momentum_lookback_days: int = 126,
        momentum_floor: float = 0.0,
        vol_lookback_days: int = 21,
        max_vol_for_3x: float = 0.55,
        drawdown_lookback_days: int = 63,
        max_drawdown_for_3x: float = -0.18,
    ) -> None:
        if candidates_3x is None:
            candidates_3x = ["TQQQ", "UPRO", "SOXL", "TECL"]
        if fallback_1x is None:
            fallback_1x = {
                "TQQQ": "QQQ",
                "UPRO": "SPY",
                "SOXL": "SOXX",
                "TECL": "XLK",
            }

        self.candidates_3x = list(candidates_3x)
        self.fallback_1x = dict(fallback_1x)
        self.safe_asset = safe_asset
        self.trend_asset = trend_asset
        self.trend_window_days = int(trend_window_days)
        self.momentum_lookback_days = int(momentum_lookback_days)
        self.momentum_floor = float(momentum_floor)
        self.vol_lookback_days = int(vol_lookback_days)
        self.max_vol_for_3x = float(max_vol_for_3x)
        self.drawdown_lookback_days = int(drawdown_lookback_days)
        self.max_drawdown_for_3x = float(max_drawdown_for_3x)

        self.assets = list(
            dict.fromkeys(
                self.candidates_3x
                + list(self.fallback_1x.values())
                + ([self.trend_asset] if self.trend_asset else [])
                + [self.safe_asset]
            )
        )
        self.params = {
            "candidates_3x": self.candidates_3x,
            "fallback_1x": self.fallback_1x,
            "safe_asset": safe_asset,
            "trend_asset": trend_asset,
            "trend_window_days": trend_window_days,
            "momentum_lookback_days": momentum_lookback_days,
            "momentum_floor": momentum_floor,
            "vol_lookback_days": vol_lookback_days,
            "max_vol_for_3x": max_vol_for_3x,
            "drawdown_lookback_days": drawdown_lookback_days,
            "max_drawdown_for_3x": max_drawdown_for_3x,
        }

    @classmethod
    def get_param_grid(cls):
        return {
            "momentum_lookback_days": [63, 126, 189],
            "max_vol_for_3x": [0.40, 0.50, 0.60],
            "max_drawdown_for_3x": [-0.12, -0.18, -0.24],
        }

    def _trend_ok(self, data: pd.DataFrame) -> bool:
        if not self.trend_asset:
            return True
        if self.trend_asset not in data.columns:
            return True
        prices = data[self.trend_asset].dropna()
        if len(prices) < self.trend_window_days:
            return True
        px = float(prices.iloc[-1])
        sma = float(prices.iloc[-self.trend_window_days :].mean())
        return bool(np.isfinite(sma) and px >= sma)

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        if data is None or data.empty:
            return Allocation({self.safe_asset: 1.0})

        if not self._trend_ok(data):
            return Allocation({self.safe_asset: 1.0})

        best_ticker = None
        best_mom = -np.inf
        for ticker in self.candidates_3x:
            if ticker not in data.columns:
                continue
            m = _momentum(data[ticker], self.momentum_lookback_days)
            if m is None:
                continue
            if m > best_mom:
                best_mom = m
                best_ticker = ticker

        if best_ticker is None or best_mom <= self.momentum_floor:
            return Allocation({self.safe_asset: 1.0})

        prices = data[best_ticker]
        vol = _annualized_vol(prices, self.vol_lookback_days)
        dd = _recent_drawdown(prices, self.drawdown_lookback_days)

        if (
            vol is not None
            and dd is not None
            and vol <= self.max_vol_for_3x
            and dd >= self.max_drawdown_for_3x
        ):
            return Allocation({best_ticker: 1.0})

        fallback = self.fallback_1x.get(best_ticker)
        if fallback and fallback in data.columns:
            return Allocation({fallback: 1.0})
        return Allocation({self.safe_asset: 1.0})


strategy = LeveredMomentumCrashGuard()
