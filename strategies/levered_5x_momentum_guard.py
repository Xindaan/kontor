"""
5x momentum strategy with risk gates and 3x/1x fallback.

Uses real 5x ETP tickers that are available in Yahoo data:
- QQQ5.DE (Leverage Shares 5x Long Nasdaq 100)
- 5SPE.DE (Leverage Shares 5x Long S&P 500)

Because these products launched later, the strategy falls back to 3x/1x
alternatives when 5x data is missing or fails risk checks.
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
    window = clean.iloc[-lookback_days:]
    peak = float(window.max())
    if peak <= 0.0:
        return None
    dd = float(window.iloc[-1] / peak - 1.0)
    return dd if np.isfinite(dd) else None


class Levered5xMomentumGuard(Strategy):
    name = "[Experimental] Levered 5x Momentum Guard (real tickers)"
    rebalance_frequency = "monthly"

    def __init__(
        self,
        candidates_5x: List[str] | None = None,
        fallback_3x: Dict[str, str] | None = None,
        fallback_1x: Dict[str, str] | None = None,
        safe_asset: str = "SPY",
        trend_asset: str | None = "SPY",
        trend_window_days: int = 200,
        momentum_lookback_days: int = 84,
        momentum_floor: float = -0.02,
        vol_lookback_days: int = 21,
        max_vol_for_5x: float = 0.75,
        max_drawdown_for_5x: float = -0.20,
        max_vol_for_3x: float = 0.90,
        max_drawdown_for_3x: float = -0.50,
    ) -> None:
        if candidates_5x is None:
            candidates_5x = ["QQQ5.DE", "5SPE.DE"]
        if fallback_3x is None:
            fallback_3x = {"QQQ5.DE": "TQQQ", "5SPE.DE": "UPRO"}
        if fallback_1x is None:
            fallback_1x = {"QQQ5.DE": "QQQ", "5SPE.DE": "SPY"}

        self.candidates_5x = list(candidates_5x)
        self.fallback_3x = dict(fallback_3x)
        self.fallback_1x = dict(fallback_1x)
        self.safe_asset = safe_asset
        self.trend_asset = trend_asset
        self.trend_window_days = int(trend_window_days)
        self.momentum_lookback_days = int(momentum_lookback_days)
        self.momentum_floor = float(momentum_floor)
        self.vol_lookback_days = int(vol_lookback_days)
        self.max_vol_for_5x = float(max_vol_for_5x)
        self.max_drawdown_for_5x = float(max_drawdown_for_5x)
        self.max_vol_for_3x = float(max_vol_for_3x)
        self.max_drawdown_for_3x = float(max_drawdown_for_3x)

        assets = (
            self.candidates_5x
            + list(self.fallback_3x.values())
            + list(self.fallback_1x.values())
            + ([self.trend_asset] if self.trend_asset else [])
            + [self.safe_asset]
        )
        self.assets = list(dict.fromkeys(assets))
        self.params = {
            "candidates_5x": self.candidates_5x,
            "fallback_3x": self.fallback_3x,
            "fallback_1x": self.fallback_1x,
            "safe_asset": safe_asset,
            "trend_asset": trend_asset,
            "trend_window_days": trend_window_days,
            "momentum_lookback_days": momentum_lookback_days,
            "momentum_floor": momentum_floor,
            "vol_lookback_days": vol_lookback_days,
            "max_vol_for_5x": max_vol_for_5x,
            "max_drawdown_for_5x": max_drawdown_for_5x,
            "max_vol_for_3x": max_vol_for_3x,
            "max_drawdown_for_3x": max_drawdown_for_3x,
        }

    @classmethod
    def get_param_grid(cls):
        return {
            "momentum_lookback_days": [42, 63, 84],
            "max_vol_for_5x": [0.75, 0.85, 0.95],
            "max_drawdown_for_5x": [-0.30, -0.20, -0.10],
            "max_vol_for_3x": [0.9, 1.0, 1.1],
            "max_drawdown_for_3x": [-0.60, -0.50, -0.40],
            "momentum_floor": [-0.02, 0.0, 0.02],
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

    def _pick_momentum_leader(self, data: pd.DataFrame, tickers: List[str]) -> tuple[Optional[str], float]:
        best_ticker: Optional[str] = None
        best_mom = -np.inf
        for ticker in tickers:
            if ticker not in data.columns:
                continue
            mom = _momentum(data[ticker], self.momentum_lookback_days)
            if mom is None:
                continue
            if mom > best_mom:
                best_mom = mom
                best_ticker = ticker
        return best_ticker, float(best_mom)

    def _passes_risk_gates(
        self,
        data: pd.DataFrame,
        ticker: str,
        max_vol: float,
        max_drawdown: float,
    ) -> bool:
        if ticker not in data.columns:
            return False
        prices = data[ticker]
        vol = _annualized_vol(prices, self.vol_lookback_days)
        dd = _recent_drawdown(prices, self.vol_lookback_days * 3)
        if vol is None or dd is None:
            return False
        return bool(vol <= max_vol and dd >= max_drawdown)

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        if data is None or data.empty:
            return Allocation({self.safe_asset: 1.0})

        if not self._trend_ok(data):
            return Allocation({self.safe_asset: 1.0})

        leader_5x, mom_5x = self._pick_momentum_leader(data, self.candidates_5x)
        if leader_5x is None or mom_5x <= self.momentum_floor:
            # No usable 5x history yet (or weak momentum): use 3x fallback basket.
            leader_3x, mom_3x = self._pick_momentum_leader(data, list(self.fallback_3x.values()))
            if leader_3x is not None and mom_3x > self.momentum_floor:
                if self._passes_risk_gates(
                    data,
                    leader_3x,
                    max_vol=self.max_vol_for_3x,
                    max_drawdown=self.max_drawdown_for_3x,
                ):
                    return Allocation({leader_3x: 1.0})
                fallback_1x = next(
                    (k for k, v in self.fallback_1x.items() if self.fallback_3x.get(k) == leader_3x),
                    None,
                )
                one_x = self.fallback_1x.get(fallback_1x) if fallback_1x else None
                if one_x and one_x in data.columns:
                    return Allocation({one_x: 1.0})
            return Allocation({self.safe_asset: 1.0})

        if self._passes_risk_gates(
            data,
            leader_5x,
            max_vol=self.max_vol_for_5x,
            max_drawdown=self.max_drawdown_for_5x,
        ):
            return Allocation({leader_5x: 1.0})

        # 5x fails risk gates -> degrade to 3x for same underlying, then 1x.
        fallback_3x = self.fallback_3x.get(leader_5x)
        if fallback_3x and fallback_3x in data.columns:
            mom_3x = _momentum(data[fallback_3x], self.momentum_lookback_days)
            if mom_3x is not None and mom_3x > self.momentum_floor:
                if self._passes_risk_gates(
                    data,
                    fallback_3x,
                    max_vol=self.max_vol_for_3x,
                    max_drawdown=self.max_drawdown_for_3x,
                ):
                    return Allocation({fallback_3x: 1.0})

        fallback_1x = self.fallback_1x.get(leader_5x)
        if fallback_1x and fallback_1x in data.columns:
            return Allocation({fallback_1x: 1.0})

        return Allocation({self.safe_asset: 1.0})


strategy = Levered5xMomentumGuard()
