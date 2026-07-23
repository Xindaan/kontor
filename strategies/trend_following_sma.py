"""
SMA Trend Filter Strategy

Risk-on when risk_asset price is above its SMA (computed on month-end prices),
otherwise allocate to safe_asset.

Default: SPY vs BND, SMA=10 months.
"""

from datetime import date
from typing import Dict

import numpy as np
import pandas as pd

from backtest.strategy import Strategy, Allocation


def _infer_periods_per_year(idx: pd.Index) -> int:
    if len(idx) < 3:
        return 252
    deltas = np.diff(pd.to_datetime(idx).values).astype("timedelta64[D]").astype(int)
    med = float(np.median(deltas))
    if med >= 20:
        return 12
    if med >= 5:
        return 52
    return 252


def _month_end_prices(prices: pd.Series) -> pd.Series:
    s = prices.dropna()
    if s.empty:
        return s
    # Use 'ME' (month-end) to avoid FutureWarning
    try:
        return s.resample("ME").last().dropna()
    except ValueError:
        # Fallback for older pandas versions
        return s.resample("M").last().dropna()


class SMATrendFilter(Strategy):
    name = "[Benchmark] SMA Trend Filter"
    def __init__(
        self,
        risk_asset: str = "SPY",
        safe_asset: str = "BND",
        sma_months: int = 10,
        fallback: str = "risk",  # "risk" or "safe"
    ):
        self.params = {
            "risk_asset": risk_asset,
            "safe_asset": safe_asset,
            "sma_months": sma_months,
            "fallback": fallback,
        }
        self.assets = [risk_asset, safe_asset]
        self._risk = risk_asset
        self._safe = safe_asset
        self._sma_m = sma_months
        self._fallback = fallback

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        if data is None or data.empty or self._risk not in data.columns or self._safe not in data.columns:
            return Allocation({self._safe: 1.0})

        px = data[self._risk]
        me = _month_end_prices(px)
        if len(me) < self._sma_m + 1:
            return Allocation({self._risk: 1.0}) if self._fallback == "risk" else Allocation({self._safe: 1.0})

        sma = me.rolling(self._sma_m).mean().iloc[-1]
        last = me.iloc[-1]

        if pd.isna(sma) or pd.isna(last):
            return Allocation({self._safe: 1.0})

        if last > sma:
            return Allocation({self._risk: 1.0})
        return Allocation({self._safe: 1.0})


def gem_like() -> SMATrendFilter:
    return SMATrendFilter(risk_asset="SPY", safe_asset="BND", sma_months=10, fallback="risk")
