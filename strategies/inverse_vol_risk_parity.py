"""
Inverse Volatility Risk Parity Strategy

Weights proportional to 1 / realized_vol over a lookback window.
Long-only, fully invested, optionally with min/max clamps.

Default: SPY + BND.
"""

from datetime import date
from typing import List, Dict

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


class InverseVolRiskParity(Strategy):
    name = "[Benchmark] Inverse Volatility Risk Parity"
    def __init__(
        self,
        assets: List[str] = None,
        lookback: int = 63,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        fallback: str = "equal",
    ):
        assets = assets or ["SPY", "BND"]
        self.params = {
            "assets": assets,
            "lookback": lookback,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "fallback": fallback,
        }
        self.assets = assets
        self._assets = assets
        self._lb = int(lookback)
        self._wmin = float(min_weight)
        self._wmax = float(max_weight)
        self._fallback = fallback

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        if data is None or data.empty:
            return self._fallback_alloc()

        cols = [a for a in self._assets if a in data.columns]
        if not cols:
            return self._fallback_alloc()

        px = data[cols].dropna(how="all")
        rets = px.pct_change(fill_method=None).dropna(how="all")
        if len(rets) < 5:
            return self._fallback_alloc(cols)

        window = rets.iloc[-self._lb:] if len(rets) >= self._lb else rets

        ppy = _infer_periods_per_year(window.index)
        vol = window.std(skipna=True) * np.sqrt(ppy)
        vol = vol.replace([0.0, np.inf, -np.inf], np.nan).dropna()

        if vol.empty:
            return self._fallback_alloc(cols)

        inv = 1.0 / vol
        w = inv / inv.sum()

        # clamp & renormalize
        w = w.clip(lower=self._wmin, upper=self._wmax)
        if w.sum() <= 0:
            return self._fallback_alloc(cols)
        w = w / w.sum()

        return Allocation({k: float(w[k]) for k in w.index})

    def _fallback_alloc(self, cols=None) -> Allocation:
        cols = cols or self._assets
        cols = list(cols)
        if not cols:
            return Allocation({})
        if self._fallback == "equal":
            w = 1.0 / len(cols)
            return Allocation({c: w for c in cols})
        # default defensive: first asset
        return Allocation({cols[0]: 1.0})
