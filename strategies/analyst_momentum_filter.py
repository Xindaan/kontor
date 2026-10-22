"""Analyst Momentum Filter Demo Strategy (Phase B T-0061).

Top-N momentum picks, filtered/reweighted by analyst consensus from an
external features provider. Without a provider the strategy behaves like
a pure equal-weight top-N momentum baseline.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.strategy import Allocation, Strategy

DEFAULT_UNIVERSE = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD")
DEFAULT_TOP_N = 3
DEFAULT_LOOKBACK = 126
DEFAULT_ANALYST_CUTOFF = 0.0
DEFAULT_ANALYST_DATASET = "synthetic_analyst_pit"


class AnalystMomentumFilter(Strategy):
    """Demo strategy that combines price momentum with analyst consensus."""

    name = "Analyst Momentum Filter"

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        top_n: int = DEFAULT_TOP_N,
        lookback: int = DEFAULT_LOOKBACK,
        analyst_cutoff: float = DEFAULT_ANALYST_CUTOFF,
        analyst_dataset: Optional[str] = DEFAULT_ANALYST_DATASET,
    ) -> None:
        if universe is None:
            universe = list(DEFAULT_UNIVERSE)
        self.assets = list(universe)
        self.params: Dict[str, Any] = {
            "universe": list(universe),
            "top_n": int(top_n),
            "lookback": int(lookback),
            "analyst_cutoff": float(analyst_cutoff),
            "analyst_dataset": analyst_dataset,
        }
        self._top_n = int(top_n)
        self._lookback = int(lookback)
        self._analyst_cutoff = float(analyst_cutoff)
        self._analyst_dataset = analyst_dataset

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        usable_cols = [t for t in self.assets if t in data.columns]
        if not usable_cols:
            return Allocation({})

        prices = data[usable_cols].dropna(how="all")
        if len(prices) < 2:
            return Allocation({})
        lookback = min(self._lookback, len(prices) - 1)
        momentum = (
            prices.iloc[-1] / prices.iloc[-1 - lookback] - 1.0
        ).sort_values(ascending=False)
        candidates = list(momentum.head(self._top_n).index)
        if not candidates:
            return Allocation({})

        analyst_scores = self._fetch_analyst_scores(current_date, candidates)

        kept = []
        for ticker in candidates:
            score = analyst_scores.get(ticker, 0.0)
            if score >= self._analyst_cutoff:
                kept.append(ticker)
        if not kept:
            kept = candidates  # fallback: never go fully to cash

        weights = self._weight(kept, analyst_scores)
        return Allocation(weights)

    def _fetch_analyst_scores(
        self,
        as_of: date,
        tickers: List[str],
    ) -> Dict[str, float]:
        provider = getattr(self, "_external_features_provider", None)
        if provider is None or not tickers:
            return {}
        try:
            if self._analyst_dataset and hasattr(provider, "snapshot_dataset"):
                snap = provider.snapshot_dataset(self._analyst_dataset, as_of=as_of, tickers=tickers)
            else:
                snap = provider.snapshot(as_of=as_of, tickers=tickers)
        except Exception:
            return {}
        if snap is None or getattr(snap, "data", None) is None:
            return {}
        df = snap.data
        if df.empty or "feature_name" not in df.columns:
            return {}
        scores: Dict[str, float] = {}
        rows = df.loc[df["feature_name"] == "analyst_score"]
        for ticker, group in rows.groupby("ticker"):
            try:
                scores[str(ticker).upper()] = float(group["feature_value"].iloc[-1])
            except (TypeError, ValueError):
                continue
        return scores

    def _weight(
        self,
        tickers: List[str],
        analyst_scores: Dict[str, float],
    ) -> Dict[str, float]:
        if not tickers:
            return {}
        # Strategy-Analyst-Score per docs: weighted average of analyst
        # scores of the target allocation; missing scores count neutral
        # 0.0. For weighting the individual sleeves we keep it simple:
        # equal weight among kept names.
        weight = 1.0 / len(tickers)
        return {t: weight for t in tickers}


strategy = AnalystMomentumFilter()
