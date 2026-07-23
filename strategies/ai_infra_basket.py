"""
[Research] AI Infrastructure Basket — rule-based, point-in-time basket.

Research candidate (NOT a master allocation). Builds a diversified basket
from the AI infra value-chain classification
(``data/universes/ai_infra_segments.csv``), with a deliberate underweight of
the semiconductor complex (``cap_group == "semi"``), because the existing
core is a 3x semi ETF.

Honesty / bias control:
- **U-A (default, promotion-grade):** point-in-time, survivorship-complete
  universe from ``ai_infra_pit.csv`` (NDX∪SP500 membership ∩ segments; via
  ``backtest.research.segments.build_ai_infra_pit_universe``). Includes
  delisted/acquired names up to their removal date.
- **U-B (foil, NOT promotable):** static list of names from a report
  (``ai_infra_ub.csv``) — look-ahead by construction, only used to measure
  the hindsight premium.
Selection uses exclusively PIT-clean signals (price momentum, trend,
realized vol). Fundamental "AI revenue share" data is historically NOT
PIT-available and is deliberately NOT used.

Methodologies:
- M1 Momentum + trend filter: 12-1 momentum, top-N, only names above the
  200d MA and with positive momentum; cash/safe fallback.
- M2 Momentum + low-vol: top-N by momentum, inverse-vol weighted.
- M3 Segment-balanced buy&hold: whole segment universe, low turnover.

Tax: single stocks -> ``BacktestConfig.equity_fund_map`` (is_equity_fund=False,
no Teilfreistellung [partial tax exemption]). Set by the audit/backtest, not
by the strategy.
"""

from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backtest.strategy import Strategy, Allocation
from backtest.universe import CsvPITUniverseProvider
from backtest.research.segments import (
    DEFAULT_AI_INFRA_PIT_CSV,
    DEFAULT_SEGMENTS_CSV,
    load_cap_groups,
    load_segment_map,
)

# Momentum utilities from the strategies/ package (same as the PIT top-N template).
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _momentum_utils import (  # noqa: E402
    compute_momentum,
    pick_top,
    inv_vol_weights,
    sma,
)


class AIInfraBasket(Strategy):
    """Rule-based AI infrastructure basket (research)."""

    name = "[Research] AI Infrastructure Basket"
    rebalance_frequency = "quarterly"

    def __init__(
        self,
        universe_csv: str = DEFAULT_AI_INFRA_PIT_CSV,
        segments_csv: str = DEFAULT_SEGMENTS_CSV,
        methodic: str = "M3",                 # "M1" | "M2" | "M3"
        weighting: str = "equal",             # "equal" | "inverse_vol"
        semi_cap: Optional[float] = 0.35,     # cap on cap_group=="semi" (None=off)
        top_n: int = 15,
        lookback_days: int = 252,
        skip_days: int = 21,
        vol_lookback: int = 63,
        max_weight: float = 0.20,
        trend_ma: int = 200,
        min_history: int = 252,
        min_names: int = 5,
        safe_asset: str = "SPY",
        rebalance_frequency: str = "quarterly",
    ):
        self.universe_csv = universe_csv
        self.segments_csv = segments_csv
        self.methodic = methodic.upper()
        self.weighting = weighting
        self.semi_cap = semi_cap
        self.top_n = top_n
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.vol_lookback = vol_lookback
        self.max_weight = max_weight
        self.trend_ma = trend_ma
        self.min_history = min_history
        self.min_names = min_names
        self.safe_asset = safe_asset
        self.rebalance_frequency = rebalance_frequency

        if self.methodic not in {"M1", "M2", "M3"}:
            raise ValueError(f"Unbekannte methodic: {methodic}")
        if self.weighting not in {"equal", "inverse_vol"}:
            raise ValueError(f"Unbekannte weighting: {weighting}")

        # Segment/cap classification (static thesis map).
        self._segment_map: Dict[str, str] = load_segment_map(segments_csv)
        self._cap_group: Dict[str, str] = load_cap_groups(segments_csv)

        # PIT universe provider.
        self._universe_provider: Optional[CsvPITUniverseProvider] = None
        self._all_tickers: List[str] = []
        try:
            self._universe_provider = CsvPITUniverseProvider(
                path=universe_csv, date_col="as_of", ticker_col="ticker",
            )
            self._all_tickers = sorted({
                t for tickers in self._universe_provider._snapshots.values()
                for t in tickers
            })
        except FileNotFoundError:
            import warnings
            warnings.warn(
                f"PIT-Universum-CSV nicht gefunden: {universe_csv}. "
                "Strategie faellt auf Safe-Asset zurueck, bis sie gebaut ist "
                "(backtest.research.segments.build_ai_infra_pit_universe)."
            )

        self.assets = sorted(set(self._all_tickers + [safe_asset]))

        self.params = {
            "universe_csv": universe_csv,
            "methodic": self.methodic,
            "weighting": self.weighting,
            "semi_cap": semi_cap,
            "top_n": top_n,
            "lookback_days": lookback_days,
            "skip_days": skip_days,
            "vol_lookback": vol_lookback,
            "max_weight": max_weight,
            "trend_ma": trend_ma,
            "rebalance_frequency": rebalance_frequency,
            "point_in_time": True,
        }

    # ------------------------------------------------------------------ #
    def _get_universe(self, as_of: date) -> List[str]:
        if self._universe_provider is None:
            return []
        return self._universe_provider.snapshot(as_of).tickers

    def _eligible(self, universe: List[str], data: pd.DataFrame) -> List[str]:
        """Names with sufficient history and an available price series."""
        out = []
        for ticker in universe:
            if ticker not in data.columns:
                continue
            prices = data[ticker].dropna()
            if len(prices) < self.lookback_days:
                continue
            out.append(ticker)
        return out

    def _select(self, eligible: List[str], data: pd.DataFrame) -> List[str]:
        """Methodology-specific selection of the names to hold."""
        if self.methodic == "M3":
            # Segment-balanced buy&hold: hold all eligible names.
            return list(eligible)

        # M1/M2: momentum ranking.
        scores: Dict[str, float] = {}
        for ticker in eligible:
            prices = data[ticker].dropna()
            mom = compute_momentum(prices, self.lookback_days, self.skip_days)
            if mom is None:
                continue
            if self.methodic == "M1":
                # Trend filter: only above the 200d MA and with positive momentum.
                ma = sma(prices, self.trend_ma)
                if ma is None or prices.iloc[-1] < ma or mom <= 0:
                    continue
            scores[ticker] = mom

        return pick_top(scores, self.top_n)

    def _base_weights(self, selected: List[str], data: pd.DataFrame) -> Dict[str, float]:
        if not selected:
            return {}
        if self.weighting == "inverse_vol":
            w = inv_vol_weights(
                data, selected, vol_lookback=self.vol_lookback, cap=self.max_weight,
            )
            return {t: v for t, v in w.items() if v > 0}
        # equal weight
        each = 1.0 / len(selected)
        return {t: min(each, self.max_weight) for t in selected}

    def _apply_semi_cap(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Cap the aggregate weight of the cap_group=='semi' names.

        Excess is distributed proportionally to diversifiers; if there are
        none, it stays as cash (Allocation treats the remainder as cash).
        """
        if self.semi_cap is None:
            return weights
        semi = {t: w for t, w in weights.items() if self._cap_group.get(t) == "semi"}
        div = {t: w for t, w in weights.items() if self._cap_group.get(t) != "semi"}
        semi_sum = sum(semi.values())
        if semi_sum <= self.semi_cap + 1e-12 or semi_sum <= 0:
            return weights

        scale = self.semi_cap / semi_sum
        for t in semi:
            weights[t] *= scale
        excess = semi_sum - self.semi_cap
        div_sum = sum(div.values())
        if div_sum > 0:
            for t in div:
                add = excess * (div[t] / div_sum)
                weights[t] = min(weights[t] + add, self.max_weight)
        return weights

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        universe = self._get_universe(current_date)
        if not universe or len(data) < self.min_history:
            return Allocation({self.safe_asset: 1.0})

        eligible = self._eligible(universe, data)
        if len(eligible) < self.min_names:
            return Allocation({self.safe_asset: 1.0})

        selected = self._select(eligible, data)
        if len(selected) < self.min_names:
            # The M1 trend filter can filter out (almost) everything during crashes.
            return Allocation({self.safe_asset: 1.0})

        weights = self._base_weights(selected, data)
        weights = self._apply_semi_cap(weights)

        # Numerical cleanup: ensure <=1.0 (Allocation raises otherwise).
        total = sum(weights.values())
        if total > 1.0:
            weights = {t: w / total for t, w in weights.items()}
        weights = {t: w for t, w in weights.items() if w > 1e-6}

        if not weights:
            return Allocation({self.safe_asset: 1.0})
        return Allocation(weights)

    @classmethod
    def get_param_grid(cls):
        return {
            "methodic": ["M1", "M2", "M3"],
            "weighting": ["equal", "inverse_vol"],
            "semi_cap": [0.30, 0.40, None],
            "rebalance_frequency": ["quarterly", "monthly"],
        }


# Default instance for CLI loading (works if ai_infra_pit.csv exists).
strategy = AIInfraBasket()
