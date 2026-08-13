"""
Sector Rotation Strategy using S&P 500 Sector ETFs.

This strategy rotates between the 11 S&P 500 sector ETFs based on
relative momentum. It selects the top-N sectors with the strongest
momentum and equal-weights them.

Sectors (SPDR Select Sector ETFs):
- XLK: Technology
- XLF: Financials
- XLV: Health Care
- XLY: Consumer Discretionary
- XLP: Consumer Staples
- XLE: Energy
- XLI: Industrials
- XLB: Materials
- XLU: Utilities
- XLRE: Real Estate
- XLC: Communication Services

This strategy does NOT need PIT constituent data because:
1. The sector ETFs themselves are the universe (static)
2. We're trading the ETFs directly, not underlying stocks
3. No survivorship bias since ETFs are actively managed

Advantages:
- Simple to implement and understand
- Lower transaction costs (11 ETFs vs 500+ stocks)
- Built-in diversification within each sector
- Can capture sector rotation alpha

Disadvantages:
- Limited to US large-cap sectors
- May lag pure stock momentum strategies
- Sectors can be highly correlated in crisis
"""

from datetime import date
from pathlib import Path
from typing import List, Dict

import pandas as pd

from backtest.strategy import Strategy, Allocation

# Import momentum utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _momentum_utils import compute_momentum, pick_top


# All 11 S&P 500 Sector ETFs
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}


class SectorRotationMomentum(Strategy):
    """
    Sector Rotation based on Momentum.

    Selects top-N S&P 500 sector ETFs based on momentum and
    equal-weights them. Rotates monthly.

    Parameters:
        top_n: Number of top sectors to hold (default: 3)
        lookback_days: Momentum lookback period (default: 252)
        skip_days: Skip recent days to avoid mean reversion (default: 21)
        safe_asset: Asset for fallback when no valid signals (default: "SPY")
        min_history: Minimum days of history required (default: 252)
        use_absolute_filter: Require positive absolute momentum (default: False)
    """

    name = "[Benchmark] Sector Rotation Momentum"
    rebalance_frequency = "monthly"

    def __init__(
        self,
        top_n: int = 3,
        lookback_days: int = 252,
        skip_days: int = 21,
        safe_asset: str = "SPY",
        min_history: int = 252,
        use_absolute_filter: bool = False,
    ):
        self.top_n = top_n
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.safe_asset = safe_asset
        self.min_history = min_history
        self.use_absolute_filter = use_absolute_filter

        # Universe is fixed: all sector ETFs + safe asset
        self.sector_tickers = list(SECTOR_ETFS.keys())
        self.assets = self.sector_tickers + [safe_asset]

        self.params = {
            "top_n": top_n,
            "lookback_days": lookback_days,
            "skip_days": skip_days,
            "use_absolute_filter": use_absolute_filter,
            "sectors": len(SECTOR_ETFS),
        }

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        """
        Generate allocation signal.

        Ranks sectors by momentum, allocates to top N.
        """
        # Need enough history
        if len(data) < self.min_history:
            return Allocation({self.safe_asset: 1.0})

        # Calculate momentum for each sector ETF
        momentum_scores: Dict[str, float] = {}

        for ticker in self.sector_tickers:
            if ticker not in data.columns:
                continue

            prices = data[ticker].dropna()

            if len(prices) < self.lookback_days:
                continue

            mom = compute_momentum(prices, self.lookback_days, self.skip_days)
            if mom is not None:
                # Apply absolute momentum filter if enabled
                if self.use_absolute_filter and mom < 0:
                    continue
                momentum_scores[ticker] = mom

        if not momentum_scores:
            return Allocation({self.safe_asset: 1.0})

        # Pick top N sectors by momentum
        top_sectors = pick_top(momentum_scores, self.top_n)

        if not top_sectors:
            return Allocation({self.safe_asset: 1.0})

        # Equal weight allocation
        weight = 1.0 / len(top_sectors)
        weights = {ticker: weight for ticker in top_sectors}

        return Allocation(weights)


class SectorRotationDualMomentum(Strategy):
    """
    Sector Rotation with Dual Momentum filter.

    Similar to SectorRotationMomentum but with an additional
    absolute momentum filter: only invest in sectors with
    positive absolute momentum, otherwise go to safe asset.

    This adds crash protection by moving to cash when all
    sectors have negative momentum.
    """

    name = "[Benchmark] Sector Rotation Dual Momentum"
    rebalance_frequency = "monthly"

    def __init__(
        self,
        top_n: int = 3,
        lookback_days: int = 252,
        skip_days: int = 21,
        safe_asset: str = "SPY",
        min_history: int = 252,
        abs_threshold: float = 0.0,
    ):
        self.top_n = top_n
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.safe_asset = safe_asset
        self.min_history = min_history
        self.abs_threshold = abs_threshold

        self.sector_tickers = list(SECTOR_ETFS.keys())
        self.assets = self.sector_tickers + [safe_asset]

        self.params = {
            "top_n": top_n,
            "lookback_days": lookback_days,
            "skip_days": skip_days,
            "abs_threshold": abs_threshold,
            "dual_momentum": True,
        }

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        """Generate allocation with dual momentum filter."""
        if len(data) < self.min_history:
            return Allocation({self.safe_asset: 1.0})

        momentum_scores: Dict[str, float] = {}

        for ticker in self.sector_tickers:
            if ticker not in data.columns:
                continue

            prices = data[ticker].dropna()

            if len(prices) < self.lookback_days:
                continue

            mom = compute_momentum(prices, self.lookback_days, self.skip_days)
            if mom is not None and mom > self.abs_threshold:
                momentum_scores[ticker] = mom

        # If no sectors pass absolute momentum filter, go to safe asset
        if not momentum_scores:
            return Allocation({self.safe_asset: 1.0})

        top_sectors = pick_top(momentum_scores, self.top_n)

        if not top_sectors:
            return Allocation({self.safe_asset: 1.0})

        weight = 1.0 / len(top_sectors)
        weights = {ticker: weight for ticker in top_sectors}

        return Allocation(weights)


# =============================================================================
# Default Strategy Instances (for CLI loading)
# =============================================================================

# Basic sector rotation - top 3 sectors
strategy = SectorRotationMomentum(
    top_n=3,
    lookback_days=252,
    skip_days=21,
    safe_asset="SPY",
)

# Alternative: with dual momentum filter for crash protection
strategy_dual = SectorRotationDualMomentum(
    top_n=3,
    lookback_days=252,
    skip_days=21,
    safe_asset="SPY",
)
