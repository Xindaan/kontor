"""
Top-N Momentum Strategy with Point-in-Time S&P 500 Universe.

This strategy uses historical S&P 500 constituent data to avoid
survivorship bias. It loads the universe from a CSV file that
contains monthly snapshots of which stocks were in the S&P 500
at each historical date.

IMPORTANT: This is the "fair" way to backtest momentum strategies
because it only uses stocks that were actually in the index at
each rebalance date, not the current index composition.

Required CSV format (data/universes/sp500_constituents.csv):
    as_of,ticker
    2010-01-01,AAPL
    2010-01-01,MSFT
    ...
    2010-02-01,AAPL
    2010-02-01,MSFT
    ...
"""

from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from backtest.strategy import Strategy, Allocation
from backtest.universe import (
    CsvPITUniverseProvider,
    UniverseProvider,
    validate_universe_for_backtest,
)

# Import momentum utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))
from _momentum_utils import compute_momentum, pick_top


class MomentumTopNPITSP500(Strategy):
    """
    Top-N Momentum with Point-in-Time S&P 500 Universe.

    Uses historical constituent data to avoid survivorship bias.
    Only considers stocks that were in the S&P 500 at each rebalance date.

    Parameters:
        csv_path: Path to CSV with historical constituents
                  (default: "data/universes/sp500_constituents.csv")
        top_n: Number of top stocks to hold (default: 10)
        lookback_days: Momentum lookback period (default: 252)
        skip_days: Skip recent days (default: 21)
        safe_asset: Asset for fallback (default: "SPY")
        min_history: Minimum days of history required per stock (default: 252)

    CSV Format:
        as_of,ticker
        2010-01-01,AAPL
        2010-01-01,MSFT
        ...
    """

    name = "[Benchmark] Momentum Top-N (PIT S&P 500)"
    rebalance_frequency = "monthly"

    def __init__(
        self,
        csv_path: str = "data/universes/sp500_constituents.csv",
        top_n: int = 10,
        lookback_days: int = 252,
        skip_days: int = 21,
        safe_asset: str = "SPY",
        min_history: int = 252,
    ):
        self.csv_path = csv_path
        self.top_n = top_n
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.safe_asset = safe_asset
        self.min_history = min_history

        # Try to load the PIT provider
        self._universe_provider: Optional[CsvPITUniverseProvider] = None
        self._all_tickers: List[str] = []

        try:
            self._universe_provider = CsvPITUniverseProvider(
                path=csv_path,
                date_col="as_of",
                ticker_col="ticker",
            )
            # Get all unique tickers for assets list
            self._all_tickers = list(set(
                ticker
                for tickers in self._universe_provider._snapshots.values()
                for ticker in tickers
            ))
        except FileNotFoundError:
            # CSV not found - will fail gracefully
            import warnings
            warnings.warn(
                f"PIT universe CSV not found: {csv_path}. "
                "Strategy will return safe asset until CSV is provided."
            )
            self._all_tickers = []

        # Assets includes all possible universe tickers + safe asset
        self.assets = list(set(self._all_tickers + [safe_asset]))

        self.params = {
            "csv_path": csv_path,
            "top_n": top_n,
            "lookback_days": lookback_days,
            "skip_days": skip_days,
            "point_in_time": True,
        }

    def _get_universe(self, as_of: date) -> List[str]:
        """Get the S&P 500 constituents as of the given date."""
        if self._universe_provider is None:
            return []

        snapshot = self._universe_provider.snapshot(as_of)

        # Validate that this is PIT data (should always be True)
        # No need to call validate_universe_for_backtest since PIT is always safe
        return snapshot.tickers

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        """
        Generate allocation signal.

        Gets PIT universe for current date, ranks by momentum, allocates to top N.
        """
        # Get universe for this date
        universe = self._get_universe(current_date)

        if not universe:
            return Allocation({self.safe_asset: 1.0})

        # Need enough history
        if len(data) < self.min_history:
            return Allocation({self.safe_asset: 1.0})

        # Calculate momentum for each ticker in the PIT universe
        momentum_scores = {}

        for ticker in universe:
            if ticker not in data.columns:
                # Ticker not in data (maybe delisted or no data)
                continue

            prices = data[ticker].dropna()

            # Check if ticker has enough history
            if len(prices) < self.lookback_days:
                continue

            mom = compute_momentum(prices, self.lookback_days, self.skip_days)
            if mom is not None:
                momentum_scores[ticker] = mom

        if not momentum_scores:
            return Allocation({self.safe_asset: 1.0})

        # Pick top N by momentum
        top_tickers = pick_top(momentum_scores, self.top_n)

        # If we have fewer than top_n valid tickers, check threshold
        if len(top_tickers) < self.top_n:
            # If very few valid tickers, go to safe asset
            if len(top_tickers) < max(1, self.top_n // 2):
                return Allocation({self.safe_asset: 1.0})

        if not top_tickers:
            return Allocation({self.safe_asset: 1.0})

        # Equal weight allocation
        weight = 1.0 / len(top_tickers)
        weights = {ticker: weight for ticker in top_tickers}

        return Allocation(weights)


# =============================================================================
# Default Strategy Instance (for CLI loading)
# =============================================================================

# This will work if the CSV exists, otherwise fall back to safe asset
strategy = MomentumTopNPITSP500(
    csv_path="data/universes/sp500_constituents.csv",
    top_n=10,
    lookback_days=252,
    skip_days=21,
    safe_asset="SPY",
)
