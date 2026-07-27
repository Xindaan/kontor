"""
Buy and Hold Strategy

Simple strategy that maintains a fixed allocation without active trading.
Rebalancing only occurs to maintain the target weights.
"""

from datetime import date
from typing import Dict

import pandas as pd

from backtest.strategy import Strategy, Allocation


class BuyAndHold(Strategy):
    """
    Buy and Hold strategy with fixed allocation.

    This strategy maintains a constant target allocation, rebalancing
    monthly to restore the original weights.

    Example:
        strategy = BuyAndHold({"SPY": 0.6, "BND": 0.4})
    """

    name = "[Benchmark] Buy & Hold"

    def __init__(self, allocation: Dict[str, float] = None):
        """
        Initialize Buy and Hold strategy.

        Args:
            allocation: Target allocation as dict of ticker -> weight.
                        Weights should sum to <= 1.0.
                        Example: {"SPY": 0.6, "BND": 0.4}
                        Default: 100% SPY
        """
        if allocation is None:
            allocation = {"SPY": 1.0}
        self.params = {"allocation": allocation}
        self.assets = list(allocation.keys())
        self._allocation = Allocation(allocation)

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        """
        Return the fixed target allocation.

        Args:
            date: Current date
            data: Historical price data (not used)

        Returns:
            Fixed target allocation
        """
        return self._allocation


# Convenience function to create common allocations
def msci_world() -> BuyAndHold:
    """100% MSCI World (iShares Core)"""
    return BuyAndHold({"URTH": 1.0})


def sp500() -> BuyAndHold:
    """100% S&P 500 (SPDR)"""
    return BuyAndHold({"SPY": 1.0})
