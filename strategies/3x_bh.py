"""
Leveraged Buy & Hold

Invest 100% in a leveraged ETF ticker (e.g., UPRO, TQQQ).
"""

from datetime import date
import pandas as pd

from backtest.strategy import Strategy, Allocation


class LeveredBuyAndHold(Strategy):
    name = "[Benchmark] Levered Buy & Hold (3x)"

    def __init__(self, levered_ticker: str = "UPRO", rebalance_frequency: str = "monthly"):
        self.rebalance_frequency = rebalance_frequency
        self.params = {"levered_ticker": levered_ticker, "rebalance_frequency": rebalance_frequency}
        self.assets = [levered_ticker]
        self._t = levered_ticker
        self._alloc = Allocation({levered_ticker: 1.0})

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        return self._alloc


def sp500_3x() -> LeveredBuyAndHold:
    return LeveredBuyAndHold("UPRO")


def nasdaq100_3x() -> LeveredBuyAndHold:
    return LeveredBuyAndHold("TQQQ")
