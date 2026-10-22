"""
Classic 60/40 Portfolio Strategy

The traditional balanced portfolio with 60% equities and 40% bonds.
"""

from datetime import date

import pandas as pd

from backtest.strategy import Strategy, Allocation


class Classic6040(Strategy):
    """
    Classic 60% Equities / 40% Bonds portfolio.

    A traditional balanced portfolio that has been a benchmark for
    diversified investing for decades.

    Example:
        strategy = Classic6040()  # Uses SPY and BND
        strategy = Classic6040(equity="URTH", bonds="AGG")  # Custom tickers
    """

    name = "[Benchmark] 60/40"

    def __init__(
        self,
        equity: str = "SPY",
        bonds: str = "BND",
        equity_weight: float = 0.6
    ):
        """
        Initialize 60/40 strategy.

        Args:
            equity: Equity ETF ticker (default: SPY)
            bonds: Bond ETF ticker (default: BND)
            equity_weight: Weight for equities (default: 0.6)
        """
        self.params = {
            "equity": equity,
            "bonds": bonds,
            "equity_weight": equity_weight,
        }
        self.assets = [equity, bonds]
        self._equity = equity
        self._bonds = bonds
        self._equity_weight = equity_weight

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        """
        Return the 60/40 allocation.

        Args:
            date: Current date
            data: Historical price data (not used)

        Returns:
            60/40 allocation
        """
        bond_weight = round(1.0 - self._equity_weight, 10)
        return Allocation({
            self._equity: self._equity_weight,
            self._bonds: bond_weight,
        })


# Convenience constructors
def conservative() -> Classic6040:
    """40% Equity / 60% Bonds (conservative)"""
    return Classic6040(equity_weight=0.4)


def aggressive() -> Classic6040:
    """80% Equity / 20% Bonds (aggressive)"""
    return Classic6040(equity_weight=0.8)
