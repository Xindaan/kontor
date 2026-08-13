"""
All Weather Strategy (ETF proxies).

Classic risk-parity inspired allocation across asset classes:
- Equities
- Long-term bonds
- Intermediate bonds
- Gold
- Commodities

Weights are static and rebalanced monthly by default.
"""

from typing import Dict, Optional

from backtest.strategy import Strategy, Allocation


class AllWeatherStrategy(Strategy):
    """All Weather strategy using liquid ETF proxies."""

    name = "[Benchmark] All Weather"
    rebalance_frequency = "monthly"

    def __init__(
        self,
        equity_ticker: str = "SPY",
        long_bond_ticker: str = "TLT",
        mid_bond_ticker: str = "IEF",
        gold_ticker: str = "GLD",
        commodity_ticker: str = "DBC",
        weights: Optional[Dict[str, float]] = None,
    ):
        self.assets = [
            equity_ticker,
            long_bond_ticker,
            mid_bond_ticker,
            gold_ticker,
            commodity_ticker,
        ]
        default_weights = {
            equity_ticker: 0.30,
            long_bond_ticker: 0.40,
            mid_bond_ticker: 0.15,
            gold_ticker: 0.075,
            commodity_ticker: 0.075,
        }
        self.params = {
            "weights": weights or default_weights,
        }

    def signal(self, date, data):
        weights = self.params["weights"]
        return Allocation(weights)
