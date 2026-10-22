"""
Dual Momentum Strategy (Gary Antonacci)

Combines absolute momentum (trend following) with relative momentum
(cross-sectional) to select the best performing asset while avoiding
downturns.

Reference: "Dual Momentum Investing" by Gary Antonacci (2014)
"""

from datetime import date
from typing import List

import pandas as pd

from backtest.strategy import Strategy, Allocation


class DualMomentum(Strategy):
    """
    Gary Antonacci's Dual Momentum strategy.

    Rules:
    1. Calculate 12-month momentum for each risk asset
    2. Select the risk asset with highest momentum (relative momentum)
    3. If best risk asset has positive momentum, invest 100% in it
    4. If best risk asset has negative momentum, invest 100% in safe asset
       (absolute momentum / trend filter)

    Example:
        strategy = DualMomentum(
            risk_assets=["SPY", "EFA"],  # US vs International stocks
            safe_asset="BND",             # Bonds as safe haven
            lookback_months=12
        )
    """

    name = "[Benchmark] Dual Momentum"

    def __init__(
        self,
        risk_assets: List[str] = None,
        safe_asset: str = "BND",
        lookback_months: int = 12
    ):
        """
        Initialize Dual Momentum strategy.

        Args:
            risk_assets: List of risk asset tickers to choose from.
                         Default: ["SPY", "EFA"] (US and International)
            safe_asset: Safe haven asset ticker (default: BND)
            lookback_months: Momentum lookback period (default: 12)
        """
        if risk_assets is None:
            risk_assets = ["SPY", "EFA"]

        self.params = {
            "risk_assets": risk_assets,
            "safe_asset": safe_asset,
            "lookback": lookback_months,
        }
        self.assets = risk_assets + [safe_asset]
        self._risk_assets = risk_assets
        self._safe_asset = safe_asset
        self._lookback = lookback_months

    def _calculate_momentum(self, data: pd.DataFrame, ticker: str) -> float:
        """
        Calculate total return momentum for a ticker.

        Args:
            data: Price data
            ticker: Ticker symbol

        Returns:
            Momentum as decimal (e.g., 0.1 = 10% return)
        """
        if ticker not in data.columns:
            return 0.0

        prices = data[ticker].dropna()
        if len(prices) < self._lookback + 1:
            return 0.0

        current = prices.iloc[-1]
        past = prices.iloc[-self._lookback - 1]

        if past == 0:
            return 0.0

        return (current / past) - 1

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        """
        Generate allocation signal based on dual momentum.

        Args:
            date: Current date
            data: Historical price data

        Returns:
            Allocation to best performing asset
        """
        # Calculate momentum for all risk assets
        momentums = {
            ticker: self._calculate_momentum(data, ticker)
            for ticker in self._risk_assets
        }

        # Find best risk asset (relative momentum)
        best_asset = max(momentums, key=momentums.get)
        best_momentum = momentums[best_asset]

        # Apply absolute momentum filter
        if best_momentum > 0:
            # Positive momentum: invest in best risk asset
            return Allocation({best_asset: 1.0})
        else:
            # Negative momentum: retreat to safe asset
            return Allocation({self._safe_asset: 1.0})


# Convenience constructors for common configurations
def classic_dual_momentum() -> DualMomentum:
    """
    Classic Dual Momentum: US vs International stocks.
    SPY (S&P 500) vs EFA (EAFE), BND as safe asset.
    """
    return DualMomentum(
        risk_assets=["SPY", "EFA"],
        safe_asset="BND",
        lookback_months=12
    )


def global_dual_momentum() -> DualMomentum:
    """
    Global Dual Momentum with more regions.
    US, Developed ex-US, Emerging Markets.
    """
    return DualMomentum(
        risk_assets=["SPY", "EFA", "EEM"],
        safe_asset="BND",
        lookback_months=12
    )
