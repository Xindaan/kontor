"""
Volatility Targeting Strategy

Adjusts equity exposure dynamically to maintain a target portfolio
volatility. When realized volatility is high, reduce equity exposure.
When volatility is low, increase equity exposure.
"""

from datetime import date

import numpy as np
import pandas as pd

from backtest.strategy import Strategy, Allocation


class VolatilityTargeting(Strategy):
    """
    Volatility Targeting strategy.

    Dynamically adjusts the allocation between a risk asset and safe asset
    to maintain a target portfolio volatility.

    Formula:
        equity_weight = target_volatility / realized_volatility

    This provides:
    - Lower exposure during volatile (often declining) markets
    - Higher exposure during calm (often rising) markets

    Example:
        strategy = VolatilityTargeting(
            risk_asset="SPY",
            safe_asset="BND",
            target_vol=0.10,      # 10% annual volatility target
            lookback_days=63      # ~3 months lookback
        )
    """

    name = "[Benchmark] Volatility Targeting"

    def __init__(
        self,
        risk_asset: str = "SPY",
        safe_asset: str = "BND",
        target_vol: float = 0.10,
        lookback_days: int = 63,
        max_leverage: float = 1.0,
        min_weight: float = 0.0
    ):
        """
        Initialize Volatility Targeting strategy.

        Args:
            risk_asset: Risky asset ticker (default: SPY)
            safe_asset: Safe haven asset ticker (default: BND)
            target_vol: Target annual volatility (default: 0.10 = 10%)
            lookback_days: Days to look back for volatility calc (default: 63)
            max_leverage: Maximum weight in risk asset (default: 1.0 = no leverage)
            min_weight: Minimum weight in risk asset (default: 0.0)
        """
        self.params = {
            "risk_asset": risk_asset,
            "safe_asset": safe_asset,
            "target_vol": target_vol,
            "lookback": lookback_days,
            "max_leverage": max_leverage,
            "min_weight": min_weight,
        }
        self.assets = [risk_asset, safe_asset]
        self._risk_asset = risk_asset
        self._safe_asset = safe_asset
        self._target_vol = target_vol
        self._lookback = lookback_days
        self._max_leverage = max_leverage
        self._min_weight = min_weight

    def _calculate_realized_vol(self, data: pd.DataFrame) -> float:
        """
        Calculate annualized realized volatility.

        Args:
            data: Price data

        Returns:
            Annualized volatility as decimal
        """
        if self._risk_asset not in data.columns:
            return self._target_vol  # Return target if no data

        prices = data[self._risk_asset].dropna()
        if len(prices) < self._lookback:
            return self._target_vol  # Not enough data

        # Calculate daily returns for lookback period
        recent_prices = prices.iloc[-self._lookback:]
        returns = recent_prices.pct_change(fill_method=None).dropna()

        if len(returns) < 2:
            return self._target_vol

        # Annualize (assuming ~21 trading days per month, 252 per year)
        # Since we're using monthly data, adjust accordingly
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        return annual_vol

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        """
        Generate allocation based on volatility targeting.

        Args:
            date: Current date
            data: Historical price data

        Returns:
            Allocation with volatility-adjusted weights
        """
        # Calculate realized volatility
        realized_vol = self._calculate_realized_vol(data)

        # Calculate target weight
        if realized_vol <= 0:
            target_weight = self._max_leverage
        else:
            target_weight = self._target_vol / realized_vol

        # Apply constraints
        target_weight = max(self._min_weight, min(target_weight, self._max_leverage))

        # Allocate remainder to safe asset
        safe_weight = 1.0 - target_weight

        return Allocation({
            self._risk_asset: target_weight,
            self._safe_asset: safe_weight,
        })


# Convenience constructors
def low_vol() -> VolatilityTargeting:
    """Low volatility target (8%)"""
    return VolatilityTargeting(target_vol=0.08)


def medium_vol() -> VolatilityTargeting:
    """Medium volatility target (12%)"""
    return VolatilityTargeting(target_vol=0.12)


def high_vol() -> VolatilityTargeting:
    """High volatility target (16%)"""
    return VolatilityTargeting(target_vol=0.16)
