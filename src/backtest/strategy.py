"""
Strategy module - Base class for all investment strategies.

Every strategy must have a unique, descriptive name. The name is used
in reports and for identification. If not explicitly set, the class
name is used as a fallback.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Any, Optional

import pandas as pd


class _CashAccessor:
    """
    Backwards-compatible cash accessor.

    - Instance usage: allocation.cash -> float cash weight
    - Class usage: Allocation.cash() -> Allocation in 100% cash
    """

    def __get__(self, instance, owner):
        if instance is None:
            return lambda: owner({})
        return 1.0 - sum(instance.weights.values())


@dataclass
class Allocation:
    """
    Represents a portfolio allocation.

    Weights must sum to <= 1.0. The remainder is held in cash.

    Attributes:
        weights: Dictionary mapping ticker symbols to their target weights.
                 Example: {"SPY": 0.6, "BND": 0.4}
    """
    weights: Dict[str, float] = field(default_factory=dict)
    cash = _CashAccessor()

    def __post_init__(self):
        total = sum(self.weights.values())
        if not (0 <= total <= 1.0 + 1e-9):  # Small epsilon for float precision
            raise ValueError(f"Weights must sum to ≤1.0, got {total:.4f}")
        # Normalize if slightly over due to float precision
        if total > 1.0:
            factor = 1.0 / total
            self.weights = {k: v * factor for k, v in self.weights.items()}

    def get(self, ticker: str, default: float = 0.0) -> float:
        """Get weight for a specific ticker."""
        return self.weights.get(ticker, default)

    def items(self):
        """Iterate over (ticker, weight) pairs."""
        return self.weights.items()

    def keys(self):
        """Return ticker symbols."""
        return self.weights.keys()

    def values(self):
        """Return weights."""
        return self.weights.values()

    def __repr__(self) -> str:
        weights_str = ", ".join(f"{k}: {v:.1%}" for k, v in self.weights.items())
        return f"Allocation({{{weights_str}}}, cash={self.cash:.1%})"


class StrategyMeta(type(ABC)):
    """
    Metaclass for Strategy that enforces naming conventions.

    If a Strategy subclass doesn't define a 'name' attribute,
    the class name is used as a fallback.
    """

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)

        # Skip the base Strategy class itself
        if name == "Strategy":
            return cls

        # If 'name' is not defined or is still "Unnamed Strategy", use class name
        if not hasattr(cls, 'name') or cls.name == "Unnamed Strategy":
            # Use class name as fallback, with spaces before capital letters
            import re
            cls.name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)

        return cls


class Strategy(ABC, metaclass=StrategyMeta):
    """
    Abstract base class for all investment strategies.

    A strategy defines how the portfolio should be allocated at each
    rebalancing point.

    Attributes:
        name: Human-readable name of the strategy (e.g., "Dual Momentum").
              If not explicitly set, defaults to the class name with spaces.
        params: Configurable parameters for the strategy
        assets: List of ticker symbols required by this strategy
        rebalance_frequency: How often to rebalance ("monthly", "quarterly", "yearly")

    Example:
        class MyStrategy(Strategy):
            name = "My Custom Strategy"  # Optional: defaults to "My Strategy"

            def __init__(self):
                self.params = {"lookback": 12}
                self.assets = ["SPY", "BND"]

            def signal(self, date, data):
                return Allocation({"SPY": 0.6, "BND": 0.4})
    """

    name: str = "Unnamed Strategy"
    params: Dict[str, Any] = {}
    assets: List[str] = []
    rebalance_frequency: str = "monthly"

    # External features provider slot.
    # The Backtester populates this for the duration of a run when
    # BacktestConfig.external_features_loader is set, then restores the
    # previous value. Strategies that want to consume external signals
    # (analyst, news, ML forecast) can read self._external_features_provider
    # inside signal(); if it is None, the provider is not active and the
    # strategy must behave exactly as without the feature. The type is
    # Optional[ExternalFeaturesLoader]; typed as Any here to keep this
    # module import-light.
    _external_features_provider: Optional[Any] = None

    @property
    def display_name(self) -> str:
        """
        Get the display name for reports.

        Returns the 'name' attribute if set, otherwise the class name.
        """
        if hasattr(self, 'name') and self.name and self.name != "Unnamed Strategy":
            return self.name
        return self.__class__.__name__

    @abstractmethod
    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        """
        Calculate target allocation for a given date.

        This method is called at each rebalancing point to determine
        the target portfolio allocation.

        Args:
            date: Current date (typically month-end)
            data: Historical price data up to and including `date`.
                  DataFrame with DatetimeIndex and ticker symbols as columns.
                  Values are adjusted close prices.

        Returns:
            Allocation object with target weights for each asset.
            Weights must sum to <= 1.0.
        """
        ...

    @classmethod
    def get_param_grid(cls) -> Optional[Dict[str, List[Any]]]:
        """
        Return parameter grid for optimization.

        Override this method in strategy subclasses to define which
        parameters should be optimized and what values to try.

        This enables auto-discovery of optimization parameters instead
        of hardcoding them in batch_optimize.py.

        Returns:
            Dictionary mapping parameter names to lists of values to try,
            or None if no optimization grid is defined.

        Example:
            @classmethod
            def get_param_grid(cls):
                return {
                    "lookback_months": [6, 9, 12, 15, 18],
                    "use_sma_filter": [True, False],
                }
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.display_name}', params={self.params})"


def validate_strategy_name(strategy: Strategy) -> bool:
    """
    Validate that a strategy has a proper name.

    Args:
        strategy: Strategy instance to validate

    Returns:
        True if the strategy has a valid name

    Raises:
        ValueError: If the strategy name is empty or "Unnamed Strategy"
    """
    name = getattr(strategy, 'name', None) or strategy.__class__.__name__
    if not name or name == "Unnamed Strategy":
        raise ValueError(
            f"Strategy {strategy.__class__.__name__} must have a 'name' attribute. "
            f"Add 'name = \"Your Strategy Name\"' to the class definition."
        )
    return True
