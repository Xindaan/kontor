"""Tests for the strategy module."""

import pytest

from backtest.strategy import Strategy, Allocation


class TestAllocation:
    """Tests for the Allocation class."""

    def test_valid_allocation(self):
        """Test creating a valid allocation."""
        alloc = Allocation({"SPY": 0.6, "BND": 0.4})
        assert alloc.weights == {"SPY": 0.6, "BND": 0.4}
        assert alloc.cash == pytest.approx(0.0)

    def test_allocation_with_cash(self):
        """Test allocation with some cash."""
        alloc = Allocation({"SPY": 0.5})
        assert alloc.cash == pytest.approx(0.5)

    def test_allocation_empty(self):
        """Test empty allocation (100% cash)."""
        alloc = Allocation({})
        assert alloc.cash == pytest.approx(1.0)

    def test_allocation_over_one_raises(self):
        """Test that allocations over 1.0 raise an error."""
        with pytest.raises(ValueError):
            Allocation({"SPY": 0.6, "BND": 0.5})

    def test_allocation_get(self):
        """Test getting weight for a ticker."""
        alloc = Allocation({"SPY": 0.6})
        assert alloc.get("SPY") == 0.6
        assert alloc.get("BND") == 0.0
        assert alloc.get("BND", 0.5) == 0.5

    def test_allocation_items(self):
        """Test iterating over allocation."""
        alloc = Allocation({"SPY": 0.6, "BND": 0.4})
        items = dict(alloc.items())
        assert items == {"SPY": 0.6, "BND": 0.4}


class TestStrategy:
    """Tests for the Strategy base class."""

    def test_strategy_is_abstract(self):
        """Test that Strategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Strategy()

    def test_strategy_subclass(self):
        """Test creating a Strategy subclass."""
        from datetime import date
        import pandas as pd

        class TestStrategy(Strategy):
            name = "Test"
            assets = ["SPY"]

            def signal(self, date: date, data: pd.DataFrame) -> Allocation:
                return Allocation({"SPY": 1.0})

        strategy = TestStrategy()
        assert strategy.name == "Test"
        assert strategy.assets == ["SPY"]

        result = strategy.signal(date.today(), pd.DataFrame())
        assert result.weights == {"SPY": 1.0}
