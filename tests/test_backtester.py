"""Tests for the backtester module."""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from backtest.strategy import Strategy, Allocation
from backtest.data import PriceData
from backtest.backtester import Backtester, BacktestConfig, Portfolio, Trade


class SimpleStrategy(Strategy):
    """Simple strategy for testing - 100% in SPY."""
    name = "Simple"
    assets = ["SPY"]

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        return Allocation({"SPY": 1.0})


class SplitStrategy(Strategy):
    """50/50 strategy for testing."""
    name = "Split"
    assets = ["SPY", "BND"]

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        return Allocation({"SPY": 0.5, "BND": 0.5})


class TestPortfolio:
    """Tests for the Portfolio class."""

    def test_initial_portfolio(self):
        """Test initial portfolio state."""
        portfolio = Portfolio(cash=10000)
        assert portfolio.cash == 10000
        assert portfolio.positions == {}

    def test_total_value_cash_only(self):
        """Test total value with only cash."""
        portfolio = Portfolio(cash=10000)
        prices = pd.Series({"SPY": 100})
        assert portfolio.total_value(prices) == 10000

    def test_total_value_with_positions(self):
        """Test total value with positions."""
        portfolio = Portfolio(cash=1000, positions={"SPY": 10})
        prices = pd.Series({"SPY": 100})
        assert portfolio.total_value(prices) == 2000  # 1000 cash + 10*100

    def test_weights(self):
        """Test weight calculation."""
        portfolio = Portfolio(cash=1000, positions={"SPY": 10})
        prices = pd.Series({"SPY": 100})
        weights = portfolio.weights(prices)
        assert weights["SPY"] == pytest.approx(0.5)

    def test_rebalance_to_buy(self):
        """Test rebalancing to buy."""
        portfolio = Portfolio(cash=10000)
        prices = pd.Series({"SPY": 100})
        prices.name = pd.Timestamp("2020-01-01")

        target = Allocation({"SPY": 1.0})
        trades = portfolio.rebalance_to(target, prices)

        assert len(trades) == 1
        assert trades[0].action == "BUY"
        assert trades[0].ticker == "SPY"
        assert portfolio.positions.get("SPY", 0) > 0
        assert portfolio.cash < 10000

    def test_rebalance_to_sell(self):
        """Test rebalancing to sell."""
        portfolio = Portfolio(cash=0, positions={"SPY": 100})
        prices = pd.Series({"SPY": 100})
        prices.name = pd.Timestamp("2020-01-01")

        target = Allocation({"SPY": 0.5})  # Sell half
        trades = portfolio.rebalance_to(target, prices)

        assert len(trades) == 1
        assert trades[0].action == "SELL"
        assert trades[0].ticker == "SPY"
        assert portfolio.positions.get("SPY", 0) < 100


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()
        assert config.initial_capital == 10000
        assert config.currency == "EUR"
        assert config.costs_pct == 0.001
        assert config.benchmark == "S&P 500"

    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_capital=50000,
            costs_pct=0.002,
            benchmark=None
        )
        assert config.initial_capital == 50000
        assert config.costs_pct == 0.002
        assert config.benchmark is None


class TestBacktester:
    """Tests for the Backtester class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        spy_prices = [100 + i * 2 for i in range(24)]
        bnd_prices = [50 + i * 0.5 for i in range(24)]

        prices = pd.DataFrame({
            "SPY": spy_prices,
            "BND": bnd_prices
        }, index=dates)

        return PriceData(
            prices=prices,
            currency={"SPY": "USD", "BND": "USD"},
            fx_rates=None
        )

    def test_backtester_creation(self, sample_data):
        """Test creating a backtester."""
        strategy = SimpleStrategy()
        backtester = Backtester(strategy, sample_data)

        assert backtester.strategy == strategy
        assert backtester.data == sample_data

    def test_backtester_run(self, sample_data):
        """Test running a backtest."""
        strategy = SimpleStrategy()
        config = BacktestConfig(benchmark=None)
        backtester = Backtester(strategy, sample_data, config)

        result = backtester.run()

        assert result.strategy == strategy
        assert len(result.equity_curve) > 0
        assert len(result.trades) > 0
        assert result.metrics is not None

    def test_backtester_equity_grows(self, sample_data):
        """Test that equity grows when prices increase."""
        strategy = SimpleStrategy()
        config = BacktestConfig(benchmark=None, costs_pct=0)
        backtester = Backtester(strategy, sample_data, config)

        result = backtester.run()

        # Since SPY prices increase, portfolio should grow
        assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]

    def test_backtester_with_costs(self, sample_data):
        """Test that costs reduce returns."""
        strategy = SimpleStrategy()

        # Without costs
        config_no_costs = BacktestConfig(benchmark=None, costs_pct=0, slippage_pct=0)
        result_no_costs = Backtester(strategy, sample_data, config_no_costs).run()

        # With costs
        config_with_costs = BacktestConfig(benchmark=None, costs_pct=0.01, slippage_pct=0.01)
        result_with_costs = Backtester(strategy, sample_data, config_with_costs).run()

        # Final value should be lower with costs
        assert result_with_costs.equity_curve.iloc[-1] <= result_no_costs.equity_curve.iloc[-1]

    def test_result_summary(self, sample_data):
        """Test result summary generation."""
        strategy = SimpleStrategy()
        config = BacktestConfig(benchmark=None)
        result = Backtester(strategy, sample_data, config).run()

        summary = result.summary()
        assert "SIMPLE" in summary
        assert "CAGR" in summary
        assert "Volatility" in summary

    def test_backtester_supports_gbp_assets_via_fx_dataframe(self):
        """GBP assets should convert to EUR without raising unsupported-currency errors."""

        class GbpStrategy(Strategy):
            name = "GBP Simple"
            assets = ["HSBA.L"]

            def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
                return Allocation({"HSBA.L": 1.0})

        dates = pd.date_range("2020-01-31", periods=12, freq="ME")
        prices = pd.DataFrame({"HSBA.L": [100 + i * 2 for i in range(12)]}, index=dates)
        fx_rates = pd.DataFrame({"GBP": [0.85 + i * 0.001 for i in range(12)]}, index=dates)
        data = PriceData(
            prices=prices,
            currency={"HSBA.L": "GBP"},
            fx_rates=fx_rates,
        )

        result = Backtester(GbpStrategy(), data, BacktestConfig(benchmark=None, costs_pct=0, slippage_pct=0)).run()

        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[-1] > 0
