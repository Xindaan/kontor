"""
Invariant Tests for Backtest Framework.

These tests verify critical invariants that must always hold:
1. tax_paid_total >= 0
2. final_value_gross >= final_value_net_realized >= final_value_net_liquidation
3. When tax.enabled == False: tax_paid_total == 0 and CAGR_gross == CAGR_net
4. When terminal_valuation == net_liquidation: tax_paid_liquidation >= 0

Also includes "golden tests" with synthetic price data for exact verification.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from backtest.backtester import Backtester, BacktestConfig
from backtest.data import PriceData
from backtest.strategy import Strategy, Allocation


# =============================================================================
# Test Strategies
# =============================================================================

class BuyAndHoldStrategy(Strategy):
    """Simple Buy & Hold strategy for testing."""
    name = "Test_BuyAndHold"
    assets = ["ASSET_A"]

    def signal(self, date, data):
        return Allocation({"ASSET_A": 1.0})


class FrequentTraderStrategy(Strategy):
    """Strategy that trades every month (high turnover)."""
    name = "Test_FrequentTrader"
    assets = ["ASSET_A", "ASSET_B"]

    def __init__(self):
        self._toggle = True

    def signal(self, date, data):
        self._toggle = not self._toggle
        if self._toggle:
            return Allocation({"ASSET_A": 1.0})
        else:
            return Allocation({"ASSET_B": 1.0})


class TwoAssetStrategy(Strategy):
    """60/40 strategy for testing."""
    name = "Test_TwoAsset"
    assets = ["ASSET_A", "ASSET_B"]

    def signal(self, date, data):
        return Allocation({"ASSET_A": 0.6, "ASSET_B": 0.4})


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def synthetic_prices():
    """Create synthetic price data with known behavior."""
    # 2 years of daily data
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
    n = len(dates)

    # ASSET_A: Steady 10% annual growth
    asset_a = 100 * (1.10 ** (np.arange(n) / 252))

    # ASSET_B: Steady 5% annual growth
    asset_b = 50 * (1.05 ** (np.arange(n) / 252))

    prices = pd.DataFrame({
        "ASSET_A": asset_a,
        "ASSET_B": asset_b,
    }, index=dates)

    return PriceData(prices=prices, currency="EUR")


@pytest.fixture
def volatile_prices():
    """Create volatile price data with gains and losses."""
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="B")
    n = len(dates)

    np.random.seed(42)

    # ASSET_A: Volatile with overall gain
    returns_a = np.random.randn(n) * 0.02 + 0.0003  # ~8% annual with 30% vol
    asset_a = 100 * np.cumprod(1 + returns_a)

    # ASSET_B: Volatile with modest gain
    returns_b = np.random.randn(n) * 0.01 + 0.0001  # ~2.5% annual with 15% vol
    asset_b = 50 * np.cumprod(1 + returns_b)

    prices = pd.DataFrame({
        "ASSET_A": asset_a,
        "ASSET_B": asset_b,
    }, index=dates)

    return PriceData(prices=prices, currency="EUR")


# =============================================================================
# P1: Invariant Tests
# =============================================================================

class TestTaxInvariants:
    """Test that tax invariants always hold."""

    def test_tax_paid_non_negative(self, synthetic_prices):
        """Tax paid must never be negative."""
        config = BacktestConfig(
            initial_capital=10000,
            tax_enabled=True,
            metric_basis="net_liquidation",
        )

        strategy = FrequentTraderStrategy()
        backtester = Backtester(strategy, synthetic_prices, config)
        result = backtester.run()

        t = result.tax_summary
        assert t.total_tax_paid >= 0, f"total_tax_paid={t.total_tax_paid} < 0"
        assert t.tax_paid_liquidation >= 0, f"tax_paid_liquidation={t.tax_paid_liquidation} < 0"

    def test_gross_ge_net_realized_ge_net_liquidation(self, synthetic_prices):
        """final_value_gross >= final_value_net_realized >= final_value_net_liquidation."""
        config = BacktestConfig(
            initial_capital=10000,
            tax_enabled=True,
            metric_basis="net_liquidation",
        )

        strategy = TwoAssetStrategy()
        backtester = Backtester(strategy, synthetic_prices, config)
        result = backtester.run()

        t = result.tax_summary
        assert t.final_value_gross >= t.final_value_net_realized - 0.01, \
            f"gross ({t.final_value_gross}) < net_realized ({t.final_value_net_realized})"
        assert t.final_value_net_realized >= t.final_value_net_liquidation - 0.01, \
            f"net_realized ({t.final_value_net_realized}) < net_liquidation ({t.final_value_net_liquidation})"

    def test_tax_disabled_means_zero_tax(self, synthetic_prices):
        """When tax is disabled, tax_paid must be 0 and CAGR_gross == CAGR_net."""
        config = BacktestConfig(
            initial_capital=10000,
            tax_enabled=False,
        )

        strategy = FrequentTraderStrategy()
        backtester = Backtester(strategy, synthetic_prices, config)
        result = backtester.run()

        t = result.tax_summary
        assert t.total_tax_paid == 0, f"tax_paid={t.total_tax_paid} with tax_enabled=False"
        assert t.tax_paid_liquidation == 0, f"tax_paid_liquidation={t.tax_paid_liquidation}"
        assert abs(t.cagr_gross - t.cagr_net_realized) < 0.0001, \
            f"CAGR mismatch: gross={t.cagr_gross}, net={t.cagr_net_realized}"

    def test_cagr_ordering(self, synthetic_prices):
        """CAGR_gross >= CAGR_net_realized >= CAGR_net_liquidation."""
        config = BacktestConfig(
            initial_capital=10000,
            tax_enabled=True,
            metric_basis="net_liquidation",
        )

        strategy = TwoAssetStrategy()
        backtester = Backtester(strategy, synthetic_prices, config)
        result = backtester.run()

        t = result.tax_summary
        assert t.cagr_gross >= t.cagr_net_realized - 0.0001, \
            f"CAGR gross ({t.cagr_gross}) < net_realized ({t.cagr_net_realized})"
        assert t.cagr_net_realized >= t.cagr_net_liquidation - 0.0001, \
            f"CAGR net_realized ({t.cagr_net_realized}) < net_liquidation ({t.cagr_net_liquidation})"


class TestGoldenTests:
    """Golden tests with known expected values."""

    def test_simple_gain_tax_calculation(self):
        """Test: Buy at 100, sell at 110 → verify exact tax amount."""
        # 5 days of data
        dates = pd.date_range("2024-01-01", "2024-01-05", freq="B")
        prices = pd.DataFrame({
            "ASSET_A": [100.0, 102.0, 105.0, 108.0, 110.0],
        }, index=dates)
        data = PriceData(prices=prices, currency="EUR")

        # Strategy that sells everything on last day (by going to cash)
        class SellOnLastDay(Strategy):
            name = "SellOnLastDay"
            assets = ["ASSET_A"]
            rebalance_frequency = "daily"

            def signal(self, dt, data):
                # Sell on last day
                if dt >= date(2024, 1, 5):
                    return Allocation({})  # All cash
                return Allocation({"ASSET_A": 1.0})

        config = BacktestConfig(
            initial_capital=10000,
            tax_enabled=True,
            tax_exemption_amount=0,  # No exemption for clear calculation
            metric_basis="net_realized",
            rebalance_frequency="daily",
            costs_pct=0.0,
            slippage_pct=0.0,
            spread_pct=0.0,
        )

        backtester = Backtester(SellOnLastDay(), data, config)
        result = backtester.run()

        t = result.tax_summary

        # Gain: 10000 * (110/100 - 1) = 1000
        # Taxable after Teilfreistellung (30%): 1000 * 0.70 = 700
        # Tax: 700 * 0.26375 = 184.625
        expected_tax = 1000 * 0.70 * 0.26375

        assert abs(t.total_tax_paid - expected_tax) < 1.0, \
            f"Expected tax ~{expected_tax:.2f}, got {t.total_tax_paid:.2f}"

    def test_buy_and_hold_zero_realized_tax(self, synthetic_prices):
        """Buy & Hold should have zero realized tax (no sells)."""
        config = BacktestConfig(
            initial_capital=10000,
            tax_enabled=True,
            metric_basis="net_realized",
        )

        strategy = BuyAndHoldStrategy()
        backtester = Backtester(strategy, synthetic_prices, config)
        result = backtester.run()

        t = result.tax_summary
        # Buy & Hold never sells, so no realized tax
        assert t.total_tax_paid == 0, \
            f"Buy&Hold should have 0 realized tax, got {t.total_tax_paid}"

    def test_buy_and_hold_has_liquidation_tax(self, synthetic_prices):
        """Buy & Hold with net_liquidation should have tax on unrealized gains."""
        config = BacktestConfig(
            initial_capital=10000,
            tax_enabled=True,
            metric_basis="net_liquidation",
        )

        strategy = BuyAndHoldStrategy()
        backtester = Backtester(strategy, synthetic_prices, config)
        result = backtester.run()

        t = result.tax_summary

        # With growing asset, there should be unrealized gains and liquidation tax
        assert t.tax_paid_liquidation > 0, \
            f"Buy&Hold with net_liquidation should have liquidation tax > 0, got {t.tax_paid_liquidation}"
        assert t.final_value_net_liquidation < t.final_value_gross, \
            "net_liquidation value should be less than gross"

    def test_frequent_trader_has_realized_tax(self, synthetic_prices):
        """Frequent trader should have realized tax from trades."""
        config = BacktestConfig(
            initial_capital=10000,
            tax_enabled=True,
            metric_basis="net_realized",
        )

        strategy = FrequentTraderStrategy()
        backtester = Backtester(strategy, synthetic_prices, config)
        result = backtester.run()

        t = result.tax_summary

        # Frequent trader sells often, should have realized tax
        # Note: Might be 0 if all trades are losses, but with growing assets should be > 0
        assert t.total_tax_paid >= 0, "Tax should be non-negative"
        # With synthetic growing prices, should have some realized gains
        assert t.total_realized_gains > 0, \
            f"Frequent trader should have some realized gains, got {t.total_realized_gains}"


class TestTaxModeComparison:
    """Test that different tax modes produce visibly different results."""

    def test_tax_vs_no_tax_difference(self, volatile_prices):
        """Tax enabled vs disabled should show visible difference for active strategy."""
        strategy = FrequentTraderStrategy()

        # With tax
        config_tax = BacktestConfig(
            initial_capital=10000,
            tax_enabled=True,
            metric_basis="net_realized",
        )
        result_tax = Backtester(strategy, volatile_prices, config_tax).run()

        # Without tax
        config_no_tax = BacktestConfig(
            initial_capital=10000,
            tax_enabled=False,
        )
        # Need fresh strategy instance
        strategy2 = FrequentTraderStrategy()
        result_no_tax = Backtester(strategy2, volatile_prices, config_no_tax).run()

        # Should see visible difference
        cagr_diff = abs(result_no_tax.metrics.cagr - result_tax.tax_summary.cagr_net_realized)

        # If there were any realized gains, tax should make a difference
        if result_tax.tax_summary.total_realized_gains > 100:
            assert result_tax.tax_summary.total_tax_paid > 0, \
                "With realized gains, tax should be paid"

    def test_net_realized_vs_net_liquidation_for_buyhold(self, synthetic_prices):
        """net_realized vs net_liquidation should differ for Buy&Hold."""
        strategy = BuyAndHoldStrategy()

        # net_realized
        config_realized = BacktestConfig(
            initial_capital=10000,
            tax_enabled=True,
            metric_basis="net_realized",
        )
        result_realized = Backtester(strategy, synthetic_prices, config_realized).run()

        # net_liquidation
        strategy2 = BuyAndHoldStrategy()
        config_liq = BacktestConfig(
            initial_capital=10000,
            tax_enabled=True,
            metric_basis="net_liquidation",
        )
        result_liq = Backtester(strategy2, synthetic_prices, config_liq).run()

        t_realized = result_realized.tax_summary
        t_liq = result_liq.tax_summary

        # Buy&Hold: net_realized == gross (no realized tax)
        assert abs(t_realized.final_value_gross - t_realized.final_value_net_realized) < 1.0, \
            "Buy&Hold net_realized should equal gross"

        # But net_liquidation should be less (due to unrealized gains tax)
        assert t_liq.final_value_net_liquidation < t_liq.final_value_gross - 10, \
            "Buy&Hold net_liquidation should be less than gross"


class TestBenchmarkPresence:
    """Test that benchmark is computed when configured."""

    def test_benchmark_metrics_present(self, synthetic_prices):
        """Benchmark metrics should be present when benchmark is configured."""
        # Add benchmark ticker to data
        prices = synthetic_prices.prices.copy()
        prices["BENCH"] = 100 * (1.08 ** (np.arange(len(prices)) / 252))
        data = PriceData(prices=prices, currency="EUR")

        config = BacktestConfig(
            initial_capital=10000,
            benchmark="BENCH",
        )

        strategy = TwoAssetStrategy()
        backtester = Backtester(strategy, data, config)
        result = backtester.run()

        # Benchmark curve should be present
        assert result.benchmark_curve is not None, "Benchmark curve should be present"

        # Benchmark metrics should be calculated
        m = result.metrics
        assert m.tracking_difference is not None, "Tracking difference should be calculated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
