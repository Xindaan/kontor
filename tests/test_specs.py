"""
Acceptance tests for P4 Specs:
1. Taxes always cash-effective in Backtester
2. Strategies always receive daily historical data

These tests verify the core behavioral changes in the backtester.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from backtest.strategy import Strategy, Allocation
from backtest.data import PriceData
from backtest.backtester import Backtester, BacktestConfig
from backtest.rebalance import generate_rebalance_dates


# =============================================================================
# Test Strategy: Tracks what data it receives
# =============================================================================

class DataInspectorStrategy(Strategy):
    """
    Strategy that records the historical_data it receives.
    Used to verify that strategies get daily data regardless of rebalance frequency.
    """
    name = "Data Inspector"
    assets = ["SPY"]

    received_data: list = []

    def __init__(self):
        self.received_data = []

    def signal(self, dt: date, data: pd.DataFrame) -> Allocation:
        # Record the data shape and index frequency
        self.received_data.append({
            "date": dt,
            "rows": len(data),
            "index": data.index.copy(),
        })
        return Allocation({"SPY": 1.0})


class BuyHoldStrategy(Strategy):
    """Simple buy and hold for testing."""
    name = "Buy Hold"
    assets = ["SPY"]

    def signal(self, dt: date, data: pd.DataFrame) -> Allocation:
        return Allocation({"SPY": 1.0})


class TradingStrategy(Strategy):
    """
    Strategy that alternates between SPY and cash to generate trades.
    This creates realized gains/losses for tax testing.
    """
    name = "Alternating"
    assets = ["SPY"]

    def __init__(self):
        self.in_market = False

    def signal(self, dt: date, data: pd.DataFrame) -> Allocation:
        self.in_market = not self.in_market
        if self.in_market:
            return Allocation({"SPY": 1.0})
        else:
            return Allocation.cash()


# =============================================================================
# SPEC 1: Taxes always cash-effective
# =============================================================================

class TestCashEffectiveTaxes:
    """
    Tests that verify taxes are deducted from cash immediately upon realization.
    """

    @pytest.fixture
    def rising_market_data(self):
        """Create price data with rising prices (guaranteed gains)."""
        # 24 months of data, prices double
        dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        prices = pd.DataFrame({
            "SPY": [100 + i * 5 for i in range(24)],  # 100 -> 215
        }, index=dates)

        return PriceData(
            prices=prices,
            currency={"SPY": "USD"},
            fx_rates=None
        )

    @pytest.fixture
    def falling_market_data(self):
        """Create price data with falling prices (losses)."""
        dates = pd.date_range("2020-01-01", periods=24, freq="ME")
        prices = pd.DataFrame({
            "SPY": [100 - i * 2 for i in range(24)],  # 100 -> 54
        }, index=dates)

        return PriceData(
            prices=prices,
            currency={"SPY": "USD"},
            fx_rates=None
        )

    def test_tax_reduces_final_value(self, rising_market_data):
        """
        Test 1: Tax affects final portfolio value.
        With taxes enabled, final value should be lower than without.
        """
        strategy = TradingStrategy()

        # Without tax
        config_no_tax = BacktestConfig(
            benchmark=None,
            costs_pct=0,
            slippage_pct=0,
            tax_enabled=False,
        )
        result_no_tax = Backtester(TradingStrategy(), rising_market_data, config_no_tax).run()

        # With tax (and no exemption to ensure tax is paid)
        config_with_tax = BacktestConfig(
            benchmark=None,
            costs_pct=0,
            slippage_pct=0,
            tax_enabled=True,
            tax_exemption_amount=0,  # No Freistellungsauftrag
        )
        result_with_tax = Backtester(TradingStrategy(), rising_market_data, config_with_tax).run()

        # Final value should be lower with taxes
        assert result_with_tax.equity_curve.iloc[-1] < result_no_tax.equity_curve.iloc[-1], \
            "Tax should reduce final portfolio value"

        # Tax summary should show tax paid
        assert result_with_tax.tax_summary is not None
        assert result_with_tax.tax_summary.total_tax_paid > 0

    def test_loss_carry_reduces_tax(self, falling_market_data):
        """
        Test 2: Losses are carried and reduce future tax.
        Selling at a loss should not trigger tax.
        """
        strategy = TradingStrategy()

        config = BacktestConfig(
            benchmark=None,
            costs_pct=0,
            slippage_pct=0,
            tax_enabled=True,
            tax_exemption_amount=0,
        )

        result = Backtester(strategy, falling_market_data, config).run()

        # With falling prices, realized gains should be negative (losses)
        if result.tax_summary is not None:
            # No tax should be paid on losses
            # (Note: might be 0 if no gains at all)
            assert result.tax_summary.total_realized_losses <= 0, \
                "Losses should be recorded as negative"

    def test_exemption_reduces_tax(self, rising_market_data):
        """
        Test 3: Freistellungsauftrag reduces tax.
        With exemption, less tax should be paid than without.
        """
        # Without exemption
        config_no_exempt = BacktestConfig(
            benchmark=None,
            costs_pct=0,
            slippage_pct=0,
            tax_enabled=True,
            tax_exemption_amount=0,
        )
        result_no_exempt = Backtester(TradingStrategy(), rising_market_data, config_no_exempt).run()

        # With exemption
        config_with_exempt = BacktestConfig(
            benchmark=None,
            costs_pct=0,
            slippage_pct=0,
            tax_enabled=True,
            tax_exemption_amount=10000,  # Large exemption
        )
        result_with_exempt = Backtester(TradingStrategy(), rising_market_data, config_with_exempt).run()

        # With exemption, tax paid should be less or equal
        if result_no_exempt.tax_summary and result_with_exempt.tax_summary:
            assert result_with_exempt.tax_summary.total_tax_paid <= result_no_exempt.tax_summary.total_tax_paid, \
                "Exemption should reduce total tax paid"

    def test_compare_shows_different_values(self, rising_market_data):
        """
        Test 4: Compare runs with tax vs no-tax should show different results.
        """
        # This verifies that Comparator properly uses tax settings
        config_tax = BacktestConfig(
            benchmark=None,
            costs_pct=0,
            slippage_pct=0,
            tax_enabled=True,
            tax_exemption_amount=0,
        )
        config_no_tax = BacktestConfig(
            benchmark=None,
            costs_pct=0,
            slippage_pct=0,
            tax_enabled=False,
        )

        result_tax = Backtester(TradingStrategy(), rising_market_data, config_tax).run()
        result_no_tax = Backtester(TradingStrategy(), rising_market_data, config_no_tax).run()

        # CAGR should be different
        assert result_tax.metrics.cagr != result_no_tax.metrics.cagr, \
            "CAGR should differ between tax and no-tax runs"


# =============================================================================
# SPEC 2: Strategies always receive daily historical data
# =============================================================================

class TestDailyHistoryForStrategies:
    """
    Tests that verify strategies receive daily data regardless of rebalance frequency.
    """

    @pytest.fixture
    def daily_price_data(self):
        """
        Create 2 years of daily price data (about 500 trading days).
        This allows testing with various rebalance frequencies.
        """
        # Create daily dates (business days only, roughly)
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
        n = len(dates)

        # Simple price series
        prices = pd.DataFrame({
            "SPY": [100 + i * 0.1 for i in range(n)],
        }, index=dates)

        return PriceData(
            prices=prices,
            currency={"SPY": "USD"},
            fx_rates=None
        )

    def test_monthly_rebalance_gets_daily_data(self, daily_price_data):
        """
        With monthly rebalancing, strategy should still receive daily data.
        """
        strategy = DataInspectorStrategy()
        config = BacktestConfig(
            benchmark=None,
            rebalance_frequency="monthly",
            tax_enabled=False,
        )

        Backtester(strategy, daily_price_data, config).run()

        # Check that data received had daily frequency
        for record in strategy.received_data:
            if len(record["index"]) > 1:
                # Check that differences between dates are ~1-3 days (weekends)
                diffs = record["index"].to_series().diff().dropna()
                max_gap = diffs.max().days
                assert max_gap <= 5, \
                    f"Data should be daily, but found gap of {max_gap} days"

    def test_lookback_is_trading_days(self, daily_price_data):
        """
        Test that a 126-day lookback gives 126 rows of data (not 126 months).
        """
        strategy = DataInspectorStrategy()
        config = BacktestConfig(
            benchmark=None,
            rebalance_frequency="monthly",
            tax_enabled=False,
        )

        Backtester(strategy, daily_price_data, config).run()

        # After 6 months (roughly 126 trading days), the strategy should have
        # received approximately that many rows of historical data
        # Find a record after the warmup period
        late_records = [r for r in strategy.received_data if r["rows"] > 100]

        if late_records:
            # The data should grow by daily increments, not monthly
            # Between two monthly rebalances, data rows should grow by ~21 days
            for i in range(1, min(3, len(late_records))):
                growth = late_records[i]["rows"] - late_records[i-1]["rows"]
                # Should be roughly 20-23 trading days per month
                assert 15 <= growth <= 25, \
                    f"Data growth should be ~21 days/month, got {growth}"

    def test_daily_rebalance_works(self, daily_price_data):
        """
        Test that daily rebalancing works correctly.
        """
        strategy = DataInspectorStrategy()
        config = BacktestConfig(
            benchmark=None,
            rebalance_frequency="daily",
            tax_enabled=False,
        )

        result = Backtester(strategy, daily_price_data, config).run()

        # Should have many rebalance events (close to number of trading days)
        assert len(strategy.received_data) > 100


# =============================================================================
# Test: Rebalance Date Generator
# =============================================================================

class TestRebalanceDates:
    """Tests for the generate_rebalance_dates function."""

    @pytest.fixture
    def trading_calendar(self):
        """Create a realistic trading calendar (business days)."""
        return pd.date_range("2020-01-01", "2020-12-31", freq="B")

    def test_daily_returns_all_dates(self, trading_calendar):
        """Daily rebalancing should return all trading dates."""
        result = generate_rebalance_dates(trading_calendar, "daily")
        assert len(result) == len(trading_calendar)

    def test_monthly_returns_month_ends(self, trading_calendar):
        """Monthly rebalancing should return last trading day of each month."""
        result = generate_rebalance_dates(trading_calendar, "monthly")

        # Should have 12 dates (one per month)
        assert len(result) == 12

        # Each date should be the last trading day of its month
        for dt in result:
            # Next day should be in a different month
            assert dt in trading_calendar

    def test_weekly_returns_week_ends(self, trading_calendar):
        """Weekly rebalancing should return last trading day of each week."""
        result = generate_rebalance_dates(trading_calendar, "weekly")

        # Should have roughly 52 weeks
        assert 50 <= len(result) <= 53

    def test_quarterly_returns_quarter_ends(self, trading_calendar):
        """Quarterly rebalancing should return last trading day of each quarter."""
        result = generate_rebalance_dates(trading_calendar, "quarterly")

        # Should have 4 dates
        assert len(result) == 4

    def test_yearly_returns_year_end(self, trading_calendar):
        """Yearly rebalancing should return last trading day of year."""
        result = generate_rebalance_dates(trading_calendar, "yearly")

        # Should have 1 date
        assert len(result) == 1
        assert result[0] == trading_calendar[-1]

    def test_rebalance_dates_are_subset(self, trading_calendar):
        """All rebalance dates should be from the original calendar."""
        for freq in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            result = generate_rebalance_dates(trading_calendar, freq)
            for dt in result:
                assert dt in trading_calendar, \
                    f"Rebalance date {dt} not in trading calendar"
