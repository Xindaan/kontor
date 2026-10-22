"""
Tests for the validation module.

Tests cover:
- Pre-run validation (config, data, strategy)
- Post-run validation (equity curve, trades, portfolio)
- Invariant validation (tax, metrics)
- ValidationResult aggregation and reporting
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime

from backtest.validation import (
    ValidationLevel,
    ValidationIssue,
    ValidationResult,
    validate_config,
    validate_price_data,
    validate_equity_curve,
    validate_trades,
    validate_tax_invariants,
    validate_metrics_consistency,
    validate_strategy_assets,
    validate_before_run,
    validate_backtest_result,
)
from backtest.backtester import BacktestConfig, Trade, TaxSummary
from backtest.data import PriceData
from backtest.strategy import Strategy, Allocation
from backtest.metrics import Metrics


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def valid_config():
    """A valid backtest configuration."""
    return BacktestConfig(
        initial_capital=10000,
        costs_pct=0.001,
        slippage_pct=0.0005,
        tax_enabled=True,
        tax_rate=0.26375,
        tax_exemption_amount=1000,
    )


@pytest.fixture
def valid_prices():
    """Valid price data with no issues."""
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="B")
    np.random.seed(42)
    prices = pd.DataFrame({
        "ASSET_A": 100 * np.cumprod(1 + np.random.randn(len(dates)) * 0.01 + 0.0002),
        "ASSET_B": 50 * np.cumprod(1 + np.random.randn(len(dates)) * 0.008 + 0.0001),
    }, index=dates)
    return prices


@pytest.fixture
def valid_equity_curve():
    """A valid equity curve with no issues."""
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="ME")
    return pd.Series(
        10000 * np.cumprod(1 + np.random.randn(len(dates)) * 0.02 + 0.005),
        index=dates
    )


class TestStrategy(Strategy):
    """Test strategy for validation tests."""
    name = "TestStrategy"
    assets = ["ASSET_A", "ASSET_B"]

    def signal(self, date, data):
        return Allocation({"ASSET_A": 0.6, "ASSET_B": 0.4})


# =============================================================================
# ValidationResult Tests
# =============================================================================

class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_empty_result_is_valid(self):
        """Empty result should be valid."""
        result = ValidationResult()
        assert result.is_valid
        assert not result.has_errors
        assert not result.has_warnings

    def test_info_does_not_affect_validity(self):
        """INFO level should not affect validity."""
        result = ValidationResult()
        result.add_info("test", "This is info")
        assert result.is_valid
        assert not result.has_errors
        assert not result.has_warnings

    def test_warning_flags_but_still_valid(self):
        """WARNING should flag but not invalidate."""
        result = ValidationResult()
        result.add_warning("test", "This is a warning")
        assert result.is_valid
        assert not result.has_errors
        assert result.has_warnings

    def test_error_invalidates(self):
        """ERROR should invalidate result."""
        result = ValidationResult()
        result.add_error("test", "This is an error")
        assert not result.is_valid
        assert result.has_errors

    def test_fatal_invalidates(self):
        """FATAL should invalidate result."""
        result = ValidationResult()
        result.add_fatal("test", "This is fatal")
        assert not result.is_valid
        assert result.has_errors

    def test_summary_counts_correctly(self):
        """Summary should count issues correctly."""
        result = ValidationResult()
        result.add_info("test", "Info 1")
        result.add_info("test", "Info 2")
        result.add_warning("test", "Warning")
        result.add_error("test", "Error")

        summary = result.summary()
        assert "1 error" in summary
        assert "1 warning" in summary
        assert "2 info" in summary
        assert "FAILED" in summary

    def test_raise_on_errors(self):
        """raise_on_errors should raise ValueError for errors."""
        result = ValidationResult()
        result.add_error("test", "Error message")

        with pytest.raises(ValueError) as excinfo:
            result.raise_on_errors()

        assert "Error message" in str(excinfo.value)

    def test_raise_on_errors_does_nothing_when_valid(self):
        """raise_on_errors should not raise when valid."""
        result = ValidationResult()
        result.add_warning("test", "Just a warning")
        result.raise_on_errors()  # Should not raise


# =============================================================================
# Config Validation Tests
# =============================================================================

class TestConfigValidation:
    """Tests for config validation."""

    def test_valid_config_passes(self, valid_config):
        """Valid config should pass validation."""
        result = validate_config(valid_config)
        assert result.is_valid

    def test_negative_capital_is_fatal(self):
        """Negative initial capital should be fatal."""
        config = BacktestConfig(initial_capital=-1000)
        result = validate_config(config)
        assert not result.is_valid
        assert any(i.level == ValidationLevel.FATAL for i in result.issues)

    def test_zero_capital_is_fatal(self):
        """Zero initial capital should be fatal."""
        config = BacktestConfig(initial_capital=0)
        result = validate_config(config)
        assert not result.is_valid

    def test_negative_costs_is_error(self):
        """Negative costs should be an error."""
        config = BacktestConfig(costs_pct=-0.001)
        result = validate_config(config)
        assert result.has_errors

    def test_high_costs_is_warning(self):
        """High costs (>5%) should trigger warning."""
        config = BacktestConfig(costs_pct=0.10)  # 10%
        result = validate_config(config)
        assert result.has_warnings
        assert any("high" in i.message.lower() for i in result.issues)

    def test_invalid_tax_rate_is_error(self):
        """Tax rate outside 0-1 should be error."""
        config = BacktestConfig(tax_enabled=True, tax_rate=1.5)
        result = validate_config(config)
        assert result.has_errors

    def test_negative_execution_lag_is_error(self):
        """Negative execution lag should be rejected."""
        config = BacktestConfig(execution_lag_days=-1)
        result = validate_config(config)
        assert result.has_errors

    def test_invalid_liquidity_missing_mode_is_error(self):
        """Unknown liquidity missing-volume mode should be rejected."""
        config = BacktestConfig(liquidity_on_missing_volume="invalid")  # type: ignore[arg-type]
        result = validate_config(config)
        assert result.has_errors


# =============================================================================
# Price Data Validation Tests
# =============================================================================

class TestPriceDataValidation:
    """Tests for price data validation."""

    def test_valid_prices_pass(self, valid_prices):
        """Valid price data should pass validation."""
        result = validate_price_data(valid_prices)
        assert result.is_valid

    def test_negative_prices_are_error(self):
        """Negative prices should be an error."""
        dates = pd.date_range("2020-01-01", "2020-01-10", freq="B")
        prices = pd.DataFrame({
            "ASSET_A": [100, 102, -50, 105, 108, 110, 112],
        }, index=dates[:7])

        result = validate_price_data(prices)
        assert result.has_errors
        assert any("negative" in i.message.lower() for i in result.issues)

    def test_zero_prices_are_warning(self):
        """Zero prices should be a warning."""
        dates = pd.date_range("2020-01-01", "2020-01-10", freq="B")
        prices = pd.DataFrame({
            "ASSET_A": [100, 102, 0, 105, 108, 110, 112],
        }, index=dates[:7])

        result = validate_price_data(prices)
        assert result.has_warnings
        assert any("zero" in i.message.lower() for i in result.issues)

    def test_extreme_returns_are_warning(self):
        """Extreme single-day returns should be warning."""
        dates = pd.date_range("2020-01-01", "2020-01-10", freq="B")
        prices = pd.DataFrame({
            "ASSET_A": [100, 102, 200, 205, 208, 210, 212],  # 96% gain
        }, index=dates[:7])

        result = validate_price_data(prices)
        assert result.has_warnings
        assert any("extreme" in i.message.lower() for i in result.issues)

    def test_empty_column_is_error(self):
        """Column with no valid data should be error."""
        dates = pd.date_range("2020-01-01", "2020-01-10", freq="B")
        prices = pd.DataFrame({
            "ASSET_A": [100, 102, 104, 105, 108, 110, 112],
            "ASSET_B": [np.nan] * 7,
        }, index=dates[:7])

        result = validate_price_data(prices)
        assert result.has_errors

    def test_few_data_points_is_warning(self):
        """Very few data points should be warning."""
        dates = pd.date_range("2020-01-01", "2020-01-10", freq="B")
        prices = pd.DataFrame({
            "ASSET_A": [100, 102, 104],
        }, index=dates[:3])

        result = validate_price_data(prices)
        assert result.has_warnings


# =============================================================================
# Equity Curve Validation Tests
# =============================================================================

class TestEquityCurveValidation:
    """Tests for equity curve validation."""

    def test_valid_curve_passes(self, valid_equity_curve):
        """Valid equity curve should pass validation."""
        result = validate_equity_curve(valid_equity_curve)
        assert result.is_valid

    def test_nan_values_are_error(self):
        """NaN values in equity curve should be error."""
        dates = pd.date_range("2020-01-01", "2020-06-01", freq="ME")
        curve = pd.Series([10000, 10500, np.nan, 11000, 11200], index=dates[:5])

        result = validate_equity_curve(curve)
        assert result.has_errors
        assert any("nan" in i.message.lower() for i in result.issues)

    def test_negative_values_are_error(self):
        """Negative equity values should be error."""
        dates = pd.date_range("2020-01-01", "2020-06-01", freq="ME")
        curve = pd.Series([10000, 10500, -500, 11000, 11200], index=dates[:5])

        result = validate_equity_curve(curve)
        assert result.has_errors
        assert any("negative" in i.message.lower() for i in result.issues)

    def test_empty_curve_is_error(self):
        """Empty equity curve should be error."""
        result = validate_equity_curve(pd.Series(dtype=float))
        assert result.has_errors

    def test_extreme_gains_are_warning(self):
        """Extreme gains (>100% single period) should be warning."""
        dates = pd.date_range("2020-01-01", "2020-06-01", freq="ME")
        curve = pd.Series([10000, 10500, 25000, 26000, 27000], index=dates[:5])

        result = validate_equity_curve(curve)
        assert result.has_warnings


# =============================================================================
# Trade Validation Tests
# =============================================================================

class TestTradeValidation:
    """Tests for trade validation."""

    def test_valid_trades_pass(self):
        """Valid trades should pass validation."""
        trades = [
            Trade(
                date=datetime(2020, 1, 15),
                ticker="ASSET_A",
                action="BUY",
                shares=10.0,
                price=100.0,
                value=1000.0,
                costs=1.0,
                slippage=0.5
            ),
            Trade(
                date=datetime(2020, 6, 15),
                ticker="ASSET_A",
                action="SELL",
                shares=10.0,
                price=110.0,
                value=1100.0,
                costs=1.1,
                slippage=0.55
            ),
        ]

        result = validate_trades(trades)
        assert result.is_valid

    def test_invalid_action_is_error(self):
        """Invalid trade action should be error."""
        trades = [
            Trade(
                date=datetime(2020, 1, 15),
                ticker="ASSET_A",
                action="HOLD",  # Invalid
                shares=10.0,
                price=100.0,
                value=1000.0,
                costs=1.0,
                slippage=0.5
            ),
        ]

        result = validate_trades(trades)
        assert result.has_errors

    def test_nan_shares_is_error(self):
        """NaN shares should be error."""
        trades = [
            Trade(
                date=datetime(2020, 1, 15),
                ticker="ASSET_A",
                action="BUY",
                shares=float('nan'),
                price=100.0,
                value=1000.0,
                costs=1.0,
                slippage=0.5
            ),
        ]

        result = validate_trades(trades)
        assert result.has_errors

    def test_negative_costs_is_error(self):
        """Negative costs should be error."""
        trades = [
            Trade(
                date=datetime(2020, 1, 15),
                ticker="ASSET_A",
                action="BUY",
                shares=10.0,
                price=100.0,
                value=1000.0,
                costs=-1.0,  # Negative
                slippage=0.5
            ),
        ]

        result = validate_trades(trades)
        assert result.has_errors


# =============================================================================
# Tax Invariant Tests
# =============================================================================

class TestTaxInvariantValidation:
    """Tests for tax invariant validation."""

    def test_valid_tax_summary_passes(self):
        """Valid tax summary should pass all invariants."""
        tax_summary = TaxSummary(
            tax_enabled=True,
            final_value_gross=12000.0,
            final_value_net_realized=11500.0,
            final_value_net_liquidation=11000.0,
            total_tax_paid=500.0,
            tax_paid_liquidation=500.0,
            cagr_gross=0.10,
            cagr_net_realized=0.09,
            cagr_net_liquidation=0.085,
            effective_tax_rate=0.20,
            tax_rate=0.26375,
            metric_basis="net_liquidation",
        )

        result = validate_tax_invariants(tax_summary)
        assert result.is_valid

    def test_gross_less_than_net_is_error(self):
        """Gross < Net Realized should be error."""
        tax_summary = TaxSummary(
            tax_enabled=True,
            final_value_gross=11000.0,  # Less than net_realized
            final_value_net_realized=11500.0,
            final_value_net_liquidation=11000.0,
            total_tax_paid=500.0,
            metric_basis="net_realized",
        )

        result = validate_tax_invariants(tax_summary)
        assert result.has_errors

    def test_negative_tax_is_error(self):
        """Negative total tax paid should be error."""
        tax_summary = TaxSummary(
            tax_enabled=True,
            final_value_gross=12000.0,
            final_value_net_realized=11500.0,
            final_value_net_liquidation=11000.0,
            total_tax_paid=-100.0,  # Negative
            metric_basis="net_realized",
        )

        result = validate_tax_invariants(tax_summary)
        assert result.has_errors

    def test_tax_disabled_with_nonzero_tax_is_error(self):
        """Tax disabled but tax paid should be error."""
        tax_summary = TaxSummary(
            tax_enabled=False,
            total_tax_paid=100.0,  # Should be 0
            cagr_gross=0.10,
            cagr_net_realized=0.09,  # Should equal gross
        )

        result = validate_tax_invariants(tax_summary)
        assert result.has_errors


# =============================================================================
# Strategy Assets Validation Tests
# =============================================================================

class TestStrategyAssetsValidation:
    """Tests for strategy assets validation."""

    def test_all_assets_available_passes(self, valid_prices):
        """When all assets are available, validation should pass."""
        strategy = TestStrategy()
        data = PriceData(prices=valid_prices, currency="EUR")

        result = validate_strategy_assets(strategy, data)
        assert result.is_valid

    def test_missing_asset_is_error(self, valid_prices):
        """Missing required asset should be error."""
        class MissingAssetStrategy(Strategy):
            name = "MissingAsset"
            assets = ["ASSET_A", "ASSET_C"]  # ASSET_C doesn't exist

            def signal(self, date, data):
                return Allocation({"ASSET_A": 1.0})

        strategy = MissingAssetStrategy()
        data = PriceData(prices=valid_prices, currency="EUR")

        result = validate_strategy_assets(strategy, data)
        assert result.has_errors
        # Check for "not found" or "missing" in message
        assert any("not found" in i.message.lower() or "missing" in i.message.lower()
                   for i in result.issues)


# =============================================================================
# Integration Tests
# =============================================================================

class TestValidateBeforeRun:
    """Integration tests for pre-run validation."""

    def test_valid_setup_passes(self, valid_config, valid_prices):
        """Valid setup should pass all pre-run checks."""
        strategy = TestStrategy()
        data = PriceData(prices=valid_prices, currency="EUR")

        result = validate_before_run(strategy, data, valid_config)
        assert result.is_valid

    def test_multiple_issues_accumulated(self):
        """Multiple issues should all be reported."""
        # Invalid config + data issues
        config = BacktestConfig(
            initial_capital=-1000,  # Fatal
            costs_pct=-0.01,  # Error
        )

        dates = pd.date_range("2020-01-01", "2020-01-10", freq="B")
        prices = pd.DataFrame({
            "ASSET_A": [-100, 102, 104, 105, 108],  # Negative price
        }, index=dates[:5])
        data = PriceData(prices=prices, currency="EUR")

        class SimpleStrategy(Strategy):
            name = "Simple"
            assets = ["ASSET_A"]
            def signal(self, date, data):
                return Allocation({"ASSET_A": 1.0})

        result = validate_before_run(SimpleStrategy(), data, config)

        # Should have multiple errors
        assert len(result.issues) >= 3
        assert not result.is_valid


# =============================================================================
# Metrics Consistency Tests
# =============================================================================

class TestMetricsConsistency:
    """Tests for metrics consistency validation."""

    def test_consistent_metrics_pass(self, valid_equity_curve):
        """Consistent metrics should pass validation."""
        from backtest.metrics import MetricsCalculator

        metrics = Metrics(
            total_return=MetricsCalculator.total_return(valid_equity_curve),
            max_drawdown=MetricsCalculator.max_drawdown(valid_equity_curve),
            win_rate_monthly=0.6,
            sharpe_ratio=1.0,
            sortino_ratio=1.5,
        )

        result = validate_metrics_consistency(valid_equity_curve, metrics)
        assert result.is_valid

    def test_positive_max_drawdown_is_error(self, valid_equity_curve):
        """Positive max drawdown should be error."""
        metrics = Metrics(
            total_return=0.5,
            max_drawdown=0.1,  # Should be negative
            win_rate_monthly=0.6,
        )

        result = validate_metrics_consistency(valid_equity_curve, metrics)
        assert result.has_errors

    def test_win_rate_out_of_bounds_is_error(self, valid_equity_curve):
        """Win rate outside 0-1 should be error."""
        metrics = Metrics(
            total_return=0.5,
            max_drawdown=-0.1,
            win_rate_monthly=1.5,  # Should be <= 1
        )

        result = validate_metrics_consistency(valid_equity_curve, metrics)
        assert result.has_errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
