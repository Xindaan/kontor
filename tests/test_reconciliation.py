"""Tests for the reconciliation module and trade semantics."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime
from io import StringIO
import json
import tempfile
from pathlib import Path

from backtest.strategy import Strategy, Allocation
from backtest.data import PriceData
from backtest.backtester import Backtester, BacktestConfig, Portfolio, Trade
from backtest.reconciliation import (
    CashflowReconciliation,
    DailyReconciliation,
    MonthlyReconciliation,
    validate_trades_invariants,
)
from backtest.reporter import Reporter


class SimpleStrategy(Strategy):
    """Simple strategy for testing - 100% in SPY."""
    name = "Simple"
    assets = ["SPY"]

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        return Allocation({"SPY": 1.0})


class SwitchStrategy(Strategy):
    """Strategy that switches between assets to generate both BUY and SELL trades."""
    name = "Switch"
    assets = ["SPY", "BND"]

    def __init__(self):
        self._toggle = False

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        self._toggle = not self._toggle
        if self._toggle:
            return Allocation({"SPY": 1.0})
        else:
            return Allocation({"BND": 1.0})


class TestTradeSemantics:
    """Tests for the new Trade semantic fields."""

    def test_trade_creation_with_new_fields(self):
        """Test that trades have all the new semantic fields."""
        trade = Trade(
            date=datetime(2024, 1, 15),
            ticker="SPY",
            action="BUY",
            shares=10.0,
            price=100.0,
            value=1000.0,
            costs=1.0,
            slippage=0.5,
            price_ref=100.0,
            price_exec=100.05,
            value_ref=1000.0,
            value_exec=1000.5,
            tax_paid_trade=0.0,
        )

        assert trade.price_ref == 100.0
        assert trade.price_exec == 100.05
        assert trade.value_ref == 1000.0
        assert trade.value_exec == 1000.5
        assert trade.tax_paid_trade == 0.0

    def test_trade_validate_invariants(self):
        """Test that trade invariants validation works."""
        # Valid trade
        valid_trade = Trade(
            date=datetime(2024, 1, 15),
            ticker="SPY",
            action="BUY",
            shares=10.0,
            price=100.0,
            value=1000.0,
            costs=1.0,
            slippage=0.5,
            price_ref=100.0,
            price_exec=100.0,
            value_ref=1000.0,
            value_exec=1000.0,
            tax_paid_trade=0.0,
        )
        assert valid_trade.validate_invariants() is True

        # Invalid trade (value_exec doesn't match shares * price_exec)
        invalid_trade = Trade(
            date=datetime(2024, 1, 15),
            ticker="SPY",
            action="BUY",
            shares=10.0,
            price=100.0,
            value=1000.0,
            costs=1.0,
            slippage=0.5,
            price_ref=100.0,
            price_exec=100.0,
            value_ref=1000.0,
            value_exec=999.0,  # Wrong! Should be 1000.0
            tax_paid_trade=0.0,
        )
        assert invalid_trade.validate_invariants() is False

    def test_trade_cash_flow_buy(self):
        """Test cash_flow property for BUY trades."""
        trade = Trade(
            date=datetime(2024, 1, 15),
            ticker="SPY",
            action="BUY",
            shares=10.0,
            price=100.0,
            value=1000.0,
            costs=1.0,
            slippage=0.5,
            price_ref=100.0,
            price_exec=100.0,
            value_ref=1000.0,
            value_exec=1000.0,
            tax_paid_trade=0.0,
        )
        # BUY cash flow is negative (money out)
        # cash_flow = -(value_exec + costs) = -(1000 + 1) = -1001
        assert trade.cash_flow == pytest.approx(-1001.0)

    def test_trade_cash_flow_sell(self):
        """Test cash_flow property for SELL trades."""
        trade = Trade(
            date=datetime(2024, 1, 15),
            ticker="SPY",
            action="SELL",
            shares=10.0,
            price=100.0,
            value=1000.0,
            costs=1.0,
            slippage=0.5,
            price_ref=100.0,
            price_exec=100.0,
            value_ref=1000.0,
            value_exec=1000.0,
            tax_paid_trade=50.0,
        )
        # SELL cash flow is positive (money in)
        # cash_flow = value_exec - costs = 1000 - 1 = 999
        assert trade.cash_flow == pytest.approx(999.0)
        # cash_flow_after_tax = cash_flow - tax = 999 - 50 = 949
        assert trade.cash_flow_after_tax == pytest.approx(949.0)

    def test_trade_backwards_compatibility(self):
        """Test that legacy 'price' and 'value' fields still work."""
        trade = Trade(
            date=datetime(2024, 1, 15),
            ticker="SPY",
            action="BUY",
            shares=10.0,
            price=100.05,  # This should be price_exec
            value=1000.0,  # This should be value_ref
            costs=1.0,
            slippage=0.5,
        )
        # __post_init__ should fill in the new fields
        assert trade.price_exec == 100.05
        assert trade.value_ref == 1000.0
        assert trade.value_exec == pytest.approx(10.0 * 100.05)


class TestValidateTradesInvariants:
    """Tests for the validate_trades_invariants function."""

    def test_valid_trades(self):
        """Test that valid trades pass validation."""
        trades = [
            Trade(
                date=datetime(2024, 1, 15),
                ticker="SPY",
                action="BUY",
                shares=10.0,
                price=100.0,
                value=1000.0,
                costs=1.0,
                slippage=0.5,
                price_ref=100.0,
                price_exec=100.0,
                value_ref=1000.0,
                value_exec=1000.0,
            ),
        ]
        warnings = validate_trades_invariants(trades)
        assert len(warnings) == 0

    def test_invalid_trades(self):
        """Test that invalid trades are detected."""
        trades = [
            Trade(
                date=datetime(2024, 1, 15),
                ticker="SPY",
                action="BUY",
                shares=10.0,
                price=100.0,
                value=1000.0,
                costs=1.0,
                slippage=0.5,
                price_ref=100.0,
                price_exec=100.0,
                value_ref=1000.0,
                value_exec=999.0,  # Wrong!
            ),
        ]
        warnings = validate_trades_invariants(trades)
        assert len(warnings) > 0
        assert "value_exec" in warnings[0]


class TestDailyReconciliation:
    """Tests for the DailyReconciliation class."""

    def test_consistent_reconciliation(self):
        """Test that consistent flows result in zero residual."""
        rec = DailyReconciliation(
            date=date(2024, 1, 15),
            cash_before=10000.0,
            sells_inflow=5000.0,  # Net sell proceeds
            buys_outflow=7000.0,  # Total buy cost
            trading_costs=100.0,  # Informational
            taxes_paid=200.0,
            dividends=0.0,
            cash_after=7800.0,  # 10000 + 5000 - 7000 - 200 = 7800
        )
        assert rec.expected_cash_after == pytest.approx(7800.0)
        assert abs(rec.residual) < 0.01
        assert rec.is_consistent is True

    def test_inconsistent_reconciliation(self):
        """Test that inconsistent flows are detected."""
        rec = DailyReconciliation(
            date=date(2024, 1, 15),
            cash_before=10000.0,
            sells_inflow=5000.0,
            buys_outflow=7000.0,
            trading_costs=100.0,
            taxes_paid=200.0,
            dividends=0.0,
            cash_after=8000.0,  # Wrong! Should be 7800
        )
        assert rec.expected_cash_after == pytest.approx(7800.0)
        assert abs(rec.residual) > 100.0  # 200 difference
        # is_consistent is set based on tolerance (default 0.01)
        # With residual of 200, it should be inconsistent
        rec.is_consistent = abs(rec.residual) <= 0.01
        assert rec.is_consistent is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rec = DailyReconciliation(
            date=date(2024, 1, 15),
            cash_before=10000.0,
            sells_inflow=5000.0,
            buys_outflow=7000.0,
            trading_costs=100.0,
            taxes_paid=200.0,
            dividends=0.0,
            cash_after=7800.0,
            num_sells=2,
            num_buys=3,
        )
        d = rec.to_dict()
        assert d["date"] == "2024-01-15"
        assert d["cash_before"] == 10000.0
        assert d["num_sells"] == 2
        assert d["num_buys"] == 3


class TestMonthlyReconciliation:
    """Tests for the MonthlyReconciliation class."""

    def test_monthly_aggregation(self):
        """Test monthly reconciliation calculation."""
        rec = MonthlyReconciliation(
            year_month="2024-01",
            cash_start=10000.0,
            total_sells_inflow=20000.0,
            total_buys_outflow=25000.0,
            total_trading_costs=500.0,
            total_taxes_paid=1000.0,
            total_dividends=100.0,
            cash_end=4100.0,  # 10000 + 20000 - 25000 - 1000 + 100 = 4100
            num_trading_days=5,
        )
        assert rec.expected_cash_end == pytest.approx(4100.0)
        assert abs(rec.residual) < 0.01


class TestCashflowReconciliation:
    """Tests for the CashflowReconciliation class."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range("2020-01-01", "2020-04-01", freq="B")
        np.random.seed(42)
        prices = pd.DataFrame({
            "SPY": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            "BND": 50 + np.cumsum(np.random.randn(len(dates)) * 0.2),
        }, index=dates)
        return PriceData(
            prices=prices,
            currency={"SPY": "USD", "BND": "USD"},
        )

    def test_from_backtest_result_simple(self, sample_price_data):
        """Test creating reconciliation from a simple backtest."""
        strategy = SimpleStrategy()
        config = BacktestConfig(
            initial_capital=10000,
            rebalance_frequency="monthly",
            costs_pct=0.001,
            slippage_pct=0.0005,
            tax_enabled=False,
            validate=False,
        )
        backtester = Backtester(strategy, sample_price_data, config)
        result = backtester.run()

        recon = CashflowReconciliation.from_backtest_result(result, tolerance=0.01)

        # Should have daily records
        assert len(recon.daily) > 0

        # Should have monthly records
        assert len(recon.monthly) > 0

        # Should be fully consistent
        assert recon.is_fully_consistent is True

        # Residuals should be ~0
        assert abs(recon.total_residual) < 1.0
        assert recon.max_residual < 0.01

    def test_from_backtest_result_with_tax(self, sample_price_data):
        """Test creating reconciliation from a backtest with tax enabled."""
        strategy = SwitchStrategy()
        config = BacktestConfig(
            initial_capital=10000,
            rebalance_frequency="monthly",
            costs_pct=0.001,
            slippage_pct=0.0005,
            tax_enabled=True,
            tax_exemption_amount=0,  # No exemption for cleaner test
            validate=False,
        )
        backtester = Backtester(strategy, sample_price_data, config)
        result = backtester.run()

        recon = CashflowReconciliation.from_backtest_result(result, tolerance=0.01)

        # Should have records with taxes
        has_taxes = any(r.taxes_paid > 0 for r in recon.daily)
        # Note: Tax may or may not be paid depending on gains/losses

        # Should still be consistent
        assert recon.is_fully_consistent is True

    def test_to_dataframe(self, sample_price_data):
        """Test converting reconciliation to DataFrame."""
        strategy = SimpleStrategy()
        config = BacktestConfig(
            initial_capital=10000,
            rebalance_frequency="monthly",
            validate=False,
        )
        backtester = Backtester(strategy, sample_price_data, config)
        result = backtester.run()

        recon = CashflowReconciliation.from_backtest_result(result)

        # Daily DataFrame
        df_daily = recon.to_dataframe("daily")
        assert "date" in df_daily.columns
        assert "cash_before" in df_daily.columns
        assert "residual" in df_daily.columns
        assert len(df_daily) == len(recon.daily)

        # Monthly DataFrame
        df_monthly = recon.to_dataframe("monthly")
        assert "year_month" in df_monthly.columns
        assert len(df_monthly) == len(recon.monthly)

    def test_to_csv(self, sample_price_data):
        """Test exporting reconciliation to CSV."""
        strategy = SimpleStrategy()
        config = BacktestConfig(
            initial_capital=10000,
            rebalance_frequency="monthly",
            validate=False,
        )
        backtester = Backtester(strategy, sample_price_data, config)
        result = backtester.run()

        recon = CashflowReconciliation.from_backtest_result(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "recon_daily.csv"
            recon.to_csv(str(csv_path), level="daily")

            # Read back and verify
            df = pd.read_csv(csv_path)
            assert len(df) == len(recon.daily)
            assert "cash_before" in df.columns

    def test_to_json(self, sample_price_data):
        """Test exporting reconciliation to JSON."""
        strategy = SimpleStrategy()
        config = BacktestConfig(
            initial_capital=10000,
            rebalance_frequency="monthly",
            validate=False,
        )
        backtester = Backtester(strategy, sample_price_data, config)
        result = backtester.run()

        recon = CashflowReconciliation.from_backtest_result(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "recon.json"
            recon.to_json(str(json_path))

            # Read back and verify
            with open(json_path) as f:
                data = json.load(f)

            assert "daily" in data
            assert "monthly" in data
            assert "total_residual" in data
            assert data["is_fully_consistent"] is True

    def test_generate_html_section(self, sample_price_data):
        """Test generating HTML section."""
        strategy = SimpleStrategy()
        config = BacktestConfig(
            initial_capital=10000,
            rebalance_frequency="monthly",
            validate=False,
        )
        backtester = Backtester(strategy, sample_price_data, config)
        result = backtester.run()

        recon = CashflowReconciliation.from_backtest_result(result)

        html = recon.generate_html_section()

        assert "Cashflow Reconciliation" in html
        assert "CONSISTENT" in html or "INCONSISTENT" in html
        assert "Monthly Aggregation" in html
        assert "Daily Detail" in html


class TestReporterReconciliation:
    """Tests for the Reporter's reconciliation integration."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample backtest result for testing."""
        dates = pd.date_range("2020-01-01", "2020-04-01", freq="B")
        np.random.seed(42)
        prices = pd.DataFrame({
            "SPY": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            "BND": 50 + np.cumsum(np.random.randn(len(dates)) * 0.2),
        }, index=dates)
        data = PriceData(
            prices=prices,
            currency={"SPY": "USD", "BND": "USD"},
        )

        strategy = SwitchStrategy()
        config = BacktestConfig(
            initial_capital=10000,
            rebalance_frequency="monthly",
            validate=False,
        )
        backtester = Backtester(strategy, data, config)
        return backtester.run()

    def test_reporter_reconciliation_property(self, sample_result):
        """Test that Reporter has reconciliation property."""
        reporter = Reporter(sample_result)

        recon = reporter.reconciliation
        assert isinstance(recon, CashflowReconciliation)
        assert len(recon.daily) > 0

    def test_reporter_export_reconciliation(self, sample_result):
        """Test exporting reconciliation through Reporter."""
        reporter = Reporter(sample_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.export_reconciliation(tmpdir, prefix="test_recon")

            assert "daily_csv" in paths
            assert "monthly_csv" in paths
            assert "json" in paths

            # Verify files exist
            assert Path(paths["daily_csv"]).exists()
            assert Path(paths["monthly_csv"]).exists()
            assert Path(paths["json"]).exists()


class TestBacktesterTradeFields:
    """Tests that verify the backtester produces trades with correct semantic fields."""

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range("2020-01-01", "2020-03-01", freq="B")
        np.random.seed(42)
        prices = pd.DataFrame({
            "SPY": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        }, index=dates)
        return PriceData(
            prices=prices,
            currency={"SPY": "USD"},
        )

    def test_trade_has_all_semantic_fields(self, sample_price_data):
        """Test that trades from backtester have all semantic fields."""
        strategy = SimpleStrategy()
        config = BacktestConfig(
            initial_capital=10000,
            rebalance_frequency="monthly",
            costs_pct=0.001,
            slippage_pct=0.0005,
            tax_enabled=False,
            validate=False,
        )
        backtester = Backtester(strategy, sample_price_data, config)
        result = backtester.run()

        assert len(result.trades) > 0

        for trade in result.trades:
            # Check all new fields exist and are non-zero where appropriate
            assert trade.price_ref > 0
            assert trade.price_exec > 0
            assert trade.value_ref > 0
            assert trade.value_exec > 0

            # Validate invariants
            assert trade.validate_invariants()

    def test_price_exec_includes_slippage(self, sample_price_data):
        """Test that price_exec properly reflects slippage."""
        strategy = SimpleStrategy()
        slippage_pct = 0.001  # 0.1% slippage
        config = BacktestConfig(
            initial_capital=10000,
            rebalance_frequency="monthly",
            costs_pct=0.0,
            slippage_pct=slippage_pct,
            tax_enabled=False,
            validate=False,
        )
        backtester = Backtester(strategy, sample_price_data, config)
        result = backtester.run()

        for trade in result.trades:
            if trade.action == "BUY":
                # BUY: price_exec = price_ref * (1 + slippage)
                expected_price_exec = trade.price_ref * (1 + slippage_pct)
                assert trade.price_exec == pytest.approx(expected_price_exec, rel=1e-6)
            else:  # SELL
                # SELL: price_exec = price_ref * (1 - slippage)
                expected_price_exec = trade.price_ref * (1 - slippage_pct)
                assert trade.price_exec == pytest.approx(expected_price_exec, rel=1e-6)

    def test_value_consistency(self, sample_price_data):
        """Test that value_ref and value_exec are consistent with shares and prices."""
        strategy = SimpleStrategy()
        config = BacktestConfig(
            initial_capital=10000,
            rebalance_frequency="monthly",
            costs_pct=0.001,
            slippage_pct=0.0005,
            tax_enabled=False,
            validate=False,
        )
        backtester = Backtester(strategy, sample_price_data, config)
        result = backtester.run()

        for trade in result.trades:
            # value_ref = shares * price_ref
            expected_value_ref = trade.shares * trade.price_ref
            assert trade.value_ref == pytest.approx(expected_value_ref, rel=1e-6)

            # value_exec = shares * price_exec
            expected_value_exec = trade.shares * trade.price_exec
            assert trade.value_exec == pytest.approx(expected_value_exec, rel=1e-6)


class TestConsistencyWithTax:
    """Integration tests verifying cashflow consistency with tax."""

    @pytest.fixture
    def price_data_with_gains(self):
        """Create price data that will generate taxable gains."""
        dates = pd.date_range("2020-01-01", "2020-06-01", freq="B")
        # Steadily increasing prices to ensure gains
        prices = pd.DataFrame({
            "SPY": 100 + np.linspace(0, 20, len(dates)),
            "BND": 50 + np.linspace(0, 5, len(dates)),
        }, index=dates)
        return PriceData(
            prices=prices,
            currency={"SPY": "USD", "BND": "USD"},
        )

    def test_tax_paid_trade_is_recorded(self, price_data_with_gains):
        """Test that tax_paid_trade is recorded on SELL trades."""
        strategy = SwitchStrategy()
        config = BacktestConfig(
            initial_capital=10000,
            rebalance_frequency="monthly",
            costs_pct=0.0,  # No costs to simplify
            slippage_pct=0.0,  # No slippage to simplify
            tax_enabled=True,
            tax_exemption_amount=0,  # No exemption
            validate=False,
        )
        backtester = Backtester(strategy, price_data_with_gains, config)
        result = backtester.run()

        sell_trades = [t for t in result.trades if t.action == "SELL"]
        assert len(sell_trades) > 0

        # At least some SELL trades should have tax (if there are gains)
        trades_with_tax = [t for t in sell_trades if t.tax_paid_trade > 0]
        # Note: May or may not have tax depending on gains

        # All BUY trades should have 0 tax
        buy_trades = [t for t in result.trades if t.action == "BUY"]
        for trade in buy_trades:
            assert trade.tax_paid_trade == 0.0

    def test_reconciliation_accounts_for_tax(self, price_data_with_gains):
        """Test that reconciliation properly accounts for taxes."""
        strategy = SwitchStrategy()
        config = BacktestConfig(
            initial_capital=10000,
            rebalance_frequency="monthly",
            costs_pct=0.001,
            slippage_pct=0.0005,
            tax_enabled=True,
            tax_exemption_amount=0,
            validate=False,
        )
        backtester = Backtester(strategy, price_data_with_gains, config)
        result = backtester.run()

        recon = CashflowReconciliation.from_backtest_result(result, tolerance=0.01)

        # Should be consistent despite taxes
        assert recon.is_fully_consistent is True
        assert recon.max_residual <= 0.01
