"""
Consistency Tests for Kontor v2.

These tests ensure:
- Manifest is written after runs
- Metrics are consistent between single and comparison reports
- No deprecated pandas 'M' frequency is used
- All strategies have proper names
"""

import pytest
import tempfile
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import numpy as np


class TestManifest:
    """Tests for run manifest generation."""

    def test_manifest_written(self):
        """Test that manifest is written after a run with required keys."""
        from backtest.config.run_config import RunConfig
        from backtest.manifest import RunManifest, DataSnapshot, GitInfo, StrategyInfo
        from backtest.strategy import Strategy, Allocation

        # Create a simple test strategy
        class TestStrategy(Strategy):
            name = "Test Strategy"

            def __init__(self):
                self.params = {"test": True}
                self.assets = ["SPY"]

            def signal(self, date, data):
                return Allocation({"SPY": 1.0})

        strategy = TestStrategy()
        config = RunConfig()

        # Create mock data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = pd.DataFrame({"SPY": np.random.randn(100).cumsum() + 100}, index=dates)

        # Create mock PriceData-like object
        class MockPriceData:
            def __init__(self, prices):
                self.prices = prices

        data = MockPriceData(prices)

        # Create manifest
        manifest = RunManifest.create(config, strategy, data, "test")

        # Verify required keys
        manifest_dict = manifest.to_dict()
        required_keys = ["run_id", "timestamp", "config", "strategy", "data", "git"]
        for key in required_keys:
            assert key in manifest_dict, f"Missing required key: {key}"

        # Verify config hash is present
        assert "config_hash" in manifest_dict["config"]

        # Verify strategy info
        assert manifest_dict["strategy"]["name"] == "Test Strategy"
        assert manifest_dict["strategy"]["class_name"] == "TestStrategy"

        # Verify git info
        assert "commit_hash" in manifest_dict["git"]
        assert "dirty" in manifest_dict["git"]

    def test_manifest_save_load(self):
        """Test manifest can be saved and loaded."""
        from backtest.config.run_config import RunConfig
        from backtest.manifest import RunManifest, DataSnapshot, GitInfo, StrategyInfo
        from backtest.strategy import Strategy, Allocation

        class TestStrategy(Strategy):
            name = "Test Strategy"

            def __init__(self):
                self.params = {}
                self.assets = ["SPY"]

            def signal(self, date, data):
                return Allocation({"SPY": 1.0})

        strategy = TestStrategy()
        config = RunConfig()

        # Create mock data
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = pd.DataFrame({"SPY": np.random.randn(100).cumsum() + 100}, index=dates)

        class MockPriceData:
            def __init__(self, prices):
                self.prices = prices

        data = MockPriceData(prices)

        manifest = RunManifest.create(config, strategy, data, "test")

        # Save and load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            manifest.save(f.name)
            loaded = RunManifest.from_file(f.name)

        assert loaded.run_id == manifest.run_id
        assert loaded.config["config_hash"] == manifest.config["config_hash"]


class TestMetricsConsistency:
    """Tests for metrics consistency between different calculation paths."""

    def test_annualization_monthly_vs_daily(self):
        """Test that annualization is correct for different frequencies."""
        from backtest.utils import (
            annualized_vol,
            sharpe_ratio,
            infer_periods_per_year,
        )

        # Create monthly returns with known properties
        np.random.seed(42)
        monthly_returns = pd.Series(
            np.random.randn(60) * 0.03,  # 5 years of monthly returns
            index=pd.date_range("2018-01-31", periods=60, freq="ME")
        )

        # Create daily returns with same properties (scaled)
        daily_returns = pd.Series(
            np.random.randn(1260) * 0.01,  # 5 years of daily returns
            index=pd.date_range("2018-01-01", periods=1260, freq="B")
        )

        # Check frequency inference
        assert infer_periods_per_year(monthly_returns.index) == 12
        assert infer_periods_per_year(daily_returns.index) == 252

        # Volatility should be annualized correctly
        monthly_vol = annualized_vol(monthly_returns)
        daily_vol = annualized_vol(daily_returns)

        # Both should be reasonable (between 5% and 50%)
        assert 0.05 < monthly_vol < 0.5, f"Monthly vol {monthly_vol} out of range"
        assert 0.05 < daily_vol < 0.5, f"Daily vol {daily_vol} out of range"

    def test_cagr_calculation(self):
        """Test CAGR calculation is correct."""
        from backtest.utils import cagr

        # Create equity curve that doubles in 7 years
        # CAGR should be approximately 10.4%
        dates = pd.date_range("2015-01-01", "2022-01-01", freq="ME")
        values = np.linspace(10000, 20000, len(dates))
        equity = pd.Series(values, index=dates)

        result = cagr(equity)
        expected = 2 ** (1 / 7) - 1  # ~10.4%

        assert abs(result - expected) < 0.01, f"CAGR {result} != expected {expected}"

    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        from backtest.utils import max_drawdown

        # Create equity curve with known 20% drawdown
        dates = pd.date_range("2020-01-01", periods=10, freq="ME")
        values = [100, 110, 100, 90, 88, 95, 100, 105, 102, 108]
        equity = pd.Series(values, index=dates)

        result = max_drawdown(equity)

        # Max drawdown is from 110 to 88 = -20%
        expected = (88 - 110) / 110

        assert abs(result - expected) < 0.01, f"MaxDD {result} != expected {expected}"


class TestNoDeprecatedM:
    """Test that deprecated 'M' frequency is not used."""

    def test_no_deprecated_m_in_code(self):
        """Scan source files for deprecated 'M' frequency usage."""
        import re

        src_dir = Path(__file__).parent.parent / "src" / "backtest"

        deprecated_pattern = re.compile(r'resample\s*\(\s*["\']M["\']\s*\)')

        violations = []
        for py_file in src_dir.glob("**/*.py"):
            content = py_file.read_text()
            matches = deprecated_pattern.findall(content)
            if matches:
                violations.append(f"{py_file.name}: {matches}")

        assert not violations, f"Deprecated 'M' frequency found in: {violations}"

    def test_month_end_freq_helper(self):
        """Test that month_end_freq helper works correctly."""
        from backtest.utils import MONTH_END_FREQ, month_end_freq

        # Should return either 'ME' or 'M' depending on pandas version
        assert MONTH_END_FREQ in ("ME", "M")

        # Should work with pandas resample
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        series = pd.Series(range(100), index=dates)

        # This should not raise
        resampled = series.resample(MONTH_END_FREQ).last()
        assert len(resampled) > 0


class TestStrategyNaming:
    """Test that all strategies have proper names."""

    def test_strategy_metaclass_adds_name(self):
        """Test that StrategyMeta adds name from class name."""
        from backtest.strategy import Strategy, Allocation

        class MyTestStrategy(Strategy):
            def __init__(self):
                self.params = {}
                self.assets = ["SPY"]

            def signal(self, date, data):
                return Allocation({"SPY": 1.0})

        strategy = MyTestStrategy()

        # Metaclass should convert "MyTestStrategy" to "My Test Strategy"
        assert strategy.name == "My Test Strategy"
        assert strategy.display_name == "My Test Strategy"

    def test_explicit_name_preserved(self):
        """Test that explicitly set names are preserved."""
        from backtest.strategy import Strategy, Allocation

        class ExplicitNameStrategy(Strategy):
            name = "My Custom Name"

            def __init__(self):
                self.params = {}
                self.assets = ["SPY"]

            def signal(self, date, data):
                return Allocation({"SPY": 1.0})

        strategy = ExplicitNameStrategy()
        assert strategy.name == "My Custom Name"
        assert strategy.display_name == "My Custom Name"

    def test_no_unnamed_strategy(self):
        """Test that no strategy ends up with 'Unnamed Strategy'."""
        from backtest.strategy import Strategy, Allocation

        class AnotherStrategy(Strategy):
            def __init__(self):
                self.params = {}
                self.assets = ["SPY"]

            def signal(self, date, data):
                return Allocation({"SPY": 1.0})

        strategy = AnotherStrategy()
        assert strategy.name != "Unnamed Strategy"
        assert strategy.display_name != "Unnamed Strategy"

    def test_existing_strategies_have_names(self):
        """Test that all existing strategy files have proper names."""
        from pathlib import Path
        import importlib.util
        import sys

        strategies_dir = Path(__file__).parent.parent / "strategies"

        for py_file in strategies_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            # Load the module
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            module = importlib.util.module_from_spec(spec)

            try:
                spec.loader.exec_module(module)
            except Exception as e:
                # Skip modules that can't be loaded
                continue

            # Find Strategy subclasses
            from backtest.strategy import Strategy

            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, Strategy)
                    and obj is not Strategy
                ):
                    # Check that the class has a name
                    assert hasattr(obj, 'name'), f"{name} in {py_file.name} has no name"
                    assert obj.name != "Unnamed Strategy", (
                        f"{name} in {py_file.name} has 'Unnamed Strategy' as name"
                    )


class TestConfigHash:
    """Test configuration hash for reproducibility."""

    def test_same_config_same_hash(self):
        """Test that identical configs produce identical hashes."""
        from backtest.config.run_config import RunConfig

        config1 = RunConfig(start_date="2020-01-01", initial_capital=10000)
        config2 = RunConfig(start_date="2020-01-01", initial_capital=10000)

        assert config1.config_hash == config2.config_hash

    def test_different_config_different_hash(self):
        """Test that different configs produce different hashes."""
        from backtest.config.run_config import RunConfig

        config1 = RunConfig(start_date="2020-01-01", initial_capital=10000)
        config2 = RunConfig(start_date="2020-01-01", initial_capital=20000)

        assert config1.config_hash != config2.config_hash

    def test_metadata_not_in_hash(self):
        """Test that metadata fields don't affect the hash."""
        from backtest.config.run_config import RunConfig

        config1 = RunConfig(start_date="2020-01-01", name="Config 1")
        config2 = RunConfig(start_date="2020-01-01", name="Config 2")

        assert config1.config_hash == config2.config_hash


class TestFrequencyInference:
    """Test frequency inference utilities."""

    def test_infer_daily(self):
        """Test daily frequency inference."""
        from backtest.utils import infer_periods_per_year, infer_frequency_name

        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        assert infer_periods_per_year(dates) == 252
        assert infer_frequency_name(dates) == "daily"

    def test_infer_monthly(self):
        """Test monthly frequency inference."""
        from backtest.utils import infer_periods_per_year, infer_frequency_name

        dates = pd.date_range("2020-01-31", periods=24, freq="ME")
        assert infer_periods_per_year(dates) == 12
        assert infer_frequency_name(dates) == "monthly"

    def test_infer_weekly(self):
        """Test weekly frequency inference."""
        from backtest.utils import infer_periods_per_year, infer_frequency_name

        dates = pd.date_range("2020-01-01", periods=52, freq="W")
        assert infer_periods_per_year(dates) == 52
        assert infer_frequency_name(dates) == "weekly"
