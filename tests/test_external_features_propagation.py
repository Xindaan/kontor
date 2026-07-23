"""Propagation tests for ExternalFeaturesConfig / loader.

Asserts that the loader reaches the strategy's _external_features_provider
slot through every backtest entry path:
- Direct Backtester
- RunConfig -> config_to_backtest_config -> Backtester
- SweepConfig -> run_single_window
- optimize_single_strategy
- walk_forward_single_strategy
- run_batch_optimization

After each run the provider must be restored to None (no leaks).
"""

from __future__ import annotations

import json
import textwrap
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.backtester import Backtester, BacktestConfig
from backtest.config.run_config import (
    CostConfig,
    RunConfig,
    TaxConfig,
    config_to_backtest_config,
)
from backtest.data import PriceData
from backtest.external_features.config import (
    ExternalFeaturesConfig,
    build_loader_from_config,
)
from backtest.strategy import Allocation, Strategy


class CapturingStrategy(Strategy):
    name = "CapturingPropagation"
    assets = ["SPY"]
    rebalance_frequency = "monthly"
    captured: list = []

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        self.__class__.captured.append(self._external_features_provider)
        return Allocation({"SPY": 1.0})


@pytest.fixture(autouse=True)
def _reset_capture():
    CapturingStrategy.captured = []


@pytest.fixture
def sample_data() -> PriceData:
    dates = pd.date_range("2020-01-01", periods=36, freq="ME")
    prices = pd.DataFrame({"SPY": np.linspace(100, 180, 36)}, index=dates)
    return PriceData(prices=prices, currency={"SPY": "USD"}, fx_rates=None)


@pytest.fixture
def enabled_ef_config(tmp_path: Path) -> ExternalFeaturesConfig:
    # Mode 'off' to skip provenance, dataset directory may stay empty;
    # ExternalFeaturesLoader.load() handles empty datasets gracefully.
    root = tmp_path / "external_features"
    (root / "snapshots" / "test_ds").mkdir(parents=True, exist_ok=True)
    return ExternalFeaturesConfig(
        enabled=True,
        dataset="test_ds",
        root=str(root),
        provenance_mode="off",
        registry_path=str(tmp_path / "prov.json"),
    )


def test_run_config_to_dict_roundtrip_preserves_external_features():
    cfg = RunConfig(
        external_features=ExternalFeaturesConfig(
            enabled=True, dataset="abc", provenance_mode="strict"
        )
    )
    payload = cfg.to_dict(include_metadata=False)
    restored = RunConfig.from_dict(payload)
    assert restored.external_features.enabled is True
    assert restored.external_features.dataset == "abc"
    assert restored.external_features.provenance_mode == "strict"


def test_config_hash_stable_for_default_external_features():
    a = RunConfig().config_hash
    b = RunConfig().config_hash
    assert a == b


def test_config_hash_changes_when_external_features_enabled():
    disabled = RunConfig().config_hash
    enabled = RunConfig(
        external_features=ExternalFeaturesConfig(enabled=True, dataset="x")
    ).config_hash
    assert enabled != disabled


def test_backtester_direct_sets_provider_and_restores(sample_data):
    sentinel = object()
    strategy = CapturingStrategy()
    config = BacktestConfig(benchmark=None, external_features_loader=sentinel)
    Backtester(strategy, sample_data, config).run()
    assert sentinel in CapturingStrategy.captured
    # All capture entries during this run must point to the sentinel.
    assert all(c is sentinel for c in CapturingStrategy.captured)
    # After the run the slot is restored to the class default.
    assert strategy._external_features_provider is None


def test_backtester_run_restores_previous_provider_value(sample_data):
    sentinel_a = object()
    sentinel_b = object()
    strategy = CapturingStrategy()
    strategy._external_features_provider = sentinel_a  # pre-existing

    config = BacktestConfig(benchmark=None, external_features_loader=sentinel_b)
    Backtester(strategy, sample_data, config).run()
    assert all(c is sentinel_b for c in CapturingStrategy.captured)
    assert strategy._external_features_provider is sentinel_a  # restored


def test_run_config_to_backtest_config_builds_loader(enabled_ef_config):
    rc = RunConfig(
        start_date="2020-01-01",
        external_features=enabled_ef_config,
    )
    bt_cfg = config_to_backtest_config(rc)
    assert bt_cfg.external_features_loader is not None
    # Disabled default → loader is None.
    bt_cfg_default = config_to_backtest_config(RunConfig())
    assert bt_cfg_default.external_features_loader is None


def test_run_config_pathway_provider_reaches_strategy(sample_data, enabled_ef_config):
    rc = RunConfig(external_features=enabled_ef_config)
    bt_cfg = config_to_backtest_config(rc)
    bt_cfg.benchmark = None  # avoid benchmark download
    strategy = CapturingStrategy()
    Backtester(strategy, sample_data, bt_cfg).run()
    assert any(c is not None for c in CapturingStrategy.captured)
    assert strategy._external_features_provider is None


def test_sweep_run_single_window_propagates(sample_data, enabled_ef_config):
    from backtest.sweep import SweepConfig, Window, run_single_window

    window = Window(
        start=pd.Timestamp("2020-01-31"),
        end=pd.Timestamp("2022-12-31"),
        years=3.0,
    )
    cfg = SweepConfig(external_features=enabled_ef_config, validate=False, warmup_days=0)
    strategy = CapturingStrategy()
    run_single_window(strategy, "(memory)", sample_data, window, cfg)
    assert any(c is not None for c in CapturingStrategy.captured)
    assert strategy._external_features_provider is None


def _write_capturing_strategy_file(tmp_path: Path) -> tuple[Path, Path]:
    """Write a strategy module that records signal() provider state to a file.

    Returns (strategy_file, log_file). The log_file accumulates one line per
    signal() call: "saw" if a provider is set, "none" otherwise. Cross-module
    state is captured via the log file so the test does not depend on whether
    importlib returns the same class object.
    """
    log_file = tmp_path / "capturing_log.txt"
    target = tmp_path / "capturing_strategy.py"
    target.write_text(
        textwrap.dedent(
            f"""
            from datetime import date
            import pandas as pd

            from backtest.strategy import Strategy, Allocation


            _LOG_PATH = r"{log_file}"


            class CapturingFileStrategy(Strategy):
                name = "CapturingFileStrategy"
                assets = ["SPY"]
                rebalance_frequency = "monthly"

                def __init__(self):
                    self.params = {{}}

                def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
                    seen = getattr(self, "_external_features_provider", None)
                    with open(_LOG_PATH, "a", encoding="utf-8") as fh:
                        fh.write("saw\\n" if seen is not None else "none\\n")
                    return Allocation({{"SPY": 1.0}})


            strategy = CapturingFileStrategy()
            """
        )
    )
    return target, log_file


def _read_log(log_file: Path) -> list[str]:
    if not log_file.exists():
        return []
    return [line.strip() for line in log_file.read_text().splitlines() if line.strip()]


def test_optimize_single_strategy_propagates(tmp_path: Path, sample_data, enabled_ef_config):
    from backtest.batch_optimize import optimize_single_strategy

    strategy_file, log_file = _write_capturing_strategy_file(tmp_path)
    optimize_single_strategy(
        strategy_file=str(strategy_file),
        data=sample_data,
        param_grid={},
        rebalance_frequencies=["monthly"],
        metric="cagr",
        external_features=enabled_ef_config,
        validate=False,
    )
    log = _read_log(log_file)
    assert log
    assert "saw" in log


def test_walk_forward_single_strategy_propagates(tmp_path: Path, sample_data, enabled_ef_config):
    from backtest.batch_optimize import walk_forward_single_strategy

    strategy_file, log_file = _write_capturing_strategy_file(tmp_path)
    walk_forward_single_strategy(
        strategy_file=str(strategy_file),
        data=sample_data,
        param_grid={},
        rebalance_frequencies=["monthly"],
        train_years=1.0,
        test_years=0.5,
        external_features=enabled_ef_config,
        validate=False,
    )
    log = _read_log(log_file)
    assert log
    assert "saw" in log


def test_run_batch_optimization_propagates(tmp_path: Path, sample_data, enabled_ef_config, monkeypatch):
    from backtest import batch_optimize as bo

    strategy_file, log_file = _write_capturing_strategy_file(tmp_path)

    # Avoid Yahoo network call: monkeypatch DataLoader.yahoo to return our data.
    def _fake_yahoo(*args, **kwargs):
        return sample_data

    monkeypatch.setattr(bo.DataLoader, "yahoo", staticmethod(_fake_yahoo))

    bo.run_batch_optimization(
        strategy_files=[str(strategy_file)],
        start="2020-01-01",
        external_features=enabled_ef_config,
        progress=False,
        validate=False,
    )
    log = _read_log(log_file)
    assert log
    assert "saw" in log
