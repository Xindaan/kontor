from datetime import date

from backtest.batch_optimize import (
    OptimizationResult,
    BatchOptimizationResult,
    run_batch_optimization,
)
from backtest.strategy import Strategy


class DummyStrategy(Strategy):
    name = "Dummy Strategy"
    assets = ["SPY"]

    def signal(self, date, data):
        return None


def test_batch_optimize_reports_loading_strategies(monkeypatch):
    dummy_instance = DummyStrategy()

    def fake_load_strategy_from_file(_path):
        return dummy_instance, DummyStrategy, dummy_instance.name

    def fake_data_loader(**_kwargs):
        class DummyPrices:
            index = [0]

        class DummyData:
            prices = DummyPrices()
            end_date = date.today()
            start_date = date.today()

        return DummyData()

    def fake_get_param_grid(_cls, _instance, _custom):
        return {}

    def fake_optimize_single_strategy(**_kwargs):
        return OptimizationResult(
            strategy_name=dummy_instance.name,
            strategy_class=DummyStrategy.__name__,
            strategy_file="strategies/dummy.py",
            best_params={},
            best_rebalance_frequency="monthly",
            best_metric_value=0.0,
            metric_name="sharpe_ratio",
            all_results=[],
        )

    monkeypatch.setattr("backtest.batch_optimize.load_strategy_from_file", fake_load_strategy_from_file)
    monkeypatch.setattr("backtest.data.DataLoader.yahoo", fake_data_loader)
    monkeypatch.setattr("backtest.batch_optimize.get_param_grid_for_strategy", fake_get_param_grid)
    monkeypatch.setattr("backtest.batch_optimize.optimize_single_strategy", fake_optimize_single_strategy)

    progress_updates = []

    def progress_callback(**kwargs):
        progress_updates.append(kwargs)

    result = run_batch_optimization(
        strategy_files=["strategies/dummy.py"],
        progress=False,
        progress_callback=progress_callback,
    )

    assert isinstance(result, BatchOptimizationResult)
    assert any(update.get("status") == "loading_strategies" for update in progress_updates)
