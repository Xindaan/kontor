from pathlib import Path

from backtest.strategy import Strategy, Allocation
from backtest.sweep import SweepConfig, SweepResult, run_sweep, SweepCancelled, run_sweep_with_strategies


class DummyStrategy(Strategy):
    name = "Dummy Strategy"
    assets = ["SPY"]

    def signal(self, date, data):
        return Allocation({"SPY": 1.0})


def test_run_sweep_reports_loading_strategies(monkeypatch):
    dummy_strategy = DummyStrategy()

    def fake_load_strategy_from_file(_path: str):
        return dummy_strategy

    def fake_run_sweep_with_strategies(**_kwargs):
        return SweepResult(
            config=SweepConfig(),
            windows=[],
            window_results=[],
            summaries=[],
        )

    monkeypatch.setattr("backtest.cli.load_strategy_from_file", fake_load_strategy_from_file)
    monkeypatch.setattr("backtest.sweep.run_sweep_with_strategies", fake_run_sweep_with_strategies)

    progress_updates = []

    def progress_callback(**kwargs):
        progress_updates.append(kwargs)

    run_sweep(
        strategy_paths=[Path("strategies/dummy.py")],
        config=SweepConfig(),
        progress=False,
        progress_callback=progress_callback,
    )

    assert any(update.get("status") == "loading_strategies" for update in progress_updates)


def test_run_sweep_with_strategies_honors_cancel_check(monkeypatch):
    dummy_strategy = DummyStrategy()

    def fake_data_loader(**_kwargs):
        class DummyPrices:
            index = [0]

        class DummyData:
            prices = DummyPrices()
        return DummyData()

    def fake_generate_windows(**_kwargs):
        return [type("Window", (), {"start": None, "end": None, "years": 1})()]

    def fake_compute_benchmark_for_window(**_kwargs):
        return None

    monkeypatch.setattr("backtest.data.DataLoader.yahoo", fake_data_loader)
    monkeypatch.setattr("backtest.sweep.generate_windows", fake_generate_windows)
    monkeypatch.setattr("backtest.sweep.compute_benchmark_for_window", fake_compute_benchmark_for_window)

    def cancel_check():
        return True

    try:
        run_sweep_with_strategies(
            strategies=[dummy_strategy],
            strategy_files={dummy_strategy.name: "strategies/dummy.py"},
            config=SweepConfig(),
            progress=False,
            cancel_check=cancel_check,
        )
    except SweepCancelled:
        pass
    else:
        assert False, "Expected sweep cancellation"
