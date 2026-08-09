from datetime import date

import pandas as pd

from backtest.backtester import BacktestConfig
from backtest.comparator import Comparator
from backtest.data import PriceData
from backtest.strategy import Allocation, Strategy


class _FrequencyStrategy(Strategy):
    assets = ["AAA"]

    def __init__(self, name: str, frequency: str) -> None:
        self.name = name
        self.rebalance_frequency = frequency

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        return Allocation({"AAA": 1.0})


def test_comparator_does_not_leak_resolved_frequency_between_strategies():
    dates = pd.bdate_range("2024-01-01", periods=80)
    data = PriceData(
        prices=pd.DataFrame({"AAA": [100.0 + i for i in range(len(dates))]}, index=dates),
        currency={"AAA": "EUR"},
    )
    config = BacktestConfig(rebalance_frequency=None, tax_enabled=False, validate=False)

    result = Comparator(
        strategies=[
            _FrequencyStrategy("weekly", "weekly"),
            _FrequencyStrategy("monthly", "monthly"),
        ],
        data=data,
        config=config,
    ).run(progress=False)

    assert [r.config.rebalance_frequency for r in result.results] == ["weekly", "monthly"]
    assert config.rebalance_frequency is None
