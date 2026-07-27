from dataclasses import dataclass

import pandas as pd

from backtest.backtester import BacktestConfig
from backtest.data import PriceData
from backtest.strategy import Allocation, Strategy
from backtest.validation import validate_before_run, validate_temporal_leakage


@dataclass
class _SimpleLeakageStrategy(Strategy):
    name: str = "LeakageTest"
    assets: list[str] = None

    def __post_init__(self):
        if self.assets is None:
            self.assets = ["AAA"]

    def signal(self, date, data):
        return Allocation({"AAA": 1.0})


def _build_prices():
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    return pd.DataFrame({"AAA": [100 + i for i in range(len(dates))]}, index=dates)


def test_validate_temporal_leakage_flags_future_macro_rows():
    prices = _build_prices()
    future_macro = pd.DataFrame(
        {"macro": [1.0]},
        index=[prices.index.max() + pd.Timedelta(days=5)],
    )
    data = PriceData(prices=prices, currency={"AAA": "USD"}, macro=future_macro)
    strategy = _SimpleLeakageStrategy()

    result = validate_temporal_leakage(strategy, data)

    assert result.has_errors
    assert any("Macro data extends beyond" in issue.message for issue in result.issues)


def test_validate_temporal_leakage_flags_invalid_lag_settings():
    prices = _build_prices()
    data = PriceData(prices=prices, currency={"AAA": "USD"})
    strategy = _SimpleLeakageStrategy()
    strategy.feature_lag_days = 0
    strategy.macro_lag_days = "abc"

    result = validate_temporal_leakage(strategy, data)

    assert result.has_errors
    assert any("feature_lag_days" in issue.message for issue in result.issues)
    assert any("macro_lag_days must be numeric" in issue.message for issue in result.issues)


def test_validate_temporal_leakage_warns_on_missing_fundamental_lag_metadata():
    prices = _build_prices()
    data = PriceData(prices=prices, currency={"AAA": "USD"})
    strategy = _SimpleLeakageStrategy()
    strategy.uses_fundamentals = True

    result = validate_temporal_leakage(strategy, data)

    assert result.has_warnings
    assert any("provides no release-lag metadata" in issue.message for issue in result.issues)


def test_validate_before_run_includes_leakage_issues():
    prices = _build_prices()
    future_macro = pd.DataFrame(
        {"macro": [1.0]},
        index=[prices.index.max() + pd.Timedelta(days=5)],
    )
    data = PriceData(prices=prices, currency={"AAA": "USD"}, macro=future_macro)
    strategy = _SimpleLeakageStrategy()
    config = BacktestConfig(initial_capital=10000)

    result = validate_before_run(strategy, data, config)

    assert result.has_errors
    assert any(issue.category == "leakage" for issue in result.issues)
