import pandas as pd
from datetime import date

from backtest.backtester import BacktestConfig, Backtester, Portfolio
from backtest.data import PriceData
from backtest.strategy import Allocation, Strategy


class _AlwaysLong(Strategy):
    name = "AlwaysLong"
    assets = ["AAPL"]

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        return Allocation({"AAPL": 1.0})


def test_volume_guard_caps_buy_shares_by_participation():
    portfolio = Portfolio(cash=10_000.0)
    prices = pd.Series({"AAPL": 100.0})
    prices.name = pd.Timestamp("2020-01-02")
    volumes = pd.Series({"AAPL": 100.0})
    volumes.name = prices.name

    trades = portfolio.execute_buys(
        target=Allocation({"AAPL": 1.0}),
        prices=prices,
        volumes=volumes,
        max_volume_participation=0.10,  # max 10 shares
        liquidity_on_missing_volume="allow",
    )

    assert len(trades) == 1
    assert trades[0].shares <= 10.0 + 1e-9
    assert portfolio.positions.get("AAPL", 0.0) <= 10.0 + 1e-9


def test_volume_guard_skips_when_volume_missing_and_configured():
    portfolio = Portfolio(cash=10_000.0)
    prices = pd.Series({"AAPL": 100.0})
    prices.name = pd.Timestamp("2020-01-02")

    trades = portfolio.execute_buys(
        target=Allocation({"AAPL": 1.0}),
        prices=prices,
        volumes=None,
        max_volume_participation=0.10,
        liquidity_on_missing_volume="skip",
    )

    assert trades == []
    assert portfolio.positions == {}


def test_volume_guard_respects_min_daily_dollar_volume():
    portfolio = Portfolio(cash=10_000.0)
    prices = pd.Series({"AAPL": 100.0})
    prices.name = pd.Timestamp("2020-01-02")
    volumes = pd.Series({"AAPL": 50.0})  # $5,000 daily notional
    volumes.name = prices.name

    trades = portfolio.execute_buys(
        target=Allocation({"AAPL": 1.0}),
        prices=prices,
        volumes=volumes,
        min_daily_dollar_volume=10_000.0,
        liquidity_on_missing_volume="allow",
    )

    assert trades == []
    assert portfolio.positions == {}


def test_execution_lag_t_plus_one_shifts_first_trade_date():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates)
    data = PriceData(prices=prices, currency={"AAPL": "USD"})

    cfg_same_day = BacktestConfig(
        initial_capital=10_000.0,
        benchmark=None,
        rebalance_frequency="daily",
        costs_pct=0.0,
        slippage_pct=0.0,
        spread_pct=0.0,
        tax_enabled=False,
        execution_lag_days=0,
        validate=False,
    )
    cfg_t1 = BacktestConfig(
        initial_capital=10_000.0,
        benchmark=None,
        rebalance_frequency="daily",
        costs_pct=0.0,
        slippage_pct=0.0,
        spread_pct=0.0,
        tax_enabled=False,
        execution_lag_days=1,
        validate=False,
    )

    same_day_result = Backtester(_AlwaysLong(), data, cfg_same_day).run()
    t1_result = Backtester(_AlwaysLong(), data, cfg_t1).run()

    first_buy_same_day = next(t for t in same_day_result.trades if t.action == "BUY")
    first_buy_t1 = next(t for t in t1_result.trades if t.action == "BUY")

    assert first_buy_same_day.date == dates[0]
    assert first_buy_t1.date == dates[1]


def test_backtester_exposes_constraint_impact_diagnostics():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    prices = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104, 105]}, index=dates)
    volumes = pd.DataFrame({"AAPL": [100, 100, 100, 100, 100, 100]}, index=dates)
    data = PriceData(prices=prices, currency={"AAPL": "USD"}, volumes=volumes)

    cfg = BacktestConfig(
        initial_capital=10_000.0,
        benchmark=None,
        rebalance_frequency="daily",
        costs_pct=0.0,
        slippage_pct=0.0,
        spread_pct=0.0,
        tax_enabled=False,
        execution_lag_days=1,
        max_volume_participation=0.10,
        liquidity_on_missing_volume="allow",
        risk_overlay={"max_position": 0.5},
        validate=False,
    )

    result = Backtester(_AlwaysLong(), data, cfg).run()
    assert result.constraint_impact is not None

    liquidity = result.constraint_impact["liquidity"]
    assert liquidity["requested_trades"] >= 1
    assert liquidity["executed_trades"] >= 1
    assert liquidity["requested_trades"] >= liquidity["executed_trades"]
    assert liquidity["clipped_by_max_participation"] >= 1

    risk_overlay = result.constraint_impact["risk_overlay"]
    assert risk_overlay["apply_calls"] >= 1
    assert risk_overlay["max_position_bindings"] >= 1
