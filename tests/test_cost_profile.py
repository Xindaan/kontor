import pandas as pd

from backtest.backtester import BacktestConfig, Backtester, Portfolio
from backtest.costs import CostModelResolver
from backtest.strategy import Allocation, Strategy
from backtest.data import PriceData


def test_cost_model_resolver_prioritizes_ticker_overrides():
    default_model = Portfolio._default_cost_model(costs_pct=0.001, slippage_pct=0.0005, spread_pct=0.0)
    profile = {
        "default": {"spread_bps": 4.0},
        "asset_classes": {
            "us_equity": {"slippage_bps": 2.0},
        },
        "ticker_asset_class": {
            "AAPL": "us_equity",
            "MSFT": "us_equity",
        },
        "tickers": {
            "AAPL": {"commission_pct": 0.0002, "commission_min": 1.0},
        },
    }

    resolver = CostModelResolver.from_profile(default_model=default_model, profile=profile)

    aapl_model = resolver.for_ticker("AAPL")
    msft_model = resolver.for_ticker("MSFT")
    spy_model = resolver.for_ticker("SPY")

    # Ticker override wins
    assert aapl_model.commission_pct == 0.0002
    assert aapl_model.commission_min == 1.0
    # Inherits class + default overlays
    assert aapl_model.slippage_bps == 2.0
    assert aapl_model.spread_bps == 4.0

    # Class-level fallback
    assert msft_model.commission_pct == default_model.commission_pct
    assert msft_model.slippage_bps == 2.0
    assert msft_model.spread_bps == 4.0

    # Global fallback
    assert spy_model.commission_pct == default_model.commission_pct
    assert spy_model.slippage_bps == default_model.slippage_bps


def test_execute_buys_respects_min_commission_and_cash_constraint():
    portfolio = Portfolio(cash=1000.0)
    prices = pd.Series({"AAPL": 100.0})
    prices.name = pd.Timestamp("2020-01-01")
    target = Allocation({"AAPL": 1.0})

    default_model = Portfolio._default_cost_model(costs_pct=0.0, slippage_pct=0.0, spread_pct=0.0)
    resolver = CostModelResolver.from_profile(
        default_model=default_model,
        profile={
            "tickers": {
                "AAPL": {
                    "commission_pct": 0.0,
                    "commission_min": 10.0,
                    "slippage_bps": 100.0,  # 1%
                    "spread_bps": 0.0,
                }
            }
        },
    )

    trades = portfolio.execute_buys(
        target=target,
        prices=prices,
        costs_pct=0.0,
        slippage_pct=0.0,
        spread_pct=0.0,
        cost_resolver=resolver,
    )

    assert len(trades) == 1
    trade = trades[0]

    # BUY execution price includes 1% slippage from profile
    assert trade.price_exec == 101.0
    # Min commission from profile is applied
    assert trade.costs == 10.0
    # Total spent must never exceed cash
    assert trade.value_exec + trade.costs <= 1000.0 + 1e-6
    assert portfolio.cash >= -1e-6


class _AlwaysLongAAPL(Strategy):
    name = "AlwaysLongAAPL"
    assets = ["AAPL"]

    def signal(self, date, data):
        return Allocation({"AAPL": 1.0})


def test_backtester_applies_cost_profile_overrides():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"AAPL": [100, 101, 102, 103, 104]}, index=dates)
    data = PriceData(prices=prices, currency={"AAPL": "USD"})

    config = BacktestConfig(
        initial_capital=10_000.0,
        costs_pct=0.0,
        slippage_pct=0.0,
        spread_pct=0.0,
        cost_profile={
            "tickers": {
                "AAPL": {
                    "commission_pct": 0.0,
                    "commission_min": 5.0,
                    "slippage_bps": 0.0,
                    "spread_bps": 0.0,
                }
            }
        },
        rebalance_frequency="daily",
        tax_enabled=False,
        validate=False,
    )

    result = Backtester(_AlwaysLongAAPL(), data, config).run()

    buy_trades = [t for t in result.trades if t.action == "BUY"]
    assert buy_trades
    assert buy_trades[0].costs == 5.0
