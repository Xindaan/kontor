from datetime import date

import pandas as pd
import pytest

from backtest.backtester import BacktestConfig, Backtester
from backtest.data import PriceData
from backtest.meta_promotion import metric_curve
from backtest.strategy import Allocation, Strategy


class _AlwaysLong(Strategy):
    name = "always long"
    assets = ["SPY"]

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        return Allocation({"SPY": 1.0})


def _price_data(prices: pd.Series) -> PriceData:
    return PriceData(
        prices=pd.DataFrame({"SPY": prices}),
        currency={"SPY": "EUR"},
        fx_rates=None,
    )


def test_daily_mtm_captures_intraperiod_drawdown_without_changing_headline_metrics():
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    prices = pd.Series(100.0, index=dates)
    prices.loc[pd.Timestamp("2021-06-30")] = 50.0

    config = BacktestConfig(
        initial_capital=100_000,
        benchmark=None,
        costs_pct=0.0,
        slippage_pct=0.0,
        tax_enabled=False,
        rebalance_frequency="yearly",
        validate=False,
    )

    result = Backtester(_AlwaysLong(), _price_data(prices), config).run()

    assert len(result.equity_curve) == 2
    assert len(result.equity_curve_daily) > len(result.equity_curve)
    assert result.metrics.max_drawdown == pytest.approx(0.0)
    assert result.metrics.max_drawdown_daily == pytest.approx(-0.4995, abs=1e-4)
    assert result.metrics_daily.max_drawdown == pytest.approx(result.metrics.max_drawdown_daily)
    assert result.metrics.underwater_days_daily is not None
    assert result.metrics.underwater_days_daily > 0


def test_metric_curve_uses_net_liquidation_headline_curve():
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    steps = pd.Series(range(len(dates)), index=dates)
    prices = 100.0 + steps.astype(float) / float(len(dates) - 1) * 100.0

    config = BacktestConfig(
        initial_capital=100_000,
        benchmark=None,
        costs_pct=0.0,
        slippage_pct=0.0,
        tax_enabled=True,
        tax_exemption_amount=0.0,
        metric_basis="net_liquidation",
        rebalance_frequency="yearly",
        validate=False,
    )

    result = Backtester(_AlwaysLong(), _price_data(prices), config).run()

    # A bare `> 0` stayed green when the number was off by a factor of 100.
    # The expected value is derivable exactly: unrealized gain, 70 % of it
    # taxable (30 % Teilfreistellung), taxed at 26.375 %.
    expected_liq_tax = (
        result.tax_summary.unrealized_gain_at_end * 0.70 * 0.26375)
    assert result.tax_summary.tax_paid_liquidation == pytest.approx(
        expected_liq_tax, abs=0.01)
    assert result.tax_summary.tax_paid_liquidation == pytest.approx(6148.01, abs=0.01)
    assert result.equity_curve.iloc[-1] < result.equity_curve_net.iloc[-1]
    pd.testing.assert_series_equal(metric_curve(result), result.equity_curve)
    assert result.equity_curve_daily.iloc[-1] < result.equity_curve_daily_net.iloc[-1]
