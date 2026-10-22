"""Cash, valuation and dividend-entitlement regressions of the backtester."""
from datetime import date

import pandas as pd
import pytest

from backtest.backtester import BacktestConfig, Backtester, TaxSummary
from backtest.data import PriceData
from backtest.metrics import MetricsCalculator
from backtest.strategy import Strategy, Allocation
from backtest.validation import validate_tax_invariants


class Scheduled(Strategy):
    name = 'Audit scheduled'

    def __init__(self, schedule, assets):
        self.schedule, self.assets = schedule, assets

    def signal(self, current_date, data):
        return Allocation(self.schedule(current_date))


def run(prices, schedule, dividends=None, **overrides):
    settings = dict(initial_capital=10000, benchmark=None, currency='EUR',
                    costs_pct=0., slippage_pct=0., spread_pct=0., tax_enabled=True,
                    tax_exemption_amount=0., rebalance_frequency='daily', validate=False,
                    tax_treatment_map={t: ('general', False) for t in prices})
    settings.update(overrides)
    return Backtester(Scheduled(schedule, list(prices)), PriceData(
        prices=prices, currency={t: 'EUR' for t in prices}, dividends=dividends,
    ), BacktestConfig(**settings)).run()


@pytest.mark.parametrize('tax', [False, True])
@pytest.mark.parametrize('basis', ['net_realized', 'net_liquidation'])
def test_final_t_plus_one_trade_cost_is_in_daily_end_value(tax, basis):
    dates = pd.bdate_range('2024-01-02', periods=5)
    r = run(pd.DataFrame({'AUDITA': 100.}, index=dates),
            lambda d: {'AUDITA': 1.} if d >= dates[-2].date() else {},
            execution_lag_days=1, costs_pct=.01, tax_enabled=tax, metric_basis=basis)
    assert len(r.trades) == 1
    assert r.trades[0].date == dates[-1]
    assert r.equity_curve_daily.iloc[-1] == pytest.approx(9900.990099009903)
    assert r.equity_curve_daily.iloc[-1] == pytest.approx(r.tax_summary.final_value_for_metric_basis)
    assert r.metrics_daily.cagr == pytest.approx(r.tax_summary.cagr_for_metric_basis)


def test_actual_final_liquidation_is_reflected_in_each_daily_curve():
    dates = pd.bdate_range('2024-01-02', periods=5)
    r = run(pd.DataFrame({'AUDITA': [100, 110, 120, 130, 140]}, index=dates),
            lambda d: {'AUDITA': 1.}, costs_pct=.01, actual_liquidation_at_end=True)
    assert r.trades[-1].action == 'SELL'
    assert r.equity_curve_daily.index.is_unique
    assert r.equity_curve_daily_gross.iloc[-1] == pytest.approx(r.tax_summary.final_value_gross)
    assert r.equity_curve_daily_net.iloc[-1] == pytest.approx(r.tax_summary.final_value_net_realized)
    assert r.equity_curve_daily.iloc[-1] == pytest.approx(r.tax_summary.final_value_net_liquidation)


@pytest.mark.parametrize('winner_first', [True, False])
def test_virtual_liquidation_does_not_depend_on_ticker_name(winner_first):
    dates = pd.bdate_range('2024-01-02', periods=5)
    win, lose = [100, 100, 120, 120, 120], [100, 100, 80, 80, 80]
    prices = pd.DataFrame({'AUDITA': win if winner_first else lose,
                           'AUDITZ': lose if winner_first else win}, index=dates)
    r = run(prices, lambda d: {'AUDITA': .5, 'AUDITZ': .5}, rebalance_min_deviation=.49)
    assert r.tax_summary.tax_paid_liquidation == pytest.approx(0.)
    assert r.equity_curve_daily.to_numpy() == pytest.approx([10000.] * 5)
    assert r.tax_summary.final_value_net_liquidation == pytest.approx(10000.)


@pytest.mark.parametrize('lag', [1, 2])
def test_purchase_on_or_after_ex_date_has_no_dividend_entitlement(lag):
    dates = pd.bdate_range('2024-01-02', periods=5)
    r = run(pd.DataFrame({'AUDITA': [100, 90, 90, 90, 90]}, index=dates),
            lambda d: {'AUDITA': 1.},
            dividends=pd.DataFrame({'AUDITA': [0, 10, 0, 0, 0]}, index=dates),
            execution_lag_days=lag, tax_enabled=False, rebalance_min_deviation=.49)
    assert not r.dividend_events
    assert r.equity_curve_daily.iloc[-1] == pytest.approx(10000.)


def test_sale_on_ex_date_retains_dividend_on_pretrade_shares():
    dates = pd.bdate_range('2024-01-02', periods=5)
    r = run(pd.DataFrame({'AUDITA': [100, 100, 90, 90, 90]}, index=dates),
            lambda d: {'AUDITA': 1.} if d == dates[0].date() else {},
            dividends=pd.DataFrame({'AUDITA': [0, 0, 10, 0, 0]}, index=dates),
            execution_lag_days=1, tax_enabled=False)
    assert r.trades[0].date == dates[1]
    assert r.trades[1].date == dates[2]
    assert len(r.dividend_events) == 1
    assert r.dividend_events[0].gross_amount == pytest.approx(999.)
    assert r.equity_curve_daily.iloc[-1] == pytest.approx(10000.)


def test_refund_cash_reaches_next_purchase_and_net_metrics_keep_earlier_tax_path():
    dates = pd.bdate_range('2024-01-02', periods=6)
    prices = pd.DataFrame({'AUDITA': [100, 120, 120, 120, 120, 120],
                           'AUDITB': [100, 100, 80, 80, 80, 80],
                           'AUDITC': [100] * 6}, index=dates)
    r = run(prices, lambda d: {'AUDITA' if d == dates[0].date() else
                              'AUDITB' if d == dates[1].date() else 'AUDITC': 1.},
            metric_basis='net_realized')
    sells = [t for t in r.trades if t.action == 'SELL']
    assert sells[0].tax_paid_trade > 0
    assert sells[1].tax_paid_trade == pytest.approx(-sells[0].tax_paid_trade)
    assert r.tax_summary.total_tax_paid == pytest.approx(0.)
    holdings = {t: 0. for t in prices}
    cash = 10000.
    for t in r.trades:
        holdings[t.ticker] += t.shares if t.action == 'BUY' else -t.shares
        cash += t.cash_flow_after_tax
    expected = cash + sum(q * prices[t].iloc[-1] for t, q in holdings.items())
    assert r.equity_curve_daily.iloc[-1] == pytest.approx(expected)
    assert r.metrics_daily_net.max_drawdown == pytest.approx(MetricsCalculator.max_drawdown(r.equity_curve_daily_net))
    assert r.metrics_daily_net.max_drawdown != pytest.approx(r.metrics_daily_gross.max_drawdown)


def test_refund_liquidation_validates_by_identity_instead_of_monotonicity():
    summary = TaxSummary(total_tax_paid=100, final_value_gross=1000,
                         final_value_net_realized=900, tax_paid_liquidation=-75,
                         final_value_net_liquidation=975)
    assert validate_tax_invariants(summary).is_valid
    summary.final_value_net_liquidation = 1000
    assert validate_tax_invariants(summary).has_errors
    summary.tax_paid_liquidation = -101
    summary.final_value_net_liquidation = 1001
    assert validate_tax_invariants(summary).has_errors


def test_gross_diagnostic_identifies_nominal_addback_not_tax_free_counterfactual():
    dates = pd.bdate_range('2024-01-02', periods=5)
    prices = pd.DataFrame({'AUDITA': [100, 200, 200, 200, 200],
                           'AUDITB': [100, 100, 200, 200, 200]}, index=dates)
    schedule = lambda d: {'AUDITA' if d == dates[0].date() else 'AUDITB': 1.}
    taxed = run(prices, schedule)
    tax_free = run(prices, schedule, tax_enabled=False)
    assert taxed.tax_summary.final_value_gross == pytest.approx(
        taxed.tax_summary.final_value_net_realized + taxed.tax_summary.total_tax_paid)
    assert taxed.tax_summary.final_value_gross < tax_free.tax_summary.final_value_gross - 2500
    assert taxed.tax_summary.to_dict()['gross_basis'] == 'net_plus_nominal_tax'
    assert taxed.tax_summary.to_dict()['tax_drag_basis'] == 'nominal_tax_addback'
    assert taxed.tax_summary.to_dict()['tax_settlement'] == 'per_event_calendar_year'
    assert 'not a tax-free comparison' in taxed.summary()
