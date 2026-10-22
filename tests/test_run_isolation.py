"""A run must color neither the caller nor the next run.

The backtester treated things it OWNS (config, strategy state, curve end) as if
they were shared -- and things that FAIL (missing price, missing lot) as if they
were the normal case. Every test here checks BEHAVIOR (number, call count,
warning), not source text.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from backtest.backtester import BacktestConfig, Backtester
from backtest.data import PriceData
from backtest.strategy import Allocation, Strategy


def _series(n=260, start='2020-01-01', growth=1.004):
    idx = pd.bdate_range(start, periods=n)
    return idx, pd.Series([100.0 * (growth ** i) for i in range(n)], index=idx)


def _data(n=260):
    idx, a = _series(n)
    b = pd.Series([100.0] * n, index=idx)
    df = pd.DataFrame({'A': a, 'B': b})
    return PriceData(prices=df, currency={'A': 'EUR', 'B': 'EUR'},
                     fx_rates=pd.Series(1.0, index=idx))


class _Counting(Strategy):
    """Holds A and counts how often it was asked for a signal."""

    name = 'counting'
    assets = ['A', 'B']
    rebalance_frequency = 'weekly'

    def __init__(self):
        super().__init__()
        self.calls = 0

    def signal(self, date, data):
        self.calls += 1
        return Allocation({'A': 1.0})


class _CountingMonthly(_Counting):
    name = 'counting_monthly'
    rebalance_frequency = 'monthly'


def _cfg(**kw):
    base = dict(initial_capital=10_000.0, validate=False, tax_enabled=False, benchmark=None)
    base.update(kw)
    return BacktestConfig(**base)


class TestConfigBelongsToTheCaller:
    def test_passed_config_stays_unchanged(self):
        cfg = _cfg()
        assert cfg.rebalance_frequency is None
        Backtester(_Counting(), _data(), cfg).run()
        assert cfg.rebalance_frequency is None, 'The run mutated the caller config'

    def test_second_strategy_keeps_its_own_frequency(self):
        """The actual damage: run 2 inherited the frequency of run 1."""
        d, cfg = _data(), _cfg()
        weekly = _Counting()
        Backtester(weekly, d, cfg).run()
        monthly = _CountingMonthly()
        Backtester(monthly, d, cfg).run()
        # 260 trading days = ~52 weeks, ~12 months.
        assert weekly.calls > 40
        assert monthly.calls < 20, 'The monthly strategy ran weekly -- frequency inherited'

    def test_effective_frequency_is_in_the_result(self):
        res = Backtester(_Counting(), _data(), _cfg()).run()
        assert res.config.rebalance_frequency == 'weekly'


class _StickyMini(Strategy):
    """Minimal hysteresis: stays with what it has once."""

    name = 'sticky_mini'
    assets = ['A', 'B']
    rebalance_frequency = 'monthly'

    def __init__(self):
        super().__init__()
        self._current = None

    def reset(self):
        super().reset()
        self._current = None

    def signal(self, date, data):
        if self._current is None:
            self._current = 'A'
        return Allocation({self._current: 1.0})


class TestStateDoesNotSurviveTheRun:
    def test_default_reset_is_a_no_op(self):
        s = _Counting()
        s.reset()
        assert s.calls == 0

    def test_backtester_calls_reset(self):
        s = _StickyMini()
        s._current = 'B'  # state from an imagined previous run
        Backtester(s, _data(), _cfg()).run()
        assert s._current == 'A', 'The run took over foreign state'

    def test_sticky_levered_twin_resets_its_overlay_state(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / 'strategies' / 'sticky_levered_vol_targeted.py'
        spec = importlib.util.spec_from_file_location('slvt_reset_probe', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        s = module.StickyLeveredVolTargeted()
        s._current, s._hold_periods, s._picked = 'TQQQ', 3, 'TQQQ'
        s._last_pick_month, s._cur_risky_weight = (2020, 1), 0.7
        s.reset()
        assert (s._current, s._hold_periods, s._picked) == (None, 0, None)
        assert s._last_pick_month is None and s._cur_risky_weight == 0.0


class _FixedTarget(Strategy):
    name = 'fixed_target'
    assets = ['A', 'B']
    rebalance_frequency = 'weekly'

    def signal(self, date, data):
        return Allocation({'A': 0.5, 'B': 0.5})


class TestMissingPriceIsNotNormal:
    """A trade without a price is dropped -- visibly, not silently."""

    def _data_with_gap(self):
        idx, a = _series(120)
        b = pd.Series([100.0] * 120, index=idx)
        df = pd.DataFrame({'A': a, 'B': b})
        # B has no price before day 60 (listed later) -> NaN, not 0.
        df.loc[df.index[:60], 'B'] = np.nan
        return PriceData(prices=df, currency={'A': 'EUR', 'B': 'EUR'},
                         fx_rates=pd.Series(1.0, index=idx))

    def _run(self):
        cfg = _cfg(initial_capital=100_000.0, rebalance_frequency='weekly')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = Backtester(_FixedTarget(), self._data_with_gap(), cfg).run()
        return res, [str(w.message) for w in caught]

    def test_skip_is_counted(self):
        res, _ = self._run()
        liq = (res.constraint_impact or {}).get('liquidity', {})
        assert liq.get('skipped_invalid_price', 0) > 0, 'The buy without a price was skipped silently'

    def test_skip_is_reported(self):
        _, messages = self._run()
        assert any('skipped_invalid_price' in m for m in messages)

    def test_diagnostics_stay_computable(self):
        """NaN used to contaminate the field for the whole run."""
        res, _ = self._run()
        liq = (res.constraint_impact or {}).get('liquidity', {})
        assert np.isfinite(liq.get('requested_shares_total', float('nan')))


class TestBandRecordsTheHoldings:
    def test_skipped_date_does_not_show_the_target(self):
        cfg = _cfg(initial_capital=100_000.0, costs_pct=0.0, slippage_pct=0.0,
                   rebalance_frequency='weekly', rebalance_min_deviation=0.20)
        res = Backtester(_FixedTarget(), _data(260), cfg).run()
        # A rises, B is flat: the actual weights drift away from 50/50 while the
        # target stays exactly 0.5. Any entry != 0.5 comes from the holdings.
        a_weights = res.allocations['A'].dropna()
        assert len(a_weights) > 5
        assert (a_weights - 0.5).abs().max() > 0.01, 'History shows the target everywhere'


class _Alternating(Strategy):
    """Trades on every date so the last trade carries costs."""

    name = 'alternating'
    assets = ['A', 'B']
    rebalance_frequency = 'weekly'

    def __init__(self):
        super().__init__()
        self.k = 0

    def signal(self, date, data):
        self.k += 1
        return Allocation({'A': 1.0} if self.k % 2 else {'B': 1.0})


class TestCurveEndsAfterTheLastTrade:
    def test_curve_end_equals_final_value(self):
        cfg = _cfg(initial_capital=100_000.0, tax_enabled=True, tax_exemption_amount=0.0,
                   rebalance_frequency='weekly', metric_basis='net_realized')
        res = Backtester(_Alternating(), _data(260), cfg).run()
        assert res.tax_summary is not None
        assert res.equity_curve.iloc[-1] == pytest.approx(
            res.tax_summary.final_value_net_realized, rel=1e-9)


class _HoldThenSwitch(Strategy):
    """Holds A through all dividends and sells it EXACTLY ONCE, so one sale
    carries the whole tax -- if it fails, the number drops to 0."""

    name = 'hold_then_switch'
    assets = ['A', 'B']
    rebalance_frequency = 'monthly'

    def __init__(self):
        super().__init__()
        self.k = 0

    def reset(self):
        super().reset()
        self.k = 0

    def signal(self, date, data):
        self.k += 1
        return Allocation({'A': 1.0} if self.k <= 6 else {'B': 1.0})


class TestDripBooksALot:
    """Reinvested shares are purchases for tax purposes."""

    def _run(self, drip):
        n = 300
        idx = pd.bdate_range('2020-01-01', periods=n)
        a = pd.Series(np.linspace(100.0, 150.0, n), index=idx)
        b = pd.Series([100.0] * n, index=idx)
        div = pd.DataFrame(0.0, index=idx, columns=['A', 'B'])
        for day in ('2020-02-14', '2020-03-13', '2020-04-15', '2020-05-15'):
            div.loc[pd.Timestamp(day), 'A'] = 1.0
        d = PriceData(prices=pd.DataFrame({'A': a, 'B': b}),
                      currency={'A': 'EUR', 'B': 'EUR'},
                      fx_rates=pd.Series(1.0, index=idx), dividends=div)
        cfg = _cfg(initial_capital=100_000.0, costs_pct=0.0, slippage_pct=0.0,
                   tax_enabled=True, tax_exemption_amount=0.0, drip_enabled=drip,
                   metric_basis='net_realized')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            res = Backtester(_HoldThenSwitch(), d, cfg).run()
        return res, [str(w.message) for w in caught]

    def test_sale_after_drip_is_taxed(self):
        without, _ = self._run(drip=False)
        with_drip, _ = self._run(drip=True)
        tax_without = sum(t.tax_paid_trade for t in without.trades if t.action == 'SELL')
        tax_with = sum(t.tax_paid_trade for t in with_drip.trades if t.action == 'SELL')
        assert tax_without > 1000.0, 'fixture carries no sale tax'
        # DRIP holds slightly more shares -> slightly higher tax. The tight bound
        # is deliberate: the failure was total (0.00).
        assert tax_with == pytest.approx(tax_without, rel=0.02), (
            'DRIP shares missing in the tax model: %.2f instead of ~%.2f' % (tax_with, tax_without))

    def test_no_tax_inconsistency_reported(self):
        _, messages = self._run(drip=True)
        assert not [m for m in messages if 'Steuer-Inkonsistenz' in m]


class TestLotHoldingsEqualPortfolioHoldings:
    """The book is kept in two places -- `portfolio.positions` and the tax lots.
    Any holdings change without a lot makes them diverge; this check catches the
    NEXT such change before a sale goes through untaxed."""

    def _run_with_tampered_holdings(self, tamper):
        from backtest.backtester import Portfolio

        cfg = BacktestConfig(initial_capital=100_000.0, validate=True, tax_enabled=True,
                             tax_exemption_amount=0.0, benchmark=None,
                             rebalance_frequency='monthly', metric_basis='net_realized')
        real_execute = Portfolio.execute_buys
        state = {'done': False}

        def patched(self, *args, **kwargs):
            trades = real_execute(self, *args, **kwargs)
            if tamper and not state['done'] and self.positions:
                # Put shares WITHOUT a lot into the portfolio -- the DRIP case.
                ticker = sorted(self.positions)[0]
                self.positions[ticker] += 10.0
                state['done'] = True
            return trades

        Portfolio.execute_buys = patched
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                Backtester(_FixedTarget(), _data(120), cfg).run()
            return [str(w.message) for w in caught]
        finally:
            Portfolio.execute_buys = real_execute

    def test_clean_run_reports_nothing(self):
        messages = self._run_with_tampered_holdings(tamper=False)
        assert not [m for m in messages if 'tax-lot holdings differ' in m]

    def test_holdings_without_lot_are_reported(self):
        messages = self._run_with_tampered_holdings(tamper=True)
        hits = [m for m in messages if 'tax-lot holdings differ' in m]
        assert hits, 'Holdings without a lot went unnoticed'
        assert 'UNTAXED' in hits[0]
