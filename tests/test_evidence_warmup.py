"""OOS windows start warm, not cold.

`run_meta_evidence_analysis` cut train and test windows exactly at their
boundaries. On the first test day a strategy with 63-day momentum therefore saw a
single row of history and fell back to its safe asset -- for the first weeks of
every window. That biases not only the level but the DIRECTION of
`oos_degradation_pct`: the five-year train window dilutes the cold start, the
one-year test window is dominated by it.
"""
import inspect

import numpy as np
import pandas as pd

from backtest.data import PriceData
from backtest.meta_evidence import _run_backtest_bundle, run_meta_evidence_analysis
from backtest.strategy import Allocation, Strategy


class _NeedsHistory(Strategy):
    """Only enters the market after seeing 60 days of history."""

    name = 'needs_history'
    assets = ['RISKY', 'SAFE']
    rebalance_frequency = 'weekly'

    def __init__(self):
        super().__init__()
        self.cold = 0
        self.warm = 0

    def reset(self):
        super().reset()
        self.cold = 0
        self.warm = 0

    def signal(self, current_date, data):
        if len(data.dropna()) < 60:
            self.cold += 1
            return Allocation({'SAFE': 1.0})
        self.warm += 1
        return Allocation({'RISKY': 1.0})


def _data(n=400):
    idx = pd.bdate_range('2020-01-01', periods=n)
    # RISKY rises clearly, SAFE is flat -> a cold start measurably loses return.
    risky = pd.Series(100.0 * (1.002 ** np.arange(n)), index=idx)
    safe = pd.Series([100.0] * n, index=idx)
    return PriceData(prices=pd.DataFrame({'RISKY': risky, 'SAFE': safe}),
                     currency={'RISKY': 'EUR', 'SAFE': 'EUR'},
                     fx_rates=pd.Series(1.0, index=idx))


WINDOW_START = pd.Timestamp('2021-01-04')


class TestWarmupInsideTheWindow:
    def _run(self, with_warmup):
        d = _data()
        if with_warmup:
            cut = d.filter_dates(WINDOW_START - pd.Timedelta(days=365), d.prices.index[-1])
        else:
            cut = d.filter_dates(WINDOW_START, d.prices.index[-1])
        s = _NeedsHistory()
        bundle = _run_backtest_bundle(s, cut, rebalance_frequency='weekly',
                                      measure_start=WINDOW_START if with_warmup else None)
        return s, bundle

    def test_without_warmup_the_strategy_starts_cold(self):
        s, _ = self._run(with_warmup=False)
        assert s.cold > 0, 'fixture produces no cold start'

    def test_with_warmup_the_measured_window_is_warm(self):
        s, bundle = self._run(with_warmup=True)
        assert s.warm > s.cold
        assert bundle['equity_curve'].index[0] >= WINDOW_START, (
            'The measured curve still contains warm-up')

    def test_cold_start_costs_measurable_return(self):
        _, cold = self._run(with_warmup=False)
        _, warm = self._run(with_warmup=True)
        assert warm['metrics']['cagr'] > cold['metrics']['cagr'] + 0.005

    def test_measure_start_cuts_the_curve(self):
        d = _data()
        without = _run_backtest_bundle(_NeedsHistory(), d, rebalance_frequency='weekly')
        with_ = _run_backtest_bundle(_NeedsHistory(), d, rebalance_frequency='weekly',
                                     measure_start=WINDOW_START)
        assert without['equity_curve'].index[0] < WINDOW_START
        assert with_['equity_curve'].index[0] >= WINDOW_START
        assert len(with_['equity_curve']) < len(without['equity_curve'])


def test_warm_start_is_the_default():
    sig = inspect.signature(run_meta_evidence_analysis)
    assert sig.parameters['warmup_days'].default == 365


def _static_strategy(path, ticker):
    path.write_text('\n'.join([
        'from backtest.strategy import Strategy, Allocation',
        'class S(Strategy):',
        "    assets = ['%s']" % ticker,
        "    rebalance_frequency = 'monthly'",
        "    name = 'S %s'" % ticker,
        '    def signal(self, current_date, data):',
        "        return Allocation({'%s': 1.0})" % ticker,
        'strategy = S()',
    ]), encoding='utf-8')
    return str(path)


def _fake_yahoo(tickers, start, end=None, currency='EUR', align='ffill',
                skip_failed=False, **kwargs):
    tickers = list(tickers)
    idx = pd.bdate_range(pd.to_datetime(start), pd.to_datetime(end or '2024-12-31'))
    steps = np.arange(len(idx), dtype=float)
    prices = {}
    for t in tickers:
        rate = 0.0006 if t == 'BBB' else 0.0002
        prices[t] = np.maximum(100.0 * (1.0 + rate * steps), 1.0)
    return PriceData(prices=pd.DataFrame(prices, index=idx),
                     currency={t: 'USD' for t in tickers})


class TestWindowGetsWarmup:
    """The wiring, not only the helper: does the warm-up reach the window?"""

    DATA_START = pd.Timestamp('2015-01-01')

    def _windows(self, monkeypatch, tmp_path, warmup_days):
        seen = []
        import backtest.meta_evidence as me
        real_bundle = me._run_backtest_bundle

        def spy(strategy, data, **kwargs):
            seen.append({'first_day': data.prices.index[0],
                         'measure_start': kwargs.get('measure_start')})
            return real_bundle(strategy, data, **kwargs)

        monkeypatch.setattr(me, '_run_backtest_bundle', spy)
        monkeypatch.setattr('backtest.data.DataLoader.yahoo', _fake_yahoo)
        me.run_meta_evidence_analysis(
            current_strategy=_static_strategy(tmp_path / 'a.py', 'AAA'),
            target_strategy=_static_strategy(tmp_path / 'b.py', 'BBB'),
            start_date='2015-01-01', as_of='2024-01-01',
            train_years=3.0, test_years=1.0, step_months=24,
            save_artifact=False, warmup_days=warmup_days,
        )
        return seen

    def _with_available_warmup(self, seen):
        # The FIRST train window starts at the data start; nothing to warm up on.
        return [g for g in seen if g['measure_start'] is not None
                and (pd.Timestamp(g['measure_start']) - self.DATA_START).days > 400]

    def test_test_window_starts_before_the_measurement(self, monkeypatch, tmp_path):
        checkable = self._with_available_warmup(self._windows(monkeypatch, tmp_path, 365))
        assert checkable, 'no window with available warm-up in the fixture'
        for g in checkable:
            assert (pd.Timestamp(g['measure_start']) - g['first_day']).days > 300

    def test_first_window_still_runs(self, monkeypatch, tmp_path):
        assert len(self._windows(monkeypatch, tmp_path, 365)) >= 4

    def test_warmup_zero_restores_the_old_state(self, monkeypatch, tmp_path):
        checkable = self._with_available_warmup(self._windows(monkeypatch, tmp_path, 0))
        assert checkable
        for g in checkable:
            assert (pd.Timestamp(g['measure_start']) - g['first_day']).days <= 4
