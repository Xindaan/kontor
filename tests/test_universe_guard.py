"""The universe look-ahead guard is wired.

`BacktestConfig.allow_universe_lookahead` existed but was read by nobody, and
`universe.validate_universe_for_backtest` had no caller in active code. FATAL
now aborts even without `strict_validation` -- otherwise the guard would only be
a warning, i.e. no protection.
"""
import datetime as dt

import pandas as pd
import pytest

from backtest.backtester import BacktestConfig, Backtester
from backtest.data import PriceData
from backtest.strategy import Allocation, Strategy
from backtest.universe import UniverseSnapshot


def _data(start='2013-01-01', n=120):
    idx = pd.bdate_range(start, periods=n)
    df = pd.DataFrame({'A': [100.0] * n}, index=idx)
    return PriceData(prices=df, currency={'A': 'EUR'}, fx_rates=pd.Series(1.0, index=idx))


class _WithUniverse(Strategy):
    """Strategy that draws its universe from a non-PIT source."""

    name = 'with_universe'
    assets = ['A']
    rebalance_frequency = 'monthly'

    def __init__(self, point_in_time=False, as_of=dt.date(2024, 1, 1)):
        super().__init__()
        self.universe_snapshot = UniverseSnapshot(
            as_of=as_of, tickers=['A'], source='yahoo_screener',
            point_in_time=point_in_time)

    def signal(self, date, data):
        return Allocation({'A': 1.0})


def _run(strategy, **cfg_kwargs):
    cfg = BacktestConfig(initial_capital=10_000.0, tax_enabled=False,
                         benchmark=None, validate=True, **cfg_kwargs)
    return Backtester(strategy, _data(), cfg).run()


class TestLookaheadIsBlocked:
    def test_future_universe_aborts(self):
        """Backtest from 2013 with a member list from 2024."""
        with pytest.raises(ValueError, match='look-ahead'):
            _run(_WithUniverse(point_in_time=False))

    def test_abort_also_without_strict_validation(self):
        assert BacktestConfig().strict_validation is False
        with pytest.raises(ValueError, match='look-ahead'):
            _run(_WithUniverse(point_in_time=False), strict_validation=False)

    def test_flag_lifts_the_guard(self):
        res = _run(_WithUniverse(point_in_time=False), allow_universe_lookahead=True)
        assert res.equity_curve.iloc[-1] > 0

    def test_pit_universe_passes(self):
        res = _run(_WithUniverse(point_in_time=True))
        assert res.equity_curve.iloc[-1] > 0

    def test_strategy_without_universe_is_unaffected(self):
        class Without(Strategy):
            name = 'without'
            assets = ['A']
            rebalance_frequency = 'monthly'

            def signal(self, date, data):
                return Allocation({'A': 1.0})

        assert _run(Without()).equity_curve.iloc[-1] > 0

    def test_private_attribute_is_found_too(self):
        class Private(Strategy):
            name = 'private'
            assets = ['A']
            rebalance_frequency = 'monthly'

            def __init__(self):
                super().__init__()
                self._universe_snapshot = UniverseSnapshot(
                    as_of=dt.date(2024, 1, 1), tickers=['A'],
                    source='yahoo_screener', point_in_time=False)

            def signal(self, date, data):
                return Allocation({'A': 1.0})

        with pytest.raises(ValueError, match='look-ahead'):
            _run(Private())
