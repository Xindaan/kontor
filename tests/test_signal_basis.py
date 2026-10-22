"""The live signal path measures the same thing as the backtest.

- The stale check used to see only HELD securities. A frozen candidate,
  however, distorts exactly the two quantities the signal is built from --
  the momentum comparison and the realized volatility.
- The strategy used to get NATIVE prices (USD for US-listed ETFs), while the
  backtester converts to EUR before every `signal()`.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from backtest.data import PriceData
from backtest.signals import Portfolio, SignalGenerator
from backtest.strategy import Allocation, Strategy


class _RecordsPrices(Strategy):
    """Remembers which frame it actually saw."""

    name = 'records_prices'
    assets = ['USD_A', 'EUR_B']
    rebalance_frequency = 'daily'

    def __init__(self):
        super().__init__()
        self.seen = None

    def signal(self, current_date, data):
        self.seen = data.copy()
        return Allocation({'USD_A': 1.0})


def _price_data(days=40, last_print=None):
    """USD_A quotes in USD, EUR_B in EUR; EURUSD rises from 1.00 to 1.25."""
    idx = pd.bdate_range('2024-01-01', periods=days)
    a = pd.Series(np.linspace(100.0, 100.0, days), index=idx)   # flat in USD
    b = pd.Series(np.linspace(50.0, 50.0, days), index=idx)
    fx = pd.Series(np.linspace(1.00, 1.25, days), index=idx)    # EUR strengthens
    prints = {'USD_A': idx[-1], 'EUR_B': idx[-1]}
    if last_print:
        prints.update(last_print)
    return PriceData(
        prices=pd.DataFrame({'USD_A': a, 'EUR_B': b}),
        currency={'USD_A': 'USD', 'EUR_B': 'EUR'},
        fx_rates=pd.DataFrame({'USD': fx}),
        last_print=prints,
    )


def _with_loader(monkeypatch, data):
    def fake_yahoo(tickers, start, end=None, currency='EUR', align='ffill',
                   skip_failed=False, **kwargs):
        return data

    monkeypatch.setattr('backtest.data.DataLoader.yahoo', fake_yahoo)


class TestSignalBasisIsEuro:
    def test_strategy_sees_eur_prices(self, monkeypatch):
        _with_loader(monkeypatch, _price_data())
        s = _RecordsPrices()
        SignalGenerator(strategy=s, portfolio=Portfolio()).generate(
            as_of=dt.date(2024, 3, 1), lookback_days=60)
        assert s.seen is not None
        # USD_A is flat at 100 in USD; in EUR it falls from 100 to 80 because
        # EURUSD rises from 1.00 to 1.25.
        first, last = s.seen['USD_A'].iloc[0], s.seen['USD_A'].iloc[-1]
        assert first == pytest.approx(100.0, rel=1e-6)
        assert last == pytest.approx(80.0, rel=1e-6), (
            'The strategy computes on native USD prices instead of EUR')

    def test_eur_ticker_stays_unchanged(self, monkeypatch):
        _with_loader(monkeypatch, _price_data())
        s = _RecordsPrices()
        SignalGenerator(strategy=s, portfolio=Portfolio()).generate(
            as_of=dt.date(2024, 3, 1), lookback_days=60)
        assert s.seen['EUR_B'].iloc[-1] == pytest.approx(50.0, rel=1e-6)

    def test_native_basis_remains_as_comparison_switch(self, monkeypatch):
        _with_loader(monkeypatch, _price_data())
        s = _RecordsPrices()
        SignalGenerator(strategy=s, portfolio=Portfolio(),
                        signal_currency=None).generate(
            as_of=dt.date(2024, 3, 1), lookback_days=60)
        assert s.seen['USD_A'].iloc[-1] == pytest.approx(100.0, rel=1e-6)

    def test_default_is_eur(self):
        assert SignalGenerator(strategy=_RecordsPrices()).signal_currency == 'EUR'

    def test_unknown_currency_is_rejected(self):
        with pytest.raises(ValueError, match='signal_currency'):
            SignalGenerator(strategy=_RecordsPrices(), signal_currency='CHF')


class TestStaleCheckSeesCandidates:
    """A frozen candidate must be flagged even when not held."""

    def _warnings(self, monkeypatch, positions):
        old = pd.Timestamp('2024-01-05')   # well before the frame end
        _with_loader(monkeypatch, _price_data(last_print={'USD_A': old}))
        report = SignalGenerator(
            strategy=_RecordsPrices(),
            portfolio=Portfolio(positions=positions, cash=1000.0),
        ).generate(as_of=dt.date(2024, 3, 1), lookback_days=60)
        return [w for w in (report.price_warnings or []) if 'USD_A' in w]

    def test_unheld_candidate_warns(self, monkeypatch):
        hits = self._warnings(monkeypatch, {'EUR_B': 10.0})
        assert hits, 'Frozen candidate went unnoticed'
        assert '[signal candidate]' in hits[0]

    def test_held_position_still_warns(self, monkeypatch):
        hits = self._warnings(monkeypatch, {'USD_A': 10.0})
        assert hits
        assert 'position' in hits[0]

    def test_fresh_series_does_not_warn(self, monkeypatch):
        _with_loader(monkeypatch, _price_data())
        report = SignalGenerator(
            strategy=_RecordsPrices(),
            portfolio=Portfolio(positions={'USD_A': 10.0}, cash=1000.0),
        ).generate(as_of=dt.date(2024, 3, 1), lookback_days=60)
        assert not [w for w in (report.price_warnings or []) if 'trading days' in w]
