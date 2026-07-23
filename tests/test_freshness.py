"""Tests for the freshness invariant.

Each test here reflects a real failure instance, not a hypothetical scenario.
"""

import datetime

import pandas as pd
import pytest

from backtest.data import PriceData
from backtest.freshness import (STALE_BDAYS, DatedPrice, FreshnessError, dated_last,
                                real_prints, fx_exact, stale_lag_bdays)

FR = datetime.date(2026, 7, 17)      # Friday; last expected close = Thu 16.07.


def _preis(value, tag, source='MU', today=FR):
    return DatedPrice(value=value, session_date=datetime.date.fromisoformat(tag),
                      source=source, stale_lag=stale_lag_bdays(tag, today=today))


class TestStaleLag:
    def test_gestriger_schluss_ist_frisch(self):
        assert stale_lag_bdays('2026-07-16', today=FR) == 0

    def test_wochenende_zaehlt_nicht_als_rueckstand(self):
        assert stale_lag_bdays('2026-07-17', today=datetime.date(2026, 7, 20)) == 0

    def test_mehrtaegige_luecke_schlaegt_an(self):
        assert stale_lag_bdays('2026-07-13', today=FR) >= STALE_BDAYS


class TestDatedPrice:
    def test_frischer_preis_ist_nicht_stale(self):
        assert not _preis(760.00, '2026-07-16').stale

    def test_alter_preis_ist_stale_und_sagt_es_in_describe(self):
        p = _preis(1300.0, '2026-07-10')
        assert p.stale
        assert 'STALE' in p.describe() and 'KEIN VERDIKT' in p.describe()

    def test_describe_traegt_immer_das_datum(self):
        """A verdict without a visible price date was an invitation for errors."""
        assert 'Stand 2026-07-16' in _preis(760.00, '2026-07-16').describe()

    def test_require_fresh_wirft_statt_still_durchzulassen(self):
        with pytest.raises(FreshnessError, match='NO VERDICT'):
            _preis(1300.0, '2026-07-10').require_fresh()
        frisch = _preis(760.00, '2026-07-16')
        assert frisch.require_fresh() is frisch


class TestDatedLast:
    def test_letzter_print_traegt_sein_echtes_datum(self):
        s = pd.Series([100.0, 101.0], index=pd.to_datetime(['2026-07-15', '2026-07-16']))
        p = dated_last(s, 'MU', today=FR)
        assert (p.value, p.session_date, p.source) == (101.0, datetime.date(2026, 7, 16), 'MU')
        assert not p.stale

    def test_eingefrorene_serie_wird_als_stale_erkannt(self):
        """Incident 1 (Jun 15): frozen SOXL -> vol collapses -> signal inverts."""
        s = pd.Series([100.0, 101.0], index=pd.to_datetime(['2026-07-09', '2026-07-10']))
        assert dated_last(s, 'SOXL', today=FR).stale

    def test_leere_serie_wirft(self):
        with pytest.raises(FreshnessError, match='no price data'):
            dated_last(pd.Series(dtype=float), 'MU', today=FR)


def _fake_yahoo(monkeypatch, frame, volumes=None):
    def fake(**kwargs):
        cols = [t for t in kwargs['tickers'] if t in frame.columns]
        return PriceData(prices=frame[cols].copy(),
                         volumes=None if volumes is None else volumes[cols].copy(),
                         currency={c: 'EUR' for c in cols})
    monkeypatch.setattr('backtest.freshness.DataLoader.yahoo', staticmethod(fake))


class TestEchtePrints:
    def test_heute_bar_wird_verworfen(self, monkeypatch):
        """Rule: always use the end-of-day close, NEVER intraday. A real
        observation showed the intraday price diverging from the prior
        close by roughly 3.5% -- enough to trigger falsely on a dip
        level."""
        _fake_yahoo(monkeypatch, pd.DataFrame(
            {'A1P0.DE': [7.65, 7.38]}, index=pd.to_datetime(['2026-07-16', '2026-07-17'])))
        s = real_prints('A1P0.DE', '2026-06-01', '2026-07-18', today=FR)
        assert list(s.index.date) == [datetime.date(2026, 7, 16)]
        assert float(s.iloc[-1]) == 7.65

    def test_heute_bar_kann_bewusst_behalten_werden(self, monkeypatch):
        _fake_yahoo(monkeypatch, pd.DataFrame(
            {'A1P0.DE': [7.65, 7.38]}, index=pd.to_datetime(['2026-07-16', '2026-07-17'])))
        s = real_prints('A1P0.DE', '2026-06-01', '2026-07-18', today=FR, drop_today=False)
        assert float(s.iloc[-1]) == 7.38

    def test_nullumsatz_zeilen_sind_platzhalter(self, monkeypatch):
        idx = pd.to_datetime(['2026-07-15', '2026-07-16'])
        _fake_yahoo(monkeypatch,
                    pd.DataFrame({'EXFC.MU': [1300.0, 1310.0]}, index=idx),
                    volumes=pd.DataFrame({'EXFC.MU': [12.0, 0.0]}, index=idx))
        s = real_prints('EXFC.MU', '2026-06-01', '2026-07-18', today=FR, require_volume=True)
        assert list(s.index.date) == [datetime.date(2026, 7, 15)]

    def test_unerreichbare_quelle_wirft(self, monkeypatch):
        _fake_yahoo(monkeypatch, pd.DataFrame(
            {'X': [1.0]}, index=pd.to_datetime(['2026-07-16'])))
        with pytest.raises(FreshnessError, match='not loaded'):
            real_prints('MU', '2026-06-01', '2026-07-18', today=FR)


class TestFxExakt:
    FX = pd.Series([1.1470258, 1.1440338],
                   index=pd.to_datetime(['2026-07-16', '2026-07-17']))

    def test_exakte_zeile_wird_genommen(self):
        assert fx_exact(self.FX, '2026-07-16', today=FR) == pytest.approx(1.1470258)

    def test_fehlende_zeile_wird_nicht_stillschweigend_vorwaerts_gefuellt(self):
        """`asof()` would have used the 07-16 price for 07-15 here."""
        with pytest.raises(FreshnessError, match='no quote'):
            fx_exact(self.FX, '2026-07-15', today=FR)

    def test_heutige_zeile_ist_ein_teiltages_wert(self):
        """fx_EURUSDX.csv carries a 07-17 row, written at 07:02 --
        the FX day runs until ~23:00 CET."""
        with pytest.raises(FreshnessError, match='in-progress FX day'):
            fx_exact(self.FX, '2026-07-17', today=FR)

    def test_ungueltige_quote_wirft(self):
        fx = pd.Series([float('nan')], index=pd.to_datetime(['2026-07-16']))
        with pytest.raises(FreshnessError, match='invalid quote'):
            fx_exact(fx, '2026-07-16', today=FR)


class TestLoaderMerktDenLetztenPrint:
    """The PRODUCER records the real print date before `align="ffill"`
    makes it unrecognizable. After that, the frame can no longer tell
    whether the tail value is today's price or a carried-forward old one --
    exactly what caused incident 1 (Jun 15, frozen SOXL inverted the
    signal)."""

    def test_ffill_frame_traegt_das_echte_print_datum_je_ticker(self, monkeypatch, tmp_path):
        from backtest.data import DataLoader

        idx = pd.bdate_range('2026-07-13', periods=4)      # Mo..Do
        roh = {'FRISCH': [10.0, 11.0, 12.0, 13.0],
               'FROZEN': [100.0, 101.0, None, None]}       # letzter Print Di 14.07.

        def fake_download(tickers, start=None, end=None, progress=False,
                          auto_adjust=True, group_by='column'):
            # yfinance returns MultiIndex columns for multiple tickers.
            return pd.DataFrame({('Close', t): roh[t] for t in tickers}, index=idx)

        monkeypatch.setattr('backtest.data.yf.download', fake_download)
        monkeypatch.setattr(DataLoader, 'CACHE_DIR', tmp_path)
        DataLoader._MEMORY_CACHE.clear()

        pdta = DataLoader.yahoo(tickers=['FRISCH', 'FROZEN'], start='2026-07-13',
                                end='2026-07-17', currency='USD', align='ffill',
                                cache=False, validate=False)

        # The ffill makes both columns look "current" on Thursday ...
        assert float(pdta.prices['FROZEN'].iloc[-1]) == 101.0
        assert pdta.prices.index.max().date() == datetime.date(2026, 7, 16)
        # ... only last_print knows the truth.
        assert pdta.last_print['FRISCH'].date() == datetime.date(2026, 7, 16)
        assert pdta.last_print['FROZEN'].date() == datetime.date(2026, 7, 14)


class TestStaleLagWochenende:
    """Finding 2026-07-19: ``roll='backward'`` rolled back TWICE over the weekend.

    Failure class: the reference day (last expected close) was one trading
    day too early on Saturdays/Sundays -> every print looked one day fresher
    than it was -> FAIL-OPEN in the freshness gate. Concretely, a Wednesday
    print passed the gate on Sunday (lag 1) that it would have been blocked
    by on Monday (lag 2).
    """

    @pytest.mark.parametrize('print_tag', [
        datetime.date(2026, 7, 17),   # Fr - frischest moeglich
        datetime.date(2026, 7, 16),   # Do
        datetime.date(2026, 7, 15),   # Wed - the case that used to flip
        datetime.date(2026, 7, 14),   # Di
    ])
    def test_wochenende_urteilt_wie_folgemontag(self, print_tag):
        """The same print must be judged the same on Sat/Sun as on the following Monday."""
        samstag = stale_lag_bdays(print_tag, today=datetime.date(2026, 7, 18))
        sonntag = stale_lag_bdays(print_tag, today=datetime.date(2026, 7, 19))
        montag = stale_lag_bdays(print_tag, today=datetime.date(2026, 7, 20))
        assert samstag == sonntag == montag, (
            'Weekend verdict differs from Monday verdict: '
            'Sat=%d Sun=%d Mon=%d' % (samstag, sonntag, montag))

    def test_freitagsprint_am_wochenende_ist_lag_null(self):
        """Friday is the freshest possible close over the weekend -> lag 0, never negative."""
        for heute in (datetime.date(2026, 7, 18), datetime.date(2026, 7, 19)):
            assert stale_lag_bdays(datetime.date(2026, 7, 17), today=heute) == 0

    def test_kein_negativer_lag_fuer_vergangene_prints(self):
        """A print from the past must never report a negative lag."""
        for tag in range(13, 18):
            for heute in (datetime.date(2026, 7, 18), datetime.date(2026, 7, 19), datetime.date(2026, 7, 20)):
                assert stale_lag_bdays(datetime.date(2026, 7, tag), today=heute) >= 0
