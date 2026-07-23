"""Tests for the stop-monitor IO layer -- no network, no portfolio file.

This layer carries the money decision: which series is authoritative, when
the native fallback kicks in, when the broker witness overrides the
verdict. Until now it was only verified via live runs -- i.e. exactly not
on the days that matter.
"""

import datetime

import pandas as pd
import pytest

from backtest.freshness import FreshnessError
from backtest.stop_monitor import StopMonitor, stop_quote, stop_rows, stop_source

FR = datetime.date(2026, 7, 17)          # Friday; last expected close = Thu 16.07.
FX = pd.Series([1.15, 1.15, 1.15, 1.15, 1.15],
               index=pd.to_datetime(['2026-07-10', '2026-07-13', '2026-07-14',
                                     '2026-07-15', '2026-07-16']))


def _fetcher(daten):
    """Fake price source: dict ticker -> Series. Unknown = FreshnessError, just like the real thing."""
    def fetch(ticker, start, end, currency='EUR', today=None, require_volume=False,
              drop_today=True):
        if ticker not in daten:
            raise FreshnessError('%s: not loaded (source unreachable)' % ticker)
        return daten[ticker].copy()
    return fetch


def _monitor(daten, tradegate=None):
    return StopMonitor(fx_eurusd=FX, end='2026-07-18', today=FR,
                       fetch=_fetcher(daten), tradegate_close=tradegate)


def _serie(werte, tage):
    return pd.Series(werte, index=pd.to_datetime(tage), dtype=float)


FOREIGN_EUR_LINE = {'name': 'Example Foreign Corp', 'price_ticker': '900001.KS', 'waehrung': 'EUR_Linie',
         'shares': 3, 'stop_eur': 700.0, 'eur_hoch': 1000.0,
         'stop_price_ticker_eur_line': 'EXFC.MU', 'stop_price_eur_line_currency': 'EUR',
         'stop_price_ticker_native': '900001.KS', 'stop_price_native_currency': 'KRW',
         'stop_tracking_start': '2026-06-01'}

AMD = {'name': 'AMD', 'price_ticker': 'AMD', 'waehrung': 'USD->EUR', 'shares': 10,
       'stop_eur': 350.00, 'eur_hoch': 500.00,
       'stop_price_ticker_eur_line': 'AMD.DE', 'stop_price_eur_line_currency': 'EUR',
       'stop_price_ticker_native': 'AMD', 'stop_price_native_currency': 'USD',
       'stop_tracking_start': '2026-06-01'}


class TestQuellenwahl:
    def test_usd_namen_nehmen_die_native_linie_nicht_die_deutsche(self):
        """A Xetra high (17:30) against a US close (22:00) shifts
        the stop by up to 3%. The eur_line fields are documentation only
        for USD-denominated names."""
        assert stop_source(dict(AMD)) == ('native_fx', 'AMD', 'USD')

    def test_echte_eur_linie_bleibt_auf_ihrer_linie(self):
        assert stop_source(dict(FOREIGN_EUR_LINE)) == ('eur_line', 'EXFC.MU', 'EUR')

    def test_eur_line_feld_faellt_auf_den_price_ticker_zurueck(self):
        assert stop_quote({'price_ticker': 'COHR', 'waehrung': 'USD->EUR'}) == ('COHR', 'USD')

    def test_stop_rows_nimmt_nur_bestand_mit_stop(self):
        rows = stop_rows([{'stop_eur': 1.0, 'shares': 5}, {'stop_eur': 1.0, 'shares': 0},
                          {'shares': 5}])
        assert len(rows) == 1


class TestNativeFallback:
    """The thin DE regional lines (.MU/.SG/.DU) have day gaps --
    their last bar is then NOT the previous day's close."""

    def test_luecke_in_der_eur_linie_zieht_den_nativen_vortagsschluss(self):
        daten = {'EXFC.MU': _serie([1000.0, 1100.0], ['2026-07-14', '2026-07-15']),
                 '900001.KS': _serie([1_500_000.0, 1_600_000.0],
                                     ['2026-07-15', '2026-07-16']),
                 'EURKRW=X': _serie([1400.0, 1400.0], ['2026-07-15', '2026-07-16'])}

        snap = _monitor(daten).snapshot(dict(FOREIGN_EUR_LINE))

        assert snap['current_date'] == '2026-07-16'           # the more recent native print
        assert snap['quote'] == '900001.KS+FX'
        assert snap['current_eur'] == pytest.approx(1_600_000.0 / 1400.0)
        assert snap['native_note'] is not None                # base change stays visible
        assert 'stale bis 2026-07-15' in snap['native_note']

    def test_ohne_luecke_bleibt_die_eur_linie_massgeblich(self):
        daten = {'EXFC.MU': _serie([1000.0, 1100.0], ['2026-07-15', '2026-07-16']),
                 '900001.KS': _serie([1_500_000.0, 1_600_000.0],
                                     ['2026-07-15', '2026-07-16']),
                 'EURKRW=X': _serie([1400.0, 1400.0], ['2026-07-15', '2026-07-16'])}

        snap = _monitor(daten).snapshot(dict(FOREIGN_EUR_LINE))

        assert snap['quote'] == 'EXFC.MU'
        assert snap['current_eur'] == 1100.0
        assert snap['native_note'] is None

    def test_native_linie_ohne_fx_zum_print_datum_kippt_den_fallback(self):
        """Prefer the (stale) EUR line with a documented date over a conversion
        that uses an FX row from who-knows-when."""
        daten = {'EXFC.MU': _serie([1000.0, 1100.0], ['2026-07-14', '2026-07-15']),
                 '900001.KS': _serie([1_600_000.0], ['2026-07-16']),
                 'EURKRW=X': _serie([1400.0], ['2026-07-15'])}      # no 07-16 row

        snap = _monitor(daten).snapshot(dict(FOREIGN_EUR_LINE))

        assert snap['quote'] == 'EXFC.MU'
        assert snap['current_date'] == '2026-07-15'

    def test_fehlende_daten_geben_none_statt_eines_verdikts(self):
        assert _monitor({}).snapshot(dict(AMD)) is None


class TestUsdUmrechnung:
    def test_usd_serie_wird_taggenau_umgerechnet(self):
        daten = {'AMD': _serie([500.0, 502.32], ['2026-07-15', '2026-07-16'])}

        snap = _monitor(daten).snapshot(dict(AMD))

        assert snap['current_eur'] == pytest.approx(502.32 / 1.15)
        assert snap['quote'] == 'AMD+FX'
        assert snap['current_date'] == '2026-07-16'

    def test_fehlende_fx_zeile_wirft_statt_still_vorwaerts_zu_fuellen(self):
        """The caller catches this and reports STOP-CHECK ERROR -> NO VERDICT."""
        daten = {'AMD': _serie([500.0], ['2026-07-17'])}    # FX endet am 16.07.

        with pytest.raises(FreshnessError):
            _monitor(daten).snapshot(dict(AMD))


class TestTradegateZeuge:
    """Tradegate is NO LONGER a price source, but it is the session witness."""

    def test_belegte_fehlende_session_kippt_das_verdikt(self):
        """Lag 1 sits under STALE_BDAYS (holiday tolerance) -- without a witness
        exactly this gap slips through, and it caused all three incidents."""
        daten = {'AMD': _serie([500.0, 502.32], ['2026-07-14', '2026-07-15'])}
        zeuge = lambda pos: (436.80, pd.Timestamp('2026-07-16'), 'Tradegate')

        snap = _monitor(daten, tradegate=zeuge).snapshot(dict(AMD))

        assert snap['verdikt'].kein_verdikt
        assert 'fehlende Session' in snap['verdikt'].grund

    def test_abweichung_bei_gleichem_datum_meldet_aber_kippt_nicht(self):
        daten = {'AMD': _serie([500.0, 502.32], ['2026-07-15', '2026-07-16'])}
        ours = 502.32 / 1.15
        zeuge = lambda pos: (ours * 1.05, pd.Timestamp('2026-07-16'), 'Tradegate')

        snap = _monitor(daten, tradegate=zeuge).snapshot(dict(AMD))

        assert snap['crosscheck_note'] is not None      # data error is reported ...
        assert not snap['verdikt'].kein_verdikt         # ... but the date is documented
        assert snap['verdikt'].kind == 'HAELT'

    def test_gleicher_kurs_gleiches_datum_ist_still(self):
        daten = {'AMD': _serie([500.0, 502.32], ['2026-07-15', '2026-07-16'])}
        zeuge = lambda pos: (502.32 / 1.15, pd.Timestamp('2026-07-16'), 'Tradegate')

        snap = _monitor(daten, tradegate=zeuge).snapshot(dict(AMD))

        assert snap['crosscheck_note'] is None
        assert snap['verdikt'].kind == 'HAELT'


class TestVerdiktUndRatchet:
    def test_stale_serie_gibt_kein_verdikt_statt_haelt(self):
        daten = {'AMD': _serie([500.0, 400.0], ['2026-07-10', '2026-07-13'])}

        snap = _monitor(daten).snapshot(dict(AMD))

        assert snap['verdikt'].kein_verdikt

    def test_ratchet_schreibt_hoch_und_stop_in_die_position(self):
        daten = {'AMD': _serie([600.0, 700.0], ['2026-07-15', '2026-07-16'])}
        pos = dict(AMD)

        snap = _monitor(daten).snapshot(pos)

        neu = round(700.0 / 1.15, 2)
        assert snap['ratchet']['runter'] is False
        assert pos['eur_hoch'] == neu                       # side effect is intentional
        assert pos['stop_eur'] == round(neu * 0.7, 2)
        assert 'Basis native_fx' in pos['stop_last_ratchet_note']

    def test_gerissener_stop_wird_als_solcher_gemeldet(self):
        daten = {'AMD': _serie([500.0, 300.0], ['2026-07-15', '2026-07-16'])}

        snap = _monitor(daten).snapshot(dict(AMD))

        assert snap['verdikt'].gerissen
        assert snap['breached'] is True
