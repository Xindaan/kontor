"""Regressions of the -30% stop rule.

Failure class: a wrong or wrongly dated price silently produces a wrong
stop verdict. In a real production run this nearly cost a wrong verdict
on a real position.
"""

import datetime

import pandas as pd
import pytest

from backtest.freshness import DatedPrice, stale_lag_bdays as _lag
from backtest.stop_rules import (derive_trailing_stop, stale_lag_bdays, stop_basis,
                                 stop_verdict)


def _series(werte):
    idx = pd.bdate_range('2026-06-01', periods=len(werte))
    return pd.Series(werte, index=idx, dtype=float)


# ---------- basis choice: never mix high and current on different bases ----------

def test_usd_namen_rechnen_auf_der_nativen_linie():
    assert stop_basis({'waehrung': 'USD->EUR'}) == 'native_fx'


def test_echte_eur_linie_bleibt_ohne_fx():
    assert stop_basis({'waehrung': 'EUR_Linie'}) == 'eur_line'


def test_explizite_basis_schlaegt_die_ableitung():
    assert stop_basis({'waehrung': 'USD->EUR', 'stop_basis': 'eur_line'}) == 'eur_line'


# ---------- peak derivation ----------

def test_peak_und_stop_kommen_aus_der_schlusskurs_serie():
    s = _series([100.0, 120.0, 110.0])

    rule = derive_trailing_stop(s, current_eur=110.0, current_date=s.index[-1])

    assert rule['high'] == 120.0
    assert rule['stop'] == 84.0
    assert rule['breached'] is False


def test_intraday_hoch_wird_bei_positiver_widerlegung_korrigiert():
    """A real case: eur_hoch 1110.00 was a broker intraday value, the real
    close peak was 1100.00. The series shows a lower close on the stored
    peak day -> positively refuted -> the correction kicks in."""
    s = _series([1000.0, 1100.00, 900.0])

    rule = derive_trailing_stop(s, current_eur=900.0, current_date=s.index[-1],
                                stored_high=1110.00, stored_stop=777.00,
                                stored_high_date=s.index[1])

    assert rule['high'] == 1100.00
    assert rule['stop'] == 770.00
    assert rule['ratchet']['runter'] is True
    assert rule['ratchet']['old_high'] == 1110.00
    assert rule['blocked'] is None


def test_peak_sinkt_nicht_ohne_beleg():
    """A Yahoo data revision or a missing peak day must not silently loosen the
    stop. Without positive refutation, the tighter peak stands."""
    s = _series([1000.0, 1020.0, 900.0])   # the peak day is missing from the series

    rule = derive_trailing_stop(s, current_eur=900.0, current_date=s.index[-1],
                                stored_high=1100.00, stored_stop=770.00,
                                stored_high_date='2026-06-25')

    assert rule['high'] == 1100.00          # gespeicherter, ENGERER Peak bleibt
    assert rule['stop'] == 770.00
    assert rule['ratchet'] is None          # nichts persistieren
    assert rule['blocked'] is not None      # aber laut melden
    assert rule['blocked']['derived_high'] == 1020.0


def test_ohne_peak_datum_wird_nicht_abgesenkt():
    s = _series([1000.0, 1020.0])

    rule = derive_trailing_stop(s, current_eur=1020.0, current_date=s.index[-1],
                                stored_high=1100.00, stored_stop=770.00,
                                stored_high_date=None)

    assert rule['high'] == 1100.00
    assert rule['blocked']['grund'] == 'kein gespeichertes Peak-Datum'


def test_hoch_ratschen_bleibt_still_und_fail_safe():
    """Asymmetry: higher = stop gets tighter = safe -> no alarm needed."""
    s = _series([100.0, 150.0])

    rule = derive_trailing_stop(s, current_eur=150.0, current_date=s.index[-1],
                                stored_high=120.0, stored_stop=84.0,
                                stored_high_date=s.index[0])

    assert rule['high'] == 150.0
    assert rule['ratchet']['runter'] is False
    assert rule['blocked'] is None


def test_neues_hoch_ratscht_hoch_und_zieht_den_stop_nach():
    s = _series([100.0, 150.0])

    rule = derive_trailing_stop(s, current_eur=150.0, current_date=s.index[-1],
                                stored_high=120.0, stored_stop=84.0)

    assert rule['high'] == 150.0
    assert rule['stop'] == 105.0
    assert rule['ratchet']['runter'] is False


def test_unveraendertes_hoch_meldet_keinen_ratchet():
    """Idempotency: a second run on the same day must not rewrite anything."""
    s = _series([100.0, 120.0, 110.0])

    rule = derive_trailing_stop(s, current_eur=110.0, current_date=s.index[-1],
                                stored_high=120.0, stored_stop=84.0)

    assert rule['ratchet'] is None


def test_frischerer_schluss_ueber_dem_serien_peak_wird_zum_hoch():
    s = _series([100.0, 120.0])

    rule = derive_trailing_stop(s, current_eur=130.0, current_date='2026-07-16')

    assert rule['high'] == 130.0
    assert rule['high_date'] == '2026-07-16'


# ---------- verdict ----------

def test_breached_stop_is_detected():
    """End-to-end on realistic numbers: a stored peak, a fresh close below the
    derived stop -> stop breached. With the stale witness price, the
    monitor would have reported 'holds' instead."""
    s = _series([1000.0, 1100.00, 805.00])

    rule = derive_trailing_stop(s, current_eur=760.00, current_date='2026-07-16',
                                stored_high=1100.00, stored_stop=770.00,
                                stored_high_date=s.index[1])

    assert rule['stop'] == 770.00
    assert rule['breached'] is True
    assert rule['buffer'] < 0


def test_verdict_haelt_auch_unter_dem_alten_intraday_peak():
    """Robustness of the real verdict: the fresh close breaches the stop both
    under the corrected peak and under the old intraday entry. The peak
    question doesn't actually decide the outcome on that day."""
    s = _series([1000.0, 1100.00, 805.00])

    rule = derive_trailing_stop(s, current_eur=760.00, current_date='2026-07-16',
                                stored_high=1110.00, stored_stop=777.00,
                                stored_high_date=None)   # -> Absenkung blockiert

    assert rule['blocked'] is not None
    assert rule['stop'] == 777.00
    assert rule['breached'] is True


def test_stale_kurs_haette_den_riss_uebertuencht():
    """Counter-check to the previous test: same stop, but a price one session
    old falsely signals the all-clear."""
    s = _series([1000.0, 1100.00, 805.00])

    rule = derive_trailing_stop(s, current_eur=800.00, current_date='2026-07-16',
                                stored_high=1100.00, stored_stop=770.00)

    assert rule['breached'] is False


def test_schluss_genau_auf_dem_stop_gilt_als_gerissen():
    s = _series([100.0, 100.0])

    rule = derive_trailing_stop(s, current_eur=70.0, current_date=s.index[-1])

    assert rule['stop'] == 70.0
    assert rule['breached'] is True


def test_leere_serie_wirft_statt_still_zu_raten():
    with pytest.raises(ValueError):
        derive_trailing_stop(pd.Series(dtype=float), current_eur=100.0,
                             current_date='2026-07-16')


# ---------- stale detector on the stop series ----------

def test_gestriger_schluss_ist_frisch():
    # Fri expects Thursday's close.
    assert stale_lag_bdays('2026-07-16', today=datetime.date(2026, 7, 17)) == 0


def test_wochenende_zaehlt_nicht_als_rueckstand():
    # Mon expects Friday's close.
    assert stale_lag_bdays('2026-07-17', today=datetime.date(2026, 7, 20)) == 0


def test_mehrtaegige_luecke_schlaegt_an():
    lag = stale_lag_bdays('2026-07-13', today=datetime.date(2026, 7, 17))

    assert lag >= 2



# ---------- Tri-state in the type ----------

FR = datetime.date(2026, 7, 17)


def _p(value, tag, source='MU+FX'):
    return DatedPrice(value=value, session_date=datetime.date.fromisoformat(tag),
                      source=source, stale_lag=_lag(tag, today=FR))


def test_stale_kurs_gibt_kein_verdikt_statt_haelt():
    """The common denominator of all three incidents: a stale price reported
    'holds'.

    Here with realistic numbers: a stop value, where the stale witness
    price (from the prior session's close) would have said 'holds'.
    """
    v = stop_verdict(_p(800.00, '2026-07-10'), stop=770.00)

    assert v.kein_verdikt and not v.gerissen
    assert 'KEIN VERDIKT' in v.label
    assert '2026-07-10' in v.grund


def test_session_zeuge_kippt_auch_einen_kalendarisch_frischen_kurs():
    """Lag 1 sits under STALE_BDAYS (holiday tolerance) -- the witness catches it."""
    v = stop_verdict(_p(800.00, '2026-07-16'), stop=770.00,
                     zusatz_stale='Tradegate belegt einen Schluss vom 2026-07-16')

    assert v.kein_verdikt
    assert 'Tradegate' in v.grund


def test_frischer_kurs_unter_stop_ist_gerissen():
    # A realistic case: a fresh price against the derived stop.
    v = stop_verdict(_p(760.00, '2026-07-16'), stop=770.00)

    assert v.gerissen and not v.kein_verdikt


def test_frischer_kurs_knapp_ueber_stop_ist_nah():
    v = stop_verdict(_p(780.0, '2026-07-16'), stop=770.00)

    assert v.nah and not v.gerissen


def test_frischer_kurs_mit_puffer_haelt():
    v = stop_verdict(_p(1000.0, '2026-07-16'), stop=770.00)

    assert v.kind == 'HAELT' and not v.kein_verdikt


def test_frische_schlaegt_den_kursvergleich():
    """Even a BREACHED price gives NO VERDICT if it's stale -- otherwise the
    monitor acts on a price nobody can substantiate."""
    v = stop_verdict(_p(100.0, '2026-07-10'), stop=770.00)

    assert v.kein_verdikt and not v.gerissen
