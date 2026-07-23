"""Pure rule logic for the -30% trailing stop on individual equity positions.

Deliberately free of network/IO: a separate monitoring script fetches the
prices and calls into this module. The money-path logic belongs under test,
not in a script that pulls from a live data provider on import.

Locked convention: -30% from the trailing DAILY-CLOSE high in EUR, price
always the daily close (never intraday), the same price basis for both the
high and the current price.
"""

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pandas as pd

# The freshness invariant lives in exactly ONE place: stop_rules is a
# consumer, not the owner. Re-exported because existing code/tests import
# from here.
from backtest.freshness import (  # noqa: F401
    STALE_BDAYS, DatedPrice, FreshnessError, stale_lag_bdays,
)

STOP_PCT = 0.30

# Buffer threshold below which the monitor reports "NEAR" (info, not an alert).
NAH_SCHWELLE = 0.10


def stop_basis(position: Mapping[str, Any]) -> str:
    """ONE price basis for both the high AND the current price -- never mix bases.

    Comparing a Xetra high (17:30 CET) against a US close (22:00 CET) can shift
    the stop by up to ~3% -- more than some buffers are wide. In addition, the
    thin German regional lines have day-gaps precisely on the volatile days when
    the stop matters most.

    'native_fx' = close of the home exchange / same-day FX (for USD->EUR names).
    'eur_line'  = a genuine EUR-denominated line without FX conversion (e.g. a
                  EUR-denominated Xetra line of a foreign stock).
    """
    explicit = position.get('stop_basis')
    if explicit in ('native_fx', 'eur_line'):
        return explicit
    return 'native_fx' if position.get('waehrung') == 'USD->EUR' else 'eur_line'


@dataclass(frozen=True)
class StopVerdict:
    """Tri-state enforced at the TYPE level.

    Previously this case distinction lived inline in the monitor script, and
    the default branch was "holding" -- a missing price would fall through to
    the reassuring answer. Here there is no default branch: anyone who cannot
    prove freshness gets NO VERDICT.
    """

    kind: str          # 'KEIN_VERDIKT' | 'GERISSEN' | 'NAH' | 'HAELT'
    label: str
    grund: Optional[str] = None

    @property
    def kein_verdikt(self) -> bool:
        return self.kind == 'KEIN_VERDIKT'

    @property
    def gerissen(self) -> bool:
        return self.kind == 'GERISSEN'

    @property
    def nah(self) -> bool:
        return self.kind == 'NAH'


def stop_verdict(preis: DatedPrice, stop: float, zusatz_stale: Optional[str] = None,
                 nah_schwelle: float = NAH_SCHWELLE) -> StopVerdict:
    """Freshness FIRST, only then the price comparison.

    ``zusatz_stale`` is external counter-evidence against freshness that isn't
    visible to the calendar (e.g. an exchange session witness proves a session
    that our own series doesn't have -- a lag of 1 is under STALE_BDAYS and
    would otherwise slip through).
    """
    if preis.stale:
        return StopVerdict('KEIN_VERDIKT', 'KEIN VERDIKT (Kurs stale)',
                           '%s: juengster Schluss %s ist %d Handelstage alt'
                           % (preis.source, preis.session_date.isoformat(), preis.stale_lag))
    if zusatz_stale:
        return StopVerdict('KEIN_VERDIKT', 'KEIN VERDIKT (Kurs stale)', zusatz_stale)
    if preis.value <= stop:
        return StopVerdict('GERISSEN', '*** GERISSEN')
    puffer = (preis.value - stop) / preis.value if preis.value > 0 else float('nan')
    if puffer < nah_schwelle:
        return StopVerdict('NAH', 'NAH (<%.0f%%)' % (nah_schwelle * 100))
    return StopVerdict('HAELT', 'haelt')


def _peak_positiv_widerlegt(series: pd.Series, stored_high: float,
                            stored_high_date: Optional[Any], tol: float = 0.001) -> bool:
    """Does the series prove that the stored high was never actually a close?

    The high is only disproven if the stored peak DATE is present in the series
    and shows a materially lower close there (a broker's intraday entry, or a
    price revision). If the date is missing entirely -- data outage, volume
    filter, start-date clipping -- NOTHING is proven, and the peak must not be
    lowered.
    """
    if stored_high_date is None:
        return False
    try:
        key = pd.Timestamp(stored_high_date).normalize()
    except (ValueError, TypeError):
        return False
    if key not in series.index:
        return False
    return float(series.loc[key]) < stored_high * (1.0 - tol)


def derive_trailing_stop(
    series: pd.Series,
    current_eur: float,
    current_date: Any,
    stored_high: Optional[float] = None,
    stored_stop: Optional[float] = None,
    stored_high_date: Optional[Any] = None,
    stop_pct: float = STOP_PCT,
) -> dict:
    """DERIVE the trailing high and stop from the closing-price series -- asymmetric.

    The peak is recomputed from scratch out of ``series`` on every run instead of
    only ever ratcheting upward. A high that was recorded too high -- e.g. a
    broker's INTRADAY high instead of a closing price -- would otherwise keep
    the stop permanently too loose, and a pure upward-only ratchet never
    corrects it.

    The two directions are NOT symmetric, though (found during a later code
    review):
    - Ratcheting UP is fail-safe (the stop tightens) -> adopt silently.
    - Ratcheting DOWN loosens the stop -> only adopt on **positive disproof**
      (the series shows a lower close on the stored peak date). Otherwise the
      stored, tighter peak is kept (``blocked``) and the caller must raise an
      alert: a data-provider revision or a missing peak date would otherwise
      silently loosen the stop -- the same failure class described above.
    """
    if series is None or len(series) == 0:
        raise ValueError('empty closing-price series')

    series_high = float(series.max())
    if current_eur >= series_high:
        high_raw, high_date = float(current_eur), current_date
    else:
        high_raw, high_date = series_high, series.idxmax()

    derived = round(high_raw, 2)
    derived_date = pd.Timestamp(high_date).date().isoformat()

    if stored_high is None:
        high, high_date_out, ratchet, blocked = derived, derived_date, None, None
        stored_stop = round(derived * (1.0 - stop_pct), 2)
    else:
        stored = round(float(stored_high), 2)
        stored_stop = (round(stored * (1.0 - stop_pct), 2) if stored_stop is None
                       else float(stored_stop))
        blocked = None
        if derived >= stored:
            high, high_date_out = derived, derived_date
            ratchet = None if derived == stored else {
                'old_high': stored, 'old_stop': stored_stop,
                'new_high': derived, 'new_stop': round(derived * (1.0 - stop_pct), 2),
                'date': derived_date, 'runter': False,
            }
        elif _peak_positiv_widerlegt(series, stored, stored_high_date):
            high, high_date_out = derived, derived_date
            ratchet = {
                'old_high': stored, 'old_stop': stored_stop,
                'new_high': derived, 'new_stop': round(derived * (1.0 - stop_pct), 2),
                'date': derived_date, 'runter': True,
            }
        else:
            # Fail-safe: keep the tighter stored peak, and raise a loud alert.
            high, high_date_out, ratchet = stored, (
                pd.Timestamp(stored_high_date).date().isoformat()
                if stored_high_date is not None else derived_date), None
            blocked = {
                'stored_high': stored,
                'derived_high': derived,
                'stored_high_date': (pd.Timestamp(stored_high_date).date().isoformat()
                                     if stored_high_date is not None else None),
                'grund': ('kein gespeichertes Peak-Datum' if stored_high_date is None
                          else 'Peak-Tag fehlt in der Serie (Ausfall/Filter/Clipping)'),
            }

    stop = round(high * (1.0 - stop_pct), 2)

    return {
        'high': high,
        'high_date': high_date_out,
        'blocked': blocked,
        'stop': stop,
        'drawdown': current_eur / high - 1.0 if high > 0 else float('nan'),
        'buffer': (current_eur - stop) / current_eur if current_eur > 0 else float('nan'),
        'breached': current_eur <= stop,
        'ratchet': ratchet,
    }
