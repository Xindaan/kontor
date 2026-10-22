"""Stop monitor: fetch the price series, choose the basis, build a snapshot with a verdict.

This logic lives in its own testable module rather than in a thin CLI/IO
script, because that kind of script typically loads the portfolio JSON and FX
data at module level and hits the network on import -- everything in it could
only be checked by an actual live run. But this layer carries the money
decision: which series is authoritative, when the native-currency fallback
kicks in, when the broker witness overturns the verdict. That deserves to be
tested, not guessed live every day.

All external contacts are injectable (``fetch``, ``tradegate_close``), so the
tests run without network access and without a portfolio file. The calling
script stays a thin IO shell: read the JSON, build a StopMonitor, print the
table.
"""

import datetime as _dt
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from backtest.freshness import (DatedPrice, FreshnessError, dated_last, real_prints,
                                fx_exact, stale_lag_bdays)
from backtest.stop_rules import derive_trailing_stop, stop_basis, stop_verdict

# Tolerance for the broker-side cross-check (Tradegate close vs. computed close
# on the same day). Both are 22:00 CET closes; anything beyond this is a data error.
TRADEGATE_TOLERANCE = 0.02

DEFAULT_TRACKING_START = '2026-05-01'


def stop_rows(positions: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Positions with an active stop and a nonzero holding."""
    return [p for p in positions if p.get('stop_eur') and p.get('shares', 0) > 0]


def date_key(dt: Any) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return pd.Timestamp(ts.date())


def stop_quote(position: Mapping[str, Any]) -> Tuple[str, str]:
    """The EUR-LINE source -- only valid when ``stop_basis`` == 'eur_line'.

    Field name kept symmetric with ``stop_price_ticker_native``: the old
    ``stop_price_ticker`` looked like the primary source, but for USD->EUR
    names it was dead (those are computed on the native-currency line times FX).
    """
    ticker = position.get('stop_price_ticker_eur_line') or position['price_ticker']
    currency = position.get('stop_price_eur_line_currency')
    if currency is None:
        currency = 'USD' if position.get('waehrung') == 'USD->EUR' else 'EUR'
    return ticker, currency


def stop_source(position: Mapping[str, Any]) -> Tuple[str, str, str]:
    """(basis, ticker, currency) of the single series that high and current come from."""
    basis = stop_basis(position)
    if basis == 'native_fx':
        ticker = position.get('stop_price_ticker_native') or position['price_ticker']
        currency = position.get('stop_price_native_currency', 'USD')
    else:
        ticker, currency = stop_quote(position)
    return basis, ticker, currency


class StopMonitor:
    """Builds stop snapshots from real closing prices.

    Args:
        fx_eurusd: EURUSD close series (USD per EUR), index = session date.
        end:       Fetch end (exclusive, like DataLoader.yahoo).
        today:     Reference day for the freshness check. Tests/reproduction only.
        fetch:     Price source (ticker, start, end, currency=...) -> pd.Series with
                   real print dates. Default ``real_prints``.
        tradegate_close: (position) -> (eur, Timestamp, source) or None. The
                   broker-side session witness; None = no cross-check.
    """

    def __init__(self, fx_eurusd: pd.Series, end: str,
                 today: Optional[_dt.date] = None,
                 fetch: Callable[..., pd.Series] = real_prints,
                 tradegate_close: Optional[Callable[[Mapping[str, Any]], Any]] = None):
        self.fx = fx_eurusd
        self.end = end
        self.today = today
        self.fetch = fetch
        self.tradegate_close = tradegate_close
        self._fx_pair_cache: Dict[str, Optional[pd.Series]] = {}

    # ---------- Currency ----------

    def series_usd_to_eur(self, series: pd.Series) -> pd.Series:
        """Converts a USD series to EUR day-for-day -- the FX date MUST match the price date.

        Previously used ``asof()``: if the FX row for day D was missing, an older
        rate would silently convert day D instead.
        """
        out = []
        for dt, value in series.items():
            key = date_key(dt)
            out.append((key, float(value) / fx_exact(self.fx, key, today=self.today)))
        return pd.Series(dict(out)).sort_index()

    def fx_pair_exakt(self, pair: str, key: Any) -> Optional[float]:
        """FX rate (units of foreign currency per EUR) for EXACTLY ``key`` -- no asof."""
        if pair == 'EURUSD=X':
            return fx_exact(self.fx, key, today=self.today)
        if pair not in self._fx_pair_cache:
            try:
                self._fx_pair_cache[pair] = self.fetch(pair, '2026-01-01', self.end,
                                                       currency='USD')
            except FreshnessError:
                self._fx_pair_cache[pair] = None
        ser = self._fx_pair_cache[pair]
        if ser is None or ser.empty:
            return None
        return fx_exact(ser, key, quelle=pair, today=self.today)

    # ---------- Series ----------

    def eur_close_series(self, ticker: str, currency: str, start: str,
                         name: str) -> Optional[pd.Series]:
        """Closing-price series in EUR with REAL print dates in the index."""
        try:
            s = self.fetch(ticker, start, self.end, currency=currency,
                           require_volume=True, today=self.today)
        except FreshnessError:
            return None
        if currency == 'USD':
            s = self.series_usd_to_eur(s)
        elif currency != 'EUR':
            raise ValueError('Unsupported stop price currency %s for %s' % (currency, name))
        return s.dropna()

    def load_stop_series_eur(self, position: Mapping[str, Any]) -> Optional[pd.Series]:
        basis, ticker, currency = stop_source(position)
        start = position.get('stop_tracking_start', DEFAULT_TRACKING_START)
        return self.eur_close_series(ticker, currency, start, position['name'])

    def native_last_eur(self, position: Mapping[str, Any]):
        """Most recent EUR value of the NATIVE-currency line, or None.

        When the preferred liquid EUR line has no published close for the
        previous day, the native exchange supplies the fallback via FX into EUR.
        The fallback stays visible, because a direct EUR close still takes priority.

        The fallback is an IMPROVEMENT, not a requirement: if it cannot be
        substantiated (source unavailable, or no FX row for the print date), it
        returns None and the EUR line remains authoritative -- its date is
        substantiated, just older, and the verdict logic decides on freshness
        from there anyway. Raising an error instead would turn a missing FX row
        for some foreign-currency pair into a full blackout, even though a
        substantiated EUR close is available.
        """
        nt = position.get('stop_price_ticker_native')
        if not nt:
            return None
        nc = position.get('stop_price_native_currency', 'USD')
        try:
            s = self.fetch(nt, position.get('stop_tracking_start', DEFAULT_TRACKING_START),
                           self.end, currency=nc, today=self.today)
            last = dated_last(s, nt, today=self.today)
            last_date = pd.Timestamp(last.session_date)
            if nc == 'EUR':
                return last.value, last_date, nt
            fx = self.fx_pair_exakt('EUR%s=X' % nc, last_date)
            if not fx:
                return None
            return last.value / fx, last_date, nt
        except FreshnessError:
            return None

    # ---------- Cross-check ----------

    def tradegate_crosscheck(self, position: Mapping[str, Any], series: pd.Series,
                             quote: str) -> Optional[Tuple[str, str]]:
        """Broker-side cross-check on the SAME trading day: ``(text, art)`` or None.

        Tradegate is deliberately NOT used as a price source: its `close` field
        is the delta reference of the displayed session and lags it by one
        session -- used as a price, it once masked a real stop breach behind a
        stale, higher value on a foreign-listed instrument. As a same-date
        cross-check it remains valuable: that is exactly what it is meant for,
        catching a stale EUR line before it does damage.

        ``art`` separates two findings that used to share the same channel:
        'session_luecke' = the source has a session that we are missing -> a
        FRESHNESS counter-proof that must overturn the verdict (the lag is below
        STALE_BDAYS and would otherwise slip through the holiday tolerance).
        'abweichung'     = same date, different value -> report a data error, but
        the date is substantiated, so the verdict remains valid.
        """
        if self.tradegate_close is None:
            return None
        tg = self.tradegate_close(position)
        if tg is None:
            return None
        tg_eur, tg_date, _ = tg
        tg_date = pd.Timestamp(tg_date)
        if tg_date not in series.index:
            if tg_date > series.index.max():
                return ('Tradegate belegt einen Schluss vom %s, unsere %s-Serie endet am %s '
                        '-> fehlende Session' %
                        (tg_date.date(), quote, pd.Timestamp(series.index.max()).date()),
                        'session_luecke')
            return None
        ours = float(series.loc[tg_date])
        if ours <= 0:
            return None
        dev = abs(tg_eur / ours - 1.0)
        if dev <= TRADEGATE_TOLERANCE:
            return None
        return ('%s: Tradegate-Schluss %s = %.2f EUR weicht %.1f%% vom gerechneten %.2f ab '
                '-> Kursquelle pruefen' %
                (position['name'], tg_date.date(), tg_eur, dev * 100, ours),
                'abweichung')

    # ---------- Snapshot ----------

    def snapshot(self, position: Dict[str, Any]) -> Optional[dict]:
        """Complete stop state including tri-state verdict, or None without data.

        SIDE EFFECT (intentional, existing behavior): on a ratchet,
        ``eur_hoch``/``stop_eur``/``stop_last_ratchet`` are updated in place on
        ``position``. Persistence is left to the caller.
        """
        basis, source_ticker, _ = stop_source(position)
        s = self.load_stop_series_eur(position)
        if s is None or s.empty:
            return None

        current_eur = float(s.iloc[-1])
        current_date = date_key(s.index[-1])
        quote = source_ticker if basis == 'eur_line' else '%s+FX' % source_ticker
        native_note = None

        # Only the eur_line basis needs a fallback: thin regional exchange lines
        # (e.g. .MU/.SG/.DU) have day gaps. In that case, prefer the native
        # exchange via FX -- a basis switch, so it is flagged -- rather than
        # masking a stop breach.
        if basis == 'eur_line':
            native = self.native_last_eur(position)
            if native is not None:
                n_eur, n_date, n_ticker = native
                if n_date > current_date:
                    native_note = ('EUR-Linie %s stale bis %s (kein juengerer Print); '
                                   'Vortagsschluss via %s+FX = %.2f statt %.2f' %
                                   (quote, current_date.date(), n_ticker, n_eur, current_eur))
                    current_eur = n_eur
                    current_date = n_date
                    quote = '%s+FX' % n_ticker

        rule = derive_trailing_stop(s, current_eur, current_date,
                                    stored_high=position.get('eur_hoch'),
                                    stored_stop=position.get('stop_eur'),
                                    stored_high_date=position.get('stop_last_ratchet'))
        if rule['ratchet']:
            position['eur_hoch'] = rule['high']
            position['stop_eur'] = rule['stop']
            position['stop_last_ratchet'] = rule['high_date']
            position['stop_last_ratchet_note'] = ('Aus Schlusskursserie %s abgeleitet (Basis %s)'
                                                  % (quote, basis))

        cross_text, cross_art = self.tradegate_crosscheck(position, s, quote) or (None, None)

        # Price as a triple, verdict derived from the type: freshness first, then price.
        preis = DatedPrice(value=current_eur,
                           session_date=pd.Timestamp(current_date).date(),
                           source=quote,
                           stale_lag=stale_lag_bdays(current_date, today=self.today))
        verdikt = stop_verdict(
            preis, rule['stop'],
            zusatz_stale=cross_text if cross_art == 'session_luecke' else None)

        snapshot = dict(rule)
        snapshot.update({
            'preis': preis,
            'verdikt': verdikt,
            'current_eur': current_eur,
            'current_date': preis.session_date.isoformat(),
            'quote': quote,
            'basis': basis,
            'native_note': native_note,
            'crosscheck_note': cross_text if cross_art == 'abweichung' else None,
        })
        return snapshot
