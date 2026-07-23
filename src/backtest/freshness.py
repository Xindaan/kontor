"""The invariant: no verdict without proven freshness.

Three incidents in four weeks, one pattern: a stale price was silently
treated as today's close, and the monitor reported "holding" instead of
"no data".

- Cache freeze: a leveraged ETF position froze -> volatility collapsed ->
  signal inverted.
- Manifest recorded the REQUESTED end instead of the delivered end -> a
  single-stock position was reported as "8% buffer / holding" while the
  real stop had been breached (-30%).
- A broker's `close` field turned out to be the delta reference of the
  PREVIOUS session -> a position was reported as "5% buffer / NEAR" while
  it had, in reality, breached its stop.

The failure class survived every individual fix because only the SOURCE
kept getting patched. This module fixes the RULE instead:

1. A price is a TRIPLE (value, session date, source). The date must be
   PROVEN by the source -- never inferred from ``today``, from the request
   window, or from a page timestamp. ``align="ffill"`` + ``iloc[-1]`` is
   exactly this forbidden inference: ffill silently re-dates every stale
   value to the most recent index date. That's why ``real_prints()`` fetches
   per ticker and without ffill -- the series then simply cannot lie.
2. TRI-STATE: BREACHED / HOLDING(as of D, fresh) / NO VERDICT(stale). Anyone
   who cannot prove a fresh date says "no data" -- never "holding".
3. PRIORITY never overrides FRESHNESS. Source priority only breaks ties on an
   identical date; a secondary source with a newer proven date is an alert,
   never a silent no-op.

There is also the INTRADAY case: a bar dated TODAY is not a closing price
during the session, it's a running quote. The locked convention (playbook
section 2) says "price: always the daily close, NEVER intraday"; intraday
and closing prices for the same instrument have been observed to diverge by
several percent on a single day -- enough to cause a false trigger.
``real_prints()`` therefore drops today-dated bars by default: the source
cannot prove that the value is a close. The daily run (07:04) loses nothing
by this, because at that time no today-dated bar exists yet.
"""

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from backtest.data import DataLoader

# Number of trading days of lag after which the most recent close counts as
# stale. 2 lets a single holiday pass through (the business-day calendar
# doesn't know about market holidays), but catches any larger data outage.
STALE_BDAYS = 2


class FreshnessError(ValueError):
    """No value with a proven, fresh session date -> NO VERDICT.

    Deliberately an exception, not a silent return value: this failure class
    arose every time a missing date was allowed to pass as something else.
    """


def stale_lag_bdays(current_date: Any, today: Optional[_dt.date] = None) -> int:
    """Trading days between the most recent close and the last expected one.

    0 = as fresh as possible (yesterday, or Friday). Greater than 0 = lag.

    ``roll='forward'`` is NOT interchangeable with ``'backward'``: on a Sunday,
    ``'backward'`` rolls to Friday first and THEN subtracts -1 -> Thursday. The
    reference day would be one trading day too early, everything would appear
    one day fresher, and a Wednesday print (true difference 3 trading days)
    would pass the gate on Sunday with lag=1, even though it gets blocked with
    lag=2 on Monday -- a FAIL-OPEN right in the invariant. ``'forward'`` rolls
    the weekend to Monday first and then subtracts -1 -> Friday, which stays
    identical for weekdays. (Found while running a stocktaking check on a
    Sunday, which reported "-1 trading days old".)
    """
    today = today or _dt.date.today()
    last_expected = np.busday_offset(today, -1, roll='forward').astype('M8[D]').astype(_dt.date)
    return int(np.busday_count(pd.Timestamp(current_date).date(), last_expected))


@dataclass(frozen=True)
class DatedPrice:
    """Price as a triple: value + PROVEN session date + source.

    No consumer may pass ``value`` along without ``session_date`` -- exactly
    that separation was the invitation for all three incidents.
    """

    value: float
    session_date: _dt.date
    source: str
    stale_lag: int

    @property
    def stale(self) -> bool:
        return self.stale_lag >= STALE_BDAYS

    def describe(self, einheit: str = 'EUR') -> str:
        text = '%.2f %s (%s, Stand %s)' % (self.value, einheit, self.source,
                                           self.session_date.isoformat())
        if self.stale:
            text += ' [STALE: %d Handelstage alt -> KEIN VERDIKT]' % self.stale_lag
        return text

    def require_fresh(self) -> 'DatedPrice':
        """Fail-closed access: returns itself, or raises.

        For callers that have no meaningful tri-state handling (e.g. a
        conversion in the middle of a chain).
        """
        if self.stale:
            raise FreshnessError(
                '%s: latest print %s is %d trading days old -> NO VERDICT'
                % (self.source, self.session_date.isoformat(), self.stale_lag))
        return self


def dated_last(series: pd.Series, source: str,
               today: Optional[_dt.date] = None) -> DatedPrice:
    """Last REAL print of a series as a triple.

    ``series`` MUST carry real print dates in its index (see ``real_prints``);
    an ffill'd series re-dates stale values and turns this function into a lie.
    """
    s = series.dropna()
    if s.empty:
        raise FreshnessError('%s: no price data' % source)
    last = s.index[-1]
    return DatedPrice(
        value=float(s.iloc[-1]),
        session_date=pd.Timestamp(last).date(),
        source=source,
        stale_lag=stale_lag_bdays(last, today=today),
    )


def real_prints(ticker: str, start: str, end: str, currency: str = 'EUR',
                 today: Optional[_dt.date] = None, drop_today: bool = True,
                 require_volume: bool = False) -> pd.Series:
    """Closing-price series of a ticker with REAL print dates in its index.

    Deliberately one ticker per call: a multi-ticker frame with ``align="ffill"``
    fills one ticker's gaps from another ticker's trading days and makes the
    print date unusable. ``align="intersection"`` on a single column means
    exactly "only that ticker's own trading days".

    Args:
        drop_today: Drop today-dated bars (default). During the session such
            a bar is a running quote, not a close -- the source cannot prove
            the opposite.
        require_volume: Drop rows without volume. A reported close without
            volume is a placeholder, not a market close.
    """
    price_data = DataLoader.yahoo(
        tickers=[ticker], start=start, end=end, currency=currency,
        align='intersection', skip_failed=True, load_dividends=False,
        load_volumes=require_volume, validate=False,
    )
    prices = price_data.prices
    if ticker not in prices.columns:
        raise FreshnessError('%s: not loaded (source unreachable)' % ticker)
    s = prices[ticker].dropna()

    if require_volume and price_data.volumes is not None \
            and ticker in price_data.volumes.columns:
        volume = price_data.volumes[ticker].reindex(s.index)
        s = s[volume.gt(0).fillna(False)]

    s = s.copy()
    s.index = pd.DatetimeIndex([pd.Timestamp(pd.Timestamp(d).date()) for d in s.index])

    if drop_today:
        heute = pd.Timestamp(today or _dt.date.today())
        s = s[s.index < heute]

    if s.empty:
        raise FreshnessError('%s: no closing prices in window %s..%s' % (ticker, start, end))
    return s


def fx_exact(fx: pd.Series, session_date: Any, quelle: str = 'EURUSD',
             today: Optional[_dt.date] = None) -> float:
    """FX rate for EXACTLY this session date -- no ffill, no ``asof``.

    ``asof()`` silently fills forward: if the row for D is missing, an older
    day's rate gets used to convert D and the error stays invisible. Measured
    over 134 US trading days in 2026: an exact EURUSD row exists for EVERY one
    of them -- so the exact-match rule costs no days and only catches genuine
    outages.

    An FX row dated TODAY is rejected: the FX day runs until ~23:00 CET, but
    the snapshot is written in the morning -- a partial-day value posing as a
    close. That can't be proven, so it doesn't count.
    """
    key = pd.Timestamp(pd.Timestamp(session_date).date())
    heute = pd.Timestamp(today or _dt.date.today())
    if key >= heute:
        raise FreshnessError(
            '%s: row dated %s is the in-progress FX day (close only at ~23:00 CET) '
            '-> als Tagesschluss nicht belegbar' % (quelle, key.date()))
    if key not in fx.index:
        raise FreshnessError('%s: no quote for %s (no ffill) -> NO VERDICT'
                            % (quelle, key.date()))
    value = float(pd.to_numeric(fx.loc[key], errors='coerce'))
    if not np.isfinite(value) or value <= 0:
        raise FreshnessError('%s: invalid quote for %s' % (quelle, key.date()))
    return value
