"""
Valuation of portfolio positions from multiple independent price sources.

Why consensus instead of a single source: the same ETP trades on multiple
exchanges and in multiple currency lines. Individual lines have demonstrably
contained bad prints -- on one occasion, a pence-denominated LSE line
printed the value from the USD-denominated line straight into the pence
field instead of converting it, an error confirmed by comparing against the
neighboring trading days' prices.

On divergence, nothing is guessed -- it is reported. Rebalancing on a bad
print costs real money.

The consensus is built on the most recent COMMON session date: simply
taking `iloc[-1]` per source would otherwise compare two different sessions
against each other. This has happened live: one source's most recent print
was from the prior trading day (a stale exchange close) while another
source's was live from a different venue on the current day -- producing a
large apparent "divergence" that was really just a false alarm caused by
comparing mismatched sessions. The more dangerous direction is common-mode
staleness (both sources frozen -> spread 0 -> "ok"); the freshness check on
the common date guards against that: if it's older than STALE_BDAYS, there
is NO verdict.
"""

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

from backtest.data import DataLoader
from backtest.freshness import STALE_BDAYS, stale_lag_bdays

# Staged spread gate, calibrated on 228 days of overlapping data between two
# venue/currency lines of the same instrument. Derivation documented
# separately.
#
# Both exchanges close within the same second (Xetra 17:30 CET == LSE 16:30
# BST == 11:30 New York), yet the data provider's closing prices still
# diverge against each other: these are LAST TRADES on two thin lines, not
# auction prices. Measured: median 0.44%, p90 2.29%, max 10.89%. The noise
# correlates neither with exchange volume (>1% rate 21.9% -> 20.4% at volume
# >= 100) nor with daily volatility (corr -0.04) -- conditioning on those
# doesn't help, only the threshold does.
#
# The old 1% gate fired on 21.9% of all days. The flag propagates into
# `action`, the monitoring script then exits with a failure code, and the
# scheduled rebalance run aborts on it: it blocked roughly one in five
# scheduled rebalances on pure noise.
#
# The threshold has to come from the DAMAGE, not from expectations. What's
# being protected is the rebalance decision, and that decision is extremely
# insensitive:
#     d(drift) = weight * (1 - weight) * d(ln px)  ->  ~0.19pp per 1% price
#     error (at weight ~ 25%)
# Even a 5% error shifts the drift by 0.9pp against a 20pp gate; only a
# ~100% error would flip a decision. Observed failure classes in practice:
# a currency-line factor of x1.34 or x100, otherwise bad prints from ~5% up.
# Hence two tiers instead of one:
DEFAULT_SPREAD_WARN = 0.03    # above the noise floor -> report (false-positive rate 5.3%)
DEFAULT_SPREAD_BLOCK = 0.10   # currency-line mixup or gross bad print -> block
DEFAULT_SPREAD_GATE = DEFAULT_SPREAD_BLOCK   # backward-compat alias


@dataclass(frozen=True)
class ConsensusPrice:
    """Result of a multi-source valuation.

    Verdict order for consumers: check ``stale`` first (NO VERDICT, do not act
    -- freshness beats any spread statement), then ``single_source`` /
    ``diverged`` / ``noisy``. ``ok`` is the only positive verdict and requires
    freshness.
    """

    price_eur: float
    quotes: Dict[str, float]
    spread: float
    block_gate: float
    warn_gate: float = DEFAULT_SPREAD_WARN
    as_of: Optional[str] = None                       # gemeinsames Session-Datum (ISO)
    source_dates: Dict[str, str] = field(default_factory=dict)  # letzter Print je Quelle
    stale_lag: int = 0                                # Handelstage hinter dem erwarteten Schluss

    @property
    def gate(self) -> float:
        """Backward-compat alias: the historical `gate` was the block threshold."""
        return self.block_gate

    @property
    def single_source(self) -> bool:
        return len(self.quotes) < 2

    @property
    def stale(self) -> bool:
        """Common session date too old -> NO VERDICT (neither ok nor alert)."""
        return self.stale_lag >= STALE_BDAYS

    @property
    def diverged(self) -> bool:
        # Tolerance against floating-point noise: 101/100-1 is 0.010000000000000009
        # and would otherwise trip a gate of exactly 1%.
        return not self.single_source and self.spread > self.block_gate + 1e-9

    @property
    def noisy(self) -> bool:
        """Above the noise floor, but too small to flip a decision."""
        return (not self.single_source and not self.diverged
                and self.spread > self.warn_gate + 1e-9)

    @property
    def ok(self) -> bool:
        return (not self.single_source and not self.stale
                and not self.diverged and not self.noisy)

    def describe(self) -> str:
        q = ", ".join("%s=%.2f" % kv for kv in self.quotes.items())
        stand = ""
        if self.as_of:
            stand = "; Stand %s" % self.as_of
            newer = {t: d for t, d in self.source_dates.items() if d != self.as_of}
            if newer:
                stand += " (%s)" % ", ".join(
                    "%s zuletzt %s" % kv for kv in sorted(newer.items()))
        marks = ""
        if self.stale:
            marks = " [STALE: %d Handelstage alt -> KEIN VERDIKT]" % self.stale_lag
        if self.single_source:
            return "%.2f EUR (NUR 1 QUELLE: %s%s)%s" % (self.price_eur, q, stand, marks)
        if self.diverged:
            marks = " [SPREAD %.1f%% -> BLOCK]" % (self.spread * 100) + marks
        elif self.noisy:
            marks = " [SPREAD %.1f%%]" % (self.spread * 100) + marks
        return "%.2f EUR (Konsens aus %s%s; Spread %.2f%%)%s" % (
            self.price_eur, q, stand, self.spread * 100, marks,
        )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def consensus_price_eur(
    tickers: Sequence[str],
    start: str,
    end: str,
    gate: float = DEFAULT_SPREAD_BLOCK,
    warn_gate: float = DEFAULT_SPREAD_WARN,
    also_fetch: Optional[Sequence[str]] = None,
    today: Optional[_dt.date] = None,
) -> ConsensusPrice:
    """
    EUR price as the median of multiple sources on the most recent COMMON
    session date.

    Per-source REAL print dates are loaded (no ffill): an ffill'd frame
    silently re-dates every stale value to the most recent index date, after
    which `iloc[-1]` would compare two different sessions against each other.
    The consensus is built on the last date on which ALL sources printed; if
    that is older than STALE_BDAYS trading days, `stale` is set and there is
    NO verdict.

    Args:
        tickers: Valuation sources for the same instrument, e.g.
                 ["EXAMPLE.DE", "EXAMPLE.L"]. Currency comes from
                 `detect_currency`, not from the suffix.
        start:   Start date of the fetch window
        end:     End date (exclusive, like DataLoader.yahoo)
        gate:    Block threshold: above this, `diverged` is set (currency-line
                 mixup or gross bad print -> do not act).
        warn_gate: Report threshold: above this, `noisy` is set (above the
                 noise floor, but too small to flip a decision).
        also_fetch: Additional tickers to load alongside (cache warming; they
                 do NOT enter the consensus).
        today:   Reference date for the freshness check (default: today). For
                 tests/reproduction only.

    Raises:
        ValueError: if no source is configured, no source returns prices, or
                 the sources share no common session date.
    """
    wanted: List[str] = list(dict.fromkeys(t for t in tickers if t))
    if not wanted:
        raise ValueError("no valuation source configured")

    # Load each ticker individually: align="intersection" on a single ticker
    # means "only that ticker's own trading days" -- exactly the real prints the
    # date-matching needs. also_fetch remains pure cache warming.
    series_by_source: Dict[str, pd.Series] = {}
    for ticker in list(dict.fromkeys(wanted + list(also_fetch or []))):
        px = DataLoader.yahoo(
            tickers=[ticker], start=start, end=end, currency="EUR",
            align="intersection", skip_failed=True, load_dividends=False,
            validate=False,
        )
        if ticker not in wanted or ticker not in px.prices.columns:
            continue
        series = px.in_eur()[ticker].dropna()
        if not series.empty:
            series_by_source[ticker] = series

    if not series_by_source:
        raise ValueError("no EUR source reachable (%s)" % ", ".join(wanted))

    common = None
    for series in series_by_source.values():
        common = series.index if common is None else common.intersection(series.index)
    if len(common) == 0:
        raise ValueError(
            "sources share no session date (%s)"
            % ", ".join(sorted(series_by_source)))

    as_of_ts = common.max()
    quotes = {t: float(s.loc[as_of_ts]) for t, s in series_by_source.items()}
    source_dates = {t: s.index.max().date().isoformat()
                    for t, s in series_by_source.items()}

    values = list(quotes.values())
    spread = max(values) / min(values) - 1.0 if len(values) > 1 else 0.0
    return ConsensusPrice(
        _median(values), quotes, spread, gate, warn_gate,
        as_of=as_of_ts.date().isoformat(),
        source_dates=source_dates,
        stale_lag=stale_lag_bdays(as_of_ts, today=today),
    )
