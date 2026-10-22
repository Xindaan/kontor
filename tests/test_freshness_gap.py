"""Gap diagnosis must keep source absence separate from calendar evidence.

A later print can follow both a holiday and a feed outage. These observations
cannot distinguish the causes; without a verified calendar both require retry.
"""

import datetime

import numpy as np
import pandas as pd
import pytest

from backtest.freshness import (GAP_DELIVERY_DELAY, GAP_NONE, GAP_SOURCES_STALLED,
                                GAP_UNEXPLAINED, FreshnessError, classify_gap)

WED = datetime.date(2026, 9, 2)          # run day; last expected close = Tue 01.09.


def _days(*iso):
    return [datetime.date.fromisoformat(d) for d in iso]


# The four trading days before 02.09.: Thu 27.08., Fri 28.08., Mon 31.08., Tue 01.09.
CONTINENT = ('2026-08-27', '2026-08-28', '2026-08-31', '2026-09-01')
UK_WITH_HOLIDAY = ('2026-08-27', '2026-08-28', '2026-09-01')     # 31.08. bank holiday

LSE, MIL = 'LINE.L', 'LINE.MI'


class TestDeliveryDelay:
    """One side has the bar, the other is not there yet."""

    def test_delay_is_classified_as_delay(self):
        b = classify_gap({LSE: _days(*UK_WITH_HOLIDAY),
                          MIL: _days(*CONTINENT[:3])}, today=WED)
        assert b.case == GAP_DELIVERY_DELAY
        assert b.retry_worthwhile is True

    def test_delayed_day_is_named_and_holiday_kept_separate(self):
        b = classify_gap({LSE: _days(*UK_WITH_HOLIDAY),
                          MIL: _days(*CONTINENT[:3])}, today=WED)
        per_day = {d.date: d for d in b.days}
        assert per_day[datetime.date(2026, 9, 1)].case == GAP_DELIVERY_DELAY
        assert per_day[datetime.date(2026, 9, 1)].laggards == (MIL,)
        assert per_day[datetime.date(2026, 9, 1)].have == (LSE,)
        # The UK holiday next to it is NOT a delay -- exactly the case where the
        # obvious rule "one side has it" goes wrong.
        assert per_day[datetime.date(2026, 8, 31)].case == GAP_UNEXPLAINED
        assert per_day[datetime.date(2026, 8, 31)].skipped == (LSE,)

    def test_text_names_the_case_not_just_stale(self):
        text = classify_gap({LSE: _days(*UK_WITH_HOLIDAY),
                             MIL: _days(*CONTINENT[:3])}, today=WED).describe()
        assert 'DELIVERY DELAY' in text and 'RETRY WORTHWHILE' in text
        assert '2026-09-01' in text

    def test_a_delayed_day_dominates_all_holidays(self):
        """The overall case is the sharpest single statement."""
        b = classify_gap({LSE: _days(*UK_WITH_HOLIDAY),
                          MIL: _days(*CONTINENT[:3])}, today=WED)
        assert {d.case for d in b.days} == {GAP_UNEXPLAINED, GAP_DELIVERY_DELAY}
        assert b.case == GAP_DELIVERY_DELAY


class TestSkippedDay:
    """Later prints alone do not prove a holiday."""

    TUE = datetime.date(2026, 9, 1)      # last expected close = Mon 31.08.
    LINES = {LSE: ('2026-08-27', '2026-08-28', '2026-09-01'),
             MIL: ('2026-08-27', '2026-08-28', '2026-08-31')}

    def _finding(self):
        return classify_gap({q: _days(*d) for q, d in self.LINES.items()}, today=self.TUE)

    def test_presumed_holiday_without_calendar_stays_a_retry_reason(self):
        b = self._finding()
        assert b.case == GAP_UNEXPLAINED
        assert b.retry_worthwhile is True

    def test_text_names_the_case(self):
        text = self._finding().describe()
        assert 'GAP UNEXPLAINED' in text and 'RETRY WORTHWHILE' in text
        assert 'DELIVERY DELAY' not in text

    def test_the_failed_source_is_not_listed_as_closed(self):
        day = next(d for d in self._finding().days if d.date == datetime.date(2026, 8, 31))
        assert day.skipped == (LSE,) and day.laggards == ()
        assert day.closed == ()

    def test_on_the_holiday_itself_the_data_cannot_tell_yet(self):
        """Without the later print there is no proof; the finding must not claim a holiday."""
        b = classify_gap({LSE: _days('2026-08-27', '2026-08-28'),
                          MIL: _days('2026-08-27', '2026-08-28', '2026-08-31')},
                         today=self.TUE)
        assert b.case == GAP_DELIVERY_DELAY and b.retry_worthwhile is True


class TestSourcesStalled:
    """All sources stop at the same edge -- the data cannot decide."""

    def test_all_sources_at_the_same_edge_claim_no_holiday(self):
        b = classify_gap({LSE: _days(*CONTINENT[:3]), MIL: _days(*CONTINENT[:3])}, today=WED)
        assert b.case == GAP_SOURCES_STALLED
        assert b.retry_worthwhile is True
        assert 'not decidable' in b.describe()

    def test_outage_is_not_the_same_case_as_a_holiday(self):
        """Otherwise a global feed outage would pass silently as a 'holiday'.

        In the holiday fixture BOTH sources reached the last expected close (the
        UK line skipped the bank holiday and printed afterwards). Nothing is
        outstanding, so a retry fetches nothing: ``retry_worthwhile is False``.
        Both sides are pinned instead of a bare ``!=``, which would also stay
        green if both cases collapsed into a third value.
        """
        stalled = classify_gap({'a': _days(*CONTINENT[:3]), 'b': _days(*CONTINENT[:3])},
                               today=WED)
        holiday = classify_gap({'a': _days(*UK_WITH_HOLIDAY), 'b': _days(*CONTINENT)},
                               today=WED)
        assert stalled.case == GAP_SOURCES_STALLED
        assert stalled.retry_worthwhile is True
        assert holiday.case == GAP_NONE
        assert holiday.retry_worthwhile is False
        assert stalled.case != holiday.case


class TestNoGap:
    def test_complete_data_is_green(self):
        b = classify_gap({LSE: _days(*CONTINENT), MIL: _days(*CONTINENT)}, today=WED)
        assert b.case == GAP_NONE
        assert b.days == () and b.describe() == 'NO GAP'

    def test_without_sources_there_is_no_verdict(self):
        with pytest.raises(FreshnessError):
            classify_gap({LSE: []}, today=WED)


class TestThroughTheConsensusPath:
    """What the consensus finally SAYS, not only the arithmetic helper."""

    def _consensus(self, monkeypatch, frame, today):
        from backtest.data import PriceData
        from backtest.portfolio_pricing import consensus_price_eur

        def fake_yahoo(**kwargs):
            cols = [t for t in kwargs['tickers'] if t in frame.columns]
            return PriceData(prices=frame[cols].copy(), currency={c: 'EUR' for c in cols})
        monkeypatch.setattr('backtest.portfolio_pricing.DataLoader.yahoo',
                            staticmethod(fake_yahoo))
        return consensus_price_eur([MIL, LSE], '2026-08-01', '2026-09-02', today=today)

    def _frame(self):
        """`.L` has 01.09., `.MI` stops at 31.08., and `.L` lacks 31.08. (bank holiday)."""
        idx = pd.to_datetime(['2026-08-27', '2026-08-28', '2026-08-31', '2026-09-01'])
        return pd.DataFrame({MIL: [150.0, 151.0, 149.0, np.nan],
                             LSE: [150.2, 151.2, np.nan, 148.8]}, index=idx)

    def test_consensus_is_stale_and_names_the_cause(self, monkeypatch):
        c = self._consensus(monkeypatch, self._frame(), datetime.date(2026, 9, 3))
        assert c.stale, 'precondition: without stale there is nothing to explain'
        finding = c.gap(today=datetime.date(2026, 9, 3))
        assert finding is not None and finding.case == GAP_DELIVERY_DELAY
        assert finding.retry_worthwhile is True

    def test_consensus_carries_the_print_dates(self, monkeypatch):
        """The diagnosis needs the whole history, not just the last print."""
        c = self._consensus(monkeypatch, self._frame(), datetime.date(2026, 9, 3))
        assert c.session_dates[LSE] == ('2026-08-27', '2026-08-28', '2026-09-01')
        assert c.session_dates[MIL] == ('2026-08-27', '2026-08-28', '2026-08-31')

    def test_without_print_dates_there_is_no_guessed_verdict(self):
        from backtest.portfolio_pricing import ConsensusPrice
        assert ConsensusPrice(1.0, {}, 0.0, 0.1).gap() is None
