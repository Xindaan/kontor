"""A snapshot without a changelog is not a point-in-time snapshot.

The changelog parsers (Wikipedia HTML, SEC N-PORT XML, PDF) silently return an
empty list when the raw material changed or is empty. The replay loop then has
nothing to undo, and `snapshot(as_of)` returns today's member list for EVERY
historical date. Flagged as `point_in_time=True`, the universe look-ahead guard
would let it pass -- a survivorship run reported as clean.
"""
import datetime as dt

import pytest

from backtest.constituents.base import ChangelogProvider
from backtest.constituents.models import (
    ConstituentChange,
    DataQuality,
    IdentifierType,
    IndexMetadata,
    MemberIdentifier,
)


def _member(symbol):
    return MemberIdentifier(primary_id=symbol, id_type=IdentifierType.TICKER, ticker=symbol)


class _Changelog(ChangelogProvider):
    """Minimal provider whose changelog the test controls."""

    def __init__(self, changes, tmp_path=None):
        super().__init__(
            index_id='TESTIDX',
            quality=DataQuality.COMMUNITY_CHANGELOG,
            initial_members=[_member('AAA'), _member('BBB')],
            initial_date=dt.date(2020, 1, 1),
            cache_dir=tmp_path,
        )
        self._test_changes = list(changes)

    @property
    def id(self):
        return 'test_changelog'

    @property
    def metadata(self):
        return IndexMetadata(index_id='TESTIDX', name='Test Index', provider='test')

    def _load_changes(self, force=False):
        return list(self._test_changes)


class TestEmptyChangelogIsNotPit:
    def test_without_changes_the_snapshot_is_not_pit(self):
        snap = _Changelog([]).snapshot(dt.date(2015, 6, 1))
        assert snap.point_in_time is False, (
            "Today's member list poses as a historical snapshot")

    def test_with_changes_the_snapshot_stays_pit(self):
        p = _Changelog([ConstituentChange(effective_date=dt.date(2021, 3, 1),
                                          action='ADD', member=_member('CCC'))])
        assert p.snapshot(dt.date(2022, 1, 1)).point_in_time is True

    def test_the_members_themselves_are_unchanged(self):
        """The fix only changes the flag, not the result."""
        snap = _Changelog([]).snapshot(dt.date(2015, 6, 1))
        assert {m.primary_id for m in snap.members} == {'AAA', 'BBB'}


class TestInteractionWithTheLookaheadGuard:
    def test_non_pit_snapshot_is_blocked_by_the_guard(self):
        from backtest.universe import (
            UniverseSnapshot,
            UniverseValidationError,
            validate_universe_for_backtest,
        )

        snap = _Changelog([]).snapshot(dt.date(2015, 6, 1))
        us = UniverseSnapshot(as_of=dt.date(2024, 1, 1),
                              tickers=[m.primary_id for m in snap.members],
                              source='test_changelog', point_in_time=snap.point_in_time)
        with pytest.raises(UniverseValidationError):
            validate_universe_for_backtest(us, dt.date(2015, 1, 1))
