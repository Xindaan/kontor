"""Tests for constituent providers."""

import pytest
from datetime import date
from pathlib import Path
import tempfile

from backtest.constituents.base import (
    ChangelogProvider,
    SnapshotProvider,
    CompositeProvider,
)
from backtest.constituents.models import (
    ConstituentChange,
    ConstituentSnapshot,
    DataQuality,
    MemberIdentifier,
)
from backtest.constituents.user_snapshot import (
    UserSnapshotProvider,
    LocalPITProvider,
)


class MockChangelogProvider(ChangelogProvider):
    """Mock changelog provider for testing."""

    def __init__(self, changes=None, initial_members=None, cache_dir=None):
        super().__init__(
            index_id="MOCK",
            quality=DataQuality.COMMUNITY_CHANGELOG,
            initial_members=initial_members or [],
            initial_date=date(2020, 1, 1),
            cache_dir=cache_dir,
        )
        self._mock_changes = changes or []

    @property
    def id(self):
        return "mock_changelog"

    @property
    def metadata(self):
        from backtest.constituents.models import IndexMetadata
        return IndexMetadata(
            index_id="MOCK",
            name="Mock Index",
            description="Test index",
            target_count=10,
            region="US",
            currency="USD",
            default_quality=DataQuality.COMMUNITY_CHANGELOG,
            available_qualities=[DataQuality.COMMUNITY_CHANGELOG],
        )

    def _load_changes(self):
        return self._mock_changes


class MockSnapshotProvider(SnapshotProvider):
    """Mock snapshot provider for testing."""

    def __init__(self, snapshots=None, cache_dir=None):
        super().__init__(
            index_id="MOCK",
            quality=DataQuality.PROXY_ETF_HOLDINGS,
            cache_dir=cache_dir,
        )
        self._mock_snapshots = snapshots or {}

    @property
    def id(self):
        return "mock_snapshot"

    @property
    def metadata(self):
        from backtest.constituents.models import IndexMetadata
        return IndexMetadata(
            index_id="MOCK",
            name="Mock Index",
            description="Test index",
            target_count=10,
            region="US",
            currency="USD",
            default_quality=DataQuality.PROXY_ETF_HOLDINGS,
            available_qualities=[DataQuality.PROXY_ETF_HOLDINGS],
        )

    def _load_snapshots(self):
        return self._mock_snapshots


class TestChangelogProvider:
    """Tests for ChangelogProvider."""

    def test_empty_changelog(self):
        """Empty changelog returns initial members."""
        initial = [MemberIdentifier.from_ticker("AAPL")]
        provider = MockChangelogProvider(initial_members=initial)

        snapshot = provider.snapshot(date(2024, 1, 15))
        assert snapshot.size == 1
        assert snapshot.contains("AAPL")

    def test_add_change(self):
        """ADD change adds member."""
        initial = [MemberIdentifier.from_ticker("AAPL")]
        changes = [
            ConstituentChange(
                effective_date=date(2024, 1, 10),
                action="ADD",
                member=MemberIdentifier.from_ticker("MSFT"),
            )
        ]
        provider = MockChangelogProvider(initial_members=initial, changes=changes)

        # Before change
        snapshot = provider.snapshot(date(2024, 1, 5))
        assert snapshot.size == 1
        assert snapshot.contains("AAPL")
        assert not snapshot.contains("MSFT")

        # After change
        snapshot = provider.snapshot(date(2024, 1, 15))
        assert snapshot.size == 2
        assert snapshot.contains("AAPL")
        assert snapshot.contains("MSFT")

    def test_remove_change(self):
        """REMOVE change removes member."""
        initial = [
            MemberIdentifier.from_ticker("AAPL"),
            MemberIdentifier.from_ticker("MSFT"),
        ]
        changes = [
            ConstituentChange(
                effective_date=date(2024, 1, 10),
                action="REMOVE",
                member=MemberIdentifier.from_ticker("MSFT"),
            )
        ]
        provider = MockChangelogProvider(initial_members=initial, changes=changes)

        # Before change
        snapshot = provider.snapshot(date(2024, 1, 5))
        assert snapshot.size == 2

        # After change
        snapshot = provider.snapshot(date(2024, 1, 15))
        assert snapshot.size == 1
        assert snapshot.contains("AAPL")
        assert not snapshot.contains("MSFT")

    def test_multiple_changes(self):
        """Multiple changes applied in order."""
        initial = [MemberIdentifier.from_ticker("AAPL")]
        changes = [
            ConstituentChange(
                effective_date=date(2024, 1, 10),
                action="ADD",
                member=MemberIdentifier.from_ticker("MSFT"),
            ),
            ConstituentChange(
                effective_date=date(2024, 2, 15),
                action="ADD",
                member=MemberIdentifier.from_ticker("GOOGL"),
            ),
            ConstituentChange(
                effective_date=date(2024, 3, 20),
                action="REMOVE",
                member=MemberIdentifier.from_ticker("AAPL"),
            ),
        ]
        provider = MockChangelogProvider(initial_members=initial, changes=changes)

        snapshot = provider.snapshot(date(2024, 4, 1))
        assert snapshot.size == 2
        assert not snapshot.contains("AAPL")
        assert snapshot.contains("MSFT")
        assert snapshot.contains("GOOGL")

    def test_available_dates(self):
        """Available dates includes change dates."""
        initial = [MemberIdentifier.from_ticker("AAPL")]
        changes = [
            ConstituentChange(
                effective_date=date(2024, 1, 10),
                action="ADD",
                member=MemberIdentifier.from_ticker("MSFT"),
            ),
            ConstituentChange(
                effective_date=date(2024, 2, 15),
                action="ADD",
                member=MemberIdentifier.from_ticker("GOOGL"),
            ),
        ]
        provider = MockChangelogProvider(initial_members=initial, changes=changes)

        dates = provider.available_dates()
        assert date(2020, 1, 1) in dates  # Initial date
        assert date(2024, 1, 10) in dates
        assert date(2024, 2, 15) in dates


class TestSnapshotProvider:
    """Tests for SnapshotProvider."""

    def test_exact_date(self):
        """Get snapshot for exact date."""
        snapshots = {
            date(2024, 1, 31): ConstituentSnapshot(
                index_id="MOCK",
                as_of=date(2024, 1, 31),
                members=[MemberIdentifier.from_ticker("AAPL")],
                quality=DataQuality.PROXY_ETF_HOLDINGS,
                source="test",
            ),
        }
        provider = MockSnapshotProvider(snapshots=snapshots)

        snapshot = provider.snapshot(date(2024, 1, 31))
        assert snapshot.size == 1
        assert snapshot.as_of == date(2024, 1, 31)

    def test_closest_date(self):
        """Get closest snapshot before requested date."""
        snapshots = {
            date(2024, 1, 31): ConstituentSnapshot(
                index_id="MOCK",
                as_of=date(2024, 1, 31),
                members=[MemberIdentifier.from_ticker("AAPL")],
                quality=DataQuality.PROXY_ETF_HOLDINGS,
                source="test",
            ),
            date(2024, 2, 29): ConstituentSnapshot(
                index_id="MOCK",
                as_of=date(2024, 2, 29),
                members=[
                    MemberIdentifier.from_ticker("AAPL"),
                    MemberIdentifier.from_ticker("MSFT"),
                ],
                quality=DataQuality.PROXY_ETF_HOLDINGS,
                source="test",
            ),
        }
        provider = MockSnapshotProvider(snapshots=snapshots)

        # Request mid-February, should get January snapshot
        snapshot = provider.snapshot(date(2024, 2, 15))
        assert snapshot.size == 1
        assert snapshot.as_of == date(2024, 1, 31)

        # Request March, should get February snapshot
        snapshot = provider.snapshot(date(2024, 3, 15))
        assert snapshot.size == 2
        assert snapshot.as_of == date(2024, 2, 29)

    def test_no_data_before_date(self):
        """Return empty snapshot if no data before date."""
        snapshots = {
            date(2024, 6, 30): ConstituentSnapshot(
                index_id="MOCK",
                as_of=date(2024, 6, 30),
                members=[MemberIdentifier.from_ticker("AAPL")],
                quality=DataQuality.PROXY_ETF_HOLDINGS,
                source="test",
            ),
        }
        provider = MockSnapshotProvider(snapshots=snapshots)

        snapshot = provider.snapshot(date(2024, 1, 15))
        assert snapshot.size == 0
        assert "warning" in snapshot.meta

    def test_available_dates(self):
        """Available dates returns all snapshot dates."""
        snapshots = {
            date(2024, 1, 31): ConstituentSnapshot(
                index_id="MOCK",
                as_of=date(2024, 1, 31),
                members=[],
                quality=DataQuality.PROXY_ETF_HOLDINGS,
                source="test",
            ),
            date(2024, 2, 29): ConstituentSnapshot(
                index_id="MOCK",
                as_of=date(2024, 2, 29),
                members=[],
                quality=DataQuality.PROXY_ETF_HOLDINGS,
                source="test",
            ),
        }
        provider = MockSnapshotProvider(snapshots=snapshots)

        dates = provider.available_dates()
        assert len(dates) == 2
        assert date(2024, 1, 31) in dates
        assert date(2024, 2, 29) in dates


class TestCompositeProvider:
    """Tests for CompositeProvider."""

    def test_uses_highest_quality(self):
        """Uses highest quality provider with data."""
        # High quality but no data for early dates
        high_quality = MockSnapshotProvider(snapshots={
            date(2024, 6, 30): ConstituentSnapshot(
                index_id="MOCK",
                as_of=date(2024, 6, 30),
                members=[MemberIdentifier.from_ticker("NEW")],
                quality=DataQuality.OFFICIAL_CHANGELOG,
                source="high",
            ),
        })
        high_quality.quality = DataQuality.OFFICIAL_CHANGELOG

        # Lower quality with more history
        low_quality = MockSnapshotProvider(snapshots={
            date(2024, 1, 31): ConstituentSnapshot(
                index_id="MOCK",
                as_of=date(2024, 1, 31),
                members=[MemberIdentifier.from_ticker("OLD")],
                quality=DataQuality.COMMUNITY_CHANGELOG,
                source="low",
            ),
        })
        low_quality.quality = DataQuality.COMMUNITY_CHANGELOG

        composite = CompositeProvider(
            index_id="MOCK",
            providers=[low_quality, high_quality],
        )

        # Early date - uses low quality (only available)
        snapshot = composite.snapshot(date(2024, 2, 15))
        assert snapshot.contains("OLD")

        # Later date - uses high quality
        snapshot = composite.snapshot(date(2024, 7, 15))
        assert snapshot.contains("NEW")

    def test_combines_available_dates(self):
        """Combines dates from all providers."""
        p1 = MockSnapshotProvider(snapshots={
            date(2024, 1, 31): ConstituentSnapshot(
                index_id="MOCK",
                as_of=date(2024, 1, 31),
                members=[],
                quality=DataQuality.PROXY_ETF_HOLDINGS,
                source="p1",
            ),
        })
        p2 = MockSnapshotProvider(snapshots={
            date(2024, 2, 29): ConstituentSnapshot(
                index_id="MOCK",
                as_of=date(2024, 2, 29),
                members=[],
                quality=DataQuality.COMMUNITY_CHANGELOG,
                source="p2",
            ),
        })

        composite = CompositeProvider(index_id="MOCK", providers=[p1, p2])
        dates = composite.available_dates()

        assert date(2024, 1, 31) in dates
        assert date(2024, 2, 29) in dates


class TestUserSnapshotProvider:
    """Tests for UserSnapshotProvider."""

    def test_load_long_format(self):
        """Load CSV in long format."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("as_of,ticker\n")
            f.write("2024-01-15,AAPL\n")
            f.write("2024-01-15,MSFT\n")
            f.write("2024-02-15,AAPL\n")
            f.write("2024-02-15,MSFT\n")
            f.write("2024-02-15,GOOGL\n")
            path = f.name

        try:
            provider = UserSnapshotProvider(path=path, index_id="TEST")
            provider._load_snapshots()

            dates = provider.available_dates()
            assert len(dates) == 2

            snapshot = provider.snapshot(date(2024, 1, 15))
            assert snapshot.size == 2

            snapshot = provider.snapshot(date(2024, 2, 15))
            assert snapshot.size == 3
        finally:
            Path(path).unlink()

    def test_quality_is_user_snapshot(self):
        """Quality should be USER_SNAPSHOT."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("as_of,ticker\n")
            f.write("2024-01-15,AAPL\n")
            path = f.name

        try:
            provider = UserSnapshotProvider(path=path)
            assert provider.quality == DataQuality.USER_SNAPSHOT
        finally:
            Path(path).unlink()
