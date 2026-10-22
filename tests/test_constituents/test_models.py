"""Tests for constituent data models."""

import pytest
from datetime import date

from backtest.constituents.models import (
    DataQuality,
    IdentifierType,
    MemberIdentifier,
    ConstituentChange,
    ConstituentSnapshot,
    IndexMetadata,
)


class TestDataQuality:
    """Tests for DataQuality enum."""

    def test_priority_ordering(self):
        """Official changelog should have highest priority."""
        assert DataQuality.OFFICIAL_CHANGELOG.priority > DataQuality.PROXY_ETF_HOLDINGS.priority
        assert DataQuality.PROXY_ETF_HOLDINGS.priority > DataQuality.COMMUNITY_CHANGELOG.priority
        assert DataQuality.COMMUNITY_CHANGELOG.priority > DataQuality.USER_SNAPSHOT.priority

    def test_descriptions(self):
        """All quality levels should have descriptions."""
        for quality in DataQuality:
            assert quality.description
            assert len(quality.description) > 10


class TestIdentifierType:
    """Tests for IdentifierType enum."""

    def test_stability_ranking(self):
        """ISIN should be most stable."""
        assert IdentifierType.ISIN.stability > IdentifierType.TICKER.stability
        assert IdentifierType.CUSIP.stability > IdentifierType.TICKER.stability
        assert IdentifierType.TICKER.stability > IdentifierType.NAME.stability


class TestMemberIdentifier:
    """Tests for MemberIdentifier."""

    def test_from_ticker(self):
        """Create identifier from ticker."""
        m = MemberIdentifier.from_ticker("AAPL", name="Apple Inc.")
        assert m.primary_id == "AAPL"
        assert m.id_type == IdentifierType.TICKER
        assert m.ticker == "AAPL"
        assert m.name == "Apple Inc."

    def test_from_isin(self):
        """Create identifier from ISIN."""
        m = MemberIdentifier.from_isin("US0378331005", ticker="AAPL", name="Apple")
        assert m.primary_id == "US0378331005"
        assert m.id_type == IdentifierType.ISIN
        assert m.isin == "US0378331005"
        assert m.ticker == "AAPL"

    def test_from_cusip(self):
        """Create identifier from CUSIP."""
        m = MemberIdentifier.from_cusip("037833100", ticker="AAPL")
        assert m.primary_id == "037833100"
        assert m.id_type == IdentifierType.CUSIP
        assert m.cusip == "037833100"

    def test_from_jp_code(self):
        """Create identifier from Japanese code."""
        m = MemberIdentifier.from_jp_code("7203", name="Toyota Motor")
        assert m.primary_id == "7203"
        assert m.id_type == IdentifierType.JP_CODE
        assert m.name == "Toyota Motor"

    def test_serialization(self):
        """Test to_dict and from_dict."""
        m = MemberIdentifier.from_ticker("MSFT", name="Microsoft")
        d = m.to_dict()

        restored = MemberIdentifier.from_dict(d)
        assert restored.primary_id == m.primary_id
        assert restored.id_type == m.id_type
        assert restored.name == m.name

    def test_equality(self):
        """Members with same ID and type are equal."""
        m1 = MemberIdentifier.from_ticker("AAPL")
        m2 = MemberIdentifier.from_ticker("AAPL")
        m3 = MemberIdentifier.from_ticker("MSFT")

        assert m1 == m2
        assert m1 != m3
        assert hash(m1) == hash(m2)


class TestConstituentChange:
    """Tests for ConstituentChange."""

    def test_create_add(self):
        """Create addition change."""
        change = ConstituentChange(
            effective_date=date(2024, 1, 15),
            action="ADD",
            member=MemberIdentifier.from_ticker("NVDA"),
        )
        assert change.action == "ADD"
        assert change.effective_date == date(2024, 1, 15)

    def test_create_remove(self):
        """Create removal change."""
        change = ConstituentChange(
            effective_date=date(2024, 1, 15),
            action="REMOVE",
            member=MemberIdentifier.from_ticker("GE"),
            reason="Spin-off",
        )
        assert change.action == "REMOVE"
        assert change.reason == "Spin-off"

    def test_invalid_action(self):
        """Invalid action raises error."""
        with pytest.raises(ValueError):
            ConstituentChange(
                effective_date=date(2024, 1, 15),
                action="INVALID",
                member=MemberIdentifier.from_ticker("AAPL"),
            )

    def test_serialization(self):
        """Test to_dict and from_dict."""
        change = ConstituentChange(
            effective_date=date(2024, 1, 15),
            action="ADD",
            member=MemberIdentifier.from_ticker("NVDA"),
            source_document="test.pdf",
        )
        d = change.to_dict()

        restored = ConstituentChange.from_dict(d)
        assert restored.effective_date == change.effective_date
        assert restored.action == change.action
        assert restored.member.primary_id == change.member.primary_id


class TestConstituentSnapshot:
    """Tests for ConstituentSnapshot."""

    def test_create_snapshot(self):
        """Create a snapshot."""
        members = [
            MemberIdentifier.from_ticker("AAPL"),
            MemberIdentifier.from_ticker("MSFT"),
            MemberIdentifier.from_ticker("GOOGL"),
        ]
        snapshot = ConstituentSnapshot(
            index_id="SP500",
            as_of=date(2024, 1, 15),
            members=members,
            quality=DataQuality.PROXY_ETF_HOLDINGS,
            source="test",
        )
        assert snapshot.size == 3
        assert snapshot.index_id == "SP500"
        assert snapshot.point_in_time is True

    def test_tickers_property(self):
        """Extract tickers from members."""
        members = [
            MemberIdentifier.from_ticker("AAPL"),
            MemberIdentifier.from_isin("US5949181045", ticker="MSFT"),
        ]
        snapshot = ConstituentSnapshot(
            index_id="SP500",
            as_of=date(2024, 1, 15),
            members=members,
            quality=DataQuality.PROXY_ETF_HOLDINGS,
            source="test",
        )
        tickers = snapshot.tickers
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_contains(self):
        """Check if identifier is in snapshot."""
        members = [
            MemberIdentifier.from_ticker("AAPL"),
            MemberIdentifier.from_isin("US5949181045", ticker="MSFT"),
        ]
        snapshot = ConstituentSnapshot(
            index_id="SP500",
            as_of=date(2024, 1, 15),
            members=members,
            quality=DataQuality.PROXY_ETF_HOLDINGS,
            source="test",
        )
        assert snapshot.contains("AAPL")
        assert snapshot.contains("MSFT")
        assert snapshot.contains("US5949181045")
        assert not snapshot.contains("GOOGL")

    def test_deduplication(self):
        """Duplicate members are removed."""
        members = [
            MemberIdentifier.from_ticker("AAPL"),
            MemberIdentifier.from_ticker("AAPL"),  # Duplicate
            MemberIdentifier.from_ticker("MSFT"),
        ]
        snapshot = ConstituentSnapshot(
            index_id="SP500",
            as_of=date(2024, 1, 15),
            members=members,
            quality=DataQuality.PROXY_ETF_HOLDINGS,
            source="test",
        )
        assert snapshot.size == 2

    def test_hash_reproducible(self):
        """Same members produce same hash."""
        members = [
            MemberIdentifier.from_ticker("AAPL"),
            MemberIdentifier.from_ticker("MSFT"),
        ]
        s1 = ConstituentSnapshot(
            index_id="SP500",
            as_of=date(2024, 1, 15),
            members=members,
            quality=DataQuality.PROXY_ETF_HOLDINGS,
            source="test",
        )
        s2 = ConstituentSnapshot(
            index_id="SP500",
            as_of=date(2024, 1, 15),
            members=list(reversed(members)),  # Different order
            quality=DataQuality.PROXY_ETF_HOLDINGS,
            source="test",
        )
        assert s1.hash == s2.hash

    def test_serialization(self):
        """Test to_dict and from_dict."""
        members = [MemberIdentifier.from_ticker("AAPL")]
        snapshot = ConstituentSnapshot(
            index_id="SP500",
            as_of=date(2024, 1, 15),
            members=members,
            quality=DataQuality.PROXY_ETF_HOLDINGS,
            source="test",
            meta={"test_key": "test_value"},
        )
        d = snapshot.to_dict()

        restored = ConstituentSnapshot.from_dict(d)
        assert restored.index_id == snapshot.index_id
        assert restored.as_of == snapshot.as_of
        assert restored.size == snapshot.size
        assert restored.quality == snapshot.quality


class TestIndexMetadata:
    """Tests for IndexMetadata."""

    def test_create_metadata(self):
        """Create index metadata."""
        meta = IndexMetadata(
            index_id="SP500",
            name="S&P 500",
            description="500 largest US companies",
            target_count=500,
            region="US",
            currency="USD",
            default_quality=DataQuality.PROXY_ETF_HOLDINGS,
            available_qualities=[
                DataQuality.PROXY_ETF_HOLDINGS,
                DataQuality.COMMUNITY_CHANGELOG,
            ],
        )
        assert meta.index_id == "SP500"
        assert meta.target_count == 500
        assert len(meta.available_qualities) == 2

    def test_serialization(self):
        """Test to_dict."""
        meta = IndexMetadata(
            index_id="SP500",
            name="S&P 500",
            description="500 largest US companies",
            target_count=500,
            region="US",
            currency="USD",
            default_quality=DataQuality.PROXY_ETF_HOLDINGS,
            available_qualities=[DataQuality.PROXY_ETF_HOLDINGS],
            coverage_start=date(2019, 1, 1),
        )
        d = meta.to_dict()
        assert d["index_id"] == "SP500"
        assert d["coverage_start"] == "2019-01-01"
