"""
Data models for historical index constituents.

Defines core data structures with explicit quality and identifier metadata.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Set
import hashlib


class DataQuality(Enum):
    """
    Quality level of constituent data source.

    Priority order (highest to lowest):
    1. OFFICIAL_CHANGELOG - Direct from index provider
    2. PROXY_ETF_HOLDINGS - ETF holdings as proxy
    3. COMMUNITY_CHANGELOG - Community-maintained (e.g., Wikipedia)
    4. USER_SNAPSHOT - User-provided data
    """

    OFFICIAL_CHANGELOG = "official_changelog"
    PROXY_ETF_HOLDINGS = "proxy_etf_holdings"
    COMMUNITY_CHANGELOG = "community_changelog"
    USER_SNAPSHOT = "user_snapshot"

    @property
    def priority(self) -> int:
        """Higher number = higher priority."""
        priorities = {
            DataQuality.OFFICIAL_CHANGELOG: 4,
            DataQuality.PROXY_ETF_HOLDINGS: 3,
            DataQuality.COMMUNITY_CHANGELOG: 2,
            DataQuality.USER_SNAPSHOT: 1,
        }
        return priorities[self]

    @property
    def description(self) -> str:
        """Human-readable description."""
        descriptions = {
            DataQuality.OFFICIAL_CHANGELOG: "Official index provider changelog",
            DataQuality.PROXY_ETF_HOLDINGS: "ETF holdings as proxy (may differ from index)",
            DataQuality.COMMUNITY_CHANGELOG: "Community-maintained data (not audit-grade)",
            DataQuality.USER_SNAPSHOT: "User-provided snapshot data",
        }
        return descriptions[self]


class IdentifierType(Enum):
    """
    Type of security identifier.

    Different sources provide different identifier types.
    ISIN and CUSIP are preferred for stability.
    """

    TICKER = "ticker"  # Exchange ticker symbol (may change)
    CUSIP = "cusip"  # CUSIP (9-char, US/Canada)
    ISIN = "isin"  # ISIN (12-char, international)
    SEDOL = "sedol"  # SEDOL (7-char, UK)
    JP_CODE = "jp_code"  # Japanese security code
    NAME = "name"  # Company name only (least stable)

    @property
    def stability(self) -> int:
        """Higher number = more stable identifier."""
        stabilities = {
            IdentifierType.ISIN: 5,
            IdentifierType.CUSIP: 4,
            IdentifierType.SEDOL: 4,
            IdentifierType.JP_CODE: 3,
            IdentifierType.TICKER: 2,
            IdentifierType.NAME: 1,
        }
        return stabilities[self]


@dataclass
class MemberIdentifier:
    """
    Security identifier with type information.

    A member may have multiple identifiers (e.g., ISIN + ticker).
    The primary_id is the most stable available identifier.
    """

    primary_id: str
    id_type: IdentifierType
    ticker: Optional[str] = None
    isin: Optional[str] = None
    cusip: Optional[str] = None
    sedol: Optional[str] = None
    name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.primary_id, self.id_type))

    def __eq__(self, other) -> bool:
        if not isinstance(other, MemberIdentifier):
            return False
        return self.primary_id == other.primary_id and self.id_type == other.id_type

    @classmethod
    def from_ticker(cls, ticker: str, name: Optional[str] = None) -> "MemberIdentifier":
        """Create identifier from ticker symbol."""
        return cls(
            primary_id=ticker,
            id_type=IdentifierType.TICKER,
            ticker=ticker,
            name=name,
        )

    @classmethod
    def from_isin(
        cls, isin: str, ticker: Optional[str] = None, name: Optional[str] = None
    ) -> "MemberIdentifier":
        """Create identifier from ISIN."""
        return cls(
            primary_id=isin,
            id_type=IdentifierType.ISIN,
            isin=isin,
            ticker=ticker,
            name=name,
        )

    @classmethod
    def from_cusip(
        cls, cusip: str, ticker: Optional[str] = None, name: Optional[str] = None
    ) -> "MemberIdentifier":
        """Create identifier from CUSIP."""
        return cls(
            primary_id=cusip,
            id_type=IdentifierType.CUSIP,
            cusip=cusip,
            ticker=ticker,
            name=name,
        )

    @classmethod
    def from_jp_code(
        cls, jp_code: str, name: Optional[str] = None
    ) -> "MemberIdentifier":
        """Create identifier from Japanese security code."""
        return cls(
            primary_id=jp_code,
            id_type=IdentifierType.JP_CODE,
            name=name,
            extra={"jp_code": jp_code},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "primary_id": self.primary_id,
            "id_type": self.id_type.value,
            "ticker": self.ticker,
            "isin": self.isin,
            "cusip": self.cusip,
            "sedol": self.sedol,
            "name": self.name,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemberIdentifier":
        """Deserialize from dictionary."""
        return cls(
            primary_id=data["primary_id"],
            id_type=IdentifierType(data["id_type"]),
            ticker=data.get("ticker"),
            isin=data.get("isin"),
            cusip=data.get("cusip"),
            sedol=data.get("sedol"),
            name=data.get("name"),
            extra=data.get("extra", {}),
        )


@dataclass
class ConstituentChange:
    """
    A single addition or removal from an index.

    Used by changelog-based providers to track membership changes.
    """

    effective_date: date
    action: str  # "ADD" or "REMOVE"
    member: MemberIdentifier
    reason: Optional[str] = None
    replacing: Optional[MemberIdentifier] = None  # For replacements
    source_document: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.effective_date, datetime):
            self.effective_date = self.effective_date.date()
        if self.action not in ("ADD", "REMOVE"):
            raise ValueError(f"action must be ADD or REMOVE, got {self.action}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effective_date": self.effective_date.isoformat(),
            "action": self.action,
            "member": self.member.to_dict(),
            "reason": self.reason,
            "replacing": self.replacing.to_dict() if self.replacing else None,
            "source_document": self.source_document,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstituentChange":
        return cls(
            effective_date=date.fromisoformat(data["effective_date"]),
            action=data["action"],
            member=MemberIdentifier.from_dict(data["member"]),
            reason=data.get("reason"),
            replacing=MemberIdentifier.from_dict(data["replacing"])
            if data.get("replacing")
            else None,
            source_document=data.get("source_document"),
        )


@dataclass
class ConstituentSnapshot:
    """
    Point-in-time snapshot of index members with quality metadata.

    This is the primary output for backtest consumption.
    """

    index_id: str
    as_of: date
    members: List[MemberIdentifier]
    quality: DataQuality
    source: str
    point_in_time: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.as_of, datetime):
            self.as_of = self.as_of.date()
        # Deduplicate members
        seen = set()
        unique_members = []
        for m in self.members:
            key = (m.primary_id, m.id_type)
            if key not in seen:
                seen.add(key)
                unique_members.append(m)
        self.members = unique_members

    @property
    def size(self) -> int:
        """Number of members in the snapshot."""
        return len(self.members)

    @property
    def tickers(self) -> List[str]:
        """Extract ticker symbols from members (if available)."""
        result = []
        for m in self.members:
            if m.ticker:
                result.append(m.ticker)
            elif m.id_type == IdentifierType.TICKER:
                result.append(m.primary_id)
        return sorted(result)

    @property
    def hash(self) -> str:
        """Compute hash for reproducibility check."""
        ids = sorted([m.primary_id for m in self.members])
        content = f"{self.index_id}|{self.as_of.isoformat()}|{','.join(ids)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def contains(self, identifier: str) -> bool:
        """Check if identifier (ticker/isin/cusip) is in snapshot."""
        identifier_upper = identifier.upper()
        for m in self.members:
            if m.primary_id.upper() == identifier_upper:
                return True
            if m.ticker and m.ticker.upper() == identifier_upper:
                return True
            if m.isin and m.isin.upper() == identifier_upper:
                return True
            if m.cusip and m.cusip.upper() == identifier_upper:
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "index_id": self.index_id,
            "as_of": self.as_of.isoformat(),
            "members": [m.to_dict() for m in self.members],
            "quality": self.quality.value,
            "source": self.source,
            "point_in_time": self.point_in_time,
            "meta": self.meta,
            "hash": self.hash,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstituentSnapshot":
        """Deserialize from dictionary."""
        return cls(
            index_id=data["index_id"],
            as_of=date.fromisoformat(data["as_of"]),
            members=[MemberIdentifier.from_dict(m) for m in data["members"]],
            quality=DataQuality(data["quality"]),
            source=data["source"],
            point_in_time=data.get("point_in_time", True),
            meta=data.get("meta", {}),
        )

    def to_universe_snapshot(self):
        """Convert to legacy UniverseSnapshot format for compatibility."""
        from ..universe import UniverseSnapshot

        return UniverseSnapshot(
            as_of=self.as_of,
            tickers=self.tickers,
            source=self.source,
            point_in_time=self.point_in_time,
            meta={
                **self.meta,
                "quality": self.quality.value,
                "index_id": self.index_id,
                "member_count": self.size,
            },
        )

    def __repr__(self) -> str:
        return (
            f"ConstituentSnapshot(index={self.index_id}, as_of={self.as_of}, "
            f"size={self.size}, quality={self.quality.value})"
        )


@dataclass
class IndexMetadata:
    """
    Metadata about an index and its data sources.
    """

    index_id: str
    name: str
    description: str
    target_count: Optional[int]  # Expected number of constituents
    region: str  # US, EU, JP, UK, etc.
    currency: str  # Primary currency
    default_quality: DataQuality
    available_qualities: List[DataQuality]
    coverage_start: Optional[date] = None
    coverage_end: Optional[date] = None
    frequency: str = "event-based"  # "event-based", "monthly", "quarterly"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_id": self.index_id,
            "name": self.name,
            "description": self.description,
            "target_count": self.target_count,
            "region": self.region,
            "currency": self.currency,
            "default_quality": self.default_quality.value,
            "available_qualities": [q.value for q in self.available_qualities],
            "coverage_start": self.coverage_start.isoformat()
            if self.coverage_start
            else None,
            "coverage_end": self.coverage_end.isoformat() if self.coverage_end else None,
            "frequency": self.frequency,
            "notes": self.notes,
        }
