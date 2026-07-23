"""
Abstract base classes for constituent data providers.

Two provider patterns:
1. ChangelogProvider - Builds membership from add/remove events
2. SnapshotProvider - Provides discrete snapshots (e.g., monthly ETF holdings)
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import json
import logging

from .models import (
    DataQuality,
    ConstituentChange,
    ConstituentSnapshot,
    MemberIdentifier,
    IndexMetadata,
)

logger = logging.getLogger(__name__)


class ConstituentProvider(ABC):
    """
    Abstract base class for all constituent data providers.

    Subclasses implement either changelog-based or snapshot-based
    data retrieval. All providers must support the snapshot() method.
    """

    def __init__(
        self,
        index_id: str,
        quality: DataQuality,
        cache_dir: Optional[Path] = None,
    ):
        self.index_id = index_id
        self.quality = quality
        self._cache_dir = cache_dir or Path("data/constituents_cache")

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this provider configuration."""
        ...

    @property
    @abstractmethod
    def metadata(self) -> IndexMetadata:
        """Get metadata about this index."""
        ...

    @abstractmethod
    def snapshot(self, as_of: date) -> ConstituentSnapshot:
        """
        Get index members as of a specific date.

        Args:
            as_of: The date to query

        Returns:
            ConstituentSnapshot with members and quality metadata
        """
        ...

    @abstractmethod
    def available_dates(self) -> List[date]:
        """
        Get list of dates with available data.

        For changelog providers, this might be all dates from first to last change.
        For snapshot providers, this is the list of snapshot dates.
        """
        ...

    def download(self, force: bool = False) -> None:
        """
        Download/refresh data from source.

        Args:
            force: If True, redownload even if cached
        """
        pass  # Optional - subclasses override if needed

    def is_cached(self) -> bool:
        """Check if data is already cached locally."""
        return False  # Override in subclasses

    def _get_cache_path(self, suffix: str = "") -> Path:
        """Get cache file path for this provider."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.index_id}_{self.quality.value}{suffix}.json"
        return self._cache_dir / filename

    def _load_cache(self, suffix: str = "") -> Optional[Dict[str, Any]]:
        """Load data from cache."""
        path = self._get_cache_path(suffix)
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {path}: {e}")
        return None

    def _save_cache(self, data: Dict[str, Any], suffix: str = "") -> None:
        """Save data to cache."""
        path = self._get_cache_path(suffix)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved cache: {path}")
        except Exception as e:
            logger.warning(f"Failed to save cache {path}: {e}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(index={self.index_id}, quality={self.quality.value})"


class ChangelogProvider(ConstituentProvider):
    """
    Base class for providers that build membership from changelogs.

    Changelogs contain ADD/REMOVE events with effective dates.
    Membership at any date is computed by replaying events.
    """

    def __init__(
        self,
        index_id: str,
        quality: DataQuality,
        initial_members: Optional[List[MemberIdentifier]] = None,
        initial_date: Optional[date] = None,
        cache_dir: Optional[Path] = None,
    ):
        super().__init__(index_id, quality, cache_dir)
        self._initial_members = initial_members or []
        self._initial_date = initial_date
        self._changes: List[ConstituentChange] = []
        self._loaded = False

    @abstractmethod
    def _load_changes(self) -> List[ConstituentChange]:
        """Load changelog from source. Must be implemented by subclasses."""
        ...

    def _ensure_loaded(self) -> None:
        """Lazy load changes if not already loaded."""
        if not self._loaded:
            self._changes = self._load_changes()
            self._changes.sort(key=lambda c: c.effective_date)
            self._loaded = True

    @property
    def changes(self) -> List[ConstituentChange]:
        """Get all changes, loading if necessary."""
        self._ensure_loaded()
        return self._changes

    def snapshot(self, as_of: date) -> ConstituentSnapshot:
        """
        Compute membership at a specific date by replaying changelog.
        """
        self._ensure_loaded()

        if isinstance(as_of, datetime):
            as_of = as_of.date()

        # Start with initial members
        members: Dict[str, MemberIdentifier] = {
            m.primary_id: m for m in self._initial_members
        }

        # Apply changes up to as_of
        for change in self._changes:
            if change.effective_date > as_of:
                break

            if change.action == "ADD":
                members[change.member.primary_id] = change.member
            elif change.action == "REMOVE":
                members.pop(change.member.primary_id, None)

        # Find the actual snapshot date (last change date <= as_of)
        snapshot_date = as_of
        for change in reversed(self._changes):
            if change.effective_date <= as_of:
                snapshot_date = change.effective_date
                break

        return ConstituentSnapshot(
            index_id=self.index_id,
            as_of=snapshot_date,
            members=list(members.values()),
            quality=self.quality,
            source=self.id,
            point_in_time=True,
            meta={
                "requested_date": as_of.isoformat(),
                "snapshot_date": snapshot_date.isoformat(),
                "changes_applied": sum(
                    1 for c in self._changes if c.effective_date <= as_of
                ),
            },
        )

    def available_dates(self) -> List[date]:
        """Get all dates where changes occurred."""
        self._ensure_loaded()
        dates = set()
        if self._initial_date:
            dates.add(self._initial_date)
        for change in self._changes:
            dates.add(change.effective_date)
        return sorted(dates)

    def changes_between(
        self, start: date, end: date
    ) -> Iterator[ConstituentChange]:
        """Get changes in a date range."""
        self._ensure_loaded()
        for change in self._changes:
            if start <= change.effective_date <= end:
                yield change

    def is_cached(self) -> bool:
        """Check if changelog is cached."""
        return self._get_cache_path("_changes").exists()


class SnapshotProvider(ConstituentProvider):
    """
    Base class for providers that have discrete snapshots.

    Examples: Monthly ETF holdings from N-PORT filings.
    """

    def __init__(
        self,
        index_id: str,
        quality: DataQuality,
        cache_dir: Optional[Path] = None,
    ):
        super().__init__(index_id, quality, cache_dir)
        self._snapshots: Dict[date, ConstituentSnapshot] = {}
        self._loaded = False

    @abstractmethod
    def _load_snapshots(self) -> Dict[date, ConstituentSnapshot]:
        """Load all snapshots from source. Must be implemented by subclasses."""
        ...

    def _ensure_loaded(self) -> None:
        """Lazy load snapshots if not already loaded."""
        if not self._loaded:
            self._snapshots = self._load_snapshots()
            self._loaded = True

    def snapshot(self, as_of: date) -> ConstituentSnapshot:
        """
        Get the most recent snapshot on or before as_of.
        """
        self._ensure_loaded()

        if isinstance(as_of, datetime):
            as_of = as_of.date()

        # Find most recent snapshot <= as_of
        available = sorted(self._snapshots.keys())
        snapshot_date = None
        for d in reversed(available):
            if d <= as_of:
                snapshot_date = d
                break

        if snapshot_date is None:
            # Return empty snapshot if no data before requested date
            return ConstituentSnapshot(
                index_id=self.index_id,
                as_of=as_of,
                members=[],
                quality=self.quality,
                source=self.id,
                point_in_time=True,
                meta={
                    "warning": "No data before requested date",
                    "earliest_available": available[0].isoformat() if available else None,
                },
            )

        snapshot = self._snapshots[snapshot_date]
        # Update meta with requested date info
        snapshot.meta["requested_date"] = as_of.isoformat()
        return snapshot

    def available_dates(self) -> List[date]:
        """Get all snapshot dates."""
        self._ensure_loaded()
        return sorted(self._snapshots.keys())

    def get_exact_snapshot(self, snapshot_date: date) -> Optional[ConstituentSnapshot]:
        """Get snapshot for exact date (None if not available)."""
        self._ensure_loaded()
        return self._snapshots.get(snapshot_date)

    def is_cached(self) -> bool:
        """Check if snapshots are cached."""
        return self._get_cache_path("_snapshots").exists()


class CompositeProvider(ConstituentProvider):
    """
    Combines multiple providers with fallback logic.

    Uses highest-quality available data for each date.
    """

    def __init__(
        self,
        index_id: str,
        providers: List[ConstituentProvider],
        cache_dir: Optional[Path] = None,
    ):
        # Use highest quality from providers as default
        qualities = [p.quality for p in providers]
        best_quality = max(qualities, key=lambda q: q.priority)
        super().__init__(index_id, best_quality, cache_dir)
        self._providers = sorted(providers, key=lambda p: -p.quality.priority)

    @property
    def id(self) -> str:
        return f"composite:{self.index_id}"

    @property
    def metadata(self) -> IndexMetadata:
        # Return metadata from highest quality provider
        return self._providers[0].metadata

    def snapshot(self, as_of: date) -> ConstituentSnapshot:
        """Get snapshot from highest quality provider that has data."""
        for provider in self._providers:
            try:
                snapshot = provider.snapshot(as_of)
                if snapshot.size > 0:
                    return snapshot
            except Exception as e:
                logger.warning(
                    f"Provider {provider.id} failed for {as_of}: {e}"
                )
                continue

        # All providers failed - return empty
        return ConstituentSnapshot(
            index_id=self.index_id,
            as_of=as_of,
            members=[],
            quality=DataQuality.USER_SNAPSHOT,
            source=self.id,
            point_in_time=False,
            meta={"error": "All providers failed"},
        )

    def available_dates(self) -> List[date]:
        """Combine available dates from all providers."""
        all_dates = set()
        for provider in self._providers:
            try:
                all_dates.update(provider.available_dates())
            except Exception:
                continue
        return sorted(all_dates)

    def download(self, force: bool = False) -> None:
        """Download from all providers."""
        for provider in self._providers:
            try:
                provider.download(force=force)
            except Exception as e:
                logger.warning(f"Download failed for {provider.id}: {e}")
