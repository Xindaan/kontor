"""
User-provided snapshot importer.

Imports constituent data from user-provided CSV files.
This is USER_SNAPSHOT quality - treat with caution.

Supports:
- Long format: as_of, ticker (one row per ticker per date)
- Wide format: date, ticker1, ticker2, ... (comma-separated)
- Snapshot files: One file per date with ticker list
"""

import csv
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

import pandas as pd

from .base import SnapshotProvider
from .models import (
    ConstituentSnapshot,
    DataQuality,
    IdentifierType,
    IndexMetadata,
    MemberIdentifier,
)

logger = logging.getLogger(__name__)


class UserSnapshotProvider(SnapshotProvider):
    """
    Provider for user-supplied constituent snapshots.

    Reads from CSV files in either long or wide format.
    This is USER_SNAPSHOT quality - lowest priority, use for validation only.
    """

    def __init__(
        self,
        path: str,
        index_id: str = "USER",
        date_col: str = "as_of",
        ticker_col: str = "ticker",
        format: str = "auto",  # "long", "wide", or "auto"
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize user snapshot provider.

        Args:
            path: Path to CSV file
            index_id: Index identifier to use
            date_col: Name of date column (long format)
            ticker_col: Name of ticker column (long format)
            format: "long", "wide", or "auto" (detect automatically)
        """
        super().__init__(
            index_id=index_id,
            quality=DataQuality.USER_SNAPSHOT,
            cache_dir=cache_dir,
        )
        self._path = Path(path)
        self._date_col = date_col
        self._ticker_col = ticker_col
        self._format = format

    @property
    def id(self) -> str:
        return f"user_snapshot:{self._path.stem}"

    @property
    def metadata(self) -> IndexMetadata:
        return IndexMetadata(
            index_id=self.index_id,
            name=f"User Snapshot ({self._path.name})",
            description=f"User-provided constituent data from {self._path}",
            target_count=None,
            region="US",
            currency="USD",
            default_quality=DataQuality.USER_SNAPSHOT,
            available_qualities=[DataQuality.USER_SNAPSHOT],
            frequency="snapshot",
            notes=[
                "User-provided data - verify before using",
                "Not suitable as primary data source",
                f"Source file: {self._path}",
            ],
        )

    def _load_snapshots(self) -> Dict[date, ConstituentSnapshot]:
        """Load snapshots from CSV file."""
        if not self._path.exists():
            logger.error(f"User snapshot file not found: {self._path}")
            return {}

        # Detect format
        format_type = self._format
        if format_type == "auto":
            format_type = self._detect_format()

        if format_type == "long":
            return self._load_long_format()
        else:
            return self._load_wide_format()

    def _detect_format(self) -> str:
        """Detect CSV format (long vs wide)."""
        try:
            df = pd.read_csv(self._path, nrows=5)

            # Long format has date and ticker columns
            if self._date_col in df.columns and self._ticker_col in df.columns:
                return "long"

            # Wide format has date in first column and tickers in rest
            first_col = df.columns[0]
            try:
                pd.to_datetime(df[first_col])
                return "wide"
            except:
                pass

            return "long"  # Default to long
        except Exception as e:
            logger.warning(f"Failed to detect format: {e}")
            return "long"

    def _load_long_format(self) -> Dict[date, ConstituentSnapshot]:
        """Load long format (one row per ticker per date)."""
        snapshots = {}

        try:
            df = pd.read_csv(self._path)

            if self._date_col not in df.columns:
                logger.error(f"Date column '{self._date_col}' not found")
                return {}

            if self._ticker_col not in df.columns:
                logger.error(f"Ticker column '{self._ticker_col}' not found")
                return {}

            # Parse dates
            df[self._date_col] = pd.to_datetime(df[self._date_col]).dt.date

            # Group by date
            for snapshot_date, group in df.groupby(self._date_col):
                tickers = group[self._ticker_col].dropna().unique().tolist()

                members = [
                    MemberIdentifier.from_ticker(t)
                    for t in tickers
                    if t and str(t).strip()
                ]

                if members:
                    snapshots[snapshot_date] = ConstituentSnapshot(
                        index_id=self.index_id,
                        as_of=snapshot_date,
                        members=members,
                        quality=self.quality,
                        source=self.id,
                        point_in_time=True,
                        meta={
                            "file": str(self._path),
                            "format": "long",
                        },
                    )

            logger.info(f"Loaded {len(snapshots)} snapshots from {self._path}")

        except Exception as e:
            logger.error(f"Failed to load long format: {e}")

        return snapshots

    def _load_wide_format(self) -> Dict[date, ConstituentSnapshot]:
        """Load wide format (date, comma-separated tickers)."""
        snapshots = {}

        try:
            df = pd.read_csv(self._path)

            date_col = df.columns[0]
            ticker_col = df.columns[1] if len(df.columns) > 1 else None

            if ticker_col is None:
                logger.error("No ticker column found in wide format")
                return {}

            df[date_col] = pd.to_datetime(df[date_col]).dt.date

            for _, row in df.iterrows():
                snapshot_date = row[date_col]
                tickers_str = row[ticker_col]

                if pd.isna(tickers_str):
                    continue

                # Parse comma-separated tickers
                tickers = [
                    t.strip()
                    for t in str(tickers_str).split(",")
                    if t.strip()
                ]

                members = [MemberIdentifier.from_ticker(t) for t in tickers]

                if members:
                    snapshots[snapshot_date] = ConstituentSnapshot(
                        index_id=self.index_id,
                        as_of=snapshot_date,
                        members=members,
                        quality=self.quality,
                        source=self.id,
                        point_in_time=True,
                        meta={
                            "file": str(self._path),
                            "format": "wide",
                        },
                    )

            logger.info(f"Loaded {len(snapshots)} snapshots from {self._path}")

        except Exception as e:
            logger.error(f"Failed to load wide format: {e}")

        return snapshots


class LocalPITProvider(SnapshotProvider):
    """
    Provider for locally cached PIT constituent data.

    Uses the existing sp500_constituents.csv and nasdaq100_constituents.csv
    files from data/universes/.
    """

    def __init__(
        self,
        index_id: str,
        csv_path: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize local PIT provider.

        Args:
            index_id: Index identifier (SP500, NDX, etc.)
            csv_path: Path to CSV file (auto-detected if None)
        """
        super().__init__(
            index_id=index_id,
            quality=DataQuality.COMMUNITY_CHANGELOG,  # Treat as community data
            cache_dir=cache_dir,
        )

        # Auto-detect path
        if csv_path:
            self._path = Path(csv_path)
        else:
            self._path = self._find_csv()

    def _find_csv(self) -> Path:
        """Find CSV file for this index."""
        base_paths = [
            Path("data/universes"),
            Path("../data/universes"),
            Path(__file__).parent.parent.parent.parent / "data" / "universes",
        ]

        filename_map = {
            "SP500": "sp500_constituents.csv",
            "NDX": "nasdaq100_constituents.csv",
            "NASDAQ100": "nasdaq100_constituents.csv",
        }

        filename = filename_map.get(self.index_id.upper())
        if not filename:
            filename = f"{self.index_id.lower()}_constituents.csv"

        for base in base_paths:
            path = base / filename
            if path.exists():
                return path

        # Return default path even if doesn't exist
        return Path("data/universes") / filename

    @property
    def id(self) -> str:
        return f"local_pit:{self.index_id}"

    @property
    def metadata(self) -> IndexMetadata:
        target_counts = {
            "SP500": 500,
            "NDX": 100,
            "NASDAQ100": 100,
        }

        return IndexMetadata(
            index_id=self.index_id,
            name=f"{self.index_id} (Local PIT)",
            description=f"Point-in-time {self.index_id} constituents from local cache",
            target_count=target_counts.get(self.index_id.upper()),
            region="US",
            currency="USD",
            default_quality=DataQuality.COMMUNITY_CHANGELOG,
            available_qualities=[DataQuality.COMMUNITY_CHANGELOG],
            frequency="daily",
            notes=[
                f"Source: {self._path}",
                "Community-maintained GitHub data",
            ],
        )

    def _load_snapshots(self) -> Dict[date, ConstituentSnapshot]:
        """Load snapshots from local PIT CSV."""
        if not self._path.exists():
            logger.error(f"Local PIT file not found: {self._path}")
            return {}

        snapshots = {}

        try:
            df = pd.read_csv(self._path)

            # Expected columns: as_of, ticker
            if "as_of" not in df.columns:
                logger.error("Expected 'as_of' column not found")
                return {}

            if "ticker" not in df.columns:
                logger.error("Expected 'ticker' column not found")
                return {}

            df["as_of"] = pd.to_datetime(df["as_of"]).dt.date

            for snapshot_date, group in df.groupby("as_of"):
                tickers = group["ticker"].dropna().unique().tolist()

                members = [
                    MemberIdentifier.from_ticker(t)
                    for t in tickers
                    if t and str(t).strip()
                ]

                if members:
                    snapshots[snapshot_date] = ConstituentSnapshot(
                        index_id=self.index_id,
                        as_of=snapshot_date,
                        members=members,
                        quality=self.quality,
                        source=self.id,
                        point_in_time=True,
                        meta={
                            "file": str(self._path),
                        },
                    )

            logger.info(f"Loaded {len(snapshots)} {self.index_id} snapshots from {self._path}")

        except Exception as e:
            logger.error(f"Failed to load local PIT data: {e}")

        return snapshots


# Factory functions

def get_user_snapshot_provider(
    path: str,
    index_id: str = "USER",
    cache_dir: Optional[Path] = None,
) -> UserSnapshotProvider:
    """Get user snapshot provider for a CSV file."""
    return UserSnapshotProvider(path=path, index_id=index_id, cache_dir=cache_dir)


def get_local_sp500_provider(cache_dir: Optional[Path] = None) -> LocalPITProvider:
    """Get S&P 500 provider using local PIT data."""
    return LocalPITProvider(index_id="SP500", cache_dir=cache_dir)


def get_local_nasdaq100_provider(cache_dir: Optional[Path] = None) -> LocalPITProvider:
    """Get Nasdaq-100 provider using local PIT data."""
    return LocalPITProvider(index_id="NDX", cache_dir=cache_dir)
