"""
Universe Framework for investment backtesting.

Provides a structured way to define and validate stock universes with
protection against look-ahead bias (survivorship/selection bias).

Key Components:
- UniverseSnapshot: Point-in-time snapshot of a universe
- UniverseProvider: Abstract base class for universe data sources
- StaticUniverseProvider: Simple static list of tickers
- CsvPITUniverseProvider: Point-in-time universe from CSV files
- YahooScreenerUniverseProvider: Universe from Yahoo Finance screeners

Exclusion List:
- EXCLUDED_TICKERS: Set of tickers with limited data history
- tickers_to_exclude(): Helper function to get exclusion set

Usage:
    from backtest.universe import YahooScreenerProvider, UniverseSnapshot

    provider = YahooScreenerProvider(screen_id="most_actives", count=100)
    snapshot = provider.fetch(as_of=date(2024, 1, 15))
    print(snapshot.tickers)  # ['AAPL', 'TSLA', 'NVDA', ...]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import json
import time
import logging

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================

class UniverseValidationError(ValueError):
    """Raised when universe validation fails (look-ahead detected)."""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UniverseSnapshot:
    """
    Point-in-time snapshot of a universe.

    Represents the state of a universe (list of tickers) at a specific
    point in time. Used for fair backtesting by ensuring the universe
    is what would have been known at that historical date.

    Attributes:
        as_of: The date this snapshot represents
        tickers: List of ticker symbols in the universe
        source: Origin of the data ("static", "yahoo_screener", "csv_pit", etc.)
        point_in_time: True if this is a genuine historical snapshot,
                       False if using current/future-looking data
        meta: Additional metadata (e.g., screener ID, file path, hash)
    """
    as_of: date
    tickers: List[str]
    source: str
    point_in_time: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize the snapshot."""
        # Ensure tickers are unique and sorted for consistency
        self.tickers = sorted(list(set(self.tickers)))

        # Convert datetime to date if needed
        if isinstance(self.as_of, datetime):
            self.as_of = self.as_of.date()

    @property
    def size(self) -> int:
        """Number of tickers in the universe."""
        return len(self.tickers)

    @property
    def hash(self) -> str:
        """Compute a hash of the snapshot for reproducibility."""
        content = f"{self.as_of.isoformat()}|{','.join(self.tickers)}|{self.source}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def __contains__(self, ticker: str) -> bool:
        """Check if ticker is in universe."""
        return ticker in self.tickers

    def __len__(self) -> int:
        return len(self.tickers)

    def __repr__(self) -> str:
        pit_str = "PIT" if self.point_in_time else "NON-PIT"
        return (f"UniverseSnapshot(as_of={self.as_of}, size={self.size}, "
                f"source={self.source}, {pit_str})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "as_of": self.as_of.isoformat(),
            "tickers": self.tickers,
            "source": self.source,
            "point_in_time": self.point_in_time,
            "meta": self.meta,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniverseSnapshot":
        """Create from dictionary."""
        return cls(
            as_of=date.fromisoformat(data["as_of"]),
            tickers=data["tickers"],
            source=data["source"],
            point_in_time=data.get("point_in_time", False),
            meta=data.get("meta", {}),
        )


# =============================================================================
# Guardrail: Look-ahead Bias Validation
# =============================================================================

def validate_universe_for_backtest(
    snapshot: UniverseSnapshot,
    backtest_start: date,
    allow_lookahead: bool = False,
) -> None:
    """
    Validate that a universe snapshot is appropriate for backtesting.

    This is the critical guardrail against look-ahead bias. It prevents
    using non-PIT (non-point-in-time) universe data for historical backtests.

    Rules:
    - If snapshot.point_in_time == True: Always allowed (genuine historical data)
    - If snapshot.point_in_time == False AND backtest_start < snapshot.as_of:
        Raises UniverseValidationError unless allow_lookahead=True

    Args:
        snapshot: The universe snapshot to validate
        backtest_start: Start date of the backtest
        allow_lookahead: If True, bypass validation (for deliberate unfair tests)

    Raises:
        UniverseValidationError: If look-ahead bias is detected and not allowed

    Example:
        >>> snapshot = UniverseSnapshot(as_of=date(2024, 1, 1), tickers=["AAPL"],
        ...                              source="yahoo_screener", point_in_time=False)
        >>> validate_universe_for_backtest(snapshot, date(2013, 1, 1))
        UniverseValidationError: Universe look-ahead detected! ...
    """
    if allow_lookahead:
        return  # User explicitly accepts look-ahead bias

    if snapshot.point_in_time:
        return  # Point-in-time data is always safe

    # Convert dates if needed
    if isinstance(backtest_start, datetime):
        backtest_start = backtest_start.date()

    # Check for look-ahead: using future universe data for past backtest
    # Allow 1-day tolerance for "today" backtests where market data may lag
    from datetime import timedelta
    days_diff = (snapshot.as_of - backtest_start).days
    if days_diff > 1:  # More than 1 day difference = clear look-ahead
        raise UniverseValidationError(
            f"Universe look-ahead detected!\n"
            f"\n"
            f"  Universe snapshot date: {snapshot.as_of}\n"
            f"  Backtest start date:    {backtest_start}\n"
            f"  Universe source:        {snapshot.source}\n"
            f"  Point-in-time:          {snapshot.point_in_time}\n"
            f"\n"
            f"You are attempting to use a non-point-in-time universe for a\n"
            f"historical backtest. This introduces survivorship/selection bias\n"
            f"because the universe reflects CURRENT constituents, not what was\n"
            f"known at the backtest start date.\n"
            f"\n"
            f"Options:\n"
            f"  1. Use a PIT (point-in-time) universe source (recommended)\n"
            f"  2. Start backtest on or after {snapshot.as_of}\n"
            f"  3. Use --allow-universe-lookahead to bypass (not recommended)\n"
        )


# =============================================================================
# Abstract Base Class
# =============================================================================

class UniverseProvider(ABC):
    """
    Abstract base class for universe providers.

    A UniverseProvider fetches a list of tickers that represent
    a tradeable universe at a specific point in time.

    Implementations should set point_in_time=True only if the data
    genuinely represents what was known at the historical date.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this provider configuration."""
        ...

    @abstractmethod
    def snapshot(self, as_of: date) -> UniverseSnapshot:
        """
        Get universe snapshot for a specific date.

        Args:
            as_of: The date to get the universe for

        Returns:
            UniverseSnapshot with tickers available at that date
        """
        ...

    # Alias for backward compatibility
    def fetch(self, as_of: date) -> UniverseSnapshot:
        """Alias for snapshot() for backward compatibility."""
        return self.snapshot(as_of)

    def _compute_hash(self, tickers: List[str]) -> str:
        """Compute a hash of the ticker list for reproducibility."""
        content = ",".join(sorted(tickers))
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _get_cache_path(self, as_of: date) -> Path:
        """Get the cache file path for a specific date."""
        cache_dir = Path("data/universe_cache") / self.id
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{as_of.isoformat()}.json"

    def _load_from_cache(self, as_of: date) -> Optional[UniverseSnapshot]:
        """Try to load from cache."""
        cache_path = self._get_cache_path(as_of)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                snapshot = UniverseSnapshot.from_dict(data)
                snapshot.meta["cached"] = True
                logger.debug(f"Loaded universe from cache: {cache_path}")
                return snapshot
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None

    def _save_to_cache(self, snapshot: UniverseSnapshot) -> None:
        """Save snapshot to cache."""
        cache_path = self._get_cache_path(snapshot.as_of)
        try:
            with open(cache_path, "w") as f:
                json.dump(snapshot.to_dict(), f, indent=2)
            logger.debug(f"Saved universe to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id!r})"


# =============================================================================
# Static Universe Provider
# =============================================================================

class StaticUniverseProvider(UniverseProvider):
    """
    Provides a static (unchanging) universe.

    This is the simplest provider - it returns the same list of tickers
    regardless of date. NOT point-in-time because the list was defined
    at a specific point (usually "now") and may include tickers that
    didn't exist or weren't accessible at historical dates.

    Use cases:
    - Quick prototyping
    - When you have a known, fixed universe
    - Combined with allow_lookahead for deliberate research

    Attributes:
        tickers: The static list of ticker symbols
        defined_as_of: When this universe was defined
        label: Human-readable label for this universe
    """

    def __init__(
        self,
        tickers: List[str],
        defined_as_of: Optional[date] = None,
        label: str = "static",
    ):
        """
        Initialize static universe provider.

        Args:
            tickers: List of ticker symbols
            defined_as_of: Date when universe was defined (default: today)
            label: Human-readable label
        """
        self._tickers = sorted(list(set(tickers)))
        self._defined_as_of = defined_as_of or date.today()
        self._label = label
        self._id = f"static:{label}"

    @property
    def id(self) -> str:
        return self._id

    @property
    def tickers(self) -> List[str]:
        """Get the static ticker list."""
        return self._tickers.copy()

    def snapshot(self, as_of: date) -> UniverseSnapshot:
        """
        Get universe snapshot.

        Note: Always returns point_in_time=False because we can't
        guarantee the static list was valid at historical dates.
        """
        return UniverseSnapshot(
            as_of=self._defined_as_of,  # Always the definition date
            tickers=self._tickers,
            source="static",
            point_in_time=False,  # Static lists are NOT point-in-time
            meta={
                "label": self._label,
                "requested_date": as_of.isoformat() if isinstance(as_of, date) else str(as_of),
                "universe_size": len(self._tickers),
            },
        )


# =============================================================================
# CSV Point-in-Time Universe Provider
# =============================================================================

class CsvPITUniverseProvider(UniverseProvider):
    """
    Provides point-in-time universe data from CSV files.

    Reads historical universe constituents from a CSV file with columns:
    - date_col: The date of each snapshot (e.g., "as_of", "date")
    - ticker_col: The ticker symbol (e.g., "ticker", "symbol")

    The CSV should contain historical snapshots, e.g., monthly S&P 500
    constituent lists. This enables FAIR backtesting by using only
    tickers that were actually in the universe at each historical date.

    Example CSV format:
        as_of,ticker
        2020-01-01,AAPL
        2020-01-01,MSFT
        2020-01-01,GOOGL
        2020-02-01,AAPL
        2020-02-01,MSFT
        ...

    Attributes:
        path: Path to the CSV file
        date_col: Name of the date column
        ticker_col: Name of the ticker column
    """

    def __init__(
        self,
        path: str,
        date_col: str = "as_of",
        ticker_col: str = "ticker",
    ):
        """
        Initialize CSV PIT universe provider.

        Args:
            path: Path to CSV file with historical universe data
            date_col: Name of column containing dates
            ticker_col: Name of column containing tickers

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        self._path = Path(path)
        self._date_col = date_col
        self._ticker_col = ticker_col
        self._id = f"csv_pit:{self._path.name}"

        # Load and validate data
        self._load_data()

    def _load_data(self) -> None:
        """Load and parse the CSV file."""
        if not self._path.exists():
            raise FileNotFoundError(f"Universe CSV not found: {self._path}")

        self._df = pd.read_csv(self._path)

        # Validate required columns
        if self._date_col not in self._df.columns:
            raise ValueError(f"Date column '{self._date_col}' not found in CSV. "
                           f"Available columns: {list(self._df.columns)}")
        if self._ticker_col not in self._df.columns:
            raise ValueError(f"Ticker column '{self._ticker_col}' not found in CSV. "
                           f"Available columns: {list(self._df.columns)}")

        # Parse dates
        self._df[self._date_col] = pd.to_datetime(self._df[self._date_col]).dt.date

        # Build index of dates -> tickers for fast lookup
        self._snapshots: Dict[date, List[str]] = {}
        for d, group in self._df.groupby(self._date_col):
            self._snapshots[d] = sorted(group[self._ticker_col].unique().tolist())

        # Sort available dates
        self._available_dates = sorted(self._snapshots.keys())

        if not self._available_dates:
            raise ValueError(f"No valid data found in {self._path}")

    @property
    def id(self) -> str:
        return self._id

    @property
    def available_dates(self) -> List[date]:
        """Get list of dates with available snapshots."""
        return self._available_dates.copy()

    @property
    def earliest_date(self) -> date:
        """Earliest available snapshot date."""
        return self._available_dates[0]

    @property
    def latest_date(self) -> date:
        """Latest available snapshot date."""
        return self._available_dates[-1]

    def snapshot(self, as_of: date) -> UniverseSnapshot:
        """
        Get universe snapshot for a specific date.

        Returns the most recent snapshot on or before the requested date.
        If no historical data exists before the date, returns empty snapshot.

        Args:
            as_of: The date to get the universe for

        Returns:
            UniverseSnapshot with point_in_time=True
        """
        if isinstance(as_of, datetime):
            as_of = as_of.date()

        # Find the most recent snapshot <= as_of
        snapshot_date = None
        for d in reversed(self._available_dates):
            if d <= as_of:
                snapshot_date = d
                break

        if snapshot_date is None:
            # No historical data before requested date
            return UniverseSnapshot(
                as_of=as_of,
                tickers=[],
                source="csv_pit",
                point_in_time=True,
                meta={
                    "file": str(self._path),
                    "requested_date": as_of.isoformat(),
                    "warning": "No data before requested date",
                    "earliest_available": self._available_dates[0].isoformat() if self._available_dates else None,
                },
            )

        tickers = self._snapshots[snapshot_date]

        return UniverseSnapshot(
            as_of=snapshot_date,
            tickers=tickers,
            source="csv_pit",
            point_in_time=True,  # This IS point-in-time data
            meta={
                "file": str(self._path),
                "requested_date": as_of.isoformat(),
                "snapshot_date": snapshot_date.isoformat(),
                "universe_size": len(tickers),
            },
        )


# =============================================================================
# Ticker Exclusion Helper
# =============================================================================

def tickers_to_exclude(mode: str = "backtest") -> set:
    """
    Return set of tickers to exclude based on mode.

    Args:
        mode: "backtest" - exclude all problematic tickers (default)
              "live" - only exclude renamed/delisted tickers

    For backtesting, we exclude tickers with insufficient history.
    For live trading/signals, we only exclude truly problematic tickers
    (renamed, delisted) since short history is not an issue.

    Returns:
        Set of ticker symbols to exclude
    """
    # Always exclude renamed/delisted tickers
    always_exclude = TICKER_ISSUES.get("renamed", set()) | TICKER_ISSUES.get("delisted", set())

    if mode == "live":
        return always_exclude

    # For backtesting, also exclude tickers with short history
    return always_exclude | TICKER_ISSUES.get("short_history", set())


# =============================================================================
# Yahoo Screener Provider
# =============================================================================

class YahooScreenerProvider(UniverseProvider):
    """
    Fetch universe from Yahoo Finance predefined screeners.

    IMPORTANT: This provider is NOT point-in-time safe for historical
    backtesting. Yahoo screeners return CURRENT data, not historical
    constituent lists. Use only for:
    - Live/forward trading
    - Research with explicit --allow-universe-lookahead

    Available screeners (scrIds):
    - most_actives: Most active by volume
    - day_gainers: Top gainers today
    - day_losers: Top losers today
    - undervalued_large_caps: Undervalued large caps
    - growth_technology_stocks: Growth tech stocks
    - aggressive_small_caps: Aggressive small caps
    - small_cap_gainers: Small cap gainers

    Note: Yahoo screeners return current data, not historical.
    For backtesting, results are cached per date to ensure reproducibility.

    Example:
        provider = YahooScreenerProvider(screen_id="most_actives", count=100)
        snapshot = provider.fetch(as_of=date.today())
    """

    BASE_URL = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"

    # Known screener IDs
    KNOWN_SCREENERS = {
        "most_actives": "Most Active",
        "day_gainers": "Day Gainers",
        "day_losers": "Day Losers",
        "undervalued_large_caps": "Undervalued Large Caps",
        "growth_technology_stocks": "Growth Technology Stocks",
        "aggressive_small_caps": "Aggressive Small Caps",
        "small_cap_gainers": "Small Cap Gainers",
        "undervalued_growth_stocks": "Undervalued Growth Stocks",
        "most_shorted_stocks": "Most Shorted Stocks",
    }

    def __init__(
        self,
        screen_id: str = "most_actives",
        count: int = 100,
        use_cache: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        excluded_tickers: set = None,
    ):
        """
        Initialize Yahoo screener provider.

        Args:
            screen_id: Yahoo screener ID (e.g., "most_actives")
            count: Maximum number of tickers to fetch
            use_cache: Whether to use file caching
            max_retries: Maximum retry attempts on failure
            retry_delay: Delay between retries in seconds
            excluded_tickers: Set of tickers to exclude (default: EXCLUDED_TICKERS)
        """
        self.screen_id = screen_id
        self.count = count
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Use global exclusion list by default, merge with any custom exclusions
        self.excluded_tickers = EXCLUDED_TICKERS.copy()
        if excluded_tickers:
            self.excluded_tickers.update(excluded_tickers)

    @property
    def id(self) -> str:
        return f"yahoo_screener_{self.screen_id}_{self.count}"

    def snapshot(self, as_of: date) -> UniverseSnapshot:
        """
        Get universe from Yahoo screener.

        IMPORTANT: Always returns point_in_time=False because Yahoo
        screeners provide CURRENT data, not historical.

        Args:
            as_of: Date for caching (screener returns current data)

        Returns:
            UniverseSnapshot with point_in_time=False and warning in meta
        """
        fetch_date = date.today()

        # Try cache first
        if self.use_cache:
            cached = self._load_from_cache(fetch_date)
            if cached is not None:
                # Override as_of to fetch_date since it's current data
                cached.meta["warning"] = "NOT PIT - Yahoo screeners return current data only"
                return cached

        # Fetch from Yahoo
        tickers = self._fetch_from_yahoo()

        snapshot = UniverseSnapshot(
            as_of=fetch_date,  # Always today, not requested date
            tickers=tickers,
            source=f"yahoo_screener:{self.screen_id}",
            point_in_time=False,  # NEVER point-in-time
            meta={
                "screen_id": self.screen_id,
                "count": self.count,
                "requested_date": as_of.isoformat() if isinstance(as_of, date) else str(as_of),
                "fetch_date": fetch_date.isoformat(),
                "warning": "NOT PIT - Yahoo screeners return current data only",
            },
        )

        # Cache the result
        if self.use_cache:
            self._save_to_cache(snapshot)

        return snapshot

    # Alias for backward compatibility
    def fetch(self, as_of: date) -> UniverseSnapshot:
        return self.snapshot(as_of)

    def _fetch_from_yahoo(self) -> List[str]:
        """Fetch tickers from Yahoo Finance screener API."""
        import requests

        url = self.BASE_URL
        params = {
            "scrIds": self.screen_id,
            "start": 0,
            "count": self.count,
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                response.raise_for_status()

                data = response.json()
                quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
                tickers = [q.get("symbol") for q in quotes if q.get("symbol")]

                # Filter out excluded tickers
                excluded = tickers_to_exclude()
                filtered = [t for t in tickers if t not in excluded]
                if len(filtered) < len(tickers):
                    logger.info(f"Excluded {len(tickers) - len(filtered)} tickers with limited history")
                tickers = filtered

                logger.info(f"Fetched {len(tickers)} tickers from Yahoo screener '{self.screen_id}'")
                return tickers

            except requests.exceptions.RequestException as e:
                logger.warning(f"Yahoo screener request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to fetch from Yahoo screener after {self.max_retries} attempts: {e}")
                    return []  # Return empty on failure instead of raising

        return []


# Alias for backward compatibility
YahooScreenerUniverseProvider = YahooScreenerProvider


# =============================================================================
# Yahoo Query Provider (using yahooquery library)
# =============================================================================

class YahooQueryScreenerProvider(UniverseProvider):
    """
    Alternative Yahoo screener using the yahooquery library.

    IMPORTANT: Like YahooScreenerProvider, this is NOT point-in-time safe.

    Requires: pip install yahooquery

    Example:
        provider = YahooQueryScreenerProvider(screen_id="most_actives", count=50)
    """

    def __init__(
        self,
        screen_id: str = "most_actives",
        count: int = 100,
        use_cache: bool = True,
    ):
        self.screen_id = screen_id
        self.count = count
        self.use_cache = use_cache

    @property
    def id(self) -> str:
        return f"yahooquery_{self.screen_id}_{self.count}"

    def snapshot(self, as_of: date) -> UniverseSnapshot:
        """Fetch using yahooquery library."""
        fetch_date = date.today()

        if self.use_cache:
            cached = self._load_from_cache(fetch_date)
            if cached is not None:
                cached.meta["warning"] = "NOT PIT - Yahoo screeners return current data only"
                return cached

        try:
            from yahooquery import Screener

            s = Screener()
            data = s.get_screeners(self.screen_id, count=self.count)

            if self.screen_id in data:
                quotes = data[self.screen_id].get("quotes", [])
                tickers = [q.get("symbol") for q in quotes if q.get("symbol")]
            else:
                logger.warning(f"Screener '{self.screen_id}' not found in yahooquery response")
                tickers = []

        except ImportError:
            logger.warning("yahooquery not installed, returning empty universe")
            tickers = []
        except Exception as e:
            logger.error(f"yahooquery screener failed: {e}")
            tickers = []

        snapshot = UniverseSnapshot(
            as_of=fetch_date,
            tickers=tickers,
            source=f"yahooquery:{self.screen_id}",
            point_in_time=False,  # NEVER point-in-time
            meta={
                "screen_id": self.screen_id,
                "count": self.count,
                "requested_date": as_of.isoformat() if isinstance(as_of, date) else str(as_of),
                "fetch_date": fetch_date.isoformat(),
                "warning": "NOT PIT - Yahoo screeners return current data only",
            },
        )

        if self.use_cache:
            self._save_to_cache(snapshot)

        return snapshot

    def fetch(self, as_of: date) -> UniverseSnapshot:
        return self.snapshot(as_of)


# =============================================================================
# Pre-built Universe Definitions
# =============================================================================

# S&P 500 proxy (50 major stocks)
SP500_PROXY_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "PFE", "KO", "PEP", "COST", "TMO", "AVGO", "WMT", "MCD",
    "CSCO", "ABT", "DHR", "ACN", "LLY", "ADBE", "CRM", "NKE", "CMCSA",
    "TXN", "VZ", "NEE", "PM", "INTC", "UNP", "BMY", "QCOM", "RTX",
    "HON", "T", "IBM", "AMGN", "LOW",
]

# Tech-focused universe
TECH_UNIVERSE_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "AVGO", "ADBE", "CRM", "CSCO", "INTC", "AMD", "QCOM",
    "TXN", "MU", "AMAT", "LRCX", "KLAC", "SNPS",
]


def get_sp500_proxy_provider(label: str = "sp500_proxy") -> StaticUniverseProvider:
    """Get a static provider with S&P 500 proxy tickers."""
    return StaticUniverseProvider(
        tickers=SP500_PROXY_TICKERS,
        label=label,
    )


def get_tech_universe_provider(label: str = "tech") -> StaticUniverseProvider:
    """Get a static provider with tech-focused tickers."""
    return StaticUniverseProvider(
        tickers=TECH_UNIVERSE_TICKERS,
        label=label,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def get_large_cap_universe(count: int = 100, as_of: Optional[date] = None) -> List[str]:
    """
    Get large cap stocks from Yahoo screener.

    Args:
        count: Number of stocks to fetch
        as_of: Date for caching (defaults to today)

    Returns:
        List of ticker symbols
    """
    provider = YahooScreenerProvider(screen_id="undervalued_large_caps", count=count)
    snapshot = provider.snapshot(as_of or date.today())
    return snapshot.tickers


def get_most_active_universe(count: int = 100, as_of: Optional[date] = None) -> List[str]:
    """
    Get most active stocks from Yahoo screener.

    Args:
        count: Number of stocks to fetch
        as_of: Date for caching (defaults to today)

    Returns:
        List of ticker symbols
    """
    provider = YahooScreenerProvider(screen_id="most_actives", count=count)
    snapshot = provider.snapshot(as_of or date.today())
    return snapshot.tickers


# =============================================================================
# Future: Index Constituents Provider
# =============================================================================

class IndexConstituentsProvider(UniverseProvider):
    """
    Historical index constituents provider.

    Uses pre-downloaded CSV files with historical index membership.
    Helps reduce survivorship bias by using point-in-time constituents.

    Data source: https://github.com/yfiua/index-constituents

    Note: This is a placeholder - actual implementation requires
    downloading and maintaining the constituent data files.
    """

    def __init__(self, index_name: str = "sp500", data_dir: str = "data/index_constituents"):
        self.index_name = index_name
        self.data_dir = Path(data_dir)

    @property
    def id(self) -> str:
        return f"index_constituents_{self.index_name}"

    def fetch(self, as_of: date) -> UniverseSnapshot:
        """
        Fetch index constituents as of a specific date.

        This implementation requires constituent CSV files to be present.
        """
        # Find the most recent file <= as_of
        constituent_file = self._find_constituent_file(as_of)

        if constituent_file is None:
            raise FileNotFoundError(
                f"No constituent data found for {self.index_name} as of {as_of}. "
                f"Please download data to {self.data_dir}"
            )

        # Load tickers from CSV
        import csv
        with open(constituent_file) as f:
            reader = csv.reader(f)
            tickers = [row[0] for row in reader if row]  # Assumes ticker in first column

        return UniverseSnapshot(
            as_of=as_of,
            provider_id=self.id,
            tickers=tickers,
            source=f"index_constituents:{self.index_name}:{constituent_file.name}",
            hash=self._compute_hash(tickers),
            cached=False,
        )

    def _find_constituent_file(self, as_of: date) -> Optional[Path]:
        """Find the constituent file for a given date."""
        if not self.data_dir.exists():
            return None

        # Look for files like sp500_2024-01.csv
        pattern = f"{self.index_name}_*.csv"
        files = sorted(self.data_dir.glob(pattern), reverse=True)

        for f in files:
            # Extract date from filename
            try:
                date_str = f.stem.split("_")[1]
                file_date = datetime.strptime(date_str, "%Y-%m").date()
                if file_date <= as_of:
                    return f
            except (IndexError, ValueError):
                continue

        return None


# =============================================================================
# Ticker Issue Categories (formerly EXCLUDED_TICKERS)
# =============================================================================
#
# Tickers are categorized by issue type:
# - renamed: Ticker symbol changed (e.g., FB -> META)
# - delisted: No longer traded
# - short_history: Insufficient historical data for backtesting
#
# For live trading/signals, only renamed/delisted tickers are excluded.
# For backtesting, all categories are excluded.

TICKER_ISSUES = {
    # Renamed tickers - always exclude (use new symbol instead)
    "renamed": {
        "FB",      # Facebook - renamed to META
    },

    # Delisted tickers - always exclude
    "delisted": set(),

    # Short history - only exclude for backtesting
    # These tickers may be valid for live trading but lack sufficient
    # historical data for accurate backtesting
    "short_history": {
        "NTR",     # Nutrien Ltd - limited history (starts 2018-01-02)
        "VG",      # Vonage - limited history (starts 2025-01-24)
        "LTM",     # LATAM Airlines - limited history (starts 2024-07-25)
        "SOLV",    # Solventum - limited history (starts 2024-03-26)
        "KSPI",    # Kaspi.kz - limited history (starts 2024-01-19)
        "CRBG",    # Corebridge Financial - limited history (starts 2022-09-16)
        "PINS",    # Pinterest - limited history (starts 2019-04-18)
        "FOX",     # Fox Corporation - limited history (starts 2019-03-13)
        "XLC",     # Communication Services SPDR - limited history (starts 2018-06-19)
        "XLRE",    # Real Estate SPDR - limited history (starts 2015-10-08)
        "PR",      # Permian Resources - limited history (starts 2016-04-15)
    },
}

# Legacy compatibility: EXCLUDED_TICKERS returns all excluded tickers (backtest mode)
EXCLUDED_TICKERS = tickers_to_exclude("backtest")
