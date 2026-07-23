"""
Wikipedia community changelog parsers.

Parses Wikipedia articles for index constituent changes.
This is COMMUNITY_CHANGELOG quality - useful but not audit-grade.

Supported:
- Nasdaq-100: Component changes table
- S&P 500: Selected changes and historical lists
"""

import json
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import requests

from .base import ChangelogProvider
from .models import (
    ConstituentChange,
    ConstituentSnapshot,
    DataQuality,
    IdentifierType,
    IndexMetadata,
    MemberIdentifier,
)

logger = logging.getLogger(__name__)

# Wikipedia API endpoint
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# Article titles
NASDAQ100_ARTICLE = "Nasdaq-100"
SP500_ARTICLE = "List_of_S&P_500_companies"


def _fetch_wikipedia_html(title: str, cache_path: Path, force: bool = False) -> str:
    """Fetch Wikipedia article HTML with caching.

    Args:
        title: Wikipedia article title (URL-encoded if needed)
        cache_path: Path to cache file
        force: If True, ignore cache and re-fetch

    Returns:
        HTML content from Wikipedia

    Raises:
        RuntimeError: If fetch fails or returns empty content
    """
    if not force and cache_path.exists():
        # Cache for 24 hours
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age < 86400:
            content = cache_path.read_text()
            if content and len(content) > 1000:  # Validate cache isn't corrupted
                logger.debug(f"Using cached HTML from {cache_path} ({len(content)} chars)")
                return content
            else:
                logger.warning(f"Cache file {cache_path} appears corrupted, re-fetching...")

    logger.info(f"Fetching Wikipedia article: {title}")

    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
    }

    headers = {
        "User-Agent": "Kontor/1.0 (+https://github.com/Xindaan/kontor)"
    }

    try:
        response = requests.get(WIKIPEDIA_API, params=params, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch Wikipedia article '{title}': {e}")

    data = response.json()

    # Check for API errors
    if "error" in data:
        raise RuntimeError(f"Wikipedia API error: {data['error'].get('info', 'Unknown error')}")

    html = data.get("parse", {}).get("text", {}).get("*", "")

    if not html or len(html) < 1000:
        raise RuntimeError(f"Wikipedia returned empty or invalid content for '{title}'")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html)
    logger.debug(f"Cached HTML to {cache_path} ({len(html)} chars)")

    return html


def _parse_date(date_str: str) -> Optional[date]:
    """Parse various date formats from Wikipedia."""
    date_str = date_str.strip()

    # Common formats
    formats = [
        "%B %d, %Y",      # January 1, 2020
        "%d %B %Y",       # 1 January 2020
        "%Y-%m-%d",       # 2020-01-01
        "%B %Y",          # January 2020 (use 1st of month)
        "%Y",             # 2020 (use Jan 1)
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.date()
        except ValueError:
            continue

    # Try to extract year and month
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", date_str)
    if year_match:
        year = int(year_match.group(1))
        month = 1

        # Try to find month
        months = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        for month_name, month_num in months.items():
            if month_name in date_str.lower():
                month = month_num
                break

        try:
            return date(year, month, 1)
        except ValueError:
            pass

    return None


class WikipediaNasdaq100Provider(ChangelogProvider):
    """
    Nasdaq-100 constituent provider using Wikipedia changelog.

    Wikipedia's Nasdaq-100 article contains:
    - Current constituents table (anchor point)
    - Component changes table with historical additions/removals

    This is COMMUNITY_CHANGELOG quality - useful but not audit-grade.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__(
            index_id="NDX",
            quality=DataQuality.COMMUNITY_CHANGELOG,
            initial_members=[],
            initial_date=date(1985, 1, 31),  # Nasdaq-100 launch
            cache_dir=cache_dir,
        )
        self._current_members: List[MemberIdentifier] = []
        self._current_date: Optional[date] = None

    @property
    def id(self) -> str:
        return "nasdaq100_wikipedia"

    @property
    def metadata(self) -> IndexMetadata:
        return IndexMetadata(
            index_id="NDX",
            name="Nasdaq-100 (Wikipedia)",
            description="100 largest non-financial Nasdaq companies (Wikipedia changelog)",
            target_count=100,
            region="US",
            currency="USD",
            default_quality=DataQuality.COMMUNITY_CHANGELOG,
            available_qualities=[
                DataQuality.COMMUNITY_CHANGELOG,
                DataQuality.PROXY_ETF_HOLDINGS,
            ],
            frequency="event-based",
            notes=[
                "Community-maintained Wikipedia data",
                "Not audit-grade - use for research only",
                "Annual reconstitution in December",
            ],
        )

    def _load_changes(self, force: bool = False) -> List[ConstituentChange]:
        """Parse Nasdaq-100 changes from Wikipedia.

        Args:
            force: If True, ignore cache and re-fetch from Wikipedia
        """
        if not force:
            cache_data = self._load_cache("_changes")
            if cache_data:
                logger.debug(f"Loading {len(cache_data)} cached changes")
                return [ConstituentChange.from_dict(c) for c in cache_data]

        try:
            html_cache = self._cache_dir / "nasdaq100_wiki.html"
            html = _fetch_wikipedia_html(NASDAQ100_ARTICLE, html_cache, force=force)
            changes = self._parse_nasdaq100_changes(html)

            if changes:
                self._save_cache([c.to_dict() for c in changes], "_changes")
            else:
                logger.warning("No changes found in Wikipedia HTML - table structure may have changed")

            return changes

        except Exception as e:
            logger.error(f"Failed to parse Nasdaq-100 Wikipedia: {e}")
            raise

    def _parse_nasdaq100_changes(self, html: str) -> List[ConstituentChange]:
        """Parse the component changes table from Wikipedia HTML."""
        changes = []

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
        except ImportError:
            logger.warning("BeautifulSoup not installed, using regex fallback")
            return self._parse_nasdaq100_regex(html)

        # Find tables with "changes" in preceding header
        tables = soup.find_all("table", class_="wikitable")

        for table in tables:
            # Check if this is a changes table
            prev = table.find_previous(["h2", "h3", "h4"])
            if prev and "change" in prev.get_text().lower():
                rows = table.find_all("tr")

                for row in rows[1:]:  # Skip header
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 3:
                        try:
                            date_text = cells[0].get_text().strip()
                            added_text = cells[1].get_text().strip() if len(cells) > 1 else ""
                            removed_text = cells[2].get_text().strip() if len(cells) > 2 else ""

                            effective_date = _parse_date(date_text)
                            if not effective_date:
                                continue

                            # Parse added companies
                            if added_text and added_text != "—" and added_text != "-":
                                # Extract ticker if available (usually in parentheses or after colon)
                                ticker_match = re.search(r"\(([A-Z]{1,5})\)", added_text)
                                ticker = ticker_match.group(1) if ticker_match else None

                                # Clean company name
                                name = re.sub(r"\([^)]*\)", "", added_text).strip()
                                name = re.sub(r"\s+", " ", name)

                                if name:
                                    changes.append(
                                        ConstituentChange(
                                            effective_date=effective_date,
                                            action="ADD",
                                            member=MemberIdentifier(
                                                primary_id=ticker or name,
                                                id_type=IdentifierType.TICKER if ticker else IdentifierType.NAME,
                                                ticker=ticker,
                                                name=name,
                                            ),
                                            source_document="Wikipedia Nasdaq-100",
                                        )
                                    )

                            # Parse removed companies
                            if removed_text and removed_text != "—" and removed_text != "-":
                                ticker_match = re.search(r"\(([A-Z]{1,5})\)", removed_text)
                                ticker = ticker_match.group(1) if ticker_match else None

                                name = re.sub(r"\([^)]*\)", "", removed_text).strip()
                                name = re.sub(r"\s+", " ", name)

                                if name:
                                    changes.append(
                                        ConstituentChange(
                                            effective_date=effective_date,
                                            action="REMOVE",
                                            member=MemberIdentifier(
                                                primary_id=ticker or name,
                                                id_type=IdentifierType.TICKER if ticker else IdentifierType.NAME,
                                                ticker=ticker,
                                                name=name,
                                            ),
                                            source_document="Wikipedia Nasdaq-100",
                                        )
                                    )

                        except Exception as e:
                            logger.debug(f"Failed to parse row: {e}")
                            continue

        logger.info(f"Parsed {len(changes)} Nasdaq-100 changes from Wikipedia")
        return changes

    def _parse_nasdaq100_regex(self, html: str) -> List[ConstituentChange]:
        """Fallback regex parser when BeautifulSoup not available."""
        changes = []

        # Find table rows with dates
        row_pattern = r"<tr[^>]*>.*?</tr>"
        cell_pattern = r"<td[^>]*>(.*?)</td>"

        rows = re.findall(row_pattern, html, re.DOTALL | re.IGNORECASE)

        for row in rows:
            cells = re.findall(cell_pattern, row, re.DOTALL)
            if len(cells) >= 3:
                date_text = re.sub(r"<[^>]+>", "", cells[0]).strip()
                added_text = re.sub(r"<[^>]+>", "", cells[1]).strip()
                removed_text = re.sub(r"<[^>]+>", "", cells[2]).strip()

                effective_date = _parse_date(date_text)
                if not effective_date:
                    continue

                if added_text and added_text not in ("—", "-", ""):
                    ticker_match = re.search(r"\(([A-Z]{1,5})\)", added_text)
                    ticker = ticker_match.group(1) if ticker_match else None
                    name = re.sub(r"\([^)]*\)", "", added_text).strip()

                    if name:
                        changes.append(
                            ConstituentChange(
                                effective_date=effective_date,
                                action="ADD",
                                member=MemberIdentifier(
                                    primary_id=ticker or name,
                                    id_type=IdentifierType.TICKER if ticker else IdentifierType.NAME,
                                    ticker=ticker,
                                    name=name,
                                ),
                                source_document="Wikipedia Nasdaq-100",
                            )
                        )

                if removed_text and removed_text not in ("—", "-", ""):
                    ticker_match = re.search(r"\(([A-Z]{1,5})\)", removed_text)
                    ticker = ticker_match.group(1) if ticker_match else None
                    name = re.sub(r"\([^)]*\)", "", removed_text).strip()

                    if name:
                        changes.append(
                            ConstituentChange(
                                effective_date=effective_date,
                                action="REMOVE",
                                member=MemberIdentifier(
                                    primary_id=ticker or name,
                                    id_type=IdentifierType.TICKER if ticker else IdentifierType.NAME,
                                    ticker=ticker,
                                    name=name,
                                ),
                                source_document="Wikipedia Nasdaq-100",
                            )
                        )

        return changes

    def _parse_current_constituents(self, html: str) -> List[MemberIdentifier]:
        """Parse the current Nasdaq-100 constituents table from Wikipedia."""
        members = []

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
        except ImportError:
            logger.warning("BeautifulSoup not installed")
            return []

        # Look for the constituents table
        tables = soup.find_all("table", class_="wikitable")

        for table in tables:
            headers = table.find_all("th")
            header_text = " ".join([h.get_text().lower() for h in headers])

            # Look for table with ticker/symbol and company columns
            if ("ticker" in header_text or "symbol" in header_text) and "company" in header_text:
                rows = table.find_all("tr")

                for row in rows[1:]:
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 2:
                        try:
                            # Find ticker - usually first or second column
                            ticker = None
                            name = None

                            for i, cell in enumerate(cells[:3]):
                                text = cell.get_text().strip()
                                # Ticker is short, uppercase
                                if re.match(r"^[A-Z]{1,5}$", text):
                                    ticker = text
                                elif not name and len(text) > 5:
                                    name = text

                            if ticker:
                                members.append(MemberIdentifier(
                                    primary_id=ticker,
                                    id_type=IdentifierType.TICKER,
                                    ticker=ticker,
                                    name=name or ticker,
                                ))

                        except Exception as e:
                            logger.debug(f"Failed to parse constituent row: {e}")
                            continue

                if members:
                    break

        logger.info(f"Parsed {len(members)} current Nasdaq-100 constituents from Wikipedia")
        return members

    def download(self, force: bool = False) -> None:
        """Download/refresh data from Wikipedia.

        Args:
            force: If True, clear cache and re-fetch from Wikipedia
        """
        if force:
            # Clear cached data
            self._loaded = False
            self._changes = []
            self._current_members = []
            self._current_date = None

            # Clear cache files
            for suffix in ["_changes", "_current"]:
                cache_path = self._get_cache_path(suffix)
                if cache_path.exists():
                    cache_path.unlink()
                    logger.debug(f"Cleared cache: {cache_path}")

            html_cache = self._cache_dir / "nasdaq100_wiki.html"
            if html_cache.exists():
                html_cache.unlink()
                logger.debug(f"Cleared HTML cache: {html_cache}")

        # Force reload
        self._changes = self._load_changes(force=force)
        self._current_members, self._current_date = self._load_current_members(force=force)
        self._loaded = True

    def _load_current_members(self, force: bool = False) -> Tuple[List[MemberIdentifier], date]:
        """Load current constituents from Wikipedia.

        Args:
            force: If True, ignore cache and re-fetch from Wikipedia
        """
        if not force:
            cache_data = self._load_cache("_current")
            if cache_data and cache_data.get("members"):
                members = [
                    MemberIdentifier(
                        primary_id=m["ticker"],
                        id_type=IdentifierType.TICKER,
                        ticker=m["ticker"],
                        name=m.get("name", m["ticker"]),
                    )
                    for m in cache_data.get("members", [])
                ]
                as_of = date.fromisoformat(cache_data.get("as_of", date.today().isoformat()))
                logger.debug(f"Loaded {len(members)} cached current members from {as_of}")
                return members, as_of

        try:
            html_cache = self._cache_dir / "nasdaq100_wiki.html"
            html = _fetch_wikipedia_html(NASDAQ100_ARTICLE, html_cache, force=force)
            members = self._parse_current_constituents(html)
            as_of = date.today()

            if not members:
                logger.error("Failed to parse current constituents - no members found in HTML")
                raise RuntimeError(
                    "Could not parse Nasdaq-100 constituents table from Wikipedia. "
                    "The table structure may have changed. Please report this issue."
                )

            cache = {
                "as_of": as_of.isoformat(),
                "members": [{"ticker": m.ticker, "name": m.name} for m in members],
            }
            self._save_cache(cache, "_current")

            logger.info(f"Loaded {len(members)} current Nasdaq-100 constituents from Wikipedia")
            return members, as_of

        except Exception as e:
            logger.error(f"Failed to load current Nasdaq-100 constituents: {e}")
            raise

    def snapshot(self, as_of: date) -> ConstituentSnapshot:
        """Get Nasdaq-100 members as of a specific date."""
        self._ensure_loaded()

        if isinstance(as_of, datetime):
            as_of = as_of.date()

        if not self._current_members:
            self._current_members, self._current_date = self._load_current_members()

        if not self._current_members:
            return super().snapshot(as_of)

        members: Dict[str, MemberIdentifier] = {
            m.primary_id: m for m in self._current_members
        }

        # Apply changes in reverse
        for change in reversed(self._changes):
            if change.effective_date <= as_of:
                break

            if change.action == "ADD":
                members.pop(change.member.primary_id, None)
            elif change.action == "REMOVE":
                members[change.member.primary_id] = change.member

        return ConstituentSnapshot(
            index_id=self.index_id,
            as_of=as_of,
            members=list(members.values()),
            quality=self.quality,
            source=self.id,
            point_in_time=True,
            meta={
                "anchor_date": self._current_date.isoformat() if self._current_date else None,
                "anchor_size": len(self._current_members),
                "changes_reversed": sum(1 for c in self._changes if c.effective_date > as_of),
            },
        )

    def generate_monthly_snapshots(
        self,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> List[ConstituentSnapshot]:
        """Generate monthly snapshots for export."""
        import pandas as pd

        if end_date is None:
            end_date = date.today()

        date_range = pd.date_range(start=start_date, end=end_date, freq="MS")

        snapshots = []
        for dt in date_range:
            snap = self.snapshot(dt.date())
            snapshots.append(snap)

        return snapshots

    def export_to_csv(
        self,
        output_path: Path,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> int:
        """Export monthly snapshots to CSV."""
        snapshots = self.generate_monthly_snapshots(start_date, end_date)

        rows = []
        for snap in snapshots:
            for member in snap.members:
                ticker = member.ticker or member.primary_id
                rows.append({"as_of": snap.as_of.isoformat(), "ticker": ticker})

        import pandas as pd
        df = pd.DataFrame(rows)
        df = df.sort_values(["as_of", "ticker"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported {len(df)} rows to {output_path}")
        return len(df)


class WikipediaSP500Provider(ChangelogProvider):
    """
    S&P 500 constituent provider using Wikipedia changelog.

    Wikipedia's "List of S&P 500 companies" article contains:
    - Current constituents table (anchor point)
    - Selected changes history

    This is COMMUNITY_CHANGELOG quality.

    The provider works by:
    1. Parsing current constituents as anchor (known state today)
    2. Parsing historical changes
    3. Reconstructing historical states by working backwards
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__(
            index_id="SP500",
            quality=DataQuality.COMMUNITY_CHANGELOG,
            initial_members=[],
            initial_date=date(1957, 3, 4),  # S&P 500 launch
            cache_dir=cache_dir,
        )
        self._current_members: List[MemberIdentifier] = []
        self._current_date: Optional[date] = None

    @property
    def id(self) -> str:
        return "sp500_wikipedia"

    @property
    def metadata(self) -> IndexMetadata:
        return IndexMetadata(
            index_id="SP500",
            name="S&P 500 (Wikipedia)",
            description="500 largest US companies by market cap (Wikipedia changelog)",
            target_count=500,
            region="US",
            currency="USD",
            default_quality=DataQuality.COMMUNITY_CHANGELOG,
            available_qualities=[
                DataQuality.COMMUNITY_CHANGELOG,
                DataQuality.PROXY_ETF_HOLDINGS,
            ],
            frequency="event-based",
            notes=[
                "Community-maintained Wikipedia data",
                "Not audit-grade - use for research only",
                "Changes occur throughout the year",
            ],
        )

    def _load_changes(self, force: bool = False) -> List[ConstituentChange]:
        """Parse S&P 500 changes from Wikipedia.

        Args:
            force: If True, ignore cache and re-fetch from Wikipedia

        Returns:
            List of constituent changes
        """
        if not force:
            cache_data = self._load_cache("_changes")
            if cache_data:
                logger.debug(f"Loading {len(cache_data)} cached changes")
                return [ConstituentChange.from_dict(c) for c in cache_data]

        try:
            html_cache = self._cache_dir / "sp500_wiki.html"
            html = _fetch_wikipedia_html(SP500_ARTICLE, html_cache, force=force)
            changes = self._parse_sp500_changes(html)

            if changes:
                self._save_cache([c.to_dict() for c in changes], "_changes")
            else:
                logger.warning("No changes found in Wikipedia HTML - table structure may have changed")

            return changes

        except Exception as e:
            logger.error(f"Failed to parse S&P 500 Wikipedia: {e}")
            raise

    def _parse_sp500_changes(self, html: str) -> List[ConstituentChange]:
        """Parse S&P 500 component changes from Wikipedia."""
        changes = []

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
        except ImportError:
            logger.warning("BeautifulSoup not installed")
            return []

        # Find the "Selected changes" section
        tables = soup.find_all("table", class_="wikitable")

        for table in tables:
            # Look for tables with Date/Added/Removed structure
            headers = table.find_all("th")
            header_text = " ".join([h.get_text().lower() for h in headers])

            if "date" in header_text and ("added" in header_text or "removed" in header_text):
                rows = table.find_all("tr")

                for row in rows[1:]:
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 2:
                        try:
                            # Parse date (first cell)
                            date_cell = cells[0]
                            date_text = date_cell.get_text().strip()
                            effective_date = _parse_date(date_text)

                            if not effective_date:
                                continue

                            # Parse added/removed tickers
                            for i, cell in enumerate(cells[1:], 1):
                                cell_text = cell.get_text().strip()
                                if not cell_text or cell_text in ("—", "-"):
                                    continue

                                # Determine action based on column header
                                is_add = i == 1  # First column after date is typically "Added"

                                # Extract ticker (usually in parentheses or is the main text)
                                ticker_match = re.search(r"\b([A-Z]{1,5})\b", cell_text)
                                ticker = ticker_match.group(1) if ticker_match else None

                                # Get company name
                                links = cell.find_all("a")
                                if links:
                                    name = links[0].get_text().strip()
                                else:
                                    name = re.sub(r"\([^)]*\)", "", cell_text).strip()

                                if ticker or name:
                                    changes.append(
                                        ConstituentChange(
                                            effective_date=effective_date,
                                            action="ADD" if is_add else "REMOVE",
                                            member=MemberIdentifier(
                                                primary_id=ticker or name,
                                                id_type=IdentifierType.TICKER if ticker else IdentifierType.NAME,
                                                ticker=ticker,
                                                name=name,
                                            ),
                                            source_document="Wikipedia S&P 500",
                                        )
                                    )

                        except Exception as e:
                            logger.debug(f"Failed to parse row: {e}")
                            continue

        logger.info(f"Parsed {len(changes)} S&P 500 changes from Wikipedia")
        return changes

    def _parse_current_constituents(self, html: str) -> List[MemberIdentifier]:
        """Parse the current S&P 500 constituents table from Wikipedia."""
        members = []

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
        except ImportError:
            logger.warning("BeautifulSoup not installed")
            return []

        # The first wikitable is usually the constituents table
        tables = soup.find_all("table", class_="wikitable")
        logger.debug(f"Found {len(tables)} wikitables")

        for table_idx, table in enumerate(tables):
            # Check if this looks like the constituents table (has Symbol/Ticker column)
            headers = table.find_all("th")
            header_text = " ".join([h.get_text().lower() for h in headers])
            logger.debug(f"Table {table_idx} headers: {header_text[:100]}...")

            if "symbol" in header_text or "ticker" in header_text:
                rows = table.find_all("tr")
                logger.debug(f"Table {table_idx} has {len(rows)} rows, header match!")

                for row in rows[1:]:  # Skip header
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 2:
                        try:
                            # First column is usually the ticker symbol
                            ticker_cell = cells[0]
                            ticker = ticker_cell.get_text().strip().upper()

                            # Clean ticker (remove notes, references)
                            ticker = re.sub(r"\[.*?\]", "", ticker).strip()
                            ticker = re.sub(r"\s+", "", ticker)

                            if not ticker or len(ticker) > 5:
                                continue

                            # Second column is usually company name
                            name_cell = cells[1] if len(cells) > 1 else None
                            name = name_cell.get_text().strip() if name_cell else ticker

                            members.append(MemberIdentifier(
                                primary_id=ticker,
                                id_type=IdentifierType.TICKER,
                                ticker=ticker,
                                name=name,
                            ))

                        except Exception as e:
                            logger.debug(f"Failed to parse constituent row: {e}")
                            continue

                # Found the table, stop looking
                if members:
                    break

        logger.info(f"Parsed {len(members)} current S&P 500 constituents from Wikipedia")
        return members

    def download(self, force: bool = False) -> None:
        """Download/refresh data from Wikipedia.

        Args:
            force: If True, clear cache and re-fetch from Wikipedia
        """
        if force:
            # Clear cached data
            self._loaded = False
            self._changes = []
            self._current_members = []
            self._current_date = None

            # Clear cache files
            for suffix in ["_changes", "_current"]:
                cache_path = self._get_cache_path(suffix)
                if cache_path.exists():
                    cache_path.unlink()
                    logger.debug(f"Cleared cache: {cache_path}")

            html_cache = self._cache_dir / "sp500_wiki.html"
            if html_cache.exists():
                html_cache.unlink()
                logger.debug(f"Cleared HTML cache: {html_cache}")

        # Force reload
        self._changes = self._load_changes(force=force)
        self._current_members, self._current_date = self._load_current_members(force=force)
        self._loaded = True

    def _load_current_members(self, force: bool = False) -> Tuple[List[MemberIdentifier], date]:
        """Load current constituents from Wikipedia.

        Args:
            force: If True, ignore cache and re-fetch from Wikipedia

        Returns:
            Tuple of (members list, as_of date)
        """
        if not force:
            cache_data = self._load_cache("_current")
            if cache_data and cache_data.get("members"):
                members = [
                    MemberIdentifier(
                        primary_id=m["ticker"],
                        id_type=IdentifierType.TICKER,
                        ticker=m["ticker"],
                        name=m.get("name", m["ticker"]),
                    )
                    for m in cache_data.get("members", [])
                ]
                as_of = date.fromisoformat(cache_data.get("as_of", date.today().isoformat()))
                logger.debug(f"Loaded {len(members)} cached current members from {as_of}")
                return members, as_of

        try:
            html_cache = self._cache_dir / "sp500_wiki.html"
            html = _fetch_wikipedia_html(SP500_ARTICLE, html_cache, force=force)
            members = self._parse_current_constituents(html)
            as_of = date.today()

            if not members:
                logger.error("Failed to parse current constituents - no members found in HTML")
                raise RuntimeError(
                    "Could not parse S&P 500 constituents table from Wikipedia. "
                    "The table structure may have changed. Please report this issue."
                )

            # Cache the result
            cache = {
                "as_of": as_of.isoformat(),
                "members": [{"ticker": m.ticker, "name": m.name} for m in members],
            }
            self._save_cache(cache, "_current")

            logger.info(f"Loaded {len(members)} current S&P 500 constituents from Wikipedia")
            return members, as_of

        except Exception as e:
            logger.error(f"Failed to load current S&P 500 constituents: {e}")
            raise

    def snapshot(self, as_of: date) -> ConstituentSnapshot:
        """
        Get S&P 500 members as of a specific date.

        Works by starting from current constituents and applying
        changes backwards to reconstruct historical state.
        """
        self._ensure_loaded()

        if isinstance(as_of, datetime):
            as_of = as_of.date()

        # Load current members if not loaded
        if not self._current_members:
            self._current_members, self._current_date = self._load_current_members()

        if not self._current_members:
            # Fallback to parent implementation
            return super().snapshot(as_of)

        # Start from current members
        members: Dict[str, MemberIdentifier] = {
            m.primary_id: m for m in self._current_members
        }

        # Apply changes in reverse from current date back to as_of
        # (undo ADDs that happened after as_of, redo REMOVEs that happened after as_of)
        for change in reversed(self._changes):
            if change.effective_date <= as_of:
                break  # Stop when we reach the target date

            # Reverse the change
            if change.action == "ADD":
                # This was added after as_of, so remove it
                members.pop(change.member.primary_id, None)
            elif change.action == "REMOVE":
                # This was removed after as_of, so add it back
                members[change.member.primary_id] = change.member

        return ConstituentSnapshot(
            index_id=self.index_id,
            as_of=as_of,
            members=list(members.values()),
            quality=self.quality,
            source=self.id,
            point_in_time=True,
            meta={
                "anchor_date": self._current_date.isoformat() if self._current_date else None,
                "anchor_size": len(self._current_members),
                "changes_reversed": sum(
                    1 for c in self._changes if c.effective_date > as_of
                ),
            },
        )

    def generate_monthly_snapshots(
        self,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> List[ConstituentSnapshot]:
        """
        Generate monthly snapshots for export to CSV.

        Args:
            start_date: First month to generate (uses 1st of month)
            end_date: Last month (default: today)

        Returns:
            List of ConstituentSnapshot objects, one per month
        """
        import pandas as pd

        if end_date is None:
            end_date = date.today()

        # Generate month-end dates
        date_range = pd.date_range(
            start=start_date,
            end=end_date,
            freq="MS",  # Month start
        )

        snapshots = []
        for dt in date_range:
            snap = self.snapshot(dt.date())
            snapshots.append(snap)

        logger.info(f"Generated {len(snapshots)} monthly snapshots")
        return snapshots

    def export_to_csv(
        self,
        output_path: Path,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> int:
        """
        Export monthly snapshots to CSV format for PIT strategies.

        Args:
            output_path: Path to output CSV file
            start_date: First month
            end_date: Last month (default: today)

        Returns:
            Number of rows written
        """
        snapshots = self.generate_monthly_snapshots(start_date, end_date)

        rows = []
        for snap in snapshots:
            for member in snap.members:
                ticker = member.ticker or member.primary_id
                rows.append({
                    "as_of": snap.as_of.isoformat(),
                    "ticker": ticker,
                })

        import pandas as pd
        df = pd.DataFrame(rows)
        df = df.sort_values(["as_of", "ticker"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported {len(df)} rows to {output_path}")
        return len(df)


# Factory functions

def get_nasdaq100_wikipedia_provider(cache_dir: Optional[Path] = None) -> WikipediaNasdaq100Provider:
    """Get Nasdaq-100 provider using Wikipedia changelog."""
    return WikipediaNasdaq100Provider(cache_dir=cache_dir)


def get_sp500_wikipedia_provider(cache_dir: Optional[Path] = None) -> WikipediaSP500Provider:
    """Get S&P 500 provider using Wikipedia changelog."""
    return WikipediaSP500Provider(cache_dir=cache_dir)
