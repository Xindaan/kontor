"""
PDF parsers for official index constituent changelogs.

Parses:
- FTSE 100: LSEG PDF with historic additions/deletions
- Nikkei 225: Nikkei PDF with component changes
- EURO STOXX 50: STOXX PDF with component changes

These are OFFICIAL_CHANGELOG quality sources.
"""

import io
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

# PDF URLs
FTSE100_PDF_URL = "https://www.lseg.com/content/dam/ftse-russell/en_us/documents/policy-documents/ftse-100-constituent-history.pdf"
NIKKEI225_PDF_URL = "https://indexes.nikkei.co.jp/nkave/archives/file/history_of_nikkei_stock_average_component_changes_en.pdf"
STOXX_NEWS_BASE = "https://www.stoxx.com/document/News"


def _download_pdf(url: str, cache_path: Path, force: bool = False) -> bytes:
    """Download PDF with caching."""
    if cache_path.exists() and not force:
        return cache_path.read_bytes()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(response.content)

    return response.content


def _parse_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF using pdfplumber or PyMuPDF.

    Tries multiple libraries for robustness.
    """
    text = ""

    # Try pdfplumber first (better table extraction)
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
                text += "\n"
        if text.strip():
            return text
    except ImportError:
        logger.debug("pdfplumber not installed, trying PyMuPDF")
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")

    # Try PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
            text += "\n"
        doc.close()
        if text.strip():
            return text
    except ImportError:
        logger.debug("PyMuPDF not installed, trying pypdf")
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e}")

    # Try pypdf as last resort
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"
        return text
    except ImportError:
        raise ImportError(
            "No PDF library available. Install one of: pdfplumber, PyMuPDF, pypdf"
        )

    return text


class FTSE100Provider(ChangelogProvider):
    """
    FTSE 100 constituent provider using official LSEG changelog PDF.

    The PDF contains historic additions and deletions with effective dates.
    This is OFFICIAL_CHANGELOG quality.
    """

    # Known initial FTSE 100 members (approximate - from early data)
    # These would need to be refined based on actual historical data
    INITIAL_MEMBERS = [
        "BP", "HSBC", "GSK", "AZN", "SHEL", "ULVR", "RIO", "DGE", "REL", "LSEG",
        "BARC", "LLOY", "STAN", "NWG", "BATS", "VOD", "BT.A", "NG", "SSE", "CNA",
    ]

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__(
            index_id="UKX",
            quality=DataQuality.OFFICIAL_CHANGELOG,
            initial_members=[MemberIdentifier.from_ticker(t) for t in self.INITIAL_MEMBERS],
            initial_date=date(1984, 1, 3),  # FTSE 100 launch date
            cache_dir=cache_dir,
        )
        self._pdf_url = FTSE100_PDF_URL

    @property
    def id(self) -> str:
        return "ftse100_official"

    @property
    def metadata(self) -> IndexMetadata:
        return IndexMetadata(
            index_id="UKX",
            name="FTSE 100",
            description="UK's 100 largest companies by market cap",
            target_count=100,
            region="UK",
            currency="GBP",
            default_quality=DataQuality.OFFICIAL_CHANGELOG,
            available_qualities=[DataQuality.OFFICIAL_CHANGELOG],
            coverage_start=date(1984, 1, 3),
            frequency="event-based",
            notes=[
                "Official changelog from LSEG/FTSE Russell",
                "Quarterly reconstitution",
            ],
        )

    def _load_changes(self) -> List[ConstituentChange]:
        """Load and parse FTSE 100 changelog from PDF."""
        cache_path = self._cache_dir / "ftse100_changelog.json"

        # Check cache
        cache_data = self._load_cache("_changes")
        if cache_data:
            return [ConstituentChange.from_dict(c) for c in cache_data]

        # Download and parse PDF
        pdf_path = self._cache_dir / "ftse100_history.pdf"
        try:
            pdf_bytes = _download_pdf(self._pdf_url, pdf_path)
            text = _parse_pdf_text(pdf_bytes)
            changes = self._parse_ftse_changelog(text)

            # Cache parsed changes
            self._save_cache([c.to_dict() for c in changes], "_changes")

            return changes
        except Exception as e:
            logger.error(f"Failed to load FTSE 100 changelog: {e}")
            return []

    def _parse_ftse_changelog(self, text: str) -> List[ConstituentChange]:
        """
        Parse FTSE 100 changelog text.

        Expected format varies but typically contains:
        - Date (various formats)
        - Company name
        - Action (Added/Removed or similar)
        """
        changes = []

        # Common date patterns
        date_patterns = [
            r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
            r"(\d{1,2})/(\d{1,2})/(\d{4})",
            r"(\d{4})-(\d{2})-(\d{2})",
        ]

        # Split into lines and process
        lines = text.split("\n")
        current_date = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to extract date
            for pattern in date_patterns:
                match = re.search(pattern, line)
                if match:
                    try:
                        if "January" in pattern or "February" in pattern:
                            # Month name format
                            day = int(match.group(1))
                            month_name = match.group(2)
                            year = int(match.group(3))
                            month_map = {
                                "January": 1, "February": 2, "March": 3, "April": 4,
                                "May": 5, "June": 6, "July": 7, "August": 8,
                                "September": 9, "October": 10, "November": 11, "December": 12,
                            }
                            month = month_map.get(month_name, 1)
                            current_date = date(year, month, day)
                        elif "/" in pattern:
                            # DD/MM/YYYY format
                            day = int(match.group(1))
                            month = int(match.group(2))
                            year = int(match.group(3))
                            current_date = date(year, month, day)
                        else:
                            # YYYY-MM-DD format
                            year = int(match.group(1))
                            month = int(match.group(2))
                            day = int(match.group(3))
                            current_date = date(year, month, day)
                    except ValueError:
                        continue
                    break

            # Look for add/remove patterns
            add_patterns = [
                r"(?:Added|Addition|Included|Joining|New)\s*[:\-]?\s*(.+)",
                r"(.+)\s+(?:added|joins|included)",
            ]
            remove_patterns = [
                r"(?:Removed|Deletion|Excluded|Leaving)\s*[:\-]?\s*(.+)",
                r"(.+)\s+(?:removed|leaves|excluded|delisted)",
            ]

            if current_date:
                for pattern in add_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        company = match.group(1).strip()
                        company = re.sub(r"\s+\(.*\)", "", company)  # Remove parenthetical
                        if company and len(company) > 1:
                            changes.append(
                                ConstituentChange(
                                    effective_date=current_date,
                                    action="ADD",
                                    member=MemberIdentifier(
                                        primary_id=company,
                                        id_type=IdentifierType.NAME,
                                        name=company,
                                    ),
                                    source_document="FTSE 100 Constituent History PDF",
                                )
                            )
                        break

                for pattern in remove_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        company = match.group(1).strip()
                        company = re.sub(r"\s+\(.*\)", "", company)
                        if company and len(company) > 1:
                            changes.append(
                                ConstituentChange(
                                    effective_date=current_date,
                                    action="REMOVE",
                                    member=MemberIdentifier(
                                        primary_id=company,
                                        id_type=IdentifierType.NAME,
                                        name=company,
                                    ),
                                    source_document="FTSE 100 Constituent History PDF",
                                )
                            )
                        break

        logger.info(f"Parsed {len(changes)} FTSE 100 changes")
        return changes


class Nikkei225Provider(ChangelogProvider):
    """
    Nikkei 225 constituent provider using official Nikkei PDF changelog.

    The PDF contains component changes with Japanese security codes.
    This is OFFICIAL_CHANGELOG quality.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__(
            index_id="N225",
            quality=DataQuality.OFFICIAL_CHANGELOG,
            initial_members=[],  # Will be populated from PDF
            initial_date=date(1950, 9, 7),  # Nikkei 225 calculation start
            cache_dir=cache_dir,
        )
        self._pdf_url = NIKKEI225_PDF_URL

    @property
    def id(self) -> str:
        return "nikkei225_official"

    @property
    def metadata(self) -> IndexMetadata:
        return IndexMetadata(
            index_id="N225",
            name="Nikkei 225",
            description="Japan's premier stock index - 225 companies on Tokyo Stock Exchange",
            target_count=225,
            region="JP",
            currency="JPY",
            default_quality=DataQuality.OFFICIAL_CHANGELOG,
            available_qualities=[DataQuality.OFFICIAL_CHANGELOG],
            coverage_start=date(1950, 9, 7),
            frequency="event-based",
            notes=[
                "Official changelog from Nikkei Inc.",
                "Uses Japanese security codes (4-digit)",
                "Annual review in October",
            ],
        )

    def _load_changes(self) -> List[ConstituentChange]:
        """Load and parse Nikkei 225 changelog from PDF."""
        cache_data = self._load_cache("_changes")
        if cache_data:
            return [ConstituentChange.from_dict(c) for c in cache_data]

        # Download and parse PDF
        pdf_path = self._cache_dir / "nikkei225_history.pdf"
        try:
            pdf_bytes = _download_pdf(self._pdf_url, pdf_path)
            text = _parse_pdf_text(pdf_bytes)
            changes = self._parse_nikkei_changelog(text)

            # Cache parsed changes
            self._save_cache([c.to_dict() for c in changes], "_changes")

            return changes
        except Exception as e:
            logger.error(f"Failed to load Nikkei 225 changelog: {e}")
            return []

    def _parse_nikkei_changelog(self, text: str) -> List[ConstituentChange]:
        """
        Parse Nikkei 225 changelog text.

        Expected format typically includes:
        - Date (YYYY/MM/DD or similar)
        - Security code (4-digit Japanese code)
        - Company name
        - Action (Added/Removed)
        """
        changes = []

        # Pattern for Japanese security codes (4 digits)
        jp_code_pattern = r"\b(\d{4})\b"

        # Date patterns
        date_patterns = [
            r"(\d{4})/(\d{1,2})/(\d{1,2})",
            r"(\d{4})\.(\d{1,2})\.(\d{1,2})",
            r"(\d{4})-(\d{2})-(\d{2})",
        ]

        lines = text.split("\n")
        current_date = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Extract date
            for pattern in date_patterns:
                match = re.search(pattern, line)
                if match:
                    try:
                        year = int(match.group(1))
                        month = int(match.group(2))
                        day = int(match.group(3))
                        if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                            current_date = date(year, month, day)
                    except ValueError:
                        continue
                    break

            if current_date:
                # Look for add/remove with security code
                add_match = re.search(
                    r"(?:Added?|New|Inclusion|採用)\s*[:\-]?\s*(\d{4})\s+(.+)",
                    line,
                    re.IGNORECASE,
                )
                remove_match = re.search(
                    r"(?:Removed?|Deletion?|Exclusion|除外)\s*[:\-]?\s*(\d{4})\s+(.+)",
                    line,
                    re.IGNORECASE,
                )

                if add_match:
                    jp_code = add_match.group(1)
                    name = add_match.group(2).strip()
                    changes.append(
                        ConstituentChange(
                            effective_date=current_date,
                            action="ADD",
                            member=MemberIdentifier.from_jp_code(jp_code, name=name),
                            source_document="Nikkei 225 Component Changes PDF",
                        )
                    )
                elif remove_match:
                    jp_code = remove_match.group(1)
                    name = remove_match.group(2).strip()
                    changes.append(
                        ConstituentChange(
                            effective_date=current_date,
                            action="REMOVE",
                            member=MemberIdentifier.from_jp_code(jp_code, name=name),
                            source_document="Nikkei 225 Component Changes PDF",
                        )
                    )

        logger.info(f"Parsed {len(changes)} Nikkei 225 changes")
        return changes


class EuroStoxx50Provider(ChangelogProvider):
    """
    EURO STOXX 50 constituent provider using STOXX PDF changelogs.

    STOXX publishes component change PDFs in their news section.
    This is OFFICIAL_CHANGELOG quality.
    """

    # Known STOXX news PDF patterns
    STOXX_PDF_PATTERN = "STOXX_Components_Changes_STOXX_Blue_Chip_Indices_{date}.pdf"

    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__(
            index_id="SX5E",
            quality=DataQuality.OFFICIAL_CHANGELOG,
            initial_members=[],
            initial_date=date(1998, 2, 26),  # EURO STOXX 50 launch
            cache_dir=cache_dir,
        )

    @property
    def id(self) -> str:
        return "eurostoxx50_official"

    @property
    def metadata(self) -> IndexMetadata:
        return IndexMetadata(
            index_id="SX5E",
            name="EURO STOXX 50",
            description="Eurozone's 50 largest companies by market cap",
            target_count=50,
            region="EU",
            currency="EUR",
            default_quality=DataQuality.OFFICIAL_CHANGELOG,
            available_qualities=[
                DataQuality.OFFICIAL_CHANGELOG,
                DataQuality.PROXY_ETF_HOLDINGS,
            ],
            coverage_start=date(1998, 2, 26),
            frequency="event-based",
            notes=[
                "Official changelog from STOXX",
                "ISINs typically available",
                "Annual review in September",
            ],
        )

    def _find_stoxx_pdfs(self) -> List[Tuple[str, date]]:
        """
        Find available STOXX component change PDFs.

        Returns list of (url, date) tuples.
        """
        pdfs = []

        # Generate URLs for recent years
        current_year = date.today().year
        for year in range(2015, current_year + 1):
            for month in range(1, 13):
                for day in [1, 15]:  # Check common release dates
                    try:
                        d = date(year, month, day)
                        if d > date.today():
                            continue

                        date_str = d.strftime("%Y%m%d")
                        month_name = d.strftime("%B")

                        # STOXX URL pattern
                        url = f"{STOXX_NEWS_BASE}/{year}/{month_name}/STOXX_Components_Changes_STOXX_Blue_Chip_Indices_{date_str}.pdf"
                        pdfs.append((url, d))
                    except ValueError:
                        continue

        return pdfs

    def _load_changes(self) -> List[ConstituentChange]:
        """Load and parse EURO STOXX 50 changelog from PDFs."""
        cache_data = self._load_cache("_changes")
        if cache_data:
            return [ConstituentChange.from_dict(c) for c in cache_data]

        changes = []

        # Try to find and parse available PDFs
        pdf_urls = self._find_stoxx_pdfs()

        for url, pdf_date in pdf_urls:
            try:
                pdf_path = self._cache_dir / f"stoxx_{pdf_date.isoformat()}.pdf"

                # Try to download (many will 404)
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    pdf_path.write_bytes(response.content)

                    text = _parse_pdf_text(response.content)
                    pdf_changes = self._parse_stoxx_changelog(text, pdf_date)
                    changes.extend(pdf_changes)

                    logger.info(f"Parsed {len(pdf_changes)} changes from {url}")

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                # Most URLs will 404, that's expected
                continue

        # Cache parsed changes
        if changes:
            self._save_cache([c.to_dict() for c in changes], "_changes")

        logger.info(f"Total {len(changes)} EURO STOXX 50 changes")
        return changes

    def _parse_stoxx_changelog(
        self, text: str, pdf_date: date
    ) -> List[ConstituentChange]:
        """
        Parse STOXX component changes PDF.

        Expected format includes ISIN and company names.
        """
        changes = []

        # ISIN pattern (2 letter country + 9 alphanumeric + 1 check digit)
        isin_pattern = r"\b([A-Z]{2}[A-Z0-9]{9}\d)\b"

        lines = text.split("\n")
        effective_date = pdf_date

        # Look for effective date in text
        date_match = re.search(
            r"(?:effective|as of|from)\s+(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
            text,
            re.IGNORECASE,
        )
        if date_match:
            try:
                day = int(date_match.group(1))
                month_name = date_match.group(2)
                year = int(date_match.group(3))
                month_map = {
                    "January": 1, "February": 2, "March": 3, "April": 4,
                    "May": 5, "June": 6, "July": 7, "August": 8,
                    "September": 9, "October": 10, "November": 11, "December": 12,
                }
                effective_date = date(year, month_map[month_name], day)
            except (ValueError, KeyError):
                pass

        for line in lines:
            line = line.strip()

            # Skip if doesn't contain EURO STOXX 50
            if "EURO STOXX 50" not in text[:1000] and "SX5E" not in text[:1000]:
                continue

            # Look for additions
            if re.search(r"\b(?:addition|added|new)\b", line, re.IGNORECASE):
                isin_match = re.search(isin_pattern, line)
                if isin_match:
                    isin = isin_match.group(1)
                    # Extract name (text before or after ISIN)
                    name = re.sub(isin_pattern, "", line).strip()
                    name = re.sub(r"\s+", " ", name)

                    changes.append(
                        ConstituentChange(
                            effective_date=effective_date,
                            action="ADD",
                            member=MemberIdentifier.from_isin(isin, name=name),
                            source_document=f"STOXX Blue Chip Indices {pdf_date}",
                        )
                    )

            # Look for deletions
            elif re.search(r"\b(?:deletion|removed|leaving)\b", line, re.IGNORECASE):
                isin_match = re.search(isin_pattern, line)
                if isin_match:
                    isin = isin_match.group(1)
                    name = re.sub(isin_pattern, "", line).strip()
                    name = re.sub(r"\s+", " ", name)

                    changes.append(
                        ConstituentChange(
                            effective_date=effective_date,
                            action="REMOVE",
                            member=MemberIdentifier.from_isin(isin, name=name),
                            source_document=f"STOXX Blue Chip Indices {pdf_date}",
                        )
                    )

        return changes


# Factory functions

def get_ftse100_provider(cache_dir: Optional[Path] = None) -> FTSE100Provider:
    """Get FTSE 100 provider using official LSEG changelog."""
    return FTSE100Provider(cache_dir=cache_dir)


def get_nikkei225_provider(cache_dir: Optional[Path] = None) -> Nikkei225Provider:
    """Get Nikkei 225 provider using official Nikkei changelog."""
    return Nikkei225Provider(cache_dir=cache_dir)


def get_eurostoxx50_provider(cache_dir: Optional[Path] = None) -> EuroStoxx50Provider:
    """Get EURO STOXX 50 provider using official STOXX changelog."""
    return EuroStoxx50Provider(cache_dir=cache_dir)
