"""
SEC EDGAR N-PORT parser for ETF holdings.

N-PORT-P filings contain monthly portfolio holdings for registered investment
companies (including ETFs). This is used as a PROXY for index constituents.

Important limitations:
- ETF holdings may differ from index due to sampling/optimization
- Monthly frequency only (not event-based)
- Data available from ~2019 onwards

SEC EDGAR endpoints:
- Company tickers: https://www.sec.gov/files/company_tickers.json
- Filings API: https://data.sec.gov/submissions/CIK{cik}.json
- N-PORT XML: https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/primary_doc.xml
"""

import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import requests

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False

from .base import SnapshotProvider
from .models import (
    ConstituentSnapshot,
    DataQuality,
    IdentifierType,
    IndexMetadata,
    MemberIdentifier,
)

logger = logging.getLogger(__name__)

# Rate limiting for SEC EDGAR (max 10 requests/second)
SEC_RATE_LIMIT_DELAY = 0.15

# User agent required by SEC EDGAR. SEC's fair-access policy asks each
# requester to identify themselves with a real contact (name + email).
# Set your own via the SEC_USER_AGENT env var, e.g.:
#   export SEC_USER_AGENT="my-project you@example.com"
# The placeholder below is intentionally not a real address.
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "kontor-backtest contact@example.com"
)

# ETF to Index mapping with CIK numbers
ETF_INDEX_MAP = {
    # S&P 500
    "SPY": {"index": "SP500", "name": "SPDR S&P 500 ETF Trust", "cik": "0000884394"},
    "IVV": {"index": "SP500", "name": "iShares Core S&P 500 ETF", "cik": "0001100663"},
    "VOO": {"index": "SP500", "name": "Vanguard S&P 500 ETF", "cik": "0000786671"},
    # Russell
    "IWM": {"index": "R2000", "name": "iShares Russell 2000 ETF", "cik": "0001100663"},
    "IWB": {"index": "R1000", "name": "iShares Russell 1000 ETF", "cik": "0001100663"},
    # Nasdaq 100
    "QQQ": {"index": "NDX", "name": "Invesco QQQ Trust", "cik": "0001067839"},
    # MSCI EAFE
    "EFA": {"index": "EAFE", "name": "iShares MSCI EAFE ETF", "cik": "0001100663"},
    # EURO STOXX 50
    "FEZ": {"index": "SX5E", "name": "SPDR EURO STOXX 50 ETF", "cik": "0001064642"},
    # S&P 500 Sectors
    "XLE": {"index": "SP500_ENERGY", "name": "Energy Select Sector SPDR", "cik": "0001064641"},
    "XLK": {"index": "SP500_TECH", "name": "Technology Select Sector SPDR", "cik": "0001064641"},
    "XLF": {"index": "SP500_FINANCIALS", "name": "Financial Select Sector SPDR", "cik": "0001064641"},
    "XLV": {"index": "SP500_HEALTHCARE", "name": "Health Care Select Sector SPDR", "cik": "0001064641"},
    "XLI": {"index": "SP500_INDUSTRIALS", "name": "Industrial Select Sector SPDR", "cik": "0001064641"},
    "XLY": {"index": "SP500_DISCRETIONARY", "name": "Consumer Discretionary SPDR", "cik": "0001064641"},
    "XLP": {"index": "SP500_STAPLES", "name": "Consumer Staples Select SPDR", "cik": "0001064641"},
    "XLU": {"index": "SP500_UTILITIES", "name": "Utilities Select Sector SPDR", "cik": "0001064641"},
    "XLB": {"index": "SP500_MATERIALS", "name": "Materials Select Sector SPDR", "cik": "0001064641"},
    "XLRE": {"index": "SP500_REALESTATE", "name": "Real Estate Select Sector SPDR", "cik": "0001064641"},
    "XLC": {"index": "SP500_COMMUNICATION", "name": "Communication Services SPDR", "cik": "0001064641"},
}


@dataclass
class NPortFiling:
    """Represents an N-PORT filing."""
    accession_number: str
    filing_date: date
    report_date: date  # Period end date
    primary_doc: str
    cik: str


class NPortParser:
    """
    Parser for SEC EDGAR N-PORT filings.

    Downloads and parses N-PORT-P XML files to extract ETF holdings.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/constituents_cache/nport")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce SEC rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < SEC_RATE_LIMIT_DELAY:
            time.sleep(SEC_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _request(self, url: str, max_retries: int = 3) -> requests.Response:
        """Make a rate-limited request to SEC EDGAR."""
        headers = {"User-Agent": SEC_USER_AGENT}

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        raise RuntimeError(f"Failed to fetch {url}")

    def get_cik_for_etf(self, ticker: str) -> Optional[str]:
        """Get CIK number for an ETF ticker."""
        if ticker in ETF_INDEX_MAP:
            return ETF_INDEX_MAP[ticker]["cik"]

        # Try to look up from SEC company tickers
        cache_path = self.cache_dir / "company_tickers.json"
        if not cache_path.exists():
            try:
                url = "https://www.sec.gov/files/company_tickers.json"
                response = self._request(url)
                with open(cache_path, "w") as f:
                    f.write(response.text)
            except Exception as e:
                logger.error(f"Failed to fetch company tickers: {e}")
                return None

        try:
            with open(cache_path) as f:
                data = json.load(f)
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
        except Exception as e:
            logger.error(f"Failed to parse company tickers: {e}")

        return None

    def find_nport_filings(
        self,
        cik: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[NPortFiling]:
        """
        Find N-PORT-P filings for a given CIK.

        Args:
            cik: The CIK number (with leading zeros)
            start_date: Filter filings after this date
            end_date: Filter filings before this date

        Returns:
            List of NPortFiling objects
        """
        # Normalize CIK
        cik = cik.lstrip("0").zfill(10)

        # Fetch submissions index
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            response = self._request(url)
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch submissions for CIK {cik}: {e}")
            return []

        filings = []

        # Parse recent filings
        recent = data.get("filings", {}).get("recent", {})
        filings.extend(self._parse_filing_list(recent, cik, start_date, end_date))

        # Also check historical filing files (for companies with many filings)
        historical_files = data.get("filings", {}).get("files", [])
        for file_info in historical_files:
            file_name = file_info.get("name", "")
            if not file_name:
                continue

            # Fetch historical filing data
            hist_url = f"https://data.sec.gov/submissions/{file_name}"
            try:
                logger.debug(f"Fetching historical filings: {file_name}")
                hist_response = self._request(hist_url)
                hist_data = hist_response.json()
                filings.extend(self._parse_filing_list(hist_data, cik, start_date, end_date))
            except Exception as e:
                logger.warning(f"Failed to fetch historical filings {file_name}: {e}")
                continue

        return sorted(filings, key=lambda f: f.report_date)

    def _parse_filing_list(
        self,
        filing_data: Dict[str, Any],
        cik: str,
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> List[NPortFiling]:
        """Parse a filing list from SEC API response."""
        filings = []

        forms = filing_data.get("form", [])
        accessions = filing_data.get("accessionNumber", [])
        filing_dates = filing_data.get("filingDate", [])
        primary_docs = filing_data.get("primaryDocument", [])
        report_dates = filing_data.get("reportDate", [])

        for i, form in enumerate(forms):
            # Look for N-PORT and NPORT-P filings (SEC uses both formats)
            # NPORT-P (no hyphen) is the actual form type used by SEC
            if not (form.startswith("N-PORT") or form.startswith("NPORT")):
                continue

            try:
                filing_date = datetime.strptime(filing_dates[i], "%Y-%m-%d").date()
                report_date_str = report_dates[i] if i < len(report_dates) else None

                # Some filings might not have report date
                if report_date_str:
                    report_date = datetime.strptime(report_date_str, "%Y-%m-%d").date()
                else:
                    report_date = filing_date

                if start_date and report_date < start_date:
                    continue
                if end_date and report_date > end_date:
                    continue

                filings.append(
                    NPortFiling(
                        accession_number=accessions[i],
                        filing_date=filing_date,
                        report_date=report_date,
                        primary_doc=primary_docs[i],
                        cik=cik,
                    )
                )
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse filing {i}: {e}")
                continue

        return filings

    def parse_nport_xml(
        self,
        filing: NPortFiling,
        target_fund_name: Optional[str] = None,
    ) -> List[MemberIdentifier]:
        """
        Parse N-PORT XML to extract holdings.

        Args:
            filing: The N-PORT filing to parse
            target_fund_name: If provided, only parse if fund name contains this string

        Returns:
            List of MemberIdentifier for each holding
        """
        # Check cache first
        cache_file = self.cache_dir / f"{filing.accession_number.replace('-', '')}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                # Check if cached data has fund_name metadata
                if isinstance(data, dict):
                    cached_fund = data.get("fund_name", "")
                    if target_fund_name and target_fund_name.upper() not in cached_fund.upper():
                        return []
                    return [MemberIdentifier.from_dict(m) for m in data.get("members", [])]
                return [MemberIdentifier.from_dict(m) for m in data]
            except Exception as e:
                logger.warning(f"Failed to load cached holdings: {e}")

        # Download XML
        accession_clean = filing.accession_number.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{filing.cik.lstrip('0')}/{accession_clean}/{filing.primary_doc}"

        try:
            response = self._request(url)
            xml_content = response.text
        except Exception as e:
            logger.error(f"Failed to download N-PORT XML: {e}")
            return []

        # Check fund name in header before full parsing
        fund_name = self._extract_fund_name(xml_content)
        if target_fund_name and fund_name:
            if target_fund_name.upper() not in fund_name.upper():
                logger.debug(f"Skipping filing - fund '{fund_name}' doesn't match target '{target_fund_name}'")
                # Cache empty result for this fund
                try:
                    with open(cache_file, "w") as f:
                        json.dump({"fund_name": fund_name, "members": []}, f)
                except Exception:
                    pass
                return []

        # Parse holdings from XML
        members = self._parse_holdings_xml(xml_content)

        # Cache results with fund name
        try:
            with open(cache_file, "w") as f:
                json.dump({
                    "fund_name": fund_name or "",
                    "members": [m.to_dict() for m in members]
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache holdings: {e}")

        return members

    def _extract_fund_name(self, xml_content: str) -> Optional[str]:
        """Extract the fund/series name from N-PORT XML header."""
        # Try to find seriesName or similar field
        patterns = [
            r'<seriesName>([^<]+)</seriesName>',
            r'<seriesNm>([^<]+)</seriesNm>',
            r'<fundName>([^<]+)</fundName>',
            r'<registrantName>([^<]+)</registrantName>',
        ]
        for pattern in patterns:
            match = re.search(pattern, xml_content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _parse_holdings_xml(self, xml_content: str) -> List[MemberIdentifier]:
        """Parse the XML content to extract holdings."""
        members = []

        # Preprocess XML content
        xml_clean = xml_content
        # Handle namespace
        xml_clean = re.sub(r'\sxmlns="[^"]+"', '', xml_clean)
        xml_clean = re.sub(r'\sxmlns:[a-z]+="[^"]+"', '', xml_clean)

        # Handle common HTML entities that aren't defined in XML
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&apos;': "'",
            '&ndash;': '-',
            '&mdash;': '--',
            '&bull;': '*',
            '&reg;': '(R)',
            '&copy;': '(C)',
            '&trade;': '(TM)',
        }
        for entity, replacement in html_entities.items():
            xml_clean = xml_clean.replace(entity, replacement)

        # Remove any remaining undefined entities
        xml_clean = re.sub(r'&[a-zA-Z]+;', '', xml_clean)

        # Try standard XML parser first
        try:
            root = ET.fromstring(xml_clean)

            # Find all investment/security entries
            for inv in root.iter():
                if inv.tag in ("invstOrSec", "InvstOrSec", "investment"):
                    member = self._parse_investment_element(inv)
                    if member:
                        members.append(member)

            if members:
                return members

        except ET.ParseError as e:
            logger.debug(f"Standard XML parser failed: {e}")

        # Try BeautifulSoup (more lenient with malformed XML)
        if HAS_BEAUTIFULSOUP:
            try:
                members = self._parse_holdings_beautifulsoup(xml_content)
                if members:
                    logger.debug(f"BeautifulSoup extracted {len(members)} holdings")
                    return members
            except Exception as e:
                logger.debug(f"BeautifulSoup parser failed: {e}")

        # Final fallback: regex parser
        members = self._parse_holdings_regex(xml_content)
        return members

    def _parse_holdings_beautifulsoup(self, xml_content: str) -> List[MemberIdentifier]:
        """Parse holdings using BeautifulSoup (more lenient parser)."""
        members = []
        seen_ids = set()

        soup = BeautifulSoup(xml_content, 'html.parser')

        # Find all invstOrSec elements (investment/security entries)
        for inv in soup.find_all(['invstorsec', 'invstOrSec']):
            cusip = None
            isin = None
            ticker = None
            name = None

            # Try to find CUSIP
            cusip_tag = inv.find('cusip')
            if cusip_tag and cusip_tag.string:
                cusip_text = cusip_tag.string.strip()
                if len(cusip_text) >= 8:
                    cusip = cusip_text[:9].upper()

            # Try to find ticker
            ticker_tag = inv.find(['ticker', 'tkr'])
            if ticker_tag and ticker_tag.string:
                ticker = ticker_tag.string.strip().upper()

            # Try to find ISIN
            isin_tag = inv.find('isin')
            if isin_tag and isin_tag.string:
                isin_text = isin_tag.string.strip().upper()
                if len(isin_text) == 12:
                    isin = isin_text

            # Try to find name
            name_tag = inv.find(['name', 'issuername', 'issuer'])
            if name_tag and name_tag.string:
                name = name_tag.string.strip()

            # Create member if we have any identifier
            if cusip and cusip not in seen_ids:
                seen_ids.add(cusip)
                members.append(
                    MemberIdentifier(
                        primary_id=cusip,
                        id_type=IdentifierType.CUSIP,
                        cusip=cusip,
                        isin=isin,
                        ticker=ticker,
                        name=name,
                    )
                )
            elif isin and isin not in seen_ids:
                seen_ids.add(isin)
                members.append(MemberIdentifier.from_isin(isin, ticker=ticker, name=name))
            elif ticker and ticker not in seen_ids:
                seen_ids.add(ticker)
                members.append(MemberIdentifier.from_ticker(ticker, name=name))

        return members

    def _parse_investment_element(self, element: ET.Element) -> Optional[MemberIdentifier]:
        """Parse a single investment element."""
        # Try to extract identifiers
        name = None
        ticker = None
        cusip = None
        isin = None

        for child in element.iter():
            tag = child.tag.lower()
            text = (child.text or "").strip()

            if not text:
                continue

            if tag in ("name", "issuername", "issuer"):
                name = text
            elif tag in ("ticker", "tkr"):
                ticker = text.upper()
            elif tag == "cusip":
                # CUSIP is 9 characters
                if len(text) >= 8:
                    cusip = text[:9]
            elif tag == "isin":
                if len(text) == 12:
                    isin = text.upper()
            elif tag == "identifiers":
                # Parse nested identifier block
                for id_child in child:
                    id_tag = id_child.tag.lower()
                    id_text = (id_child.text or "").strip()
                    if id_tag == "cusip" and len(id_text) >= 8:
                        cusip = id_text[:9]
                    elif id_tag == "isin" and len(id_text) == 12:
                        isin = id_text.upper()
                    elif id_tag == "ticker":
                        ticker = id_text.upper()

        # Skip if no usable identifier
        if not (cusip or isin or ticker):
            return None

        # Prefer CUSIP as primary ID for US securities
        if cusip:
            return MemberIdentifier(
                primary_id=cusip,
                id_type=IdentifierType.CUSIP,
                cusip=cusip,
                isin=isin,
                ticker=ticker,
                name=name,
            )
        elif isin:
            return MemberIdentifier.from_isin(isin, ticker=ticker, name=name)
        else:
            return MemberIdentifier.from_ticker(ticker, name=name)

    def _parse_holdings_regex(self, xml_content: str) -> List[MemberIdentifier]:
        """Fallback regex parser for holdings."""
        members = []
        seen_ids = set()

        # Multiple patterns for different N-PORT XML formats
        # Pattern 1: Standard CUSIP tags
        cusip_patterns = [
            r"<cusip>([A-Z0-9]{8,9})</cusip>",
            r"<identifiers>.*?<cusip[^>]*>([A-Z0-9]{8,9})</cusip>",
            r'"cusip"\s*:\s*"([A-Z0-9]{8,9})"',  # JSON format
        ]

        # Pattern 2: Ticker tags
        ticker_patterns = [
            r"<ticker>([A-Z]{1,6})</ticker>",
            r"<tkr>([A-Z]{1,6})</tkr>",
            r'"ticker"\s*:\s*"([A-Z]{1,6})"',
        ]

        # Try to find CUSIPs
        for pattern in cusip_patterns:
            cusips = re.findall(pattern, xml_content, re.IGNORECASE | re.DOTALL)
            for cusip in cusips:
                cusip = cusip.upper()[:9]
                if cusip not in seen_ids and len(cusip) >= 8:
                    seen_ids.add(cusip)
                    members.append(
                        MemberIdentifier(
                            primary_id=cusip,
                            id_type=IdentifierType.CUSIP,
                            cusip=cusip,
                        )
                    )

        # If no CUSIPs found, try tickers
        if not members:
            for pattern in ticker_patterns:
                tickers = re.findall(pattern, xml_content, re.IGNORECASE)
                for ticker in tickers:
                    ticker = ticker.upper()
                    if ticker not in seen_ids and len(ticker) >= 1:
                        seen_ids.add(ticker)
                        members.append(
                            MemberIdentifier(
                                primary_id=ticker,
                                id_type=IdentifierType.TICKER,
                                ticker=ticker,
                            )
                        )

        if members:
            logger.debug(f"Regex fallback extracted {len(members)} holdings")

        return members


class NPortProvider(SnapshotProvider):
    """
    Constituent provider using SEC N-PORT ETF holdings as proxy.

    This provides monthly snapshots of ETF holdings, which serve as
    a proxy for index constituents. Note that ETF holdings may differ
    from the actual index due to sampling, optimization, or timing.
    """

    def __init__(
        self,
        etf_ticker: str,
        index_id: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.etf_ticker = etf_ticker.upper()
        self._index_id = index_id or self._get_index_id()

        super().__init__(
            index_id=self._index_id,
            quality=DataQuality.PROXY_ETF_HOLDINGS,
            cache_dir=cache_dir,
        )

        self._parser = NPortParser(cache_dir=self._cache_dir / "nport_raw")
        self._etf_info = ETF_INDEX_MAP.get(self.etf_ticker, {})

    def _get_index_id(self) -> str:
        """Get index ID from ETF mapping."""
        if self.etf_ticker in ETF_INDEX_MAP:
            return ETF_INDEX_MAP[self.etf_ticker]["index"]
        return f"ETF_{self.etf_ticker}"

    @property
    def id(self) -> str:
        return f"nport:{self.etf_ticker}"

    @property
    def metadata(self) -> IndexMetadata:
        target_counts = {
            "SP500": 500,
            "R1000": 1000,
            "R2000": 2000,
            "NDX": 100,
            "EAFE": 900,
            "SX5E": 50,
        }

        return IndexMetadata(
            index_id=self._index_id,
            name=self._etf_info.get("name", f"{self.etf_ticker} Holdings"),
            description=f"ETF holdings from {self.etf_ticker} N-PORT filings (proxy for {self._index_id})",
            target_count=target_counts.get(self._index_id),
            region="US" if self._index_id in ("SP500", "R1000", "R2000", "NDX") else "INTL",
            currency="USD",
            default_quality=DataQuality.PROXY_ETF_HOLDINGS,
            available_qualities=[DataQuality.PROXY_ETF_HOLDINGS],
            frequency="monthly",
            notes=[
                "ETF holdings may differ from actual index constituents",
                "Data available from ~2019 onwards",
                "Monthly granularity only",
            ],
        )

    def _load_snapshots(self) -> Dict[date, ConstituentSnapshot]:
        """Load all available N-PORT snapshots."""
        # Check cache
        cache_data = self._load_cache("_snapshots")
        if cache_data:
            result = {}
            for date_str, snapshot_data in cache_data.items():
                d = date.fromisoformat(date_str)
                result[d] = ConstituentSnapshot.from_dict(snapshot_data)
            logger.info(f"Loaded {len(result)} snapshots from cache for {self.etf_ticker}")
            return result

        return {}

    def download(self, force: bool = False, start_date: Optional[date] = None) -> None:
        """
        Download N-PORT filings and extract holdings.

        Args:
            force: If True, redownload even if cached
            start_date: Start date for filings (default: 2019-01-01)
        """
        if not force and self.is_cached():
            logger.info(f"Using cached data for {self.etf_ticker}")
            return

        cik = self._parser.get_cik_for_etf(self.etf_ticker)
        if not cik:
            logger.error(f"Could not find CIK for {self.etf_ticker}")
            return

        start = start_date or date(2019, 1, 1)
        end = date.today()

        # Get fund name for filtering (for multi-fund trusts like iShares)
        target_fund_name = self._etf_info.get("name", self.etf_ticker)

        logger.info(f"Downloading N-PORT filings for {self.etf_ticker} (CIK: {cik})")
        logger.info(f"Filtering for fund: {target_fund_name}")
        filings = self._parser.find_nport_filings(cik, start_date=start, end_date=end)
        logger.info(f"Found {len(filings)} N-PORT filings")

        snapshots = {}
        matched_filings = 0
        for i, filing in enumerate(filings):
            if i % 100 == 0 and i > 0:
                logger.info(f"Progress: {i}/{len(filings)} filings processed, {matched_filings} matched")
            logger.debug(f"Parsing {filing.accession_number} ({filing.report_date})")
            members = self._parser.parse_nport_xml(filing, target_fund_name=target_fund_name)

            if members:
                matched_filings += 1
                snapshot = ConstituentSnapshot(
                    index_id=self._index_id,
                    as_of=filing.report_date,
                    members=members,
                    quality=DataQuality.PROXY_ETF_HOLDINGS,
                    source=self.id,
                    point_in_time=True,
                    meta={
                        "etf_ticker": self.etf_ticker,
                        "filing_date": filing.filing_date.isoformat(),
                        "accession_number": filing.accession_number,
                    },
                )
                snapshots[filing.report_date] = snapshot

        logger.info(f"Processed {len(filings)} filings, {matched_filings} matched fund '{target_fund_name}'")

        # Cache results
        cache_data = {d.isoformat(): s.to_dict() for d, s in snapshots.items()}
        self._save_cache(cache_data, "_snapshots")

        self._snapshots = snapshots
        self._loaded = True

        logger.info(f"Downloaded {len(snapshots)} snapshots for {self.etf_ticker}")


# Factory functions

def get_sp500_provider(cache_dir: Optional[Path] = None) -> NPortProvider:
    """Get S&P 500 provider using SPY ETF."""
    return NPortProvider("SPY", cache_dir=cache_dir)


def get_russell2000_provider(cache_dir: Optional[Path] = None) -> NPortProvider:
    """Get Russell 2000 provider using IWM ETF."""
    return NPortProvider("IWM", cache_dir=cache_dir)


def get_russell1000_provider(cache_dir: Optional[Path] = None) -> NPortProvider:
    """Get Russell 1000 provider using IWB ETF."""
    return NPortProvider("IWB", cache_dir=cache_dir)


def get_nasdaq100_provider(cache_dir: Optional[Path] = None) -> NPortProvider:
    """Get Nasdaq-100 provider using QQQ ETF."""
    return NPortProvider("QQQ", cache_dir=cache_dir)


def get_eafe_provider(cache_dir: Optional[Path] = None) -> NPortProvider:
    """Get MSCI EAFE provider using EFA ETF."""
    return NPortProvider("EFA", cache_dir=cache_dir)


def get_eurostoxx50_etf_provider(cache_dir: Optional[Path] = None) -> NPortProvider:
    """Get EURO STOXX 50 provider using FEZ ETF (proxy)."""
    return NPortProvider("FEZ", cache_dir=cache_dir)


def get_sector_provider(sector: str, cache_dir: Optional[Path] = None) -> NPortProvider:
    """
    Get S&P 500 sector provider.

    Args:
        sector: Sector ETF ticker (XLE, XLK, XLF, etc.)
    """
    sector = sector.upper()
    if sector not in ETF_INDEX_MAP:
        raise ValueError(f"Unknown sector ETF: {sector}")
    return NPortProvider(sector, cache_dir=cache_dir)
