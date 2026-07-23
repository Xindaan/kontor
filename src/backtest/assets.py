"""
Asset registry - Maps WKN/names to Yahoo Finance tickers.

This module provides a centralized registry of assets with their
WKN (German security identification number), human-readable names,
Yahoo Finance ticker symbols, and tax-relevant information.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Asset:
    """
    Represents a tradeable asset.

    Attributes:
        wkn: German security identification number (Wertpapierkennnummer)
        name: Human-readable name of the asset
        yahoo_ticker: Ticker symbol used by Yahoo Finance
        currency: Base currency of the asset (USD, EUR, GBP, etc.)
        asset_class: Type of asset (equity, bonds, commodity, mixed)
        teilfreistellung: Tax-free portion for German tax calculation
                          (0.30 for equity funds with >51% equities, 0.0 otherwise)
    """
    wkn: str
    name: str
    yahoo_ticker: str
    currency: str
    asset_class: str
    teilfreistellung: float = 0.0

    def __repr__(self) -> str:
        return f"Asset({self.wkn}, '{self.name}', {self.yahoo_ticker})"


# Registry of known assets (WKN -> Asset)
ASSET_REGISTRY: Dict[str, Asset] = {
    # US ETFs (traded in USD)
    "A0AET0": Asset("A0AET0", "SPDR S&P 500", "SPY", "USD", "equity", 0.30),
    "A0RPWH": Asset("A0RPWH", "iShares Core MSCI World", "IWDA.AS", "USD", "equity", 0.30),
    "A2PKXG": Asset("A2PKXG", "Vanguard FTSE All-World", "VWCE.DE", "EUR", "equity", 0.30),
    # VT has longer history than VWCE.DE (2008 vs 2019) - use for backtesting
    "VT": Asset("VT", "Vanguard Total World Stock", "VT", "USD", "equity", 0.30),
    "A0RPWJ": Asset("A0RPWJ", "iShares MSCI EM", "IEMA.AS", "USD", "equity", 0.30),
    "A0F5UH": Asset("A0F5UH", "iShares MSCI World", "URTH", "USD", "equity", 0.30),
    "A1JMDF": Asset("A1JMDF", "iShares MSCI EAFE", "EFA", "USD", "equity", 0.30),

    # US Bond ETFs
    "A0NECU": Asset("A0NECU", "iShares $ Treasury Bond 7-10yr", "IBTM.DE", "USD", "bonds", 0.0),
    "A0RL83": Asset("A0RL83", "Vanguard Intermediate-Term Treasury", "VGIT", "USD", "bonds", 0.0),
    "A14071": Asset("A14071", "iShares Core US Aggregate Bond", "AGG", "USD", "bonds", 0.0),
    "A0LGQL": Asset("A0LGQL", "iShares US Treasury Bond", "GOVT", "USD", "bonds", 0.0),
    "A1W0MQ": Asset("A1W0MQ", "Vanguard Total Bond Market", "BND", "USD", "bonds", 0.0),

    # European Bond ETFs
    "A0LGQM": Asset("A0LGQM", "iShares € Govt Bond 7-10yr", "IBGL.DE", "EUR", "bonds", 0.0),
    "A0RM44": Asset("A0RM44", "iShares € Govt Bond 3-5yr", "IBGM.DE", "EUR", "bonds", 0.0),

    # European Equity ETFs
    "263530": Asset("263530", "iShares STOXX Europe 600", "EXSA.DE", "EUR", "equity", 0.30),
    "593393": Asset("593393", "iShares Core DAX", "EXS1.DE", "EUR", "equity", 0.30),
    "A0YEDL": Asset("A0YEDL", "iShares EURO STOXX 50", "EUEA.DE", "EUR", "equity", 0.30),
    "A0YEDG": Asset("A0YEDG", "iShares Core S&P 500 UCITS ETF USD (Acc)", "SXR8.DE", "USD", "equity", 0.30),
    "A0F5UF": Asset("A0F5UF", "iShares NASDAQ-100 UCITS ETF (DE)", "EXXT.DE", "USD", "equity", 0.30),
    "A142N1": Asset("A142N1", "iShares S&P 500 Information Technology Sector UCITS ETF", "QDVE.DE", "USD", "equity", 0.30),
    "A142NY": Asset("A142NY", "iShares S&P 500 Financials Sector UCITS ETF", "QDVH.DE", "USD", "equity", 0.30),
    "A142N0": Asset("A142N0", "iShares S&P 500 Industrials Sector UCITS ETF", "IUIS.L", "USD", "equity", 0.30),
    "A142NX": Asset("A142NX", "iShares S&P 500 Energy Sector UCITS ETF", "QDVF.DE", "USD", "equity", 0.30),
    "A142NZ": Asset("A142NZ", "iShares S&P 500 Health Care Sector UCITS ETF", "QDVG.DE", "USD", "equity", 0.30),
    "A142N2": Asset("A142N2", "iShares S&P 500 Materials Sector UCITS ETF", "IUMS.L", "USD", "equity", 0.30),
    "A142N3": Asset("A142N3", "iShares S&P 500 Utilities Sector UCITS ETF", "2B7A.DE", "USD", "equity", 0.30),
    "A0LEW6": Asset("A0LEW6", "iShares US Property Yield UCITS ETF (Dist)", "IQQ7.DE", "USD", "equity", 0.30),
    "A2P1KY": Asset("A2P1KY", "iShares US Property Yield UCITS ETF (Acc)", "IUSI.DE", "USD", "equity", 0.30),
    "A2QC5J": Asset("A2QC5J", "VanEck Semiconductor UCITS ETF", "VVSM.DE", "USD", "equity", 0.30),

    # European leveraged ETPs used as broker-compatible proxies
    "A3GL7E": Asset("A3GL7E", "WisdomTree NASDAQ 100 3x Daily Leveraged", "QQQ3.L", "GBP", "equity", 0.0),
    "A1VBKR": Asset("A1VBKR", "WisdomTree S&P 500 3x Daily Leveraged", "3LUS.L", "GBP", "equity", 0.0),
    "A4ANZ5": Asset("A4ANZ5", "WisdomTree PHLX Semiconductor 3x Daily Leveraged", "3SEM.L", "USD", "equity", 0.0),
    "A3GWDS": Asset("A3GWDS", "Leverage Shares 3x Long Financials ETP Securities", "3XFE.L", "USD", "equity", 0.0),
    "A3GWD0": Asset("A3GWD0", "Leverage Shares 3x Long Oil & Gas ETP Securities", "3XEE.L", "USD", "equity", 0.0),

    # Commodities
    "A0LP78": Asset("A0LP78", "Xetra-Gold", "4GLD.DE", "EUR", "commodity", 0.0),
    "A0S9GB": Asset("A0S9GB", "SPDR Gold Shares", "GLD", "USD", "commodity", 0.0),

    # Benchmarks (internal use, prefixed with underscore)
    "_SPY": Asset("_SPY", "S&P 500 (Benchmark)", "SPY", "USD", "equity", 0.30),
    "_MSCI_WORLD": Asset("_MSCI_WORLD", "MSCI World (Benchmark)", "URTH", "USD", "equity", 0.30),
}

# Mapping from ticker suffix to currency
CURRENCY_MAP: Dict[str, str] = {
    ".DE": "EUR",   # Xetra (Germany)
    ".F": "EUR",    # Frankfurt
    ".PA": "EUR",   # Paris
    ".AS": "EUR",   # Amsterdam
    ".MI": "EUR",   # Milan
    ".L": "GBP",    # London
    ".TO": "CAD",   # Toronto
    ".HK": "HKD",   # Hong Kong
    ".T": "JPY",    # Tokyo
    "": "USD",      # Default (US tickers without suffix)
}

# The suffix heuristic is wrong on the LSE: several currency lines of the
# same ETP trade there (GBP, GBp=pence, USD). Blindly reading ".L" as GBP
# throws off the conversion by GBPUSD (~1.34x) for USD lines and by exactly
# 100x for pence lines. The authoritative source is the Yahoo `currency`
# field, not the suffix.
TICKER_CURRENCY_OVERRIDES: Dict[str, str] = {
    "3SEM.L": "USD",   # WisdomTree PHLX Semiconductor 3x Daily Lev
    "QQQ3.L": "USD",   # WisdomTree NASDAQ 100 3x Daily Lev
    "3LUS.L": "GBp",   # WisdomTree S&P 500 3x Daily Lev (pence line)
    "3USL.L": "USD",   # same ETP as 3LUS.L, but the USD line
}

# The point from which a series quotes in its declared currency. Before that
# it is a DIFFERENT line and therefore unusable -- the DataLoader masks that
# range.
#
# T-0433: `3LUS.L` is a spliced series. Up to 2017-03-15 the prints are
# US cents (value/100 == 3USL.L, median error 0.82% over 1054 days), from
# 2017-03-16 on they are true pence (median error 0.30% over 2348 days).
# Reading the deep history as pence throws off the conversion by GBPUSD.
# For long histories, use `3USL.L` instead
# (clean USD line, same 3406 days, no splice).
TICKER_HISTORY_VALID_FROM: Dict[str, str] = {
    "3LUS.L": "2017-03-16",
}


def detect_currency(ticker: str) -> str:
    """
    Detect currency for a ticker.

    Per-ticker overrides win over the suffix heuristic, because the suffix
    only identifies the exchange, not the quoted currency line.

    Args:
        ticker: Yahoo Finance ticker symbol

    Returns:
        Currency code (USD, EUR, GBP, GBp, ...). "GBp" means pence.
    """
    if ticker in TICKER_CURRENCY_OVERRIDES:
        return TICKER_CURRENCY_OVERRIDES[ticker]
    for suffix, currency in CURRENCY_MAP.items():
        if suffix and ticker.endswith(suffix):
            return currency
    return "USD"


def resolve_asset(identifier: str) -> Asset:
    """
    Find an asset by WKN, name, or Yahoo ticker.

    Searches in the following order:
    1. Exact WKN match
    2. Exact Yahoo ticker match
    3. Case-insensitive partial name match

    Args:
        identifier: WKN, ticker symbol, or asset name

    Returns:
        Matching Asset object

    Raises:
        ValueError: If no matching asset is found

    Examples:
        >>> resolve_asset("A0AET0")           # By WKN
        Asset(A0AET0, 'SPDR S&P 500', SPY)
        >>> resolve_asset("SPY")              # By ticker
        Asset(A0AET0, 'SPDR S&P 500', SPY)
        >>> resolve_asset("MSCI World")       # By partial name
        Asset(A0RPWH, 'iShares Core MSCI World', IWDA.AS)
    """
    # Exact WKN match
    if identifier in ASSET_REGISTRY:
        return ASSET_REGISTRY[identifier]

    # Exact Yahoo ticker match
    for asset in ASSET_REGISTRY.values():
        if asset.yahoo_ticker.upper() == identifier.upper():
            return asset

    # Case-insensitive partial name match
    # Only do partial matching for identifiers >= 5 chars to avoid false matches
    # e.g., "ALL" (Allstate ticker) should NOT match "All-World" (VWCE.DE)
    # Also skip if identifier looks like a ticker (all uppercase, 1-5 chars)
    is_likely_ticker = identifier.isupper() and len(identifier) <= 5
    if not is_likely_ticker and len(identifier) >= 5:
        identifier_lower = identifier.lower()
        for asset in ASSET_REGISTRY.values():
            if identifier_lower in asset.name.lower():
                return asset

    raise ValueError(f"Unknown asset: {identifier}")


def get_yahoo_ticker(identifier: str) -> str:
    """
    Get Yahoo Finance ticker for an asset identifier.

    If the identifier is already a valid Yahoo ticker (not in registry),
    it is returned as-is.

    Args:
        identifier: WKN, ticker symbol, or asset name

    Returns:
        Yahoo Finance ticker symbol
    """
    try:
        asset = resolve_asset(identifier)
        return asset.yahoo_ticker
    except ValueError:
        # Not in registry, assume it's already a valid Yahoo ticker
        return identifier


def get_benchmark_ticker(name: str) -> str:
    """
    Get the Yahoo ticker for a named benchmark.

    Args:
        name: Benchmark name ("S&P 500" or "MSCI World")

    Returns:
        Yahoo Finance ticker symbol

    Raises:
        ValueError: If benchmark name is unknown
    """
    benchmarks = {
        "S&P 500": "SPY",
        "SP500": "SPY",
        "SPY": "SPY",
        "MSCI World": "URTH",
        "MSCI_WORLD": "URTH",
        "URTH": "URTH",
    }

    normalized = name.upper().replace(" ", "_").replace("&", "")
    for key, ticker in benchmarks.items():
        if normalized == key.upper().replace(" ", "_").replace("&", ""):
            return ticker

    raise ValueError(f"Unknown benchmark: {name}. Available: S&P 500, MSCI World")


def list_assets() -> str:
    """
    Return a formatted table of all registered assets.

    Returns:
        Formatted string table of assets
    """
    lines = [
        f"{'WKN':<10} {'Name':<35} {'Ticker':<12} {'Currency':<8}",
        "─" * 70,
    ]

    for wkn, asset in sorted(ASSET_REGISTRY.items()):
        if not wkn.startswith("_"):  # Skip internal benchmarks
            lines.append(
                f"{asset.wkn:<10} {asset.name:<35} {asset.yahoo_ticker:<12} {asset.currency:<8}"
            )

    return "\n".join(lines)
