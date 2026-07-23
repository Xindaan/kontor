"""
Index Registry - Central factory for constituent providers.

Provides unified access to all index constituent data sources
with automatic fallback and quality-based selection.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Type, Callable, Any

from .base import ConstituentProvider, CompositeProvider
from .models import DataQuality, IndexMetadata

logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """Configuration for an index."""

    index_id: str
    name: str
    description: str
    region: str
    currency: str
    target_count: Optional[int] = None
    default_quality: DataQuality = DataQuality.PROXY_ETF_HOLDINGS
    available_sources: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# Index configurations
INDEX_CONFIGS: Dict[str, IndexConfig] = {
    # US Indices
    "SP500": IndexConfig(
        index_id="SP500",
        name="S&P 500",
        description="500 largest US companies by market cap",
        region="US",
        currency="USD",
        target_count=500,
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_spy", "local_pit", "wikipedia"],
        notes=["Official changelog not freely available", "Use SPY ETF as proxy"],
    ),
    "NDX": IndexConfig(
        index_id="NDX",
        name="Nasdaq-100",
        description="100 largest non-financial Nasdaq companies",
        region="US",
        currency="USD",
        target_count=100,
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_qqq", "local_pit", "wikipedia"],
        notes=["Wikipedia changelog available", "Annual reconstitution in December"],
    ),
    "R1000": IndexConfig(
        index_id="R1000",
        name="Russell 1000",
        description="1000 largest US companies",
        region="US",
        currency="USD",
        target_count=1000,
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_iwb"],
        notes=["Official changelog requires subscription", "Annual reconstitution in June"],
    ),
    "R2000": IndexConfig(
        index_id="R2000",
        name="Russell 2000",
        description="Small-cap US companies (ranks 1001-3000)",
        region="US",
        currency="USD",
        target_count=2000,
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_iwm"],
        notes=["Official changelog requires subscription", "Annual reconstitution in June"],
    ),
    # International Indices
    "EAFE": IndexConfig(
        index_id="EAFE",
        name="MSCI EAFE",
        description="Developed markets excluding US and Canada",
        region="INTL",
        currency="USD",
        target_count=900,
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_efa"],
        notes=["Official changelog requires MSCI subscription"],
    ),
    "SX5E": IndexConfig(
        index_id="SX5E",
        name="EURO STOXX 50",
        description="50 largest Eurozone companies",
        region="EU",
        currency="EUR",
        target_count=50,
        default_quality=DataQuality.OFFICIAL_CHANGELOG,
        available_sources=["stoxx_pdf", "nport_fez"],
        notes=["Official PDFs from STOXX", "Annual review in September"],
    ),
    "UKX": IndexConfig(
        index_id="UKX",
        name="FTSE 100",
        description="100 largest UK companies",
        region="UK",
        currency="GBP",
        target_count=100,
        default_quality=DataQuality.OFFICIAL_CHANGELOG,
        available_sources=["ftse_pdf"],
        notes=["Official PDF from LSEG", "Quarterly reconstitution"],
    ),
    "N225": IndexConfig(
        index_id="N225",
        name="Nikkei 225",
        description="225 largest Japanese companies",
        region="JP",
        currency="JPY",
        target_count=225,
        default_quality=DataQuality.OFFICIAL_CHANGELOG,
        available_sources=["nikkei_pdf"],
        notes=["Official PDF from Nikkei Inc.", "Annual review in October"],
    ),
    # S&P 500 Sectors
    "SP500_ENERGY": IndexConfig(
        index_id="SP500_ENERGY",
        name="S&P 500 Energy",
        description="S&P 500 Energy sector",
        region="US",
        currency="USD",
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_xle"],
    ),
    "SP500_TECH": IndexConfig(
        index_id="SP500_TECH",
        name="S&P 500 Technology",
        description="S&P 500 Technology sector",
        region="US",
        currency="USD",
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_xlk"],
    ),
    "SP500_FINANCIALS": IndexConfig(
        index_id="SP500_FINANCIALS",
        name="S&P 500 Financials",
        description="S&P 500 Financials sector",
        region="US",
        currency="USD",
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_xlf"],
    ),
    "SP500_HEALTHCARE": IndexConfig(
        index_id="SP500_HEALTHCARE",
        name="S&P 500 Health Care",
        description="S&P 500 Health Care sector",
        region="US",
        currency="USD",
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_xlv"],
    ),
    "SP500_INDUSTRIALS": IndexConfig(
        index_id="SP500_INDUSTRIALS",
        name="S&P 500 Industrials",
        description="S&P 500 Industrials sector",
        region="US",
        currency="USD",
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_xli"],
    ),
    "SP500_DISCRETIONARY": IndexConfig(
        index_id="SP500_DISCRETIONARY",
        name="S&P 500 Consumer Discretionary",
        description="S&P 500 Consumer Discretionary sector",
        region="US",
        currency="USD",
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_xly"],
    ),
    "SP500_STAPLES": IndexConfig(
        index_id="SP500_STAPLES",
        name="S&P 500 Consumer Staples",
        description="S&P 500 Consumer Staples sector",
        region="US",
        currency="USD",
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_xlp"],
    ),
    "SP500_UTILITIES": IndexConfig(
        index_id="SP500_UTILITIES",
        name="S&P 500 Utilities",
        description="S&P 500 Utilities sector",
        region="US",
        currency="USD",
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_xlu"],
    ),
    "SP500_MATERIALS": IndexConfig(
        index_id="SP500_MATERIALS",
        name="S&P 500 Materials",
        description="S&P 500 Materials sector",
        region="US",
        currency="USD",
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_xlb"],
    ),
    "SP500_REALESTATE": IndexConfig(
        index_id="SP500_REALESTATE",
        name="S&P 500 Real Estate",
        description="S&P 500 Real Estate sector",
        region="US",
        currency="USD",
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_xlre"],
    ),
    "SP500_COMMUNICATION": IndexConfig(
        index_id="SP500_COMMUNICATION",
        name="S&P 500 Communication Services",
        description="S&P 500 Communication Services sector",
        region="US",
        currency="USD",
        default_quality=DataQuality.PROXY_ETF_HOLDINGS,
        available_sources=["nport_xlc"],
    ),
}

# Aliases for common names
INDEX_ALIASES = {
    "SPX": "SP500",
    "SPY": "SP500",
    "NASDAQ100": "NDX",
    "NASDAQ_100": "NDX",
    "NASDAQ-100": "NDX",
    "QQQ": "NDX",
    "RUSSELL1000": "R1000",
    "RUSSELL2000": "R2000",
    "IWM": "R2000",
    "IWB": "R1000",
    "EFA": "EAFE",
    "MSCI_EAFE": "EAFE",
    "EUROSTOXX50": "SX5E",
    "EURO_STOXX_50": "SX5E",
    "FEZ": "SX5E",
    "FTSE100": "UKX",
    "FTSE_100": "UKX",
    "NIKKEI225": "N225",
    "NIKKEI_225": "N225",
    # Sector aliases
    "XLE": "SP500_ENERGY",
    "XLK": "SP500_TECH",
    "XLF": "SP500_FINANCIALS",
    "XLV": "SP500_HEALTHCARE",
    "XLI": "SP500_INDUSTRIALS",
    "XLY": "SP500_DISCRETIONARY",
    "XLP": "SP500_STAPLES",
    "XLU": "SP500_UTILITIES",
    "XLB": "SP500_MATERIALS",
    "XLRE": "SP500_REALESTATE",
    "XLC": "SP500_COMMUNICATION",
}


class IndexRegistry:
    """
    Central registry for index constituent providers.

    Provides factory methods to get providers for any supported index,
    with automatic source selection and fallback.
    """

    _cache_dir: Path = Path("data/constituents_cache")
    _providers: Dict[str, ConstituentProvider] = {}

    @classmethod
    def set_cache_dir(cls, path: Path) -> None:
        """Set the cache directory for all providers."""
        cls._cache_dir = path
        cls._cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def normalize_index_id(cls, index_id: str) -> str:
        """Normalize index ID using aliases."""
        normalized = index_id.upper().replace("-", "_").replace(" ", "_")
        return INDEX_ALIASES.get(normalized, normalized)

    @classmethod
    def get_config(cls, index_id: str) -> Optional[IndexConfig]:
        """Get configuration for an index."""
        normalized = cls.normalize_index_id(index_id)
        return INDEX_CONFIGS.get(normalized)

    @classmethod
    def list_indices(cls) -> List[str]:
        """List all supported index IDs."""
        return list(INDEX_CONFIGS.keys())

    @classmethod
    def list_indices_by_region(cls, region: str) -> List[str]:
        """List indices for a specific region."""
        return [
            idx for idx, config in INDEX_CONFIGS.items()
            if config.region.upper() == region.upper()
        ]

    @classmethod
    def get_provider(
        cls,
        index_id: str,
        quality: Optional[DataQuality] = None,
        use_cache: bool = True,
    ) -> ConstituentProvider:
        """
        Get a constituent provider for an index.

        Args:
            index_id: Index identifier (e.g., "SP500", "NDX", "UKX")
            quality: Preferred quality level (uses default if None)
            use_cache: Whether to cache provider instance

        Returns:
            ConstituentProvider for the requested index

        Raises:
            ValueError: If index is not supported
        """
        normalized = cls.normalize_index_id(index_id)
        config = INDEX_CONFIGS.get(normalized)

        if not config:
            raise ValueError(
                f"Unknown index: {index_id}. "
                f"Supported indices: {', '.join(INDEX_CONFIGS.keys())}"
            )

        cache_key = f"{normalized}:{quality.value if quality else 'default'}"
        if use_cache and cache_key in cls._providers:
            return cls._providers[cache_key]

        # Create provider based on available sources
        provider = cls._create_provider(config, quality)

        if use_cache:
            cls._providers[cache_key] = provider

        return provider

    @classmethod
    def _create_provider(
        cls,
        config: IndexConfig,
        quality: Optional[DataQuality] = None,
    ) -> ConstituentProvider:
        """Create provider based on config and quality preference."""
        target_quality = quality or config.default_quality

        # Import providers lazily to avoid circular imports
        from .nport import NPortProvider
        from .pdf_parsers import (
            FTSE100Provider,
            Nikkei225Provider,
            EuroStoxx50Provider,
        )
        from .wikipedia import (
            WikipediaNasdaq100Provider,
            WikipediaSP500Provider,
        )
        from .user_snapshot import LocalPITProvider

        # Source mapping
        source_factory: Dict[str, Callable[[], ConstituentProvider]] = {
            # N-PORT providers
            "nport_spy": lambda: NPortProvider("SPY", cache_dir=cls._cache_dir),
            "nport_qqq": lambda: NPortProvider("QQQ", cache_dir=cls._cache_dir),
            "nport_iwm": lambda: NPortProvider("IWM", cache_dir=cls._cache_dir),
            "nport_iwb": lambda: NPortProvider("IWB", cache_dir=cls._cache_dir),
            "nport_efa": lambda: NPortProvider("EFA", cache_dir=cls._cache_dir),
            "nport_fez": lambda: NPortProvider("FEZ", cache_dir=cls._cache_dir),
            "nport_xle": lambda: NPortProvider("XLE", cache_dir=cls._cache_dir),
            "nport_xlk": lambda: NPortProvider("XLK", cache_dir=cls._cache_dir),
            "nport_xlf": lambda: NPortProvider("XLF", cache_dir=cls._cache_dir),
            "nport_xlv": lambda: NPortProvider("XLV", cache_dir=cls._cache_dir),
            "nport_xli": lambda: NPortProvider("XLI", cache_dir=cls._cache_dir),
            "nport_xly": lambda: NPortProvider("XLY", cache_dir=cls._cache_dir),
            "nport_xlp": lambda: NPortProvider("XLP", cache_dir=cls._cache_dir),
            "nport_xlu": lambda: NPortProvider("XLU", cache_dir=cls._cache_dir),
            "nport_xlb": lambda: NPortProvider("XLB", cache_dir=cls._cache_dir),
            "nport_xlre": lambda: NPortProvider("XLRE", cache_dir=cls._cache_dir),
            "nport_xlc": lambda: NPortProvider("XLC", cache_dir=cls._cache_dir),
            # PDF providers
            "ftse_pdf": lambda: FTSE100Provider(cache_dir=cls._cache_dir),
            "nikkei_pdf": lambda: Nikkei225Provider(cache_dir=cls._cache_dir),
            "stoxx_pdf": lambda: EuroStoxx50Provider(cache_dir=cls._cache_dir),
            # Wikipedia providers
            "wikipedia": lambda: (
                WikipediaNasdaq100Provider(cache_dir=cls._cache_dir)
                if config.index_id == "NDX"
                else WikipediaSP500Provider(cache_dir=cls._cache_dir)
            ),
            # Local PIT
            "local_pit": lambda: LocalPITProvider(
                index_id=config.index_id, cache_dir=cls._cache_dir
            ),
        }

        # Find best source for requested quality
        providers = []
        for source in config.available_sources:
            if source in source_factory:
                try:
                    provider = source_factory[source]()
                    providers.append(provider)
                except Exception as e:
                    logger.warning(f"Failed to create provider {source}: {e}")

        if not providers:
            raise ValueError(f"No providers available for {config.index_id}")

        # If multiple providers, create composite
        if len(providers) > 1:
            return CompositeProvider(
                index_id=config.index_id,
                providers=providers,
                cache_dir=cls._cache_dir,
            )

        return providers[0]

    @classmethod
    def download_all(cls, force: bool = False) -> Dict[str, bool]:
        """
        Download data for all indices.

        Returns dict of index_id -> success
        """
        results = {}

        for index_id in INDEX_CONFIGS:
            try:
                provider = cls.get_provider(index_id)
                provider.download(force=force)
                results[index_id] = True
                logger.info(f"Downloaded {index_id}")
            except Exception as e:
                logger.error(f"Failed to download {index_id}: {e}")
                results[index_id] = False

        return results

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """Get summary of all available indices."""
        summary = {
            "total_indices": len(INDEX_CONFIGS),
            "by_region": {},
            "by_quality": {},
            "indices": [],
        }

        for index_id, config in INDEX_CONFIGS.items():
            # Count by region
            region = config.region
            summary["by_region"][region] = summary["by_region"].get(region, 0) + 1

            # Count by quality
            quality = config.default_quality.value
            summary["by_quality"][quality] = summary["by_quality"].get(quality, 0) + 1

            # Add index info
            summary["indices"].append({
                "id": index_id,
                "name": config.name,
                "region": region,
                "quality": quality,
                "sources": config.available_sources,
            })

        return summary


# Convenience functions

def get_provider(index_id: str, quality: Optional[DataQuality] = None) -> ConstituentProvider:
    """Get constituent provider for an index."""
    return IndexRegistry.get_provider(index_id, quality)


def list_indices() -> List[str]:
    """List all supported indices."""
    return IndexRegistry.list_indices()


def snapshot(index_id: str, as_of: date) -> "ConstituentSnapshot":
    """Get index snapshot for a date."""
    from .models import ConstituentSnapshot
    provider = IndexRegistry.get_provider(index_id)
    return provider.snapshot(as_of)
