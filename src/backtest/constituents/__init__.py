"""
Historical Index Constituents Framework.

Provides point-in-time (PIT) index membership data for survivorship-bias-free
backtesting. Supports multiple data sources with explicit quality labeling.

Quality Levels (priority order):
- official_changelog: Direct from index provider (e.g., FTSE PDF changelogs)
- proxy_etf_holdings: ETF holdings as proxy (e.g., SEC N-PORT filings)
- community_changelog: Community-maintained data (e.g., Wikipedia)
- user_snapshot: User-provided snapshots

Key Components:
- ConstituentSnapshot: Point-in-time snapshot with quality metadata
- ConstituentProvider: Abstract base for data sources
- IndexRegistry: Factory for index-specific providers

Supported Indices:
- S&P 500 (SP500) - proxy_etf_holdings via SPY N-PORT
- S&P 500 Sectors - proxy_etf_holdings via Sector ETFs
- Russell 1000 (R1000) - proxy_etf_holdings via IWB N-PORT
- Russell 2000 (R2000) - proxy_etf_holdings via IWM N-PORT
- Nasdaq-100 (NDX) - proxy_etf_holdings via QQQ + community_changelog
- MSCI EAFE (EAFE) - proxy_etf_holdings via EFA N-PORT
- EURO STOXX 50 (SX5E) - official_changelog via STOXX PDFs
- FTSE 100 (UKX) - official_changelog via FTSE PDF
- Nikkei 225 (N225) - official_changelog via Nikkei PDF

Usage:
    from backtest.constituents import IndexRegistry, DataQuality

    # Get provider for an index
    provider = IndexRegistry.get_provider("SP500")

    # Get members as of a specific date
    snapshot = provider.snapshot(date(2020, 6, 15))
    print(snapshot.members)  # List of member identifiers
    print(snapshot.quality)  # DataQuality.PROXY_ETF_HOLDINGS

    # Get health report
    from backtest.constituents import HealthReport
    report = HealthReport.generate(provider)
    print(report.coverage_start, report.coverage_end)
"""

from .models import (
    DataQuality,
    IdentifierType,
    MemberIdentifier,
    ConstituentChange,
    ConstituentSnapshot,
    IndexMetadata,
)
from .base import ConstituentProvider, ChangelogProvider, SnapshotProvider
from .registry import IndexRegistry, INDEX_CONFIGS
from .health import HealthReport, IntegrityCheck

__all__ = [
    # Models
    "DataQuality",
    "IdentifierType",
    "MemberIdentifier",
    "ConstituentChange",
    "ConstituentSnapshot",
    "IndexMetadata",
    # Providers
    "ConstituentProvider",
    "ChangelogProvider",
    "SnapshotProvider",
    # Registry
    "IndexRegistry",
    "INDEX_CONFIGS",
    # Health
    "HealthReport",
    "IntegrityCheck",
]
