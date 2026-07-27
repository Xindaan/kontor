# Historical Index Constituents

Point-in-time (PIT) index membership data for survivorship-bias-free backtesting.

## Overview

This module provides historical index constituent data from multiple free/public sources,
with explicit quality labeling to help you understand the reliability of each data source.

### Supported Indices

| Index | ID | Region | Default Source | Quality |
|-------|-----|--------|----------------|---------|
| S&P 500 | SP500 | US | SPY N-PORT | proxy_etf_holdings |
| Nasdaq-100 | NDX | US | QQQ N-PORT + Wikipedia | proxy_etf_holdings |
| Russell 1000 | R1000 | US | IWB N-PORT | proxy_etf_holdings |
| Russell 2000 | R2000 | US | IWM N-PORT | proxy_etf_holdings |
| MSCI EAFE | EAFE | INTL | EFA N-PORT | proxy_etf_holdings |
| EURO STOXX 50 | SX5E | EU | STOXX PDFs | official_changelog |
| FTSE 100 | UKX | UK | LSEG PDF | official_changelog |
| Nikkei 225 | N225 | JP | Nikkei PDF | official_changelog |
| S&P 500 Sectors | SP500_* | US | Sector ETF N-PORT | proxy_etf_holdings |

### Quality Levels

1. **official_changelog** - Direct from index provider (highest quality)
   - FTSE 100: LSEG PDF changelog
   - Nikkei 225: Nikkei PDF changelog
   - EURO STOXX 50: STOXX PDF announcements

2. **proxy_etf_holdings** - ETF holdings as proxy
   - SEC N-PORT filings (monthly)
   - ETF ≠ Index (sampling/optimization may differ)
   - Available from ~2019

3. **community_changelog** - Community-maintained data
   - Wikipedia component change tables
   - Not audit-grade

4. **user_snapshot** - User-provided data
   - Local CSV files
   - Lowest priority

## Quick Start

```python
from datetime import date
from backtest.constituents import IndexRegistry, DataQuality

# Get provider for an index
provider = IndexRegistry.get_provider("SP500")

# Get members as of a specific date
snapshot = provider.snapshot(date(2020, 6, 15))
print(f"Members: {snapshot.size}")
print(f"Quality: {snapshot.quality.value}")
print(f"Tickers: {snapshot.tickers[:10]}")  # First 10

# Check if a ticker was in the index
if snapshot.contains("AAPL"):
    print("AAPL was in S&P 500")
```

## CLI Usage

```bash
# List all supported indices
python -m backtest.constituents.cli list

# Download data for an index
python -m backtest.constituents.cli download SP500

# Query members as of a date
python -m backtest.constituents.cli query SP500 2020-06-15

# Generate health report
python -m backtest.constituents.cli health SP500

# Export to CSV
python -m backtest.constituents.cli export SP500 sp500_history.csv --start 2020-01-01
```

## Data Sources

### SEC EDGAR N-PORT (ETF Holdings Proxy)

For US indices without free official changelogs, we use ETF holdings from SEC N-PORT filings:

| ETF | Tracks | CIK |
|-----|--------|-----|
| SPY | S&P 500 | 0000884394 |
| QQQ | Nasdaq-100 | 0001067839 |
| IWM | Russell 2000 | 0001100663 |
| IWB | Russell 1000 | 0001100663 |
| EFA | MSCI EAFE | 0001100663 |

**Limitations:**
- Monthly frequency only
- ETF holdings may differ from index (sampling, optimization)
- Available from ~2019 onwards

### Official PDF Changelogs

For some international indices, official changelogs are freely available:

- **FTSE 100**: [LSEG PDF](https://www.lseg.com/content/dam/ftse-russell/en_us/documents/policy-documents/ftse-100-constituent-history.pdf)
- **Nikkei 225**: [Nikkei PDF](https://indexes.nikkei.co.jp/nkave/archives/file/history_of_nikkei_stock_average_component_changes_en.pdf)
- **EURO STOXX 50**: STOXX News PDFs (crawled from news section)

### Wikipedia (Community)

Community-maintained changelogs for:
- Nasdaq-100: Component changes table
- S&P 500: Selected changes history

**Note:** Not audit-grade, use for research only.

## Integration with Backtester

```python
from backtest.constituents import IndexRegistry
from backtest.universe import CsvPITUniverseProvider

# Option 1: Use directly with IndexRegistry
provider = IndexRegistry.get_provider("SP500")
snapshot = provider.snapshot(rebalance_date)
universe = snapshot.tickers

# Option 2: Convert to UniverseSnapshot for compatibility
universe_snapshot = snapshot.to_universe_snapshot()

# Option 3: Export to CSV and use existing CsvPITUniverseProvider
# python -m backtest.constituents.cli export SP500 data/universes/sp500_nport.csv
```

## Health Reports

Generate health reports to check data quality:

```python
from backtest.constituents import IndexRegistry, HealthReport

provider = IndexRegistry.get_provider("SP500")
report = HealthReport.generate(provider)

print(f"Coverage: {report.coverage_start} to {report.coverage_end}")
print(f"Snapshots: {report.total_snapshots}")
print(f"Avg members: {report.avg_member_count:.0f}")
print(f"Stable ID coverage: {report.pct_with_stable_id:.1f}%")

for issue in report.issues:
    print(f"[{issue.severity}] {issue.message}")
```

## Architecture

```
backtest/constituents/
├── __init__.py          # Public API
├── models.py            # Data models (Snapshot, Change, etc.)
├── base.py              # Abstract provider classes
├── registry.py          # Index registry and factory
├── nport.py             # SEC EDGAR N-PORT parser
├── pdf_parsers.py       # PDF parsers (FTSE, Nikkei, STOXX)
├── wikipedia.py         # Wikipedia parsers
├── user_snapshot.py     # User CSV importer
├── health.py            # Health reports
└── cli.py               # Command-line interface
```

## Identifier Types

Different sources provide different identifier types:

| Type | Stability | Example | Sources |
|------|-----------|---------|---------|
| ISIN | Highest | US0378331005 | STOXX PDFs |
| CUSIP | High | 037833100 | N-PORT |
| SEDOL | High | 2046251 | FTSE |
| JP_CODE | Medium | 7203 | Nikkei |
| TICKER | Low | AAPL | All |
| NAME | Lowest | Apple Inc. | All |

## Limitations

1. **Russell/MSCI** - Official changelogs require expensive subscriptions.
   We use ETF proxies which may differ from the actual index.

2. **S&P 500 Sectors** - True GICS sector history is not free.
   Sector ETF holdings serve as proxies.

3. **Historical Depth** - N-PORT data available from ~2019.
   For earlier data, use community sources or local PIT files.

4. **Frequency** - N-PORT is monthly only.
   Official changelogs are event-based (more precise).

## Contributing

To add a new index:

1. Create a provider class inheriting from `ChangelogProvider` or `SnapshotProvider`
2. Add index config to `INDEX_CONFIGS` in `registry.py`
3. Add factory function in `registry.py`
4. Add tests and documentation
