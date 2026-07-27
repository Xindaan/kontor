"""
Backtest - A modular Python framework for systematic backtesting of investment strategies.

Version 2.0 Features:
- Unified RunConfig for reproducible runs
- Run manifests for audit trails
- Centralized metrics with proper annualization
- German tax model (Abgeltungssteuer)
- Research tools (parameter sweeps, walk-forward)
"""

__version__ = "2.0.0"

# Pandas compatibility patches must be applied before module imports.
from backtest.pandas_compat import enable_month_end_alias
enable_month_end_alias()

from backtest.strategy import Strategy, Allocation
from backtest.data import DataLoader, PriceData, CurrencyConverter
from backtest.backtester import Backtester, BacktestConfig, BacktestResult, Trade, Portfolio
from backtest.metrics import MetricsCalculator, Metrics
from backtest.reporter import Reporter
from backtest.reconciliation import (
    CashflowReconciliation,
    DailyReconciliation,
    MonthlyReconciliation,
    validate_trades_invariants,
)
from backtest.comparator import Comparator, ComparisonResult
from backtest.assets import Asset, ASSET_REGISTRY, resolve_asset
from backtest.fundamentals import FundamentalsLoader, FundamentalsSnapshot
from backtest.provenance import (
    ManualDataProvenanceEntry,
    ManualDataProvenanceRegistry,
    QUALITY_TAGS as PROVENANCE_QUALITY_TAGS,
)

# v2 modules
from backtest.config import RunConfig, CostConfig, TaxConfig
from backtest.manifest import RunManifest
from backtest.costs import TransactionCostModel, CostBreakdown, CostModelResolver
from backtest.tax import GermanTaxModel
from backtest.risk import RiskOverlayConfig, RiskOverlayEngine
from backtest.meta_allocator import MetaAllocatorStrategy

# Universe providers (P2)
from backtest.universe import (
    UniverseProvider,
    UniverseSnapshot,
    StaticUniverseProvider,
    YahooScreenerProvider,
)

# Signal generation (live trading)
from backtest.signals import (
    TradingSignal,
    SignalReport,
    SignalGenerator,
    format_signal_report,
)

# Validation
from backtest.validation import (
    ValidationResult,
    ValidationIssue,
    ValidationLevel,
    validate_before_run,
    validate_backtest_result,
    validate_config,
    validate_price_data,
    validate_equity_curve,
    validate_temporal_leakage,
    validate_tax_invariants,
)

__all__ = [
    # Strategy
    "Strategy",
    "Allocation",
    # Data
    "DataLoader",
    "PriceData",
    "CurrencyConverter",
    # Backtester
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "Portfolio",
    # Metrics
    "MetricsCalculator",
    "Metrics",
    # Reporter
    "Reporter",
    # Reconciliation
    "CashflowReconciliation",
    "DailyReconciliation",
    "MonthlyReconciliation",
    "validate_trades_invariants",
    # Comparator
    "Comparator",
    "ComparisonResult",
    # Assets
    "Asset",
    "ASSET_REGISTRY",
    "resolve_asset",
    # Fundamentals
    "FundamentalsLoader",
    "FundamentalsSnapshot",
    # Manual data provenance
    "ManualDataProvenanceEntry",
    "ManualDataProvenanceRegistry",
    "PROVENANCE_QUALITY_TAGS",
    # v2: Config
    "RunConfig",
    "CostConfig",
    "TaxConfig",
    # v2: Manifest
    "RunManifest",
    # v2: Costs
    "TransactionCostModel",
    "CostBreakdown",
    "CostModelResolver",
    # v2: Tax
    "GermanTaxModel",
    # v2: Risk Overlay
    "RiskOverlayConfig",
    "RiskOverlayEngine",
    # v2: Meta Allocator
    "MetaAllocatorStrategy",
    # v2: Universe
    "UniverseProvider",
    "UniverseSnapshot",
    "StaticUniverseProvider",
    "YahooScreenerProvider",
    # Signals (live trading)
    "TradingSignal",
    "SignalReport",
    "SignalGenerator",
    "format_signal_report",
    # Validation
    "ValidationResult",
    "ValidationIssue",
    "ValidationLevel",
    "validate_before_run",
    "validate_backtest_result",
    "validate_config",
    "validate_price_data",
    "validate_equity_curve",
    "validate_temporal_leakage",
    "validate_tax_invariants",
]
