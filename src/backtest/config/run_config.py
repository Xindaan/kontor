"""
Unified Run Configuration for Investment-Backtest v2.

This module provides a single source of truth for all backtest parameters,
ensuring that single reports and comparison reports use identical configurations.

Key features:
- All parameters in one place
- Config hash for reproducibility verification
- Serialization to/from JSON for manifest storage
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Tuple, Dict, List, Literal, Optional, Any
import hashlib
import json

from backtest.external_features.config import ExternalFeaturesConfig


@dataclass
class CostConfig:
    """
    Transaction cost model configuration.

    All costs are applied on trade execution.

    Attributes:
        commission_pct: Commission as percentage of trade value (e.g., 0.001 = 0.1%)
        commission_min: Minimum commission per trade in portfolio currency
        spread_bps: Bid-ask spread in basis points (e.g., 5 = 0.05%)
        slippage_bps: Slippage in basis points (e.g., 5 = 0.05%)
    """
    commission_pct: float = 0.001  # 0.1%
    commission_min: float = 0.0
    spread_bps: float = 0.0  # 0 bps
    slippage_bps: float = 5.0  # 5 bps

    @property
    def spread_pct(self) -> float:
        """Spread as decimal percentage."""
        return self.spread_bps / 10000

    @property
    def slippage_pct(self) -> float:
        """Slippage as decimal percentage."""
        return self.slippage_bps / 10000

    @property
    def total_friction_bps(self) -> float:
        """Total trading friction in basis points (one-way)."""
        return self.commission_pct * 10000 + self.spread_bps + self.slippage_bps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def zero(cls) -> "CostConfig":
        """Create a zero-cost configuration (for testing/comparison)."""
        return cls(
            commission_pct=0.0,
            commission_min=0.0,
            spread_bps=0.0,
            slippage_bps=0.0
        )

    @classmethod
    def low(cls) -> "CostConfig":
        """Low-cost broker profile (e.g., Interactive Brokers)."""
        return cls(
            commission_pct=0.0005,  # 0.05%
            commission_min=1.0,
            spread_bps=2.0,
            slippage_bps=2.0
        )

    @classmethod
    def medium(cls) -> "CostConfig":
        """Medium-cost broker profile (typical discount broker)."""
        return cls(
            commission_pct=0.001,  # 0.1%
            commission_min=5.0,
            spread_bps=5.0,
            slippage_bps=5.0
        )

    @classmethod
    def high(cls) -> "CostConfig":
        """High-cost broker profile (full-service or high-friction)."""
        return cls(
            commission_pct=0.002,  # 0.2%
            commission_min=10.0,
            spread_bps=10.0,
            slippage_bps=10.0
        )


@dataclass
class TaxConfig:
    """
    Tax model configuration.

    German tax model (Abgeltungssteuer) without Vorabpauschale.

    Attributes:
        enabled: Whether tax calculations are applied
        tax_rate: Tax rate on realized gains (26.375% = 25% + 5.5% Soli)
        partial_exemption_equity: Teilfreistellung for equity funds (30%)
        exemption_amount: Freistellungsauftrag (1000 single, 2000 joint)
        cost_basis_method: FIFO or AVGCOST for lot tracking
    """
    enabled: bool = False
    tax_rate: float = 0.26375  # 25% + 5.5% Soli
    partial_exemption_equity: float = 0.30  # 30% tax-free for equity funds
    exemption_amount: float = 1000.0  # Freistellungsauftrag (single)
    cost_basis_method: Literal["FIFO", "AVGCOST"] = "FIFO"
    # Historical note: this layer used to know neither equity_fund_map nor
    # tax_treatment_map -- anything routed through RunConfig got the default
    # treatment for every instrument (equity fund, 30% Teilfreistellung, equity
    # Verlusttopf). For leveraged debt ETPs that is the wrong tax class, and it
    # went unnoticed because research scripts build BacktestConfig directly.
    equity_fund_map: Optional[Dict[str, bool]] = None
    tax_treatment_map: Optional[Dict[str, Tuple[str, bool]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaxConfig":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def german_single(cls) -> "TaxConfig":
        """German tax config for single filer."""
        return cls(
            enabled=True,
            tax_rate=0.26375,
            partial_exemption_equity=0.30,
            exemption_amount=1000.0,
            cost_basis_method="FIFO"
        )

    @classmethod
    def german_joint(cls) -> "TaxConfig":
        """German tax config for joint filers."""
        return cls(
            enabled=True,
            tax_rate=0.26375,
            partial_exemption_equity=0.30,
            exemption_amount=2000.0,
            cost_basis_method="FIFO"
        )


@dataclass
class RunConfig:
    """
    Unified configuration for a backtest run.

    This is the single source of truth for all backtest parameters.
    Both single-strategy reports and comparison reports should use the same
    RunConfig instance to ensure consistent results.

    Attributes:
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD or None for today)
        initial_capital: Starting capital in portfolio currency
        currency: Portfolio currency (EUR or USD)
        rebalance_frequency: How often to rebalance
        data_frequency: Price data frequency
        benchmark: Benchmark for comparison (ticker or name)
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        universe: List of allowed tickers (optional filter)
        costs: Transaction cost configuration
        tax: Tax model configuration
    """
    # Time period
    start_date: str = "2010-01-01"
    end_date: Optional[str] = None

    # Portfolio
    initial_capital: float = 10_000.0
    currency: Literal["EUR", "USD"] = "EUR"

    # Frequency
    rebalance_frequency: Literal["daily", "weekly", "monthly", "quarterly", "yearly"] = "monthly"
    data_frequency: Literal["daily", "monthly", "infer"] = "daily"

    # Benchmark
    benchmark: Optional[str] = "S&P 500"
    risk_free_rate: float = 0.02  # 2% annual
    cash_rate: float = 0.0  # Annual cash interest rate

    # Universe filter (optional)
    universe: Optional[List[str]] = None

    # Sub-configurations
    costs: CostConfig = field(default_factory=CostConfig)
    tax: TaxConfig = field(default_factory=TaxConfig)
    # Optional per-ticker/asset-class cost profile (JSON-compatible dict)
    cost_profile: Optional[Dict[str, Any]] = None
    # Optional execution realism settings
    execution_lag_days: int = 0
    max_volume_participation: Optional[float] = None
    min_daily_dollar_volume: float = 0.0
    liquidity_on_missing_volume: Literal["allow", "skip"] = "allow"
    # Optional portfolio-level risk overlays
    exposure_policy: Optional[Dict[str, Any]] = None
    risk_overlay: Optional[Dict[str, Any]] = None

    # External features pipeline (Phase A plumbing).
    # Serializable config only — the loader is built on demand so that
    # config_hash stays stable for JSON roundtrips.
    external_features: ExternalFeaturesConfig = field(default_factory=ExternalFeaturesConfig)

    # Metadata (not used in hash)
    name: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert nested dicts to config objects if needed
        if isinstance(self.costs, dict):
            self.costs = CostConfig.from_dict(self.costs)
        if isinstance(self.tax, dict):
            self.tax = TaxConfig.from_dict(self.tax)
        if isinstance(self.external_features, dict):
            self.external_features = ExternalFeaturesConfig.from_dict(self.external_features)

        # Validate dates
        if self.start_date:
            datetime.strptime(self.start_date, "%Y-%m-%d")
        if self.end_date:
            datetime.strptime(self.end_date, "%Y-%m-%d")

    @property
    def config_hash(self) -> str:
        """
        Generate a short hash of the configuration.

        Used for reproducibility verification - same hash = same config.
        Excludes metadata fields (name, description).
        """
        return create_config_hash(self)

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Args:
            include_metadata: Whether to include name/description

        Returns:
            Dictionary representation
        """
        data = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "currency": self.currency,
            "rebalance_frequency": self.rebalance_frequency,
            "data_frequency": self.data_frequency,
            "benchmark": self.benchmark,
            "risk_free_rate": self.risk_free_rate,
            "cash_rate": self.cash_rate,
            "universe": self.universe,
            "costs": self.costs.to_dict(),
            "tax": self.tax.to_dict(),
            "cost_profile": self.cost_profile,
            "execution_lag_days": self.execution_lag_days,
            "max_volume_participation": self.max_volume_participation,
            "min_daily_dollar_volume": self.min_daily_dollar_volume,
            "liquidity_on_missing_volume": self.liquidity_on_missing_volume,
            "exposure_policy": self.exposure_policy,
            "risk_overlay": self.risk_overlay,
            "external_features": self.external_features.to_dict(),
        }

        if include_metadata:
            data["name"] = self.name
            data["description"] = self.description
            data["config_hash"] = self.config_hash

        return data

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        """
        Create from dictionary.

        Args:
            data: Dictionary with config values

        Returns:
            RunConfig instance
        """
        # Remove computed fields
        data = data.copy()
        data.pop("config_hash", None)

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "RunConfig":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> "RunConfig":
        """Load from JSON file."""
        from pathlib import Path
        return cls.from_json(Path(path).read_text())

    def save(self, path: str) -> None:
        """Save to JSON file."""
        from pathlib import Path
        Path(path).write_text(self.to_json())

    def with_costs(self, costs: CostConfig) -> "RunConfig":
        """Create a copy with different cost config."""
        return RunConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            currency=self.currency,
            rebalance_frequency=self.rebalance_frequency,
            data_frequency=self.data_frequency,
            benchmark=self.benchmark,
            risk_free_rate=self.risk_free_rate,
            universe=self.universe,
            costs=costs,
            tax=self.tax,
            cost_profile=self.cost_profile,
            execution_lag_days=self.execution_lag_days,
            max_volume_participation=self.max_volume_participation,
            min_daily_dollar_volume=self.min_daily_dollar_volume,
            liquidity_on_missing_volume=self.liquidity_on_missing_volume,
            exposure_policy=self.exposure_policy,
            risk_overlay=self.risk_overlay,
            external_features=self.external_features,
            name=self.name,
            description=self.description,
        )

    def with_tax(self, tax: TaxConfig) -> "RunConfig":
        """Create a copy with different tax config."""
        return RunConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            currency=self.currency,
            rebalance_frequency=self.rebalance_frequency,
            data_frequency=self.data_frequency,
            benchmark=self.benchmark,
            risk_free_rate=self.risk_free_rate,
            universe=self.universe,
            costs=self.costs,
            tax=tax,
            cost_profile=self.cost_profile,
            execution_lag_days=self.execution_lag_days,
            max_volume_participation=self.max_volume_participation,
            min_daily_dollar_volume=self.min_daily_dollar_volume,
            liquidity_on_missing_volume=self.liquidity_on_missing_volume,
            exposure_policy=self.exposure_policy,
            risk_overlay=self.risk_overlay,
            external_features=self.external_features,
            name=self.name,
            description=self.description,
        )

    # Preset configurations
    @classmethod
    def default(cls) -> "RunConfig":
        """Default configuration for quick testing."""
        return cls()

    @classmethod
    def realistic(cls) -> "RunConfig":
        """Realistic configuration with medium costs."""
        return cls(
            start_date="2010-01-01",
            costs=CostConfig.medium(),
        )

    @classmethod
    def low_cost(cls) -> "RunConfig":
        """Configuration for low-cost broker simulation."""
        return cls(
            start_date="2010-01-01",
            costs=CostConfig.low(),
        )


def create_config_hash(config: RunConfig) -> str:
    """
    Create a deterministic hash of the configuration.

    Used to verify that two runs used identical parameters.
    Excludes metadata fields (name, description).

    Args:
        config: RunConfig to hash

    Returns:
        8-character hex hash
    """
    # Create deterministic dict (exclude metadata)
    data = config.to_dict(include_metadata=False)

    # Sort for determinism
    json_str = json.dumps(data, sort_keys=True)

    # Create hash
    hash_bytes = hashlib.sha256(json_str.encode()).digest()
    return hash_bytes[:4].hex()  # 8 hex chars


# Backward compatibility adapter
def config_to_backtest_config(run_config: RunConfig):
    """
    Convert RunConfig to legacy BacktestConfig.

    This adapter allows gradual migration to the new config system.
    """
    from backtest.backtester import BacktestConfig
    from backtest.external_features.config import build_loader_from_config

    return BacktestConfig(
        initial_capital=run_config.initial_capital,
        currency=run_config.currency,
        costs_pct=run_config.costs.commission_pct,
        slippage_pct=run_config.costs.slippage_pct,
        spread_pct=run_config.costs.spread_pct,
        benchmark=run_config.benchmark,
        risk_free_rate=run_config.risk_free_rate,
        cash_rate=run_config.cash_rate,
        rebalance_frequency=run_config.rebalance_frequency,
        # Tax model settings
        tax_enabled=run_config.tax.enabled,
        tax_rate=run_config.tax.tax_rate,
        tax_partial_exemption=run_config.tax.partial_exemption_equity,
        tax_exemption_amount=run_config.tax.exemption_amount,
        # Historical note: previously dropped silently. cost_basis_method was even a
        # DEAD config field -- it existed on TaxConfig but was never wired through,
        # so "AVGCOST" silently ran as FIFO.
        cost_basis_method=run_config.tax.cost_basis_method,
        equity_fund_map=run_config.tax.equity_fund_map,
        tax_treatment_map=run_config.tax.tax_treatment_map,
        cost_profile=run_config.cost_profile,
        execution_lag_days=run_config.execution_lag_days,
        max_volume_participation=run_config.max_volume_participation,
        min_daily_dollar_volume=run_config.min_daily_dollar_volume,
        liquidity_on_missing_volume=run_config.liquidity_on_missing_volume,
        exposure_policy=run_config.exposure_policy,
        risk_overlay=run_config.risk_overlay,
        external_features_loader=build_loader_from_config(run_config.external_features),
    )
