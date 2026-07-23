"""
Transaction Cost Model v2.

Provides a transparent and decomposable cost model for realistic backtesting.
All costs are clearly separated and reported individually.

Components:
- Commission: Fixed per trade and/or percentage of trade value
- Spread: Bid-ask spread as cost (half-spread applied per trade)
- Slippage: Market impact / execution slippage

Usage:
    from backtest.costs import TransactionCostModel

    model = TransactionCostModel(
        commission_pct=0.001,  # 0.1%
        commission_min=1.0,    # $1 minimum
        spread_bps=5.0,        # 5 basis points
        slippage_bps=5.0,      # 5 basis points
    )

    costs = model.calculate(trade_value=10000, action="BUY")
    print(costs)  # CostBreakdown with all components
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from backtest.backtester import Trade


@dataclass
class CostBreakdown:
    """
    Detailed breakdown of transaction costs.

    All values are in portfolio currency (absolute, not percentage).

    Attributes:
        commission: Commission cost
        spread: Half-spread cost (bid-ask)
        slippage: Slippage/market impact cost
        total: Total costs
        trade_value: Original trade value before costs
    """
    commission: float
    spread: float
    slippage: float
    trade_value: float

    @property
    def total(self) -> float:
        """Total transaction costs."""
        return self.commission + self.spread + self.slippage

    @property
    def total_bps(self) -> float:
        """Total costs in basis points of trade value."""
        if self.trade_value == 0:
            return 0.0
        return (self.total / self.trade_value) * 10000

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "commission": self.commission,
            "spread": self.spread,
            "slippage": self.slippage,
            "total": self.total,
            "trade_value": self.trade_value,
            "total_bps": self.total_bps,
        }

    def __repr__(self) -> str:
        return (
            f"CostBreakdown(commission={self.commission:.2f}, "
            f"spread={self.spread:.2f}, slippage={self.slippage:.2f}, "
            f"total={self.total:.2f} [{self.total_bps:.1f} bps])"
        )


@dataclass
class TransactionCostModel:
    """
    Configurable transaction cost model.

    Supports three cost components:
    1. Commission: Fixed minimum + percentage of trade value
    2. Spread: Bid-ask spread (half applied per trade)
    3. Slippage: Market impact / execution slippage

    All percentage values are in decimal form (0.001 = 0.1%).
    Basis points are in standard form (5 = 0.05%).

    Attributes:
        commission_pct: Commission as percentage of trade value
        commission_min: Minimum commission per trade
        spread_bps: Bid-ask spread in basis points
        slippage_bps: Slippage in basis points
    """
    commission_pct: float = 0.001  # 0.1%
    commission_min: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 5.0  # 5 bps

    @property
    def spread_pct(self) -> float:
        """Half-spread as percentage (applied per trade)."""
        return self.spread_bps / 10000 / 2

    @property
    def slippage_pct(self) -> float:
        """Slippage as percentage."""
        return self.slippage_bps / 10000

    @property
    def total_friction_bps(self) -> float:
        """Total one-way friction in basis points."""
        commission_bps = self.commission_pct * 10000
        half_spread_bps = self.spread_bps / 2
        return commission_bps + half_spread_bps + self.slippage_bps

    def calculate(
        self,
        trade_value: float,
        action: Literal["BUY", "SELL"] = "BUY"
    ) -> CostBreakdown:
        """
        Calculate transaction costs for a trade.

        Args:
            trade_value: Absolute value of the trade
            action: "BUY" or "SELL" (currently treated the same)

        Returns:
            CostBreakdown with all cost components
        """
        # Commission: max of (percentage * value, minimum)
        commission = max(
            self.commission_pct * abs(trade_value),
            self.commission_min
        )

        # Spread: half-spread applied per trade
        spread = self.spread_pct * abs(trade_value)

        # Slippage
        slippage = self.slippage_pct * abs(trade_value)

        return CostBreakdown(
            commission=commission,
            spread=spread,
            slippage=slippage,
            trade_value=abs(trade_value),
        )

    def effective_price(
        self,
        price: float,
        action: Literal["BUY", "SELL"]
    ) -> float:
        """
        Calculate effective execution price after costs.

        Args:
            price: Market price
            action: "BUY" or "SELL"

        Returns:
            Effective price including spread and slippage
        """
        friction = self.spread_pct + self.slippage_pct

        if action == "BUY":
            # Buyer pays more
            return price * (1 + friction)
        else:
            # Seller receives less
            return price * (1 - friction)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "commission_pct": self.commission_pct,
            "commission_min": self.commission_min,
            "spread_bps": self.spread_bps,
            "slippage_bps": self.slippage_bps,
            "total_friction_bps": self.total_friction_bps,
        }

    @classmethod
    def from_config(cls, config) -> "TransactionCostModel":
        """
        Create from CostConfig or RunConfig.

        Args:
            config: CostConfig, RunConfig, or dict with cost parameters

        Returns:
            TransactionCostModel instance
        """
        if hasattr(config, 'costs'):
            # RunConfig
            cost_config = config.costs
        elif hasattr(config, 'commission_pct'):
            # CostConfig
            cost_config = config
        else:
            # Dict
            cost_config = config

        return cls(
            commission_pct=getattr(cost_config, 'commission_pct', cost_config.get('commission_pct', 0.001)),
            commission_min=getattr(cost_config, 'commission_min', cost_config.get('commission_min', 0.0)),
            spread_bps=getattr(cost_config, 'spread_bps', cost_config.get('spread_bps', 0.0)),
            slippage_bps=getattr(cost_config, 'slippage_bps', cost_config.get('slippage_bps', 5.0)),
        )

    # Preset cost profiles
    @classmethod
    def zero(cls) -> "TransactionCostModel":
        """Zero-cost model for testing."""
        return cls(
            commission_pct=0.0,
            commission_min=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
        )

    @classmethod
    def low(cls) -> "TransactionCostModel":
        """Low-cost broker (e.g., Interactive Brokers)."""
        return cls(
            commission_pct=0.0005,  # 0.05%
            commission_min=1.0,
            spread_bps=2.0,
            slippage_bps=2.0,
        )

    @classmethod
    def medium(cls) -> "TransactionCostModel":
        """Medium-cost broker (typical discount broker)."""
        return cls(
            commission_pct=0.001,  # 0.1%
            commission_min=5.0,
            spread_bps=5.0,
            slippage_bps=5.0,
        )

    @classmethod
    def high(cls) -> "TransactionCostModel":
        """High-cost broker (full-service or high-friction)."""
        return cls(
            commission_pct=0.002,  # 0.2%
            commission_min=10.0,
            spread_bps=10.0,
            slippage_bps=10.0,
        )


@dataclass
class TradingCostSummary:
    """
    Summary of trading costs over a backtest period.

    Attributes:
        total_costs: Total costs in portfolio currency
        total_trades: Number of trades executed
        total_traded_value: Total value traded
        costs_per_year: Annualized costs
        costs_pct_of_final: Costs as percentage of final portfolio value
        avg_cost_per_trade: Average cost per trade
        avg_cost_bps: Average cost in basis points per trade
        turnover_annual: Annual portfolio turnover
    """
    total_costs: float = 0.0
    total_trades: int = 0
    total_traded_value: float = 0.0
    costs_per_year: float = 0.0
    costs_pct_of_final: float = 0.0
    avg_cost_per_trade: float = 0.0
    avg_cost_bps: float = 0.0
    turnover_annual: float = 0.0

    # Breakdown by component
    commission_total: float = 0.0
    spread_total: float = 0.0
    slippage_total: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_costs": self.total_costs,
            "total_trades": self.total_trades,
            "total_traded_value": self.total_traded_value,
            "costs_per_year": self.costs_per_year,
            "costs_pct_of_final": self.costs_pct_of_final,
            "avg_cost_per_trade": self.avg_cost_per_trade,
            "avg_cost_bps": self.avg_cost_bps,
            "turnover_annual": self.turnover_annual,
            "commission_total": self.commission_total,
            "spread_total": self.spread_total,
            "slippage_total": self.slippage_total,
        }


def calculate_trade_costs(
    trades: List["Trade"],
    cost_model: TransactionCostModel,
    equity_curve: pd.Series,
    final_value: float,
) -> TradingCostSummary:
    """
    Calculate comprehensive trading cost summary.

    Args:
        trades: List of executed trades
        cost_model: Cost model used
        equity_curve: Portfolio equity curve
        final_value: Final portfolio value

    Returns:
        TradingCostSummary with all metrics
    """
    if not trades:
        return TradingCostSummary()

    # Calculate costs for each trade
    total_costs = 0.0
    total_traded = 0.0
    commission_total = 0.0
    spread_total = 0.0
    slippage_total = 0.0

    for trade in trades:
        breakdown = cost_model.calculate(trade.value, trade.action)
        total_costs += breakdown.total
        total_traded += abs(trade.value)
        commission_total += breakdown.commission
        spread_total += breakdown.spread
        slippage_total += breakdown.slippage

    # Calculate time period
    if len(equity_curve) >= 2:
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    else:
        years = 1.0

    years = max(years, 0.01)  # Avoid division by zero

    # Average portfolio value
    avg_value = equity_curve.mean() if len(equity_curve) > 0 else final_value

    return TradingCostSummary(
        total_costs=total_costs,
        total_trades=len(trades),
        total_traded_value=total_traded,
        costs_per_year=total_costs / years,
        costs_pct_of_final=total_costs / final_value * 100 if final_value > 0 else 0.0,
        avg_cost_per_trade=total_costs / len(trades) if trades else 0.0,
        avg_cost_bps=(total_costs / total_traded * 10000) if total_traded > 0 else 0.0,
        turnover_annual=(total_traded / avg_value) / years if avg_value > 0 else 0.0,
        commission_total=commission_total,
        spread_total=spread_total,
        slippage_total=slippage_total,
    )


def _normalize_cost_overrides(raw: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Normalize cost config keys to TransactionCostModel constructor keys.

    Supported inputs:
    - commission_pct
    - commission_min
    - spread_bps (full spread) OR spread_pct (one-way friction)
    - slippage_bps OR slippage_pct
    """
    if not raw:
        return {}

    normalized: Dict[str, float] = {}
    if "commission_pct" in raw and raw["commission_pct"] is not None:
        normalized["commission_pct"] = float(raw["commission_pct"])
    if "commission_min" in raw and raw["commission_min"] is not None:
        normalized["commission_min"] = float(raw["commission_min"])

    if "spread_bps" in raw and raw["spread_bps"] is not None:
        normalized["spread_bps"] = float(raw["spread_bps"])
    elif "spread_pct" in raw and raw["spread_pct"] is not None:
        # spread_pct is interpreted as one-way friction.
        normalized["spread_bps"] = float(raw["spread_pct"]) * 10000.0 * 2.0

    if "slippage_bps" in raw and raw["slippage_bps"] is not None:
        normalized["slippage_bps"] = float(raw["slippage_bps"])
    elif "slippage_pct" in raw and raw["slippage_pct"] is not None:
        normalized["slippage_bps"] = float(raw["slippage_pct"]) * 10000.0

    return normalized


@dataclass
class CostModelResolver:
    """
    Resolve per-trade transaction cost models by ticker.

    Resolution order:
    1) explicit ticker override
    2) mapped asset-class override
    3) default model
    """

    default_model: TransactionCostModel
    ticker_models: Dict[str, TransactionCostModel] = field(default_factory=dict)
    asset_class_models: Dict[str, TransactionCostModel] = field(default_factory=dict)
    ticker_asset_class: Dict[str, str] = field(default_factory=dict)

    def for_ticker(self, ticker: str) -> TransactionCostModel:
        """Return cost model for ticker (case-insensitive)."""
        key = str(ticker).upper()
        if key in self.ticker_models:
            return self.ticker_models[key]

        asset_class = self.ticker_asset_class.get(key)
        if asset_class and asset_class in self.asset_class_models:
            return self.asset_class_models[asset_class]

        return self.default_model

    @classmethod
    def from_profile(
        cls,
        default_model: TransactionCostModel,
        profile: Optional[Dict[str, Any]] = None,
    ) -> "CostModelResolver":
        """
        Build resolver from a profile dictionary.

        Expected schema (all keys optional):
            {
              "default": {...},
              "asset_classes": {"equity_us": {...}},
              "ticker_asset_class": {"AAPL": "equity_us"},
              "tickers": {"AAPL": {...}}
            }
        """
        profile = profile or {}

        base_params = default_model.to_dict()
        base_params.pop("total_friction_bps", None)

        # Optional profile-level default overrides.
        profile_default = _normalize_cost_overrides(profile.get("default"))
        if profile_default:
            base_params.update(profile_default)

        resolved_default = TransactionCostModel(**base_params)

        # Build asset class models as overlays on top of default.
        asset_class_models: Dict[str, TransactionCostModel] = {}
        raw_asset_classes = profile.get("asset_classes", {}) or {}
        for class_name, class_cfg in raw_asset_classes.items():
            merged = dict(base_params)
            merged.update(_normalize_cost_overrides(class_cfg))
            asset_class_models[str(class_name)] = TransactionCostModel(**merged)

        # Map ticker -> asset class.
        ticker_asset_class = {
            str(ticker).upper(): str(class_name)
            for ticker, class_name in (profile.get("ticker_asset_class", {}) or {}).items()
        }

        # Build ticker models with class fallback.
        ticker_models: Dict[str, TransactionCostModel] = {}
        raw_ticker_models = profile.get("tickers", {}) or {}
        for ticker, ticker_cfg in raw_ticker_models.items():
            ticker_key = str(ticker).upper()
            merged = dict(base_params)

            class_name = ticker_asset_class.get(ticker_key)
            if class_name and class_name in asset_class_models:
                class_params = asset_class_models[class_name].to_dict()
                class_params.pop("total_friction_bps", None)
                merged.update(class_params)

            merged.update(_normalize_cost_overrides(ticker_cfg))
            ticker_models[ticker_key] = TransactionCostModel(**merged)

        return cls(
            default_model=resolved_default,
            ticker_models=ticker_models,
            asset_class_models=asset_class_models,
            ticker_asset_class=ticker_asset_class,
        )
