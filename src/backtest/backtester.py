"""
Backtester module - Core backtesting engine.

This module provides the main backtesting functionality:
- Portfolio simulation with realistic trading costs
- Trade execution with slippage and transaction costs
- Equity curve generation
"""

import warnings
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.strategy import Strategy, Allocation
from backtest.data import PriceData
from backtest.assets import get_yahoo_ticker
from backtest.metrics import MetricsCalculator, Metrics
from backtest.tax import GermanTaxModel, TaxResult
from backtest.tax.de_tax_model import (  # separate error classes
    InsufficientSharesError,
    NoLotsError,
)
from backtest.rebalance import generate_rebalance_dates
from backtest.validation import (
    validate_before_run,
    validate_backtest_result,
    validate_portfolio_state,
    ValidationResult,
)
from backtest.costs import CostModelResolver, TransactionCostModel
from backtest.risk.exposure_policy import ExposurePolicyEngine
from backtest.risk.overlay import RiskOverlayEngine

# Truth Test: set BT_TRUTH_TEST=1 to enable debug output
TRUTH_TEST = os.getenv("BT_TRUTH_TEST", "0") == "1"


@dataclass
class Trade:
    """
    Represents a single trade (buy or sell).

    Attributes:
        date: Trade execution date
        ticker: Asset symbol
        action: "BUY" or "SELL"
        shares: Number of shares traded
        price: Execution price per share (legacy alias for price_exec)
        value: Total trade value (legacy alias for value_ref)
        costs: Transaction costs (commission)
        slippage: Slippage costs
        price_ref: Reference price before slippage (e.g., Close/Adj Close)
        price_exec: Execution price after slippage (cash-effective)
        value_ref: Reference value = shares * price_ref
        value_exec: Execution value = shares * price_exec (cash-effective before costs)
        tax_paid_trade: Tax paid on this trade (SELL only, 0 for BUY)

    Cash Flow Semantics:
        For SELL: cash_inflow = value_exec - costs = shares * price_exec - costs
        For BUY:  cash_outflow = value_exec + costs = shares * price_exec + costs

    Note on price vs value:
        - price_ref/value_ref: The "market" price/value before execution friction
        - price_exec/value_exec: The actual execution price/value after slippage
        - Invariant: abs(value_exec - shares * price_exec) < 1e-6

    Legacy Compatibility:
        - 'price' returns price_exec (execution price)
        - 'value' returns value_ref (reference value, for backwards compatibility)
    """
    date: datetime
    ticker: str
    action: Literal["BUY", "SELL"]
    shares: float
    # Legacy fields (kept for compatibility, now computed from new fields)
    price: float  # = price_exec
    value: float  # = value_ref
    costs: float
    slippage: float
    # New semantic fields for cash flow clarity
    price_ref: float = 0.0  # Raw price before slippage
    price_exec: float = 0.0  # Execution price after slippage
    value_ref: float = 0.0  # shares * price_ref
    value_exec: float = 0.0  # shares * price_exec (cash-effective)
    tax_paid_trade: float = 0.0  # Tax paid on this trade (SELL only)

    def __post_init__(self):
        """Initialize computed fields if not set (for backwards compatibility)."""
        # If new fields not set, infer from legacy fields
        if self.price_exec == 0.0 and self.price != 0.0:
            self.price_exec = self.price
        if self.value_ref == 0.0 and self.value != 0.0:
            self.value_ref = self.value
        # Compute value_exec if not set
        if self.value_exec == 0.0 and self.shares != 0.0 and self.price_exec != 0.0:
            self.value_exec = self.shares * self.price_exec
        # Compute price_ref from value_ref if not set
        if self.price_ref == 0.0 and self.value_ref != 0.0 and self.shares != 0.0:
            self.price_ref = self.value_ref / self.shares

    @property
    def total_costs(self) -> float:
        """Total trading costs (transaction + slippage)."""
        return self.costs + self.slippage

    @property
    def cash_flow(self) -> float:
        """
        Net cash flow from this trade.

        Returns:
            Positive for SELL (cash in), negative for BUY (cash out).
        """
        if self.action == "SELL":
            return self.value_exec - self.costs
        else:  # BUY
            return -(self.value_exec + self.costs)

    @property
    def cash_flow_after_tax(self) -> float:
        """
        Net cash flow after tax (for SELL trades with tax).

        Returns:
            Cash flow minus tax_paid_trade.
        """
        return self.cash_flow - self.tax_paid_trade

    def validate_invariants(self) -> bool:
        """
        Validate trade invariants.

        Returns:
            True if all invariants hold, False otherwise.
        """
        # Invariant 1: value_exec should equal shares * price_exec
        expected_value_exec = self.shares * self.price_exec
        if abs(self.value_exec - expected_value_exec) > 1e-6:
            return False
        # Invariant 2: value_ref should equal shares * price_ref
        expected_value_ref = self.shares * self.price_ref
        if abs(self.value_ref - expected_value_ref) > 1e-6:
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"Trade({self.date.strftime('%Y-%m-%d')}, {self.ticker}, "
            f"{self.action}, {self.shares:.2f} @ {self.price_exec:.2f})"
        )


@dataclass
class DividendEvent:
    """
    Represents a dividend payment and optional reinvestment.

    Attributes:
        date: Ex-dividend date
        ticker: Asset symbol
        dividend_per_share: Dividend amount per share
        shares_held: Number of shares held at ex-dividend date
        gross_amount: Total dividend before tax (dividend_per_share * shares_held)
        tax_paid: Tax withheld on dividend (if applicable)
        net_amount: Dividend after tax
        shares_purchased: Shares bought with reinvestment (if DRIP enabled)
        reinvest_price: Price per share for reinvestment
    """
    date: datetime
    ticker: str
    dividend_per_share: float
    shares_held: float
    gross_amount: float
    tax_paid: float = 0.0
    net_amount: Optional[float] = None  # None = auto-calculate
    shares_purchased: float = 0.0
    reinvest_price: float = 0.0

    def __post_init__(self):
        """Calculate net amount if not explicitly set."""
        # Only auto-calculate if net_amount was not provided (None)
        # This allows explicit net_amount=0.0 for 100% withholding tax scenarios
        if self.net_amount is None:
            self.net_amount = self.gross_amount - self.tax_paid

    def __repr__(self) -> str:
        return (
            f"DividendEvent({self.date.strftime('%Y-%m-%d')}, {self.ticker}, "
            f"€{self.gross_amount:.2f}, reinvest={self.shares_purchased:.4f} shares)"
        )


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.

    Note:
        For new code, consider using `RunConfig` from `backtest.config.run_config`
        which provides a more structured approach with nested CostConfig and TaxConfig.
        Use `config_to_backtest_config()` to convert RunConfig to BacktestConfig.

    Attributes:
        initial_capital: Starting capital in portfolio currency
        currency: Portfolio currency (EUR or USD)
        costs_pct: Transaction costs as percentage of trade value (e.g., 0.001 = 0.1%)
        slippage_pct: Slippage as percentage of trade value (e.g., 0.0005 = 0.05%)
        spread_pct: Bid-ask spread percentage (v2 feature)
        cost_profile: Optional per-ticker/asset-class cost overrides
        execution_lag_days: Delay between signal day and execution day (0=same-day, 1=T+1)
        max_volume_participation: Optional max traded share as fraction of daily volume (0.1 = 10%)
        min_daily_dollar_volume: Optional liquidity floor; trades skipped below this daily notional
        liquidity_on_missing_volume: Behavior when volume is missing ("allow" or "skip")
        exposure_policy: Optional exposure controller (3x -> 1x/core/safe)
        risk_overlay: Optional risk-overlay configuration (max position, sector caps, turnover budget, drawdown brake)
        benchmark: Benchmark for comparison ("S&P 500" or "MSCI World")
        risk_free_rate: Annual risk-free rate for Sharpe calculation
        cash_rate: Annual cash interest rate applied to uninvested cash
        rebalance_frequency: Rebalancing frequency ("daily", "weekly", "monthly", "quarterly", "yearly")
        tax_enabled: Whether to apply German tax model
        tax_rate: Tax rate (26.375% = 25% Abgeltungssteuer + 5.5% Soli)
        tax_partial_exemption: Teilfreistellung for equity funds (30%)
        tax_exemption_amount: Freistellungsauftrag (1000 single, 2000 joint)
        metric_basis: How to value portfolio at window end:
            - "gross": Mark-to-market without tax consideration
            - "net_realized": Only taxes on realized gains (default)
            - "net_liquidation": Virtual liquidation with tax on unrealized gains
    """
    initial_capital: float = 10_000.0
    currency: str = "EUR"
    costs_pct: float = 0.001  # 0.1%
    slippage_pct: float = 0.0005  # 0.05%
    spread_pct: float = 0.0  # v2
    # Optional per-ticker/asset-class cost profile (JSON-compatible dict)
    cost_profile: Optional[Dict] = None
    # Optional execution realism settings (Sprint 2)
    execution_lag_days: int = 0
    max_volume_participation: Optional[float] = None
    min_daily_dollar_volume: float = 0.0
    liquidity_on_missing_volume: Literal["allow", "skip"] = "allow"
    exposure_policy: Optional[Dict] = None
    risk_overlay: Optional[Dict] = None
    benchmark: Optional[str] = "S&P 500"
    risk_free_rate: float = 0.02  # 2%
    cash_rate: float = 0.0  # Annual interest on cash
    rebalance_frequency: Optional[Literal["daily", "weekly", "monthly", "quarterly", "yearly"]] = None
    # German Tax Model (P2)
    tax_enabled: bool = True  # Default: enabled
    tax_rate: float = 0.26375  # 25% + 5.5% Soli
    tax_partial_exemption: float = 0.30  # 30% Teilfreistellung
    tax_exemption_amount: float = 1000.0  # Freistellungsauftrag (single)
    # Metric basis for valuation
    metric_basis: Literal["gross", "net_realized", "net_liquidation"] = "net_liquidation"
    # Universe look-ahead protection
    allow_universe_lookahead: bool = False  # Default: block non-PIT universes for historical backtests
    # Validation settings
    validate: bool = True  # Enable pre/post validation checks
    strict_validation: bool = False  # Raise errors on validation failures
    # Liquidation at end (P3)
    actual_liquidation_at_end: bool = False  # Actually sell all positions at end (creates SELL trades)
    # Dividend Reinvestment Plan (DRIP)
    drip_enabled: bool = False  # Automatically reinvest dividends into the paying stock
    # External features provider (Phase A plumbing).
    # When set, Backtester.run() installs it on the strategy as
    # _external_features_provider for the duration of the run, then restores.
    # Default None keeps existing behaviour fully backward compatible.
    external_features_loader: Optional[Any] = None
    # Per-ticker tax classification (research: single stocks vs. equity funds).
    # Maps ticker -> is_equity_fund. Equity funds (ETFs/Fonds) get the 30%
    # Teilfreistellung; single stocks do NOT (full 26.375% on the whole gain)
    # and route losses to the Aktienverlusttopf. When None (default), ALL
    # instruments keep the legacy treatment (is_equity_fund=True,
    # instrument_class="general") -> bit-identical behaviour for existing
    # strategies/tests. Tickers absent from a provided map also keep the
    # legacy ETF treatment via _instrument_tax_flags().
    # TaxConfig introduced cost_basis_method, but BacktestConfig didn't know about it ->
    # the field had no effect. Now passed through into the tax model.
    # The daily net_liquidation curve used to carry the liquidation tax ONLY at the
    # very last point -- every other day was valued net of REALIZED tax only. MaxDD and
    # underwater duration were therefore measured on a curve that ignored the latent tax
    # liability. Default since 2026-07-19: dCAGR is 0.00 everywhere (the final value is
    # identical, only the path differs), this calculation is the more correct one, and
    # it means all project figures move exactly ONCE instead of twice.
    daily_liquidation_tax: bool = True
    cost_basis_method: str = "FIFO"
    equity_fund_map: Optional[Dict[str, bool]] = None

    # DECOUPLED tax axes. `equity_fund_map` is a bool and thereby forces the coupling
    # 'no Teilfreistellung => equity loss pot'. For leveraged debt ETPs (leveraged ETPs
    # are debt instruments) this is wrong: they have 0% Teilfreistellung, but their
    # losses belong in the GENERAL pot -- section 20 EStG restricts the special
    # offsetting to STOCK sales, section 20 InvStG assigns the 30% Teilfreistellung to
    # equity FUNDS. This map takes precedence and names both axes explicitly:
    # ticker -> (instrument_class, is_equity_fund).
    tax_treatment_map: Optional[Dict[str, Tuple[str, bool]]] = None


@dataclass
class TaxSummary:
    """
    Summary of tax impact on backtest results.

    Attributes:
        tax_enabled: Whether tax model was active
        tax_accounting_mode: "cash_effective" or "shadow"
        metric_basis: How portfolio was valued at end ("gross", "net_realized", "net_liquidation")
        total_tax_paid: Total tax paid over the backtest period (realized only)
        total_realized_gains: Total realized gains before tax
        total_realized_losses: Total realized losses
        net_realized_gain: Net gain after losses
        exemption_used: Total Freistellungsauftrag used
        effective_tax_rate: Effective tax rate on net gains
        tax_drag_cagr_pp: Tax drag as CAGR difference in percentage points
        tax_drag_final_pct: Tax drag as percentage of gross final value
        loss_pot_used: Total amount offset from loss pots (Verlustvortrag)
        loss_pot_added: Total amount added to loss pots
        loss_pot_equity_final: Final equity loss pot balance (Aktienverlusttopf)
        loss_pot_general_final: Final general loss pot balance (allg. Verlusttopf)
        tax_rate: Applied tax rate
        partial_exemption: Applied Teilfreistellung
        exemption_amount: Applied Freistellungsauftrag
        tax_paid_liquidation: Additional tax from virtual end-of-window liquidation
        unrealized_gain_at_end: Unrealized gain at window end (before liquidation tax)
        final_value_gross: Final value without any tax consideration
        final_value_net_realized: Final value with realized taxes only
        final_value_net_liquidation: Final value after virtual liquidation tax
    """
    tax_enabled: bool = True
    tax_accounting_mode: Literal["cash_effective", "shadow"] = "cash_effective"
    metric_basis: Literal["gross", "net_realized", "net_liquidation"] = "net_liquidation"
    total_tax_paid: float = 0.0  # Realized taxes only
    total_realized_gains: float = 0.0
    total_realized_losses: float = 0.0
    net_realized_gain: float = 0.0
    exemption_used: float = 0.0
    effective_tax_rate: float = 0.0
    tax_drag_cagr_pp: float = 0.0  # CAGR_gross - CAGR_net in percentage points
    tax_drag_final_pct: float = 0.0  # 1 - final_net/final_gross
    # Loss pot (Verlustvortrag) tracking
    loss_pot_used: float = 0.0  # Total offset from loss pots
    loss_pot_added: float = 0.0  # Total added to loss pots
    loss_pot_equity_final: float = 0.0  # Final Aktienverlusttopf balance
    loss_pot_general_final: float = 0.0  # Final allg. Verlusttopf balance
    # Tax parameters used
    tax_rate: float = 0.26375
    partial_exemption: float = 0.30
    exemption_amount: float = 1000.0
    # Virtual liquidation fields (only populated when metric_basis == "net_liquidation")
    tax_paid_liquidation: float = 0.0  # Additional tax from virtual liquidation
    unrealized_gain_at_end: float = 0.0  # Unrealized gain at window end
    final_value_gross: float = 0.0  # Mark-to-market value
    final_value_net_realized: float = 0.0  # After realized taxes only
    final_value_net_liquidation: float = 0.0  # After virtual liquidation tax
    # CAGR variants for correct reporting
    initial_capital: float = 0.0
    years: float = 0.0
    cagr_gross: float = 0.0
    cagr_net_realized: float = 0.0
    cagr_net_liquidation: float = 0.0

    # Legacy alias for backwards compatibility
    @property
    def tax_drag_pct(self) -> float:
        """Legacy: returns tax_drag_final_pct for backwards compatibility."""
        return self.tax_drag_final_pct

    @property
    def total_tax_including_liquidation(self) -> float:
        """Total tax including virtual liquidation."""
        return self.total_tax_paid + self.tax_paid_liquidation

    @property
    def cagr_for_metric_basis(self) -> float:
        """Return CAGR based on current metric_basis mode."""
        if self.metric_basis == "gross":
            return self.cagr_gross
        elif self.metric_basis == "net_liquidation":
            return self.cagr_net_liquidation
        else:
            return self.cagr_net_realized

    @property
    def final_value_for_metric_basis(self) -> float:
        """Return final value based on current metric_basis mode."""
        if self.metric_basis == "gross":
            return self.final_value_gross
        elif self.metric_basis == "net_liquidation":
            return self.final_value_net_liquidation
        else:
            return self.final_value_net_realized

    @property
    def tax_drag_for_metric_basis_pp(self) -> float:
        """Return tax drag in percentage points based on metric_basis mode."""
        if self.metric_basis == "gross":
            return 0.0
        elif self.metric_basis == "net_liquidation":
            return (self.cagr_gross - self.cagr_net_liquidation) * 100
        else:
            return (self.cagr_gross - self.cagr_net_realized) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "tax_enabled": self.tax_enabled,
            "tax_accounting_mode": self.tax_accounting_mode,
            "metric_basis": self.metric_basis,
            "total_tax_paid": self.total_tax_paid,
            "total_realized_gains": self.total_realized_gains,
            "total_realized_losses": self.total_realized_losses,
            "net_realized_gain": self.net_realized_gain,
            "exemption_used": self.exemption_used,
            "effective_tax_rate": self.effective_tax_rate,
            "tax_drag_cagr_pp": self.tax_drag_cagr_pp,
            "tax_drag_final_pct": self.tax_drag_final_pct,
            "loss_pot_used": self.loss_pot_used,
            "loss_pot_added": self.loss_pot_added,
            "loss_pot_equity_final": self.loss_pot_equity_final,
            "loss_pot_general_final": self.loss_pot_general_final,
            "tax_rate": self.tax_rate,
            "partial_exemption": self.partial_exemption,
            "exemption_amount": self.exemption_amount,
            "tax_paid_liquidation": self.tax_paid_liquidation,
            "unrealized_gain_at_end": self.unrealized_gain_at_end,
            "final_value_gross": self.final_value_gross,
            "final_value_net_realized": self.final_value_net_realized,
            "final_value_net_liquidation": self.final_value_net_liquidation,
            "initial_capital": self.initial_capital,
            "years": self.years,
            "cagr_gross": self.cagr_gross,
            "cagr_net_realized": self.cagr_net_realized,
            "cagr_net_liquidation": self.cagr_net_liquidation,
        }


@dataclass
class Portfolio:
    """
    Tracks portfolio state during backtesting.

    Attributes:
        cash: Current cash balance
        positions: Dictionary of ticker -> number of shares
    """
    cash: float = 0.0
    positions: Dict[str, float] = field(default_factory=dict)

    def total_value(self, prices: pd.Series) -> float:
        """
        Calculate total portfolio value at given prices.

        Args:
            prices: Series with ticker -> price mapping

        Returns:
            Total portfolio value (cash + positions)
        """
        positions_value = sum(
            shares * prices.get(ticker, 0.0)
            for ticker, shares in self.positions.items()
        )
        return self.cash + positions_value

    def position_values(self, prices: pd.Series) -> Dict[str, float]:
        """
        Calculate value of each position.

        Args:
            prices: Series with ticker -> price mapping

        Returns:
            Dictionary of ticker -> position value
        """
        return {
            ticker: shares * prices.get(ticker, 0.0)
            for ticker, shares in self.positions.items()
        }

    def weights(self, prices: pd.Series) -> Dict[str, float]:
        """
        Calculate current portfolio weights.

        Args:
            prices: Series with ticker -> price mapping

        Returns:
            Dictionary of ticker -> weight
        """
        total = self.total_value(prices)
        if total == 0:
            return {}
        return {
            ticker: value / total
            for ticker, value in self.position_values(prices).items()
        }

    @staticmethod
    def _default_cost_model(
        costs_pct: float,
        slippage_pct: float,
        spread_pct: float,
    ) -> TransactionCostModel:
        """
        Convert legacy scalar cost settings to TransactionCostModel.

        Backward-compatibility note:
        - legacy spread_pct was used as one-way friction in execution;
          TransactionCostModel expects full spread in bps and applies half per side.
        """
        return TransactionCostModel(
            commission_pct=max(0.0, float(costs_pct)),
            commission_min=0.0,
            spread_bps=max(0.0, float(spread_pct)) * 10000.0 * 2.0,
            slippage_bps=max(0.0, float(slippage_pct)) * 10000.0,
        )

    @classmethod
    def _cost_model_for_ticker(
        cls,
        ticker: str,
        costs_pct: float,
        slippage_pct: float,
        spread_pct: float,
        cost_resolver: Optional[CostModelResolver] = None,
    ) -> TransactionCostModel:
        if cost_resolver is not None:
            return cost_resolver.for_ticker(ticker)
        return cls._default_cost_model(costs_pct, slippage_pct, spread_pct)

    @staticmethod
    def _build_trade(
        current_date: datetime,
        ticker: str,
        action: Literal["BUY", "SELL"],
        shares: float,
        price_ref: float,
        cost_model: TransactionCostModel,
    ) -> Trade:
        """Create Trade with consistent cost/slippage semantics."""
        price_exec = cost_model.effective_price(price_ref, action)
        value_ref = shares * price_ref
        value_exec = shares * price_exec
        breakdown = cost_model.calculate(value_ref, action)
        # Keep legacy fields compatible:
        # - costs: explicit commission
        # - slippage: implicit price impact (spread + slippage)
        price_impact = breakdown.spread + breakdown.slippage

        return Trade(
            date=current_date,
            ticker=ticker,
            action=action,
            shares=shares,
            price=price_exec,
            value=value_ref,
            costs=breakdown.commission,
            slippage=price_impact,
            price_ref=price_ref,
            price_exec=price_exec,
            value_ref=value_ref,
            value_exec=value_exec,
            tax_paid_trade=0.0,
        )

    @staticmethod
    def _cap_shares_for_liquidity_with_meta(
        requested_shares: float,
        ticker: str,
        price_ref: float,
        volumes: Optional[pd.Series],
        max_volume_participation: Optional[float],
        min_daily_dollar_volume: float,
        liquidity_on_missing_volume: Literal["allow", "skip"],
    ) -> tuple[float, Dict[str, Any]]:
        """
        Apply simplified liquidity guards to requested shares.

        Guards:
        - Max participation of reported daily volume.
        - Minimum daily dollar-volume floor.
        - Optional skip when volume data is missing.
        """
        shares = max(0.0, float(requested_shares))
        meta: Dict[str, Any] = {
            "requested_shares": shares,
            "final_shares": shares,
            "used_volume_data": False,
            "missing_volume": False,
            "skipped_missing_volume": False,
            "skipped_min_notional": False,
            "clipped_by_max_participation": False,
            "max_participation_cap_shares": None,
        }
        if shares <= 0.0:
            return 0.0, meta

        participation = float(max_volume_participation) if max_volume_participation is not None else None
        needs_volume = (
            (participation is not None and participation > 0.0)
            or float(min_daily_dollar_volume) > 0.0
        )
        if not needs_volume:
            meta["final_shares"] = shares
            return shares, meta

        daily_volume = None
        if volumes is not None:
            raw_volume = volumes.get(ticker, np.nan)
            if pd.notna(raw_volume):
                vol = float(raw_volume)
                if vol > 0.0:
                    daily_volume = vol

        if daily_volume is None:
            meta["missing_volume"] = True
            if liquidity_on_missing_volume == "skip":
                meta["skipped_missing_volume"] = True
                meta["final_shares"] = 0.0
                return 0.0, meta
            meta["final_shares"] = shares
            return shares, meta

        meta["used_volume_data"] = True
        if float(min_daily_dollar_volume) > 0.0:
            if daily_volume * max(float(price_ref), 0.0) < float(min_daily_dollar_volume):
                meta["skipped_min_notional"] = True
                meta["final_shares"] = 0.0
                return 0.0, meta

        if participation is not None and participation > 0.0:
            max_shares = daily_volume * participation
            meta["max_participation_cap_shares"] = max_shares
            if shares > max_shares + 1e-12:
                meta["clipped_by_max_participation"] = True
            shares = min(shares, max_shares)

        shares = max(0.0, shares)
        meta["final_shares"] = shares
        return shares, meta

    @staticmethod
    def _cap_shares_for_liquidity(
        requested_shares: float,
        ticker: str,
        price_ref: float,
        volumes: Optional[pd.Series],
        max_volume_participation: Optional[float],
        min_daily_dollar_volume: float,
        liquidity_on_missing_volume: Literal["allow", "skip"],
    ) -> float:
        """Compatibility wrapper that returns only capped shares."""
        capped, _ = Portfolio._cap_shares_for_liquidity_with_meta(
            requested_shares=requested_shares,
            ticker=ticker,
            price_ref=price_ref,
            volumes=volumes,
            max_volume_participation=max_volume_participation,
            min_daily_dollar_volume=min_daily_dollar_volume,
            liquidity_on_missing_volume=liquidity_on_missing_volume,
        )
        return capped

    def calculate_rebalance_trades(
        self,
        target: Allocation,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        costs_pct: float = 0.0,
        slippage_pct: float = 0.0,
        spread_pct: float = 0.0,
        cost_resolver: Optional[CostModelResolver] = None,
        max_volume_participation: Optional[float] = None,
        min_daily_dollar_volume: float = 0.0,
        liquidity_on_missing_volume: Literal["allow", "skip"] = "allow",
        liquidity_stats: Optional[Dict[str, float]] = None,
        plan_buys: bool = True,
    ) -> tuple[List[Trade], List[Trade]]:
        """
        Calculate trades needed to rebalance, separated into sells and buys.

        This is a planning function that does NOT execute trades.
        Returns (sell_trades, buy_trades) so caller can apply tax between phases.

        Args:
            target: Target allocation
            prices: Current prices
            volumes: Optional daily traded volume per ticker
            costs_pct: Transaction cost percentage
            slippage_pct: Slippage percentage

        Returns:
            Tuple of (sell_trades, buy_trades) to be executed
        """
        current_date = prices.name if hasattr(prices, "name") else datetime.now()
        total_value = self.total_value(prices)

        # Calculate target values
        target_values = {
            ticker: weight * total_value
            for ticker, weight in target.items()
        }

        # Calculate current values
        current_values = self.position_values(prices)

        # Get all tickers involved -- sorted DETERMINISTICALLY.
        # This set drives the SELL order below, and thereby two sequential states:
        # (1) the loss-offsetting pot in the tax model -- whoever sells first gets
        # offset first -> a different order means a different tax outcome; (2) the cash
        # that funds the subsequent buys. Via a `set` both were hash-dependent.
        all_tickers = sorted(set(target_values.keys()) | set(current_values.keys()))

        # Identify sells (positions that need to be reduced)
        sell_trades = []
        for ticker in all_tickers:
            current_val = current_values.get(ticker, 0.0)
            target_val = target_values.get(ticker, 0.0)
            diff = target_val - current_val

            if diff < -0.01:  # Need to sell (small threshold to avoid tiny trades)
                price_ref = prices.get(ticker, 0.0)
                if price_ref <= 0:
                    continue

                requested_shares = min(abs(diff) / price_ref, self.positions.get(ticker, 0.0))
                if liquidity_stats is not None:
                    liquidity_stats["requested_sell_trades"] = liquidity_stats.get("requested_sell_trades", 0.0) + 1.0
                    liquidity_stats["requested_shares_total"] = liquidity_stats.get("requested_shares_total", 0.0) + float(
                        requested_shares
                    )

                shares_to_sell, cap_meta = self._cap_shares_for_liquidity_with_meta(
                    requested_shares=requested_shares,
                    ticker=ticker,
                    price_ref=price_ref,
                    volumes=volumes,
                    max_volume_participation=max_volume_participation,
                    min_daily_dollar_volume=min_daily_dollar_volume,
                    liquidity_on_missing_volume=liquidity_on_missing_volume,
                )
                if liquidity_stats is not None:
                    if cap_meta["skipped_missing_volume"]:
                        liquidity_stats["skipped_missing_volume"] = liquidity_stats.get("skipped_missing_volume", 0.0) + 1.0
                    if cap_meta["skipped_min_notional"]:
                        liquidity_stats["skipped_min_daily_dollar_volume"] = liquidity_stats.get(
                            "skipped_min_daily_dollar_volume", 0.0
                        ) + 1.0
                    if cap_meta["clipped_by_max_participation"]:
                        liquidity_stats["clipped_by_max_participation"] = liquidity_stats.get(
                            "clipped_by_max_participation", 0.0
                        ) + 1.0
                        liquidity_stats["clipped_shares_total"] = liquidity_stats.get("clipped_shares_total", 0.0) + max(
                            0.0, float(cap_meta["requested_shares"]) - float(cap_meta["final_shares"])
                        )
                if shares_to_sell <= 0.0001:
                    continue
                model = self._cost_model_for_ticker(
                    ticker=ticker,
                    costs_pct=costs_pct,
                    slippage_pct=slippage_pct,
                    spread_pct=spread_pct,
                    cost_resolver=cost_resolver,
                )
                sell_trades.append(
                    self._build_trade(
                        current_date=current_date,
                        ticker=ticker,
                        action="SELL",
                        shares=shares_to_sell,
                        price_ref=price_ref,
                        cost_model=model,
                    )
                )
                if liquidity_stats is not None:
                    liquidity_stats["planned_sell_trades"] = liquidity_stats.get("planned_sell_trades", 0.0) + 1.0
                    liquidity_stats["planned_shares_total"] = liquidity_stats.get("planned_shares_total", 0.0) + float(
                        shares_to_sell
                    )

        # Identify buys (positions that need to be increased)
        # Note: These are calculated based on CURRENT state, will be recalculated
        # after sells are executed in execute_sells()
        buy_trades = []
        if not plan_buys:
            return sell_trades, buy_trades

        for ticker in all_tickers:
            current_val = current_values.get(ticker, 0.0)
            target_val = target_values.get(ticker, 0.0)
            diff = target_val - current_val

            if diff > 0.01:  # Need to buy (small threshold to avoid tiny trades)
                price_ref = prices.get(ticker, 0.0)
                if price_ref <= 0:
                    continue

                requested_shares = diff / price_ref
                if liquidity_stats is not None:
                    liquidity_stats["requested_buy_trades"] = liquidity_stats.get("requested_buy_trades", 0.0) + 1.0
                    liquidity_stats["requested_shares_total"] = liquidity_stats.get("requested_shares_total", 0.0) + float(
                        requested_shares
                    )

                shares_to_buy, cap_meta = self._cap_shares_for_liquidity_with_meta(
                    requested_shares=requested_shares,
                    ticker=ticker,
                    price_ref=price_ref,
                    volumes=volumes,
                    max_volume_participation=max_volume_participation,
                    min_daily_dollar_volume=min_daily_dollar_volume,
                    liquidity_on_missing_volume=liquidity_on_missing_volume,
                )
                if liquidity_stats is not None:
                    if cap_meta["skipped_missing_volume"]:
                        liquidity_stats["skipped_missing_volume"] = liquidity_stats.get("skipped_missing_volume", 0.0) + 1.0
                    if cap_meta["skipped_min_notional"]:
                        liquidity_stats["skipped_min_daily_dollar_volume"] = liquidity_stats.get(
                            "skipped_min_daily_dollar_volume", 0.0
                        ) + 1.0
                    if cap_meta["clipped_by_max_participation"]:
                        liquidity_stats["clipped_by_max_participation"] = liquidity_stats.get(
                            "clipped_by_max_participation", 0.0
                        ) + 1.0
                        liquidity_stats["clipped_shares_total"] = liquidity_stats.get("clipped_shares_total", 0.0) + max(
                            0.0, float(cap_meta["requested_shares"]) - float(cap_meta["final_shares"])
                        )
                if shares_to_buy <= 0.0001:
                    continue
                model = self._cost_model_for_ticker(
                    ticker=ticker,
                    costs_pct=costs_pct,
                    slippage_pct=slippage_pct,
                    spread_pct=spread_pct,
                    cost_resolver=cost_resolver,
                )
                buy_trades.append(
                    self._build_trade(
                        current_date=current_date,
                        ticker=ticker,
                        action="BUY",
                        shares=shares_to_buy,
                        price_ref=price_ref,
                        cost_model=model,
                    )
                )
                if liquidity_stats is not None:
                    liquidity_stats["planned_buy_trades"] = liquidity_stats.get("planned_buy_trades", 0.0) + 1.0
                    liquidity_stats["planned_shares_total"] = liquidity_stats.get("planned_shares_total", 0.0) + float(
                        shares_to_buy
                    )

        return sell_trades, buy_trades

    def execute_sells(self, sell_trades: List[Trade]) -> None:
        """
        Execute sell trades and update portfolio cash.

        Cash flow calculation uses value_exec (execution value) minus costs.
        This is the cash-effective amount received from the sale.

        Args:
            sell_trades: List of SELL trades to execute
        """
        for trade in sell_trades:
            if trade.action != "SELL":
                raise ValueError(f"Expected SELL trade, got {trade.action}")

            # Cash inflow = value_exec - costs (execution value minus commission)
            # Note: slippage is already reflected in value_exec vs value_ref
            actual_value = trade.value_exec - trade.costs
            self.positions[trade.ticker] = self.positions.get(trade.ticker, 0) - trade.shares
            if abs(self.positions[trade.ticker]) < 0.0001:
                del self.positions[trade.ticker]
            self.cash += actual_value

    def execute_buys(
        self,
        target: Allocation,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        costs_pct: float = 0.0,
        slippage_pct: float = 0.0,
        spread_pct: float = 0.0,
        cost_resolver: Optional[CostModelResolver] = None,
        max_volume_participation: Optional[float] = None,
        min_daily_dollar_volume: float = 0.0,
        liquidity_on_missing_volume: Literal["allow", "skip"] = "allow",
        liquidity_stats: Optional[Dict[str, float]] = None,
    ) -> List[Trade]:
        """
        Execute buy trades based on current cash (after sells and tax).

        Recalculates buy amounts based on available cash.

        Args:
            target: Target allocation
            prices: Current prices
            volumes: Optional daily traded volume per ticker
            costs_pct: Transaction cost percentage
            slippage_pct: Slippage percentage

        Returns:
            List of BUY trades executed
        """
        current_date = prices.name if hasattr(prices, "name") else datetime.now()

        # Recalculate what we need to buy based on current state
        current_values = self.position_values(prices)
        total_value = self.total_value(prices)

        # Calculate target values with updated total
        target_values = {
            ticker: weight * total_value
            for ticker, weight in target.items()
        }

        buy_trades = []
        cash_epsilon = 1e-6
        # DETERMINISTIC order: each buy sequentially consumes cash
        # (`min(diff, self.cash * 0.999)`), so the iteration order decides who gets
        # filled first when cash is tight. Via a `set` this was hash-dependent -> the
        # same configuration produced a CAGR that varied by 0.2-0.3pp depending on
        # PYTHONHASHSEED. Sorted, the result is reproducible.
        # PROPORTIONAL PRIORITY: if cash isn't enough for all buys, they used to be
        # served ALPHABETICALLY -- the first ticker got everything, the last got the
        # remainder. That's a systematic bias favoring early ticker names and has
        # nothing to do with economics. Now the shortfall is distributed proportionally
        # across ALL buys, from the same cash snapshot. If cash is sufficient, the
        # factor is 1.0 and behavior is bit-identical to before -- the change only
        # kicks in during a shortfall.
        _verfuegbar = self.cash * 0.999
        _bedarf = 0.0
        for _t in sorted(set(target_values.keys()) | set(current_values.keys())):
            _d = target_values.get(_t, 0.0) - current_values.get(_t, 0.0)
            if _d > 0.01 and prices.get(_t, 0.0) > 0:
                _bedarf += _d
        kauf_skalierung = 1.0
        if _bedarf > _verfuegbar > 0:
            kauf_skalierung = _verfuegbar / _bedarf

        for ticker in sorted(set(target_values.keys()) | set(current_values.keys())):
            current_val = current_values.get(ticker, 0.0)
            target_val = target_values.get(ticker, 0.0)
            diff = target_val - current_val

            if diff > 0.01:  # Need to buy
                price_ref = prices.get(ticker, 0.0)
                if price_ref <= 0:
                    continue

                model = self._cost_model_for_ticker(
                    ticker=ticker,
                    costs_pct=costs_pct,
                    slippage_pct=slippage_pct,
                    spread_pct=spread_pct,
                    cost_resolver=cost_resolver,
                )

                # Calculate how much we can actually buy with available cash.
                # kauf_skalierung distributes a cash shortfall proportionally instead of
                # alphabetically; with sufficient cash it is 1.0.
                trade_value = min(diff * kauf_skalierung, self.cash * 0.999)  # Leave small buffer
                requested_shares = trade_value / price_ref
                if liquidity_stats is not None:
                    liquidity_stats["requested_buy_trades"] = liquidity_stats.get("requested_buy_trades", 0.0) + 1.0
                    liquidity_stats["requested_shares_total"] = liquidity_stats.get("requested_shares_total", 0.0) + float(
                        requested_shares
                    )
                shares_to_buy, cap_meta = self._cap_shares_for_liquidity_with_meta(
                    requested_shares=requested_shares,
                    ticker=ticker,
                    price_ref=price_ref,
                    volumes=volumes,
                    max_volume_participation=max_volume_participation,
                    min_daily_dollar_volume=min_daily_dollar_volume,
                    liquidity_on_missing_volume=liquidity_on_missing_volume,
                )
                if liquidity_stats is not None:
                    if cap_meta["skipped_missing_volume"]:
                        liquidity_stats["skipped_missing_volume"] = liquidity_stats.get("skipped_missing_volume", 0.0) + 1.0
                    if cap_meta["skipped_min_notional"]:
                        liquidity_stats["skipped_min_daily_dollar_volume"] = liquidity_stats.get(
                            "skipped_min_daily_dollar_volume", 0.0
                        ) + 1.0
                    if cap_meta["clipped_by_max_participation"]:
                        liquidity_stats["clipped_by_max_participation"] = liquidity_stats.get(
                            "clipped_by_max_participation", 0.0
                        ) + 1.0
                        liquidity_stats["clipped_shares_total"] = liquidity_stats.get("clipped_shares_total", 0.0) + max(
                            0.0, float(cap_meta["requested_shares"]) - float(cap_meta["final_shares"])
                        )
                if shares_to_buy <= 0.0001:
                    continue

                trade = self._build_trade(
                    current_date=current_date,
                    ticker=ticker,
                    action="BUY",
                    shares=shares_to_buy,
                    price_ref=price_ref,
                    cost_model=model,
                )
                total_cost = trade.value_exec + trade.costs

                # Scale down if cash is insufficient (handles commission_min too).
                if total_cost > self.cash and total_cost > 0:
                    if liquidity_stats is not None:
                        liquidity_stats["cash_scaled_buy_trades"] = liquidity_stats.get("cash_scaled_buy_trades", 0.0) + 1.0
                    shares_to_buy = shares_to_buy * (self.cash / total_cost)
                    for _ in range(3):
                        if shares_to_buy <= 0:
                            break
                        trade = self._build_trade(
                            current_date=current_date,
                            ticker=ticker,
                            action="BUY",
                            shares=shares_to_buy,
                            price_ref=price_ref,
                            cost_model=model,
                        )
                        total_cost = trade.value_exec + trade.costs
                        if total_cost <= self.cash + cash_epsilon:
                            break
                        shares_to_buy = shares_to_buy * (self.cash / total_cost)

                if shares_to_buy > 0.0001:
                    trade = self._build_trade(
                        current_date=current_date,
                        ticker=ticker,
                        action="BUY",
                        shares=shares_to_buy,
                        price_ref=price_ref,
                        cost_model=model,
                    )
                    total_cost = trade.value_exec + trade.costs
                    if total_cost > self.cash + cash_epsilon:
                        continue

                    self.positions[ticker] = self.positions.get(ticker, 0.0) + shares_to_buy
                    self.cash -= total_cost
                    if self.cash < 0 and abs(self.cash) <= cash_epsilon:
                        self.cash = 0.0
                    buy_trades.append(trade)
                    if liquidity_stats is not None:
                        liquidity_stats["planned_buy_trades"] = liquidity_stats.get("planned_buy_trades", 0.0) + 1.0
                        liquidity_stats["planned_shares_total"] = liquidity_stats.get("planned_shares_total", 0.0) + float(
                            shares_to_buy
                        )

        return buy_trades

    def rebalance_to(
        self,
        target: Allocation,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None,
        costs_pct: float = 0.0,
        slippage_pct: float = 0.0,
        spread_pct: float = 0.0,
        cost_resolver: Optional[CostModelResolver] = None,
        max_volume_participation: Optional[float] = None,
        min_daily_dollar_volume: float = 0.0,
        liquidity_on_missing_volume: Literal["allow", "skip"] = "allow",
    ) -> List[Trade]:
        """
        Rebalance portfolio to target allocation.

        For legacy compatibility. Use calculate_rebalance_trades(), execute_sells(),
        and execute_buys() for phased execution with tax between sells and buys.

        Args:
            target: Target allocation
            prices: Current prices
            volumes: Optional daily traded volume per ticker
            costs_pct: Transaction cost percentage
            slippage_pct: Slippage percentage

        Returns:
            List of trades executed
        """
        sell_trades, _ = self.calculate_rebalance_trades(
            target,
            prices,
            volumes,
            costs_pct,
            slippage_pct,
            spread_pct,
            cost_resolver,
            max_volume_participation,
            min_daily_dollar_volume,
            liquidity_on_missing_volume,
        )
        self.execute_sells(sell_trades)
        buy_trades = self.execute_buys(
            target,
            prices,
            volumes,
            costs_pct,
            slippage_pct,
            spread_pct,
            cost_resolver,
            max_volume_participation,
            min_daily_dollar_volume,
            liquidity_on_missing_volume,
        )
        return sell_trades + buy_trades


@dataclass
class BacktestResult:
    """
    Results of a backtest run.

    Attributes:
        strategy: The strategy that was tested
        config: Backtest configuration
        equity_curve: Portfolio values (NET if tax enabled, else GROSS)
        equity_curve_gross: Portfolio values without tax deductions
        equity_curve_net: Portfolio values with tax deductions (None if tax disabled)
        equity_curve_daily_gross: Daily mark-to-market gross values (additive diagnostic)
        equity_curve_daily_net: Daily mark-to-market net values (None if tax disabled)
        allocations: Historical allocations at each rebalancing point
        raw_allocations: Strategy allocations before optional exposure policy / risk overlay
        trades: List of all trades executed
        metrics: Performance metrics (NET if tax enabled, else GROSS)
        metrics_gross: Performance metrics without tax impact
        metrics_net: Performance metrics with tax impact (None if tax disabled)
        metrics_daily_gross: Daily mark-to-market metrics without tax impact
        metrics_daily_net: Daily mark-to-market metrics with tax impact
        benchmark_curve: Benchmark equity curve (if configured)
        tax_summary: Tax impact summary (if tax_enabled)
        constraint_impact: Execution/risk-constraint diagnostics (liquidity + overlays)
    """
    strategy: Strategy
    config: BacktestConfig
    equity_curve: pd.Series  # Primary curve: NET if tax on, else GROSS
    allocations: pd.DataFrame
    trades: List[Trade]
    metrics: Metrics  # Primary metrics: NET if tax on, else GROSS
    raw_allocations: Optional[pd.DataFrame] = None
    # Gross/Net separation
    equity_curve_gross: Optional[pd.Series] = None
    equity_curve_net: Optional[pd.Series] = None
    equity_curve_daily_gross: Optional[pd.Series] = None
    equity_curve_daily_net: Optional[pd.Series] = None
    equity_curve_daily_net_liquidation: Optional[pd.Series] = None
    metrics_gross: Optional[Metrics] = None
    metrics_net: Optional[Metrics] = None
    metrics_daily_gross: Optional[Metrics] = None
    metrics_daily_net: Optional[Metrics] = None
    metrics_daily_net_liquidation: Optional[Metrics] = None
    # Benchmark and tax
    benchmark_curve: Optional[pd.Series] = None
    tax_summary: Optional[TaxSummary] = None
    # Dividend events
    dividend_events: Optional[List[DividendEvent]] = None
    # Execution/risk diagnostics
    constraint_impact: Optional[Dict[str, Any]] = None
    exposure_policy_decisions: Optional[List[Dict[str, Any]]] = None

    @property
    def headline_metric_basis(self) -> str:
        """Returns the metric basis used for headline metrics."""
        if self.tax_summary is not None and self.tax_summary.tax_enabled:
            return self.tax_summary.metric_basis
        return "gross"

    @property
    def equity_curve_daily(self) -> Optional[pd.Series]:
        """Daily mark-to-market curve matching the headline metric basis."""
        if self.tax_summary is not None and self.tax_summary.tax_enabled:
            if self.tax_summary.metric_basis == "gross":
                return self.equity_curve_daily_gross
            if self.tax_summary.metric_basis == "net_liquidation":
                if self.equity_curve_daily_net_liquidation is not None:
                    return self.equity_curve_daily_net_liquidation
                return self.equity_curve_daily_net
            return self.equity_curve_daily_net
        return self.equity_curve_daily_gross

    @property
    def metrics_daily(self) -> Optional[Metrics]:
        """Daily mark-to-market metrics matching the headline metric basis."""
        if self.tax_summary is not None and self.tax_summary.tax_enabled:
            if self.tax_summary.metric_basis == "gross":
                return self.metrics_daily_gross
            if self.tax_summary.metric_basis == "net_liquidation":
                if self.metrics_daily_net_liquidation is not None:
                    return self.metrics_daily_net_liquidation
                return self.metrics_daily_net
            return self.metrics_daily_net
        return self.metrics_daily_gross

    def summary(self) -> str:
        """Generate a text summary of the backtest results."""
        m = self.metrics
        mg = self.metrics_gross
        mn = self.metrics_net
        t = self.tax_summary

        # Determine metric basis label
        basis = self.headline_metric_basis.upper()
        tax_status = "ON" if (t and t.tax_enabled) else "OFF"

        lines = [
            "",
            "═" * 60,
            f" {self.strategy.name.upper()} | {self.equity_curve.index[0].strftime('%Y-%m-%d')} → {self.equity_curve.index[-1].strftime('%Y-%m-%d')}",
            "═" * 60,
            "",
            f" Taxes: {tax_status} | Return Basis: {basis}",
            "" if not (t and t.tax_enabled) else f" Tax Accounting Mode: {t.tax_accounting_mode}",
            "",
            f" Assets:  {', '.join(self.strategy.assets[:3])}{'...' if len(self.strategy.assets) > 3 else ''}",
            f" Rebalance Frequency: {self.config.rebalance_frequency.capitalize()}",
            "",
            f" Starting Capital:    €{self.config.initial_capital:,.0f}",
            f" Ending Capital:      €{self.equity_curve.iloc[-1]:,.0f}",
            "",
        ]

        # Show Gross vs Net comparison if tax is enabled
        if t and t.tax_enabled and mg and mn:
            lines.extend([
                f" {'Metric':<22} {'Gross':>12} {'Net':>12} {'Tax Drag':>12}",
                "─" * 60,
                f" {'Final Value':<22} €{self.equity_curve_gross.iloc[-1]:>10,.0f} €{self.equity_curve_net.iloc[-1]:>10,.0f} {t.tax_drag_final_pct:>11.1f}%",
                f" {'CAGR':<22} {mg.cagr:>11.1%} {mn.cagr:>11.1%} {t.tax_drag_cagr_pp:>10.1f}pp",
                f" {'Volatility':<22} {mg.volatility:>11.1%} {mn.volatility:>11.1%}",
                f" {'Sharpe Ratio':<22} {mg.sharpe_ratio:>11.2f} {mn.sharpe_ratio:>11.2f}",
                f" {'Sortino Ratio':<22} {mg.sortino_ratio:>11.2f} {mn.sortino_ratio:>11.2f}",
                f" {'Max Drawdown':<22} {mg.max_drawdown:>11.1%} {mn.max_drawdown:>11.1%}",
                f" {'Calmar Ratio':<22} {mg.calmar_ratio:>11.2f} {mn.calmar_ratio:>11.2f}",
                "",
            ])
        else:
            # Single column (gross only)
            lines.extend([
                f" {'Metric':<22} {'Strategy':>12}",
                "─" * 60,
                f" {'CAGR':<22} {m.cagr:>11.1%}",
                f" {'Volatility':<22} {m.volatility:>11.1%}",
                f" {'Sharpe Ratio':<22} {m.sharpe_ratio:>11.2f}",
                f" {'Sortino Ratio':<22} {m.sortino_ratio:>11.2f}",
                f" {'Max Drawdown':<22} {m.max_drawdown:>11.1%}",
                f" {'Calmar Ratio':<22} {m.calmar_ratio:>11.2f}",
                "",
            ])

        lines.extend([
            f" {'Win Rate (Monthly)':<22} {m.win_rate_monthly:>11.1%}",
            f" {'Best Month':<22} {m.best_month:>11.1%}",
            f" {'Worst Month':<22} {m.worst_month:>11.1%}",
            "",
            f" {'Trades':<22} {m.num_trades:>11}",
            f" {'Annual Turnover':<22} {m.turnover_annual:>11.1%}",
            f" {'Total Costs':<22} €{m.total_costs:>10,.0f}",
            f" {'Costs p.a.':<22} €{m.costs_per_year:>10,.0f}",
            "",
        ])

        # Add execution/risk constraint diagnostics if available.
        if self.constraint_impact:
            liquidity = self.constraint_impact.get("liquidity", {})
            exposure_policy = self.constraint_impact.get("exposure_policy", {})
            risk_overlay = self.constraint_impact.get("risk_overlay", {})
            lines.extend([
                "─" * 60,
                " CONSTRAINT IMPACT",
                "─" * 60,
            ])
            if isinstance(liquidity, dict):
                lines.extend([
                    f" {'Requested Trades':<22} {int(liquidity.get('requested_trades', 0.0)):>11}",
                    f" {'Executed Trades':<22} {int(liquidity.get('executed_trades', 0.0)):>11}",
                    f" {'Skipped Trades':<22} {int(liquidity.get('skipped_trades', 0.0)):>11}",
                    f" {'Clipped Trades':<22} {int(liquidity.get('clipped_by_max_participation', 0.0)):>11}",
                    f" {'Req→Exec Ratio':<22} {liquidity.get('requested_to_executed_ratio', 1.0):>11.1%}",
                    "",
                ])
            if isinstance(exposure_policy, dict) and exposure_policy:
                pure_comparison = exposure_policy.get("pure_strategy_comparison", {})
                lines.extend([
                    f" {'Exposure Apply Calls':<22} {int(exposure_policy.get('apply_calls', 0.0)):>11}",
                    f" {'Exposure Changed':<22} {int(exposure_policy.get('changed_calls', 0.0)):>11}",
                    f" {'Guard Activations':<22} {int(exposure_policy.get('guard_activations', 0.0)):>11}",
                    f" {'3x→1x Calls':<22} {int(exposure_policy.get('deleveraged_1x_calls', 0.0)):>11}",
                    f" {'Core/Safe Calls':<22} {int(exposure_policy.get('safe_calls', 0.0)):>11}",
                ])
                if isinstance(pure_comparison, dict) and pure_comparison:
                    lines.extend([
                        f" {'Policy CAGR':<22} {pure_comparison.get('policy_cagr', 0.0):>11.1%}",
                        f" {'Pure CAGR':<22} {pure_comparison.get('pure_cagr', 0.0):>11.1%}",
                        f" {'CAGR Δ pp':<22} {pure_comparison.get('cagr_delta_pp', 0.0):>11.2f}",
                        f" {'Policy MaxDD':<22} {pure_comparison.get('policy_max_drawdown', 0.0):>11.1%}",
                        f" {'Pure MaxDD':<22} {pure_comparison.get('pure_max_drawdown', 0.0):>11.1%}",
                        f" {'MaxDD Δ pp':<22} {pure_comparison.get('max_drawdown_delta_pp', 0.0):>11.2f}",
                        f" {'Recovery Δ Days':<22} {int(pure_comparison.get('recovery_days_delta', 0)):>11}",
                    ])
                lines.append("")
            if isinstance(risk_overlay, dict) and risk_overlay:
                lines.extend([
                    f" {'Overlay Apply Calls':<22} {int(risk_overlay.get('apply_calls', 0.0)):>11}",
                    f" {'Overlay Changed':<22} {int(risk_overlay.get('changed_calls', 0.0)):>11}",
                    f" {'DD Brake Activations':<22} {int(risk_overlay.get('drawdown_brake_activations', 0.0)):>11}",
                    "",
                ])

        # Add dividend/DRIP summary if enabled
        if self.dividend_events:
            total_gross = sum(d.gross_amount for d in self.dividend_events)
            total_tax = sum(d.tax_paid for d in self.dividend_events)
            total_net = sum(d.net_amount for d in self.dividend_events)
            total_shares = sum(d.shares_purchased for d in self.dividend_events)
            lines.extend([
                "─" * 60,
                " DIVIDENDS (DRIP)",
                "─" * 60,
                f" {'Dividend Events':<22} {len(self.dividend_events):>11}",
                f" {'Total Gross':<22} €{total_gross:>10,.2f}",
                f" {'Total Tax Paid':<22} €{total_tax:>10,.2f}",
                f" {'Total Net':<22} €{total_net:>10,.2f}",
                f" {'Shares Reinvested':<22} {total_shares:>11.2f}",
                "",
            ])

        # Add benchmark comparison if available
        if m.tracking_difference is not None:
            lines.extend([
                "─" * 60,
                f" {'Tracking Difference':<22} {m.tracking_difference:>+10.1%}",
                f" {'Alpha':<22} {(m.alpha or 0):>+10.1%}",
                f" {'Beta':<22} {(m.beta or 0):>11.2f}",
                f" {'Info Ratio':<22} {(m.information_ratio or 0):>11.2f}",
                "",
            ])

        # Add detailed tax summary if available
        if t and t.tax_enabled:
            lines.extend([
                "─" * 60,
                " TAX IMPACT (German Model)",
                "─" * 60,
                f" {'Total Tax Paid':<22} €{t.total_tax_paid:>10,.0f}",
                f" {'Realized Gains':<22} €{t.total_realized_gains:>10,.0f}",
                f" {'Realized Losses':<22} €{t.total_realized_losses:>10,.0f}",
                f" {'Net Realized':<22} €{t.net_realized_gain:>10,.0f}",
                f" {'Exemption Used':<22} €{t.exemption_used:>10,.0f}",
                f" {'Effective Tax Rate':<22} {t.effective_tax_rate:>11.1%}",
                f" {'Tax Drag (CAGR)':<22} {t.tax_drag_cagr_pp:>10.1f}pp",
                f" {'Tax Drag (Final)':<22} {t.tax_drag_final_pct:>10.1f}%",
                "",
            ])
            # Show loss pot (Verlustvortrag) details if there was any activity
            if t.loss_pot_used > 0 or t.loss_pot_added > 0 or t.loss_pot_equity_final > 0 or t.loss_pot_general_final > 0:
                lines.extend([
                    " LOSS POTS (Verlustvortrag)",
                    "─" * 60,
                    f" {'Loss Pot Used':<22} €{t.loss_pot_used:>10,.0f}",
                    f" {'Loss Pot Added':<22} €{t.loss_pot_added:>10,.0f}",
                    f" {'Equity Pot (Final)':<22} €{t.loss_pot_equity_final:>10,.0f}",
                    f" {'General Pot (Final)':<22} €{t.loss_pot_general_final:>10,.0f}",
                    "",
                ])

        lines.append("═" * 60)
        return "\n".join(lines)

    def to_json(self, path: str) -> None:
        """Export results to JSON file."""
        import json

        data = {
            "strategy": {
                "name": self.strategy.name,
                "params": self.strategy.params,
                "assets": self.strategy.assets,
            },
            "config": {
                "initial_capital": self.config.initial_capital,
                "currency": self.config.currency,
                "costs_pct": self.config.costs_pct,
                "slippage_pct": self.config.slippage_pct,
                "spread_pct": self.config.spread_pct,
                "cost_profile": self.config.cost_profile,
                "execution_lag_days": self.config.execution_lag_days,
                # Tax PROVENANCE goes into the artifact -- without it, the ETP tax
                # treatment behind a result can't be reconstructed.
                "tax_enabled": self.config.tax_enabled,
                "tax_rate": self.config.tax_rate,
                "tax_partial_exemption": self.config.tax_partial_exemption,
                "tax_exemption_amount": self.config.tax_exemption_amount,
                "equity_fund_map": self.config.equity_fund_map,
                "tax_treatment_map": (
                    {k: list(v) for k, v in self.config.tax_treatment_map.items()}
                    if self.config.tax_treatment_map else None
                ),
                "max_volume_participation": self.config.max_volume_participation,
                "min_daily_dollar_volume": self.config.min_daily_dollar_volume,
                "liquidity_on_missing_volume": self.config.liquidity_on_missing_volume,
                "exposure_policy": self.config.exposure_policy,
                "benchmark": self.config.benchmark,
                "risk_free_rate": self.config.risk_free_rate,
                "rebalance_frequency": self.config.rebalance_frequency,
            },
            "metrics": {
                "total_return": self.metrics.total_return,
                "cagr": self.metrics.cagr,
                "volatility": self.metrics.volatility,
                "max_drawdown": self.metrics.max_drawdown,
                "max_drawdown_daily": self.metrics.max_drawdown_daily,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "sortino_ratio": self.metrics.sortino_ratio,
                "calmar_ratio": self.metrics.calmar_ratio,
                "max_drawdown_duration_daily": self.metrics.max_drawdown_duration_daily,
                "underwater_days_daily": self.metrics.underwater_days_daily,
                "win_rate_monthly": self.metrics.win_rate_monthly,
                "best_month": self.metrics.best_month,
                "worst_month": self.metrics.worst_month,
                "num_trades": self.metrics.num_trades,
                "turnover_annual": self.metrics.turnover_annual,
                "total_costs": self.metrics.total_costs,
            },
            "equity_curve": {
                "dates": [d.strftime("%Y-%m-%d") for d in self.equity_curve.index],
                "values": self.equity_curve.tolist(),
            },
            "trades": [
                {
                    "date": t.date.strftime("%Y-%m-%d"),
                    "ticker": t.ticker,
                    "action": t.action,
                    "shares": round(t.shares, 4),
                    "price_ref": round(t.price_ref, 4),
                    "price_exec": round(t.price_exec, 4),
                    "value_ref": round(t.value_ref, 2),
                    "value_exec": round(t.value_exec, 2),
                    "costs": round(t.costs, 2),
                    "slippage": round(t.slippage, 2),
                    "tax_paid_trade": round(t.tax_paid_trade, 2),
                    # Legacy fields for backwards compatibility
                    "price": round(t.price, 4),
                    "value": round(t.value, 2),
                }
                for t in self.trades
            ],
        }

        daily_curve = self.equity_curve_daily
        if daily_curve is not None:
            data["equity_curve_daily"] = {
                "basis": self.headline_metric_basis,
                "dates": [d.strftime("%Y-%m-%d") for d in daily_curve.index],
                "values": daily_curve.tolist(),
            }
        if self.metrics_daily is not None:
            data["metrics_daily"] = {
                "total_return": self.metrics_daily.total_return,
                "cagr": self.metrics_daily.cagr,
                "volatility": self.metrics_daily.volatility,
                "max_drawdown": self.metrics_daily.max_drawdown,
                "max_drawdown_duration": self.metrics_daily.max_drawdown_duration,
                "underwater_days": MetricsCalculator.underwater_days(daily_curve) if daily_curve is not None else None,
                "sharpe_ratio": self.metrics_daily.sharpe_ratio,
                "sortino_ratio": self.metrics_daily.sortino_ratio,
                "calmar_ratio": self.metrics_daily.calmar_ratio,
            }

        # Add tax summary if available
        if self.tax_summary is not None:
            data["tax"] = self.tax_summary.to_dict()
        if self.constraint_impact is not None:
            data["constraint_impact"] = self.constraint_impact
        if self.raw_allocations is not None:
            data["raw_allocations"] = {
                "dates": [d.strftime("%Y-%m-%d") for d in self.raw_allocations.index],
                "rows": self.raw_allocations.reset_index(drop=True).to_dict(orient="records"),
            }
        if self.exposure_policy_decisions is not None:
            data["exposure_policy_decisions"] = self.exposure_policy_decisions

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def _exposure_policy_comparison(
    policy_metrics: Metrics,
    pure_metrics: Metrics,
) -> Dict[str, float]:
    """Summarize policy result against the raw strategy path."""
    pure_recovery = int(pure_metrics.max_drawdown_duration)
    policy_recovery = int(policy_metrics.max_drawdown_duration)
    recovery_improvement = (
        (pure_recovery - policy_recovery) / pure_recovery
        if pure_recovery > 0
        else 0.0
    )
    return {
        "policy_cagr": float(policy_metrics.cagr),
        "pure_cagr": float(pure_metrics.cagr),
        "cagr_delta_pp": float((policy_metrics.cagr - pure_metrics.cagr) * 100.0),
        "policy_max_drawdown": float(policy_metrics.max_drawdown),
        "pure_max_drawdown": float(pure_metrics.max_drawdown),
        "max_drawdown_delta_pp": float((policy_metrics.max_drawdown - pure_metrics.max_drawdown) * 100.0),
        "policy_sharpe": float(policy_metrics.sharpe_ratio),
        "pure_sharpe": float(pure_metrics.sharpe_ratio),
        "sharpe_delta": float(policy_metrics.sharpe_ratio - pure_metrics.sharpe_ratio),
        "policy_recovery_days": float(policy_recovery),
        "pure_recovery_days": float(pure_recovery),
        "recovery_days_delta": float(policy_recovery - pure_recovery),
        "recovery_days_improvement_pct": float(recovery_improvement),
    }


class Backtester:
    """
    Main backtesting engine.

    Runs a strategy against historical price data and generates results.
    """

    def __init__(
        self,
        strategy: Strategy,
        data: PriceData,
        config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtester.

        Args:
            strategy: Strategy to test
            data: Historical price data
            config: Backtest configuration (uses defaults if not provided)
        """
        self.strategy = strategy
        self.data = data
        self.config = config or BacktestConfig()

    def run(self) -> BacktestResult:
        """Public entrypoint.

        Wires the external-features provider onto the strategy for the
        duration of this run using try/finally, so parallel sweeps and
        batch optimizations cannot leak state across runs. The actual
        backtest logic lives in :meth:`_run_inner`.
        """

        prev_provider = getattr(self.strategy, "_external_features_provider", None)
        external_provider = getattr(self.config, "external_features_loader", None)
        if external_provider is not None:
            self.strategy._external_features_provider = external_provider
        try:
            return self._run_inner()
        finally:
            self.strategy._external_features_provider = prev_provider

    def _run_inner(self) -> BacktestResult:
        """
        Execute the backtest.

        IMPORTANT: Strategies always receive daily historical data, regardless
        of rebalance frequency. Rebalance frequency only controls WHEN signal()
        is called, not the data resolution passed to it.

        This ensures:
        - Lookbacks like "126 trading days" work correctly at any rebalance frequency
        - Monthly rebalancing doesn't cause strategies to see "126 months" instead

        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        # PRE-RUN VALIDATION
        if self.config.validate:
            pre_validation = validate_before_run(self.strategy, self.data, self.config)
            if pre_validation.has_errors:
                pre_validation.emit_warnings()
                if self.config.strict_validation:
                    pre_validation.raise_on_errors()

        rebalance_frequency = (
            self.config.rebalance_frequency
            or getattr(self.strategy, "rebalance_frequency", None)
            or "monthly"
        )
        self.config.rebalance_frequency = rebalance_frequency
        rebalance_frequency = rebalance_frequency.lower()

        # Convert prices to EUR if needed
        if self.config.currency == "EUR" and self.data.fx_rates is not None:
            base_data = PriceData(
                prices=self.data.in_eur(),
                currency=self.data.currency.copy(),
                fx_rates=None,
                volumes=self.data.volumes.copy() if self.data.volumes is not None else None,
                dividends=self.data.dividends.copy() if self.data.dividends is not None else None,
                macro=self.data.macro.copy() if self.data.macro is not None else None,
            )
        else:
            base_data = self.data

        # CRITICAL: Keep daily prices for strategies
        # Rebalance frequency only determines WHEN signal() is called,
        # not the data resolution passed to strategies
        daily_prices = base_data.prices
        daily_volumes = base_data.volumes

        # Generate rebalance dates from the daily price index
        # These are the actual trading days when rebalancing occurs
        rebalance_dates = generate_rebalance_dates(daily_prices.index, rebalance_frequency)

        # PERFORMANCE OPTIMIZATION: Precompute integer positions for rebalance dates
        # This allows us to use .iloc[:i] instead of .loc[:date] for slicing,
        # which is O(1) instead of O(n) per iteration
        date_to_iloc = {d: i for i, d in enumerate(daily_prices.index)}
        rebalance_ilocs = [date_to_iloc[d] for d in rebalance_dates]
        execution_lag_days = max(0, int(getattr(self.config, "execution_lag_days", 0) or 0))

        if TRUTH_TEST:
            print(f"[TRUTH] {self.strategy.name}: Using {len(daily_prices)} daily prices, "
                  f"{len(rebalance_dates)} rebalance dates ({rebalance_frequency})")

        # Initialize portfolio
        portfolio = Portfolio(cash=self.config.initial_capital)
        pure_strategy_portfolio: Optional[Portfolio] = None

        # Build transaction cost resolver (default + optional per-ticker profile).
        default_cost_model = Portfolio._default_cost_model(
            costs_pct=self.config.costs_pct,
            slippage_pct=self.config.slippage_pct,
            spread_pct=self.config.spread_pct,
        )
        cost_resolver = CostModelResolver.from_profile(
            default_model=default_cost_model,
            profile=self.config.cost_profile,
        )
        exposure_policy_engine = ExposurePolicyEngine.from_raw(getattr(self.config, "exposure_policy", None))
        risk_overlay_engine = RiskOverlayEngine.from_raw(getattr(self.config, "risk_overlay", None))
        if exposure_policy_engine.enabled:
            pure_strategy_portfolio = Portfolio(cash=self.config.initial_capital)
        liquidity_stats: Dict[str, float] = {
            "requested_sell_trades": 0.0,
            "requested_buy_trades": 0.0,
            "planned_sell_trades": 0.0,
            "planned_buy_trades": 0.0,
            "executed_sell_trades": 0.0,
            "executed_buy_trades": 0.0,
            "requested_shares_total": 0.0,
            "planned_shares_total": 0.0,
            "executed_shares_total": 0.0,
            "skipped_missing_volume": 0.0,
            "skipped_min_daily_dollar_volume": 0.0,
            "clipped_by_max_participation": 0.0,
            "clipped_shares_total": 0.0,
            "cash_scaled_buy_trades": 0.0,
            "liquidation_requested_trades": 0.0,
            "liquidation_executed_trades": 0.0,
        }

        # Initialize tax model if enabled
        tax_model: Optional[GermanTaxModel] = None
        if self.config.tax_enabled:
            tax_model = GermanTaxModel(
                tax_rate=self.config.tax_rate,
                partial_exemption=self.config.tax_partial_exemption,
                exemption_amount=self.config.tax_exemption_amount,
                cost_basis_method=self.config.cost_basis_method,
            )

        # Track results
        trades: List[Trade] = []
        dividend_events: List[DividendEvent] = []  # Track dividend payments
        equity_history_net: List[Dict] = []  # Net values (after tax deductions)
        equity_history_gross: List[Dict] = []  # Gross values (before tax deductions)
        equity_history_daily_net: List[Dict] = []  # Daily net mark-to-market diagnostics
        equity_history_daily_gross: List[Dict] = []  # Daily gross mark-to-market diagnostics
        daily_marked_dates: set[pd.Timestamp] = set()
        pure_strategy_equity_history: List[Dict] = []
        allocation_history: List[Dict] = []
        raw_allocation_history: List[Dict] = []
        exposure_policy_decisions: List[Dict[str, Any]] = []
        pure_strategy_trades: List[Trade] = []

        # Track cumulative tax paid for gross calculation
        cumulative_tax_paid = 0.0

        equity_history_daily_liq: List[dict] = []

        def _virtuelle_liquidationssteuer(
            mark_prices: pd.Series,
        ) -> float:
            """Tax that would be due if everything were sold TODAY.

            Same calculation as at the endpoint, just applied to an arbitrary day.
            Operates on a ``clone()`` -- the real tax state stays untouched.
            """
            if tax_model is None:
                return 0.0
            virtuell = tax_model.clone()
            steuer = 0.0
            for tkr, anteile in sorted(portfolio.positions.items()):
                if anteile <= 1e-9:
                    continue
                kurs = mark_prices.get(tkr, 0.0)
                if kurs <= 0:
                    continue
                try:
                    _ic, _eq = self._instrument_tax_flags(tkr)
                    steuer += virtuell.apply_sale(
                        ticker=tkr, shares_sold=anteile, sale_price=kurs,
                        sale_date=mark_prices.name.date()
                        if hasattr(mark_prices.name, "date") else mark_prices.name,
                        instrument_class=_ic, is_equity_fund=_eq,
                    ).tax_due
                except NoLotsError:
                    pass
                except InsufficientSharesError:
                    pass
            return steuer

        def _record_daily_mark(
            mark_date: pd.Timestamp,
            mark_prices: pd.Series,
            tax_paid_snapshot: float,
        ) -> None:
            """Record one additive daily MTM point without changing headline sampling."""
            ts = pd.Timestamp(mark_date)
            if ts in daily_marked_dates:
                return
            net_value = portfolio.total_value(mark_prices)
            equity_history_daily_net.append({"date": ts, "value": net_value})
            equity_history_daily_gross.append({"date": ts, "value": net_value + tax_paid_snapshot})
            if self.config.daily_liquidation_tax and self.config.metric_basis == "net_liquidation":
                equity_history_daily_liq.append(
                    {"date": ts, "value": max(0.0, net_value - _virtuelle_liquidationssteuer(mark_prices))})
            daily_marked_dates.add(ts)

        def _record_daily_segment(
            start_iloc: int,
            end_iloc_exclusive: int,
            tax_paid_snapshot: float,
        ) -> None:
            """Record daily MTM points for the current portfolio state."""
            start = max(0, int(start_iloc))
            end = min(len(daily_prices), int(end_iloc_exclusive))
            if end <= start:
                return
            for mark_date, mark_prices in daily_prices.iloc[start:end].iterrows():
                _record_daily_mark(mark_date, mark_prices, tax_paid_snapshot)

        # Get dividend data if available (for DRIP)
        dividends_df = None
        if hasattr(self.data, 'dividends') and self.data.dividends is not None:
            dividends_df = self.data.dividends.copy()  # Don't modify original
            # Convert to EUR if needed
            if self.config.currency == "EUR" and self.data.fx_rates is not None:
                # Align FX rates to dividend dates with forward/backward fill.
                fx_aligned = self.data.fx_rates.reindex(dividends_df.index).ffill().bfill()
                for ticker in dividends_df.columns:
                    currency = self.data.currency.get(ticker, "USD")
                    if currency in {"USD", "GBP"}:
                        rate_series = PriceData._fx_series_for_currency(fx_aligned, currency)
                        # Use .values to avoid index alignment issues
                        dividend_values = pd.to_numeric(dividends_df[ticker], errors="coerce").values
                        fx_values = pd.to_numeric(rate_series, errors="coerce").values
                        dividends_df[ticker] = dividend_values / fx_values

        # Run simulation on rebalance dates
        for idx, date in enumerate(rebalance_dates):
            signal_iloc = rebalance_ilocs[idx]
            # Signal-day market snapshot.
            prices = daily_prices.iloc[signal_iloc]
            prices.name = date

            # Execution-day snapshot (supports optional T+N execution lag).
            # Previously `min(signal_iloc + lag, last index)` -- this let the LAST
            # signal execute at the same close despite execution_lag_days=1, i.e.
            # exactly the lookahead the lag is supposed to eliminate. A signal whose
            # execution day is no longer in the data is not tradeable and gets
            # skipped (effect measured at ~0.001pp, but the code was wrong regardless).
            execution_iloc = signal_iloc + execution_lag_days
            # Only tradeable if the execution day is still within the data. The skip
            # must lock ONLY the trade path -- cash interest, dividends, and the daily
            # mark-to-market below must keep running (counter-check found that an
            # earlier `continue` here lost interest and dividends of the last signal).
            signal_is_executable = execution_iloc <= len(daily_prices) - 1
            execution_date = None
            execution_prices = None
            execution_volumes = None
            if signal_is_executable:
                execution_date = daily_prices.index[execution_iloc]
                execution_prices = daily_prices.iloc[execution_iloc]
                execution_prices.name = execution_date
                if daily_volumes is not None:
                    execution_volumes = daily_volumes.iloc[execution_iloc]
                    execution_volumes.name = execution_date

            # Apply cash interest for the period since last rebalance
            if self.config.cash_rate and idx > 0:
                prev_date = rebalance_dates[idx - 1]
                days = (date - prev_date).days
                if days > 0:
                    growth = (1 + self.config.cash_rate) ** (days / 365.25)
                    portfolio.cash *= growth
                    if pure_strategy_portfolio is not None:
                        pure_strategy_portfolio.cash *= growth

            # Process dividends since last rebalance date
            if dividends_df is not None:
                prev_date = daily_prices.index[0] if idx == 0 else rebalance_dates[idx - 1]
                period_events = self._process_period_dividends(
                    portfolio=portfolio,
                    dividends_df=dividends_df,
                    prev_date=prev_date,
                    current_date=date,
                    prices=prices,
                    drip_enabled=self.config.drip_enabled,
                    tax_model=tax_model,
                )
                if period_events:
                    dividend_events.extend(period_events)
                    cumulative_tax_paid += sum(event.tax_paid for event in period_events)
                if pure_strategy_portfolio is not None:
                    self._process_period_dividends(
                        portfolio=pure_strategy_portfolio,
                        dividends_df=dividends_df,
                        prev_date=prev_date,
                        current_date=date,
                        prices=prices,
                        drip_enabled=self.config.drip_enabled,
                        tax_model=None,
                    )

            # Get current portfolio value (this is NET after tax deductions)
            current_value_net = portfolio.total_value(prices)
            # GROSS = NET + cumulative tax already paid (what value would be without tax)
            current_value_gross = current_value_net + cumulative_tax_paid

            equity_history_net.append({"date": date, "value": current_value_net})
            equity_history_gross.append({"date": date, "value": current_value_gross})
            _record_daily_mark(date, prices, cumulative_tax_paid)
            if not signal_is_executable:
                # Execution day is past the end of the data -> NO trade (otherwise it
                # would be the same-close lookahead). Interest/dividends/valuation above
                # have already run; the remaining days until the end of the data are
                # still marked daily here so the curve isn't cut off.
                _record_daily_segment(signal_iloc + 1, len(daily_prices), cumulative_tax_paid)
                continue
            if execution_iloc > signal_iloc:
                _record_daily_segment(signal_iloc + 1, execution_iloc + 1, cumulative_tax_paid)
            if pure_strategy_portfolio is not None:
                pure_strategy_equity_history.append(
                    {"date": date, "value": pure_strategy_portfolio.total_value(prices)}
                )

            # CRITICAL: Pass DAILY historical data to strategy, not resampled!
            # Use .iloc[:i+1] for O(1) slicing instead of .loc[:date] which is O(n)
            historical_data = daily_prices.iloc[:rebalance_ilocs[idx] + 1]

            # Truth Test: check for missing tickers and data availability
            if TRUTH_TEST:
                req = set(getattr(self.strategy, "assets", []) or [])
                cols = set(historical_data.columns) if hasattr(historical_data, "columns") else set()
                missing = sorted(req - cols)

                if missing:
                    print(f"[TRUTH] {self.strategy.name} @ {date.date()}: MISSING tickers: {missing}")

                # Show data availability for each required asset
                for t in sorted(req):
                    if t in cols:
                        nn = int(historical_data[t].notna().sum())
                        print(f"[TRUTH] {self.strategy.name} @ {date.date()}: {t} non-NaN rows = {nn}")

            target_allocation_raw = self.strategy.signal(date.date(), historical_data)

            # Normalize allocation tickers (e.g., WKN/name -> Yahoo ticker) to match price columns
            normalized_weights: Dict[str, float] = {}
            for ticker, weight in target_allocation_raw.items():
                mapped = get_yahoo_ticker(ticker)
                if mapped in daily_prices.columns:
                    target_ticker = mapped
                elif ticker in daily_prices.columns:
                    target_ticker = ticker
                else:
                    target_ticker = mapped
                normalized_weights[target_ticker] = normalized_weights.get(target_ticker, 0.0) + weight

            target_allocation = Allocation(normalized_weights)
            pure_strategy_target_allocation = target_allocation

            if exposure_policy_engine.enabled:
                raw_alloc_record = {"date": date}
                raw_alloc_record.update(target_allocation.weights)
                raw_alloc_record["cash"] = target_allocation.cash
                raw_allocation_history.append(raw_alloc_record)

                exposure_decision = exposure_policy_engine.apply(
                    target=target_allocation,
                    historical_prices=historical_data,
                )
                decision_record = {
                    "date": date.strftime("%Y-%m-%d"),
                    **exposure_decision.to_dict(),
                }
                exposure_policy_decisions.append(decision_record)
                target_allocation = exposure_decision.allocation

            if pure_strategy_portfolio is not None:
                pure_strategy_trades.extend(
                    self._rebalance_portfolio_no_tax(
                        portfolio=pure_strategy_portfolio,
                        target=pure_strategy_target_allocation,
                        prices=execution_prices,
                        volumes=execution_volumes,
                        cost_resolver=cost_resolver,
                    )
                )

            if risk_overlay_engine.enabled:
                target_allocation = risk_overlay_engine.apply(
                    target=target_allocation,
                    current_weights=portfolio.weights(prices),
                    current_equity=current_value_net,
                )

            if TRUTH_TEST:
                print(f"[TRUTH] {self.strategy.name} @ {date.date()}: alloc = {target_allocation}")

            # Track allocation
            alloc_record = {"date": date}
            alloc_record.update(target_allocation.weights)
            alloc_record["cash"] = target_allocation.cash
            allocation_history.append(alloc_record)

            # PHASED EXECUTION: SELL → Tax → BUY
            # This ensures tax is deducted before buys, preventing cash overcommitment

            # Phase 1: Calculate and execute SELL trades
            sell_trades, _ = portfolio.calculate_rebalance_trades(
                target=target_allocation,
                prices=execution_prices,
                volumes=execution_volumes,
                costs_pct=self.config.costs_pct,
                slippage_pct=self.config.slippage_pct,
                spread_pct=self.config.spread_pct,
                cost_resolver=cost_resolver,
                max_volume_participation=self.config.max_volume_participation,
                min_daily_dollar_volume=self.config.min_daily_dollar_volume,
                liquidity_on_missing_volume=self.config.liquidity_on_missing_volume,
                liquidity_stats=liquidity_stats,
                plan_buys=False,
            )
            portfolio.execute_sells(sell_trades)
            trades.extend(sell_trades)
            liquidity_stats["executed_sell_trades"] += float(len(sell_trades))
            liquidity_stats["executed_shares_total"] += float(sum(t.shares for t in sell_trades))

            # Phase 2: Apply tax on SELL trades (before BUYs)
            if tax_model is not None:
                tax_paid = self._apply_taxes_to_sells(
                    tax_model,
                    portfolio,
                    sell_trades,
                    execution_date,
                )
                cumulative_tax_paid += tax_paid

            # Phase 3: Execute BUY trades (with cash reduced by tax)
            buy_trades = portfolio.execute_buys(
                target=target_allocation,
                prices=execution_prices,
                volumes=execution_volumes,
                costs_pct=self.config.costs_pct,
                slippage_pct=self.config.slippage_pct,
                spread_pct=self.config.spread_pct,
                cost_resolver=cost_resolver,
                max_volume_participation=self.config.max_volume_participation,
                min_daily_dollar_volume=self.config.min_daily_dollar_volume,
                liquidity_on_missing_volume=self.config.liquidity_on_missing_volume,
                liquidity_stats=liquidity_stats,
            )
            trades.extend(buy_trades)
            liquidity_stats["executed_buy_trades"] += float(len(buy_trades))
            liquidity_stats["executed_shares_total"] += float(sum(t.shares for t in buy_trades))

            # Phase 4: Record BUY trades in tax model for cost basis tracking
            if tax_model is not None:
                for trade in buy_trades:
                    if trade.shares <= 0:
                        continue
                    _ic, _eq = self._instrument_tax_flags(trade.ticker)
                    tax_model.record_purchase(
                        ticker=trade.ticker,
                        shares=trade.shares,
                        price_per_share=(trade.value_exec + trade.costs) / trade.shares,
                        purchase_date=trade.date.date() if hasattr(trade.date, 'date') else trade.date,
                        instrument_class=_ic,  # "general" (ETF/Fund) or "equity" (single stock)
                        is_equity_fund=_eq,  # single stocks: no Teilfreistellung
                    )

            next_signal_iloc = rebalance_ilocs[idx + 1] if idx + 1 < len(rebalance_ilocs) else len(daily_prices) - 1
            _record_daily_segment(execution_iloc + 1, next_signal_iloc + 1, cumulative_tax_paid)

        # Actual liquidation at end (if enabled)
        # This creates real SELL trades for all remaining positions
        final_date = daily_prices.index[-1]
        final_prices = daily_prices.iloc[-1]
        final_prices.name = final_date
        final_volumes = None
        if daily_volumes is not None:
            final_volumes = daily_volumes.iloc[-1]
            final_volumes.name = final_date

        if self.config.actual_liquidation_at_end:
            liquidation_trades = []

            # Create SELL trades for all positions
            for ticker, shares in list(portfolio.positions.items()):
                if shares > 1e-9:
                    price_ref = final_prices.get(ticker, 0.0)
                    if price_ref <= 0:
                        continue

                    liquidity_stats["liquidation_requested_trades"] += 1.0
                    liquidity_stats["requested_sell_trades"] += 1.0
                    liquidity_stats["requested_shares_total"] += float(shares)
                    shares_to_sell, cap_meta = Portfolio._cap_shares_for_liquidity_with_meta(
                        requested_shares=shares,
                        ticker=ticker,
                        price_ref=price_ref,
                        volumes=final_volumes,
                        max_volume_participation=self.config.max_volume_participation,
                        min_daily_dollar_volume=self.config.min_daily_dollar_volume,
                        liquidity_on_missing_volume=self.config.liquidity_on_missing_volume,
                    )
                    if cap_meta["skipped_missing_volume"]:
                        liquidity_stats["skipped_missing_volume"] += 1.0
                    if cap_meta["skipped_min_notional"]:
                        liquidity_stats["skipped_min_daily_dollar_volume"] += 1.0
                    if cap_meta["clipped_by_max_participation"]:
                        liquidity_stats["clipped_by_max_participation"] += 1.0
                        liquidity_stats["clipped_shares_total"] += max(
                            0.0, float(cap_meta["requested_shares"]) - float(cap_meta["final_shares"])
                        )
                    if shares_to_sell <= 1e-9:
                        continue

                    cost_model = cost_resolver.for_ticker(ticker)
                    liquidation_trade = Portfolio._build_trade(
                        current_date=final_date,
                        ticker=ticker,
                        action="SELL",
                        shares=shares_to_sell,
                        price_ref=price_ref,
                        cost_model=cost_model,
                    )
                    liquidation_trades.append(liquidation_trade)
                    liquidity_stats["planned_sell_trades"] += 1.0
                    liquidity_stats["planned_shares_total"] += float(shares_to_sell)

            # Execute sells
            portfolio.execute_sells(liquidation_trades)
            liquidity_stats["executed_sell_trades"] += float(len(liquidation_trades))
            liquidity_stats["liquidation_executed_trades"] += float(len(liquidation_trades))
            liquidity_stats["executed_shares_total"] += float(sum(t.shares for t in liquidation_trades))

            # Apply tax on liquidation sells
            if tax_model is not None:
                for trade in liquidation_trades:
                    try:
                        _ic, _eq = self._instrument_tax_flags(trade.ticker)
                        tax_result = tax_model.apply_sale(
                            ticker=trade.ticker,
                            shares_sold=trade.shares,
                            sale_price=(trade.value_exec - trade.costs) / trade.shares,
                            sale_date=trade.date.date() if hasattr(trade.date, 'date') else trade.date,
                            instrument_class=_ic,
                            is_equity_fund=_eq,
                        )
                        trade.tax_paid_trade = tax_result.tax_due
                        # Deduct tax from cash
                        portfolio.cash -= tax_result.tax_due
                        cumulative_tax_paid += tax_result.tax_due
                    except NoLotsError:
                        pass  # No position in the tax model -- legitimate
                    except InsufficientSharesError as exc:
                        # NEVER swallow silently -- this would lose tax.
                        warnings.warn(f"Steuer-Inkonsistenz (Liquidation): {exc}", RuntimeWarning)

            # Add liquidation trades to the list
            trades.extend(liquidation_trades)

            # Record final state after liquidation
            liq_net_value = portfolio.total_value(final_prices)
            liq_gross_value = liq_net_value + cumulative_tax_paid
            equity_history_net.append({"date": final_date, "value": liq_net_value})
            equity_history_gross.append({"date": final_date, "value": liq_gross_value})

            if pure_strategy_portfolio is not None:
                pure_strategy_trades.extend(
                    self._liquidate_portfolio_no_tax(
                        portfolio=pure_strategy_portfolio,
                        prices=final_prices,
                        volumes=final_volumes,
                        cost_resolver=cost_resolver,
                    )
                )
                pure_strategy_equity_history.append(
                    {"date": final_date, "value": pure_strategy_portfolio.total_value(final_prices)}
                )

        # Final values (using daily prices at last rebalance date)
        final_value_net_realized = portfolio.total_value(final_prices)
        final_value_gross = final_value_net_realized + cumulative_tax_paid

        # Virtual liquidation calculation (for net_liquidation mode)
        tax_paid_liquidation = 0.0
        unrealized_gain_at_end = 0.0
        final_value_net_liquidation = final_value_net_realized

        if tax_model is not None and self.config.metric_basis == "net_liquidation":
            # Calculate unrealized gains and virtual liquidation tax
            # We need to create a copy of the tax model state to not pollute it
            # Using clone() instead of deepcopy() for better performance
            virtual_tax_model = tax_model.clone()

            for ticker, shares in portfolio.positions.items():
                if shares > 1e-9:
                    current_price = final_prices.get(ticker, 0.0)
                    if current_price > 0:
                        # Get cost basis from tax model
                        total_shares, total_cost = virtual_tax_model.get_position(ticker)
                        if total_shares > 0:
                            avg_cost = total_cost / total_shares
                            unrealized_pnl = (current_price - avg_cost) * shares
                            unrealized_gain_at_end += unrealized_pnl

                            # Apply virtual sale through tax model
                            try:
                                _ic, _eq = self._instrument_tax_flags(ticker)
                                tax_result = virtual_tax_model.apply_sale(
                                    ticker=ticker,
                                    shares_sold=shares,
                                    sale_price=current_price,
                                    sale_date=final_date.date() if hasattr(final_date, 'date') else final_date,
                                    instrument_class=_ic,
                                    is_equity_fund=_eq,
                                )
                                tax_paid_liquidation += tax_result.tax_due
                                if TRUTH_TEST:
                                    print(f"[TRUTH] {self.strategy.name} @ {final_date.date()}: "
                                          f"VIRTUAL LIQ {ticker}: unrealized={unrealized_pnl:.2f}, "
                                          f"tax={tax_result.tax_due:.2f}")
                            except NoLotsError:
                                pass  # No position in the tax model -- legitimate
                            except InsufficientSharesError as exc:
                                warnings.warn(f"Steuer-Inkonsistenz (virtuelle Liquidation): {exc}",
                                              RuntimeWarning)

            final_value_net_liquidation = final_value_net_realized - tax_paid_liquidation

        # Backwards compatibility: use net_realized as primary net value
        final_value_net = final_value_net_realized

        # Create equity curves
        equity_curve_net = pd.Series(
            [e["value"] for e in equity_history_net],
            index=pd.DatetimeIndex([e["date"] for e in equity_history_net]),
            name="equity_net"
        )
        equity_curve_gross = pd.Series(
            [e["value"] for e in equity_history_gross],
            index=pd.DatetimeIndex([e["date"] for e in equity_history_gross]),
            name="equity_gross"
        )
        equity_curve_daily_net = pd.Series(
            [e["value"] for e in equity_history_daily_net],
            index=pd.DatetimeIndex([e["date"] for e in equity_history_daily_net]),
            name="equity_daily_net"
        )
        equity_curve_daily_gross = pd.Series(
            [e["value"] for e in equity_history_daily_gross],
            index=pd.DatetimeIndex([e["date"] for e in equity_history_daily_gross]),
            name="equity_daily_gross"
        )
        pure_strategy_equity_curve: Optional[pd.Series] = None
        if pure_strategy_equity_history:
            pure_strategy_equity_curve = pd.Series(
                [e["value"] for e in pure_strategy_equity_history],
                index=pd.DatetimeIndex([e["date"] for e in pure_strategy_equity_history]),
                name="equity_pure_strategy",
            )
        equity_curve_net_liquidation: Optional[pd.Series] = None
        if tax_model is not None and self.config.metric_basis == "net_liquidation":
            equity_curve_net_liquidation = equity_curve_net.copy()
            if len(equity_curve_net_liquidation) > 0:
                equity_curve_net_liquidation.iloc[-1] = max(
                    0.0,
                    equity_curve_net_liquidation.iloc[-1] - tax_paid_liquidation,
                )
                equity_curve_net_liquidation.name = "equity_net_liquidation"
        equity_curve_daily_net_liquidation: Optional[pd.Series] = None
        if tax_model is not None and self.config.metric_basis == "net_liquidation":
            if self.config.daily_liquidation_tax and equity_history_daily_liq:
                _df_liq = pd.DataFrame(equity_history_daily_liq).drop_duplicates(
                    subset="date", keep="last").set_index("date").sort_index()
                equity_curve_daily_net_liquidation = _df_liq["value"]
                equity_curve_daily_net_liquidation.name = "equity_daily_net_liquidation"
                return_value_marker_t0478e = True
            else:
                return_value_marker_t0478e = False
            equity_curve_daily_net_liquidation = (
                equity_curve_daily_net_liquidation if return_value_marker_t0478e
                else equity_curve_daily_net.copy())
            if not return_value_marker_t0478e and len(equity_curve_daily_net_liquidation) > 0:
                equity_curve_daily_net_liquidation.iloc[-1] = max(
                    0.0,
                    equity_curve_daily_net_liquidation.iloc[-1] - tax_paid_liquidation,
                )
                equity_curve_daily_net_liquidation.name = "equity_daily_net_liquidation"

        # Load benchmark if configured
        benchmark_curve = None
        if self.config.benchmark:
            benchmark_candidates: List[str] = []
            benchmark_input = str(self.config.benchmark).strip()
            if benchmark_input:
                benchmark_candidates.append(benchmark_input)
                try:
                    from backtest.assets import get_benchmark_ticker

                    mapped_ticker = get_benchmark_ticker(benchmark_input)
                    if mapped_ticker not in benchmark_candidates:
                        benchmark_candidates.append(mapped_ticker)
                except Exception:
                    # Unknown benchmark name; treat input as potential raw ticker
                    pass

            for benchmark_ticker in benchmark_candidates:
                if benchmark_ticker not in daily_prices.columns:
                    continue
                # Use full daily benchmark prices for smooth chart
                bench_daily = daily_prices[benchmark_ticker].dropna()
                # Filter to same date range as strategy
                bench_daily = bench_daily.loc[
                    (bench_daily.index >= rebalance_dates[0]) &
                    (bench_daily.index <= rebalance_dates[-1])
                ]
                if len(bench_daily) > 0 and bench_daily.iloc[0] > 0:
                    initial_shares = self.config.initial_capital / bench_daily.iloc[0]
                    benchmark_curve = bench_daily * initial_shares
                    break

        # Create allocations DataFrame
        allocations = pd.DataFrame(allocation_history).set_index("date")
        raw_allocations = (
            pd.DataFrame(raw_allocation_history).set_index("date")
            if raw_allocation_history
            else None
        )

        # Calculate metrics for both curves
        metrics_gross = MetricsCalculator.calculate_all(
            equity_curve=equity_curve_gross,
            trades=trades,
            config=self.config,
            benchmark_curve=benchmark_curve,
            allocations=allocations,
        )
        metrics_daily_gross = MetricsCalculator.calculate_all(
            equity_curve=equity_curve_daily_gross,
            trades=trades,
            config=self.config,
            benchmark_curve=benchmark_curve,
            allocations=allocations,
        )

        # Net metrics only meaningful if tax was actually applied
        metrics_net: Optional[Metrics] = None
        metrics_daily_net: Optional[Metrics] = None
        if tax_model is not None and cumulative_tax_paid > 0:
            metrics_net = MetricsCalculator.calculate_all(
                equity_curve=equity_curve_net,
                trades=trades,
                config=self.config,
                benchmark_curve=benchmark_curve,
                allocations=allocations,
            )
            metrics_daily_net = MetricsCalculator.calculate_all(
                equity_curve=equity_curve_daily_net,
                trades=trades,
                config=self.config,
                benchmark_curve=benchmark_curve,
                allocations=allocations,
            )
        elif tax_model is not None:
            # Tax enabled but no tax paid (e.g., no realized gains)
            metrics_net = metrics_gross
            metrics_daily_net = metrics_daily_gross
        metrics_net_liquidation: Optional[Metrics] = None
        if equity_curve_net_liquidation is not None:
            metrics_net_liquidation = MetricsCalculator.calculate_all(
                equity_curve=equity_curve_net_liquidation,
                trades=trades,
                config=self.config,
                benchmark_curve=benchmark_curve,
                allocations=allocations,
            )
        metrics_daily_net_liquidation: Optional[Metrics] = None
        if equity_curve_daily_net_liquidation is not None:
            metrics_daily_net_liquidation = MetricsCalculator.calculate_all(
                equity_curve=equity_curve_daily_net_liquidation,
                trades=trades,
                config=self.config,
                benchmark_curve=benchmark_curve,
                allocations=allocations,
            )

        # Determine primary equity curve and metrics
        # If tax is cash-effective and enabled, NET is primary; otherwise GROSS
        if tax_model is not None:
            if self.config.metric_basis == "gross":
                equity_curve = equity_curve_gross
                metrics = metrics_gross
            elif self.config.metric_basis == "net_liquidation" and equity_curve_net_liquidation is not None:
                equity_curve = equity_curve_net_liquidation
                metrics = metrics_net_liquidation if metrics_net_liquidation is not None else metrics_net
            else:
                equity_curve = equity_curve_net
                metrics = metrics_net if metrics_net is not None else metrics_gross
        else:
            equity_curve = equity_curve_gross
            metrics = metrics_gross

        def _annotate_daily_metrics(headline: Optional[Metrics], daily: Optional[Metrics], curve: Optional[pd.Series]) -> None:
            if headline is None or daily is None or curve is None:
                return
            headline.max_drawdown_daily = daily.max_drawdown
            headline.max_drawdown_duration_daily = daily.max_drawdown_duration
            headline.underwater_days_daily = MetricsCalculator.underwater_days(curve)

        _annotate_daily_metrics(metrics_gross, metrics_daily_gross, equity_curve_daily_gross)
        _annotate_daily_metrics(metrics_net, metrics_daily_net, equity_curve_daily_net if tax_model is not None else None)
        _annotate_daily_metrics(
            metrics_net_liquidation,
            metrics_daily_net_liquidation,
            equity_curve_daily_net_liquidation,
        )
        if tax_model is None:
            _annotate_daily_metrics(metrics, metrics_daily_gross, equity_curve_daily_gross)
        elif self.config.metric_basis == "gross":
            _annotate_daily_metrics(metrics, metrics_daily_gross, equity_curve_daily_gross)
        elif self.config.metric_basis == "net_liquidation":
            _annotate_daily_metrics(metrics, metrics_daily_net_liquidation, equity_curve_daily_net_liquidation)
        else:
            _annotate_daily_metrics(metrics, metrics_daily_net, equity_curve_daily_net)

        # Build tax summary
        tax_summary = self._build_tax_summary(
            tax_model=tax_model,
            trades=trades,
            metrics_gross=metrics_gross,
            metrics_net=metrics_net,
            equity_curve_gross=equity_curve_gross,
            final_value_gross=final_value_gross,
            final_value_net_realized=final_value_net_realized,
            final_value_net_liquidation=final_value_net_liquidation,
            tax_paid_liquidation=tax_paid_liquidation,
            unrealized_gain_at_end=unrealized_gain_at_end,
        )

        exposure_policy_diagnostics = exposure_policy_engine.diagnostics() if exposure_policy_engine.enabled else {}
        if exposure_policy_engine.enabled and pure_strategy_equity_curve is not None:
            pure_strategy_metrics = MetricsCalculator.calculate_all(
                equity_curve=pure_strategy_equity_curve,
                trades=pure_strategy_trades,
                config=self.config,
                benchmark_curve=benchmark_curve,
                allocations=raw_allocations,
            )
            exposure_policy_diagnostics["pure_strategy_comparison"] = _exposure_policy_comparison(
                policy_metrics=metrics,
                pure_metrics=pure_strategy_metrics,
            )

        requested_trades = liquidity_stats["requested_sell_trades"] + liquidity_stats["requested_buy_trades"]
        executed_trades = liquidity_stats["executed_sell_trades"] + liquidity_stats["executed_buy_trades"]
        planned_trades = liquidity_stats["planned_sell_trades"] + liquidity_stats["planned_buy_trades"]
        skipped_trades = liquidity_stats["skipped_missing_volume"] + liquidity_stats["skipped_min_daily_dollar_volume"]
        clipped_trades = liquidity_stats["clipped_by_max_participation"]
        constraint_impact: Dict[str, Any] = {
            "execution_lag_days": float(execution_lag_days),
            "rebalance_signals": float(len(rebalance_dates)),
            "liquidity": {
                **liquidity_stats,
                "requested_trades": requested_trades,
                "planned_trades": planned_trades,
                "executed_trades": executed_trades,
                "skipped_trades": skipped_trades,
                "requested_to_executed_ratio": (executed_trades / requested_trades) if requested_trades > 0 else 1.0,
                "skipped_trade_ratio": (skipped_trades / requested_trades) if requested_trades > 0 else 0.0,
                "clipped_trade_ratio": (clipped_trades / requested_trades) if requested_trades > 0 else 0.0,
            },
            "exposure_policy": exposure_policy_diagnostics,
            "risk_overlay": risk_overlay_engine.diagnostics() if risk_overlay_engine.enabled else {},
        }

        # Sanity checks (invariants) - legacy method
        if self.config.validate:
            self._validate_invariants(tax_summary)

        # Build result
        backtest_result = BacktestResult(
            strategy=self.strategy,
            config=self.config,
            equity_curve=equity_curve,
            allocations=allocations,
            trades=trades,
            metrics=metrics,
            raw_allocations=raw_allocations,
            equity_curve_gross=equity_curve_gross,
            equity_curve_net=equity_curve_net if tax_model is not None else None,
            equity_curve_daily_gross=equity_curve_daily_gross,
            equity_curve_daily_net=equity_curve_daily_net if tax_model is not None else None,
            equity_curve_daily_net_liquidation=equity_curve_daily_net_liquidation,
            metrics_gross=metrics_gross,
            metrics_net=metrics_net,
            metrics_daily_gross=metrics_daily_gross,
            metrics_daily_net=metrics_daily_net,
            metrics_daily_net_liquidation=metrics_daily_net_liquidation,
            benchmark_curve=benchmark_curve,
            tax_summary=tax_summary,
            dividend_events=dividend_events if dividend_events else None,
            constraint_impact=constraint_impact,
            exposure_policy_decisions=exposure_policy_decisions if exposure_policy_decisions else None,
        )

        # POST-RUN VALIDATION
        if self.config.validate:
            post_validation = validate_backtest_result(backtest_result)
            if post_validation.has_warnings or post_validation.has_errors:
                post_validation.emit_warnings()
                if self.config.strict_validation and post_validation.has_errors:
                    post_validation.raise_on_errors()

        return backtest_result

    def _rebalance_portfolio_no_tax(
        self,
        portfolio: Portfolio,
        target: Allocation,
        prices: pd.Series,
        volumes: Optional[pd.Series],
        cost_resolver: CostModelResolver,
    ) -> List[Trade]:
        """Execute a shadow rebalance without tax accounting."""
        sell_trades, _ = portfolio.calculate_rebalance_trades(
            target=target,
            prices=prices,
            volumes=volumes,
            costs_pct=self.config.costs_pct,
            slippage_pct=self.config.slippage_pct,
            spread_pct=self.config.spread_pct,
            cost_resolver=cost_resolver,
            max_volume_participation=self.config.max_volume_participation,
            min_daily_dollar_volume=self.config.min_daily_dollar_volume,
            liquidity_on_missing_volume=self.config.liquidity_on_missing_volume,
            plan_buys=False,
        )
        portfolio.execute_sells(sell_trades)
        buy_trades = portfolio.execute_buys(
            target=target,
            prices=prices,
            volumes=volumes,
            costs_pct=self.config.costs_pct,
            slippage_pct=self.config.slippage_pct,
            spread_pct=self.config.spread_pct,
            cost_resolver=cost_resolver,
            max_volume_participation=self.config.max_volume_participation,
            min_daily_dollar_volume=self.config.min_daily_dollar_volume,
            liquidity_on_missing_volume=self.config.liquidity_on_missing_volume,
        )
        return sell_trades + buy_trades

    def _liquidate_portfolio_no_tax(
        self,
        portfolio: Portfolio,
        prices: pd.Series,
        volumes: Optional[pd.Series],
        cost_resolver: CostModelResolver,
    ) -> List[Trade]:
        """Liquidate a shadow portfolio without tax accounting."""
        trades: List[Trade] = []
        current_date = prices.name if hasattr(prices, "name") else datetime.now()
        for ticker, shares in list(portfolio.positions.items()):
            if shares <= 1e-9:
                continue
            price_ref = prices.get(ticker, 0.0)
            if price_ref <= 0:
                continue
            shares_to_sell, _ = Portfolio._cap_shares_for_liquidity_with_meta(
                requested_shares=shares,
                ticker=ticker,
                price_ref=price_ref,
                volumes=volumes,
                max_volume_participation=self.config.max_volume_participation,
                min_daily_dollar_volume=self.config.min_daily_dollar_volume,
                liquidity_on_missing_volume=self.config.liquidity_on_missing_volume,
            )
            if shares_to_sell <= 1e-9:
                continue
            trades.append(
                Portfolio._build_trade(
                    current_date=current_date,
                    ticker=ticker,
                    action="SELL",
                    shares=shares_to_sell,
                    price_ref=price_ref,
                    cost_model=cost_resolver.for_ticker(ticker),
                )
            )
        portfolio.execute_sells(trades)
        return trades

    def _validate_invariants(self, tax_summary: TaxSummary) -> None:
        """
        Validate metric invariants (sanity checks).

        Raises warnings if invariants are violated (does not halt execution).

        Invariants checked:
        - final_value_gross >= final_value_net_realized >= final_value_net_liquidation
        - total_tax_paid >= 0
        - tax_paid_liquidation >= 0 (when net_liquidation mode)
        """
        import warnings

        ts = tax_summary
        tolerance = 0.01  # Allow 1 cent tolerance for floating point

        # Invariant 1: Gross >= Net Realized >= Net Liquidation
        if ts.final_value_gross + tolerance < ts.final_value_net_realized:
            warnings.warn(
                f"Invariant violation: final_value_gross ({ts.final_value_gross:.2f}) < "
                f"final_value_net_realized ({ts.final_value_net_realized:.2f}) "
                f"for strategy {self.strategy.name}",
                RuntimeWarning
            )

        if ts.final_value_net_realized + tolerance < ts.final_value_net_liquidation:
            warnings.warn(
                f"Invariant violation: final_value_net_realized ({ts.final_value_net_realized:.2f}) < "
                f"final_value_net_liquidation ({ts.final_value_net_liquidation:.2f}) "
                f"for strategy {self.strategy.name}",
                RuntimeWarning
            )

        # Invariant 2: Total tax paid must be non-negative
        if ts.total_tax_paid < -tolerance:
            warnings.warn(
                f"Invariant violation: total_tax_paid ({ts.total_tax_paid:.2f}) < 0 "
                f"for strategy {self.strategy.name}",
                RuntimeWarning
            )

        # Invariant 3: Liquidation tax must be non-negative (when applicable)
        if ts.metric_basis == "net_liquidation" and ts.tax_paid_liquidation < -tolerance:
            warnings.warn(
                f"Invariant violation: tax_paid_liquidation ({ts.tax_paid_liquidation:.2f}) < 0 "
                f"for strategy {self.strategy.name}",
                RuntimeWarning
            )

    def _process_period_dividends(
        self,
        portfolio: Portfolio,
        dividends_df: pd.DataFrame,
        prev_date: pd.Timestamp,
        current_date: pd.Timestamp,
        prices: pd.Series,
        drip_enabled: bool,
        tax_model: Optional[GermanTaxModel],
    ) -> List[DividendEvent]:
        """
        Process dividends for a period and perform optional DRIP reinvestment.

        Args:
            portfolio: Current portfolio state
            dividends_df: DataFrame of dividend data
            prev_date: Start of period (exclusive)
            current_date: End of period (inclusive)
            prices: Current prices for reinvestment

        Returns:
            List of DividendEvent objects for the period
        """
        events = []

        # Find dividends in this period
        div_mask = (dividends_df.index > prev_date) & (dividends_df.index <= current_date)
        period_dividends = dividends_df.loc[div_mask]

        for div_date, div_row in period_dividends.iterrows():
            for ticker in portfolio.positions:
                if ticker in div_row.index and div_row[ticker] > 0:
                    shares_held = portfolio.positions[ticker]
                    dividend_per_share = div_row[ticker]
                    gross_amount = shares_held * dividend_per_share

                    tax_paid = 0.0
                    net_amount = gross_amount
                    if tax_model is not None:
                        _, _eq = self._instrument_tax_flags(ticker)
                        tax_result = tax_model.apply_dividend(
                            amount=gross_amount,
                            payout_date=div_date.date() if hasattr(div_date, "date") else div_date,
                            is_equity_fund=_eq,
                        )
                        tax_paid = tax_result.tax_due
                        net_amount = gross_amount - tax_paid

                    # Get price for reinvestment
                    reinvest_price = prices.get(ticker, 0)
                    shares_purchased = 0.0
                    if drip_enabled and reinvest_price > 0:
                        # Calculate shares to purchase with dividend (net after tax)
                        shares_purchased = net_amount / reinvest_price
                        portfolio.positions[ticker] += shares_purchased
                    else:
                        # DRIP disabled OR no price available - add to cash as fallback
                        portfolio.cash += net_amount

                    # Record dividend event
                    events.append(DividendEvent(
                        date=div_date,
                        ticker=ticker,
                        dividend_per_share=dividend_per_share,
                        shares_held=shares_held,
                        gross_amount=gross_amount,
                        tax_paid=tax_paid,
                        net_amount=net_amount,
                        shares_purchased=shares_purchased,
                        reinvest_price=reinvest_price,
                    ))

                    if TRUTH_TEST:
                        reinvest_note = "cash" if not drip_enabled else f"reinvest={shares_purchased:.4f} shares"
                        print(f"[TRUTH] DIV @ {div_date.strftime('%Y-%m-%d')}: "
                              f"{ticker} div={dividend_per_share:.4f}/share, "
                              f"gross={gross_amount:.2f}, tax={tax_paid:.2f}, net={net_amount:.2f}, {reinvest_note}")

        return events

    def _instrument_tax_flags(self, ticker: str) -> Tuple[str, bool]:
        """Resolve (instrument_class, is_equity_fund) for a ticker.

        Falls back to the legacy hardcoded defaults ("general", True) when no
        equity_fund_map is configured, OR when the ticker is absent from the
        map -> guarantees bit-identical results for existing strategies/tests.
        A ticker mapped to False is treated as a single stock: no
        Teilfreistellung (full 26.375%) and instrument_class "equity" so losses
        route to the Aktienverlusttopf (per German tax law).
        """
        # The explicit two-axis map takes PRECEDENCE (decouples Teilfreistellung from
        # the loss-pot class). Without a match, everything falls back to the legacy
        # path -> bit-identical for existing strategies/tests.
        tm = getattr(self.config, "tax_treatment_map", None)
        if tm and ticker in tm:
            inst_class, is_eq = tm[ticker]
            return inst_class, bool(is_eq)

        m = getattr(self.config, "equity_fund_map", None)
        if not m:
            return "general", True  # legacy default — unchanged
        is_eq = m.get(ticker, True)  # unknown tickers keep legacy ETF treatment
        return ("general" if is_eq else "equity"), is_eq

    def _apply_taxes_to_sells(
        self,
        tax_model: GermanTaxModel,
        portfolio: Portfolio,
        sell_trades: List[Trade],
        date: pd.Timestamp,
    ) -> float:
        """
        Apply tax to SELL trades and deduct from portfolio cash.

        Args:
            tax_model: German tax model instance
            portfolio: Current portfolio (cash will be modified)
            sell_trades: List of SELL trades to apply tax to
            date: Trade date

        Returns:
            Total tax paid on these trades
        """
        tax_paid = 0.0

        for trade in sell_trades:
            if trade.shares <= 0:
                continue
            try:
                _ic, _eq = self._instrument_tax_flags(trade.ticker)
                tax_result = tax_model.apply_sale(
                    ticker=trade.ticker,
                    shares_sold=trade.shares,
                    sale_price=(trade.value_exec - trade.costs) / trade.shares,
                    sale_date=trade.date.date() if hasattr(trade.date, 'date') else trade.date,
                    instrument_class=_ic,
                    is_equity_fund=_eq,
                )
                # Store tax paid on the trade for reconciliation
                trade.tax_paid_trade = tax_result.tax_due
                # Deduct tax from cash immediately
                if tax_result.tax_due > 0:
                    portfolio.cash -= tax_result.tax_due
                    tax_paid += tax_result.tax_due
                    if TRUTH_TEST:
                        print(f"[TRUTH] {self.strategy.name} @ {date.date()}: "
                              f"TAX on {trade.ticker}: €{tax_result.tax_due:.2f} "
                              f"(gain: €{tax_result.taxable_gain.raw_gain:.2f}, "
                              f"loss_pot_offset: €{tax_result.loss_pot_offset:.2f})")
                elif tax_result.added_to_loss_pot > 0:
                    if TRUTH_TEST:
                        print(f"[TRUTH] {self.strategy.name} @ {date.date()}: "
                              f"LOSS on {trade.ticker}: €{tax_result.added_to_loss_pot:.2f} "
                              f"added to loss pot")
            except NoLotsError:
                # No position to sell (normal on the first rebalance)
                pass
            except InsufficientSharesError as exc:
                # A real bookkeeping inconsistency -- surface it, don't swallow it.
                warnings.warn(f"Steuer-Inkonsistenz (Trade {trade.ticker}): {exc}", RuntimeWarning)

        return tax_paid

    def _build_tax_summary(
        self,
        tax_model: Optional[GermanTaxModel],
        trades: List[Trade],
        metrics_gross: "Metrics",
        metrics_net: Optional["Metrics"],
        equity_curve_gross: pd.Series,
        final_value_gross: float,
        final_value_net_realized: float,
        final_value_net_liquidation: float,
        tax_paid_liquidation: float,
        unrealized_gain_at_end: float,
    ) -> TaxSummary:
        """
        Build the TaxSummary object from tax model state and final values.

        Args:
            tax_model: German tax model (or None if tax disabled)
            trades: List of all trades
            metrics_gross: Gross metrics
            metrics_net: Net metrics (may be None)
            equity_curve_gross: Gross equity curve
            final_value_gross: Final gross value
            final_value_net_realized: Final net realized value
            final_value_net_liquidation: Final net liquidation value
            tax_paid_liquidation: Tax paid on virtual liquidation
            unrealized_gain_at_end: Unrealized gain at end

        Returns:
            TaxSummary object
        """
        initial_capital = self.config.initial_capital
        start_date = equity_curve_gross.index[0]
        end_date = equity_curve_gross.index[-1]
        years = (end_date - start_date).days / 365.25

        if tax_model is not None:
            # Aggregate across all years
            all_years = set(tax_model._annual_tax_results.keys())
            for trade in trades:
                trade_year = trade.date.year if hasattr(trade.date, 'year') else trade.date.date().year
                all_years.add(trade_year)

            total_gains = 0.0
            total_losses = 0.0
            total_exemption = 0.0

            for year in all_years:
                summary = tax_model.get_annual_summary(year)
                total_gains += summary.total_gains
                total_losses += summary.total_losses
                total_exemption += summary.exemption_used

            net_realized = total_gains + total_losses
            effective_rate = (tax_model.total_tax_paid / net_realized) if net_realized > 0 else 0.0

            # Tax Drag calculations
            tax_drag_cagr_pp = (metrics_gross.cagr - metrics_net.cagr) * 100 if metrics_net else 0.0
            tax_drag_final_pct = (1 - final_value_net_realized / final_value_gross) * 100 if final_value_gross > 0 else 0.0

            # Calculate CAGR variants
            if years > 0 and initial_capital > 0:
                cagr_gross = (final_value_gross / initial_capital) ** (1 / years) - 1
                cagr_net_realized = (final_value_net_realized / initial_capital) ** (1 / years) - 1
                cagr_net_liquidation = (final_value_net_liquidation / initial_capital) ** (1 / years) - 1
            else:
                cagr_gross = cagr_net_realized = cagr_net_liquidation = 0.0

            return TaxSummary(
                tax_enabled=True,
                tax_accounting_mode="cash_effective",
                metric_basis=self.config.metric_basis,
                total_tax_paid=tax_model.total_tax_paid,
                total_realized_gains=total_gains,
                total_realized_losses=total_losses,
                net_realized_gain=net_realized,
                exemption_used=total_exemption,
                effective_tax_rate=effective_rate,
                tax_drag_cagr_pp=tax_drag_cagr_pp,
                tax_drag_final_pct=tax_drag_final_pct,
                loss_pot_used=tax_model._total_loss_pot_used,
                loss_pot_added=tax_model._total_loss_pot_added,
                loss_pot_equity_final=tax_model.loss_pot_equity,
                loss_pot_general_final=tax_model.loss_pot_general,
                tax_rate=self.config.tax_rate,
                partial_exemption=self.config.tax_partial_exemption,
                exemption_amount=self.config.tax_exemption_amount,
                tax_paid_liquidation=tax_paid_liquidation,
                unrealized_gain_at_end=unrealized_gain_at_end,
                final_value_gross=final_value_gross,
                final_value_net_realized=final_value_net_realized,
                final_value_net_liquidation=final_value_net_liquidation,
                initial_capital=initial_capital,
                years=years,
                cagr_gross=cagr_gross,
                cagr_net_realized=cagr_net_realized,
                cagr_net_liquidation=cagr_net_liquidation,
            )
        else:
            # Tax disabled
            cagr_gross = metrics_gross.cagr if metrics_gross else 0.0

            return TaxSummary(
                tax_enabled=False,
                tax_accounting_mode="cash_effective",
                metric_basis=self.config.metric_basis,
                final_value_gross=final_value_gross,
                final_value_net_realized=final_value_gross,
                final_value_net_liquidation=final_value_gross,
                initial_capital=initial_capital,
                years=years,
                cagr_gross=cagr_gross,
                cagr_net_realized=cagr_gross,
                cagr_net_liquidation=cagr_gross,
            )
