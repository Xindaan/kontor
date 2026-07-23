"""
Signal generation for live/forward trading.

This module provides functionality to generate BUY/SELL/HOLD signals
for portfolio management, as opposed to historical backtesting.

Key differences from backtesting:
- Uses current universe (no historical point-in-time restrictions)
- Compares against current portfolio positions
- Generates actionable trade recommendations
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Literal, Optional, Any, Set, Tuple
import json
import math

import pandas as pd

from backtest.strategy import Strategy, Allocation
from backtest.assets import detect_currency
from backtest.data import DataLoader, PriceData
from backtest.freshness import STALE_BDAYS, stale_lag_bdays
from backtest.risk.exposure_policy import ExposurePolicyEngine


@dataclass
class TradingSignal:
    """
    A single trading signal for one ticker.

    Attributes:
        ticker: Stock ticker symbol
        action: BUY, SELL, or HOLD
        current_weight: Current portfolio weight (0 if not held)
        target_weight: Target weight according to strategy
        momentum_score: Momentum value (if applicable)
        momentum_rank: Rank in universe by momentum
        reason: Human-readable explanation
    """
    ticker: str
    action: Literal["BUY", "SELL", "HOLD"]
    current_weight: float
    target_weight: float
    momentum_score: Optional[float] = None
    momentum_rank: Optional[int] = None
    reason: str = ""
    current_shares: Optional[float] = None
    target_shares: Optional[float] = None
    shares_delta: Optional[float] = None
    current_value: Optional[float] = None
    target_value: Optional[float] = None
    value_delta: Optional[float] = None
    drift_bps: Optional[float] = None
    drift_in_tolerance: Optional[bool] = None

    @property
    def weight_change(self) -> float:
        """Difference between target and current weight."""
        return self.target_weight - self.current_weight

    @property
    def order_action(self) -> Literal["BUY", "SELL", "HOLD"]:
        """Action implied by sized order delta."""
        if self.shares_delta is None:
            return "HOLD"
        if self.shares_delta > 1e-9:
            return "BUY"
        if self.shares_delta < -1e-9:
            return "SELL"
        return "HOLD"

    @staticmethod
    def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
        """Round float values while preserving None/invalid inputs."""
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return round(parsed, digits)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "ticker": self.ticker,
            "action": self.action,
            "current_weight": round(self.current_weight, 4),
            "target_weight": round(self.target_weight, 4),
            "weight_change": round(self.weight_change, 4),
            "momentum_score": self._round_or_none(self.momentum_score),
            "momentum_rank": self.momentum_rank,
            "reason": self.reason,
            "order_action": self.order_action,
            "current_shares": self._round_or_none(self.current_shares),
            "target_shares": self._round_or_none(self.target_shares),
            "shares_delta": self._round_or_none(self.shares_delta),
            "current_value": self._round_or_none(self.current_value, digits=2),
            "target_value": self._round_or_none(self.target_value, digits=2),
            "value_delta": self._round_or_none(self.value_delta, digits=2),
            "drift_bps": self._round_or_none(self.drift_bps, digits=1),
            "drift_in_tolerance": self.drift_in_tolerance,
        }


@dataclass
class SignalReport:
    """
    Complete signal report for a given date.

    Attributes:
        as_of: Date the signals are generated for
        strategy_name: Name of the strategy used
        strategy_params: Strategy parameters used
        universe_size: Total number of tickers in universe
        signals: All trading signals
    """
    as_of: date
    strategy_name: str
    strategy_params: Dict[str, Any]
    universe_size: int
    signals: List[TradingSignal] = field(default_factory=list)
    rebalance_frequency: str = "monthly"
    next_rebalance: Optional[date] = None
    days_until_rebalance: Optional[int] = None
    portfolio_value: Optional[float] = None
    current_cash_weight: Optional[float] = None
    target_cash_weight: Optional[float] = None
    drift_tolerance: float = 0.005
    missing_prices: List[str] = field(default_factory=list)
    price_warnings: List[str] = field(default_factory=list)
    meta_decision: Optional[Dict[str, Any]] = None
    exposure_policy: Optional[Dict[str, Any]] = None

    @property
    def buys(self) -> List[TradingSignal]:
        """Signals with action=BUY."""
        return [s for s in self.signals if s.action == "BUY"]

    @property
    def sells(self) -> List[TradingSignal]:
        """Signals with action=SELL."""
        return [s for s in self.signals if s.action == "SELL"]

    @property
    def holds(self) -> List[TradingSignal]:
        """Signals with action=HOLD."""
        return [s for s in self.signals if s.action == "HOLD"]

    @property
    def order_buys(self) -> List[TradingSignal]:
        """Order proposals with BUY action."""
        return [s for s in self.signals if s.order_action == "BUY"]

    @property
    def order_sells(self) -> List[TradingSignal]:
        """Order proposals with SELL action."""
        return [s for s in self.signals if s.order_action == "SELL"]

    @property
    def order_holds(self) -> List[TradingSignal]:
        """Signals that currently do not need a sized order."""
        return [s for s in self.signals if s.order_action == "HOLD"]

    @property
    def actionable_orders(self) -> List[TradingSignal]:
        """Order proposals that require execution."""
        return [s for s in self.signals if s.order_action in ("BUY", "SELL")]

    @property
    def gross_weight_drift(self) -> float:
        """Total absolute weight drift across tracked positions."""
        return sum(abs(s.weight_change) for s in self.signals)

    @property
    def max_abs_weight_drift(self) -> float:
        """Largest absolute weight drift."""
        if not self.signals:
            return 0.0
        return max(abs(s.weight_change) for s in self.signals)

    @property
    def drift_out_of_tolerance(self) -> int:
        """Number of positions whose drift exceeds tolerance."""
        return sum(
            1
            for s in self.signals
            if s.drift_in_tolerance is False
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        cash_drift = None
        if self.current_cash_weight is not None and self.target_cash_weight is not None:
            cash_drift = self.target_cash_weight - self.current_cash_weight

        order_rows = [
            {
                "ticker": s.ticker,
                "action": s.order_action,
                "current_shares": TradingSignal._round_or_none(s.current_shares),
                "target_shares": TradingSignal._round_or_none(s.target_shares),
                "shares_delta": TradingSignal._round_or_none(s.shares_delta),
                "current_value": TradingSignal._round_or_none(s.current_value, digits=2),
                "target_value": TradingSignal._round_or_none(s.target_value, digits=2),
                "value_delta": TradingSignal._round_or_none(s.value_delta, digits=2),
                "reason": s.reason,
            }
            for s in self.signals
        ]

        drift_rows = [
            {
                "ticker": s.ticker,
                "current_weight": round(s.current_weight, 4),
                "target_weight": round(s.target_weight, 4),
                "weight_drift": round(s.weight_change, 4),
                "drift_bps": TradingSignal._round_or_none(s.drift_bps, digits=1),
                "in_tolerance": s.drift_in_tolerance,
                "reason": s.reason,
            }
            for s in self.signals
        ]

        return {
            "as_of": self.as_of.isoformat(),
            "strategy_name": self.strategy_name,
            "strategy_params": self.strategy_params,
            "rebalance_frequency": self.rebalance_frequency,
            "next_rebalance": self.next_rebalance.isoformat() if self.next_rebalance else None,
            "days_until_rebalance": self.days_until_rebalance,
            "universe_size": self.universe_size,
            "portfolio_value": TradingSignal._round_or_none(self.portfolio_value, digits=2),
            "current_cash_weight": TradingSignal._round_or_none(self.current_cash_weight),
            "target_cash_weight": TradingSignal._round_or_none(self.target_cash_weight),
            "missing_prices": self.missing_prices,
            "price_warnings": self.price_warnings,
            "exposure_policy": self.exposure_policy,
            "summary": {
                "buys": len(self.buys),
                "sells": len(self.sells),
                "holds": len(self.holds),
                "orders": {
                    "buy": len(self.order_buys),
                    "sell": len(self.order_sells),
                    "hold": len(self.order_holds),
                    "actionable": len(self.actionable_orders),
                },
                "drift": {
                    "tolerance": TradingSignal._round_or_none(self.drift_tolerance),
                    "gross_weight_drift": round(self.gross_weight_drift, 4),
                    "max_abs_weight_drift": round(self.max_abs_weight_drift, 4),
                    "out_of_tolerance": self.drift_out_of_tolerance,
                    "cash_weight_drift": TradingSignal._round_or_none(cash_drift),
                },
            },
            "buys": [s.to_dict() for s in self.buys],
            "sells": [s.to_dict() for s in self.sells],
            "holds": [s.to_dict() for s in self.holds],
            "orders": order_rows,
            "drift_reconciliation": {
                "tolerance": TradingSignal._round_or_none(self.drift_tolerance),
                "cash": {
                    "current_weight": TradingSignal._round_or_none(self.current_cash_weight),
                    "target_weight": TradingSignal._round_or_none(self.target_cash_weight),
                    "weight_drift": TradingSignal._round_or_none(cash_drift),
                },
                "positions": drift_rows,
            },
            "meta_decision": self.meta_decision,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class Portfolio:
    """
    Simple portfolio representation for signal generation.

    Attributes:
        positions: Ticker -> number of shares
        cash: Available cash
        last_rebalance: Date of last rebalance (optional)
    """
    positions: Dict[str, float] = field(default_factory=dict)
    cash: float = 0.0
    last_rebalance: Optional[date] = None
    position_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_path: Optional[str] = None
    raw_payload: Optional[Dict[str, Any]] = None

    @classmethod
    def from_json(cls, path: str) -> "Portfolio":
        """Load portfolio from JSON file.

        Supported formats:
        - Legacy: ``{"positions": {"TICKER": shares}, "cash": ...}``
        - Manual broker files: ``{"positionen": [{"price_ticker": ..., "shares": ...}]}``
          where shares are the source of truth and market values are derived
          from live prices.
        """
        with open(path, "r") as f:
            data = json.load(f)

        positions, metadata, cash_from_positions = cls._parse_positions(data)
        cash = cls._parse_cash(data, default=cash_from_positions)
        last_rebalance = None
        if data.get("last_rebalance"):
            last_rebalance = date.fromisoformat(data["last_rebalance"])

        return cls(
            positions=positions,
            cash=cash,
            last_rebalance=last_rebalance,
            position_metadata=metadata,
            source_path=str(path),
            raw_payload=data,
        )

    @staticmethod
    def _parse_cash(data: Dict[str, Any], *, default: float = 0.0) -> float:
        for key in ("cash", "cash_eur", "eur_cash", "liquiditaet_eur"):
            if key in data and data.get(key) is not None:
                try:
                    return float(data.get(key) or 0.0)
                except (TypeError, ValueError):
                    return 0.0
        return float(default or 0.0)

    @classmethod
    def _parse_positions(
        cls,
        data: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]], float]:
        raw_positions = data.get("positions")
        if raw_positions is None:
            raw_positions = data.get("positionen", {})

        positions: Dict[str, float] = {}
        metadata: Dict[str, Dict[str, Any]] = {}
        cash_from_positions = 0.0

        if isinstance(raw_positions, dict):
            for key, value in raw_positions.items():
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("price_ticker", key)
                    ticker, shares, cash_value = cls._parse_position_row(row)
                    if cash_value:
                        cash_from_positions += cash_value
                    if ticker is None:
                        continue
                    positions[ticker] = positions.get(ticker, 0.0) + shares
                    metadata[ticker] = row
                else:
                    try:
                        shares = float(value or 0.0)
                    except (TypeError, ValueError):
                        continue
                    ticker = str(key).strip().upper()
                    if ticker:
                        positions[ticker] = positions.get(ticker, 0.0) + shares
            return positions, metadata, cash_from_positions

        if isinstance(raw_positions, list):
            for raw_row in raw_positions:
                if not isinstance(raw_row, dict):
                    continue
                row = dict(raw_row)
                ticker, shares, cash_value = cls._parse_position_row(row)
                if cash_value:
                    cash_from_positions += cash_value
                if ticker is None:
                    continue
                positions[ticker] = positions.get(ticker, 0.0) + shares
                metadata[ticker] = row

        return positions, metadata, cash_from_positions

    @staticmethod
    def _parse_position_row(row: Dict[str, Any]) -> Tuple[Optional[str], float, float]:
        role = str(row.get("rolle") or row.get("role") or "").strip().lower()
        ticker = str(row.get("price_ticker") or row.get("ticker") or "").strip().upper()
        if ticker in {"", "NONE", "N/A", "CASH", "EUR"} or role in {"cash", "bar", "liquiditaet"}:
            cash_value = (
                row.get("value_eur")
                or row.get("wert_eur")
                or row.get("cash")
                or row.get("shares")
                or 0.0
            )
            try:
                return None, 0.0, float(cash_value or 0.0)
            except (TypeError, ValueError):
                return None, 0.0, 0.0

        try:
            shares = float(row.get("shares") or 0.0)
        except (TypeError, ValueError):
            shares = 0.0
        return ticker, shares, 0.0

    def manual_price_eur(self, ticker: str) -> Optional[float]:
        """
        Return an explicit broker/manual EUR quote for a ticker, if present.

        A legacy "last broker EUR price" field is deliberately NOT consulted:
        it is an audit reference, not a price source, since it can drift from
        the exchange close. Valuation runs on the two-source consensus
        instead.
        """
        meta = self.position_metadata.get(str(ticker).upper()) or {}
        for key in (
            "broker_price_eur",
            "price_eur",
            "last_price_eur",
            "manual_price_eur",
        ):
            if meta.get(key) is None:
                continue
            try:
                value = float(meta[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                return value
        return None

    @staticmethod
    def _row_pence(currency_raw: str) -> bool:
        """True if a portfolio-row currency string denotes London pence (GBp/GBX).

        Case-sensitive on the GBp/GBP pair: "GBp" is pence, "GBP" is pounds.
        `.upper()` would collapse the two, which is exactly the mistake that
        once made every ".L" ticker look like pence.
        """
        cur = currency_raw.strip()
        if not cur or cur == "GBP":
            return False
        if cur == "GBp":
            return True
        return cur.lower() in {"gbx", "gbpence", "gbppence", "gbp_pence", "pence"}

    def _price_line_is_pence(self, ticker_norm: str, meta: Dict[str, Any]) -> bool:
        """Decide whether a live price line is quoted in pence.

        Source of truth is the same per-ticker detection the DataLoader uses
        (`assets.detect_currency`), NOT a blanket ".L"==pence rule -- the LSE
        lists USD, GBP, and pence lines of the same underlying ETP (e.g. a
        USD-denominated LSE ETP line and a pence-denominated LSE line for the
        same product). An explicit pence marker on the portfolio row wins,
        because detect_currency only knows pence for tickers it has an
        override for (e.g. GLEN.L resolves to GBP via the suffix, not GBp).
        """
        currency_raw = str(meta.get("waehrung") or meta.get("currency") or "")
        if self._row_pence(currency_raw):
            return True
        return detect_currency(ticker_norm) == "GBp"

    def price_scale_for(self, ticker: str) -> float:
        """Return a raw-price scale before FX conversion for live valuation.

        Pence lines are divided to GBP here so the downstream GBP->EUR conversion
        applies exactly once. Which lines are pence is decided per ticker (see
        `_price_line_is_pence`), not by the ".L" suffix.
        """
        ticker_norm = str(ticker).upper()
        meta = self.position_metadata.get(ticker_norm) or {}
        if meta.get("price_scale") is not None:
            try:
                scale = float(meta["price_scale"])
            except (TypeError, ValueError):
                scale = 1.0
            return scale if math.isfinite(scale) and scale > 0 else 1.0

        if self._price_line_is_pence(ticker_norm, meta):
            return 0.01
        return 1.0

    def currency_override_for(self, ticker: str) -> Optional[str]:
        """Return the FX currency override for live valuation of a ticker.

        Pence lines report GBP (price_scale_for already divided them to GBP, so a
        "GBp" label would divide by 100 a second time). Every other line keeps the
        loader's per-ticker currency (which is `detect_currency`) -- the portfolio
        row `waehrung` is NOT trusted as the currency source here, because it can
        name the desired valuation currency rather than the price line's own
        (e.g. a USD-denominated LSE ETP line can carry waehrung="EUR" while
        its price_ticker prints in USD).
        """
        ticker_norm = str(ticker).upper()
        meta = self.position_metadata.get(ticker_norm) or {}
        if self._price_line_is_pence(ticker_norm, meta):
            return "GBP"
        return None

    def price_validation_warnings(self, prices: Dict[str, float]) -> List[str]:
        """Build explicit live-price warnings for manually maintained broker rows."""
        warnings: List[str] = []
        for ticker, meta in sorted(self.position_metadata.items()):
            if str(ticker).upper() == "VERIFY" or meta.get("ticker_verify") is True:
                price = prices.get(ticker)
                if ticker == "VERIFY":
                    warnings.append(
                        "VERIFY ticker placeholder in portfolio file; replace price_ticker before valuation."
                    )
                    continue
                if price is None:
                    warnings.append(f"{ticker}: ticker_verify=true but no live EUR price was loaded.")
                elif not math.isfinite(float(price)) or float(price) <= 0:
                    warnings.append(f"{ticker}: ticker_verify=true returned invalid price {price!r}.")
        return warnings

    def get_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate current portfolio weights based on prices.

        Args:
            prices: Ticker -> current price

        Returns:
            Ticker -> weight (0-1)
        """
        position_values = self.get_position_values(prices)
        total_value = self.cash + sum(position_values.values())

        if total_value <= 0:
            return {}

        # Convert to weights
        return {
            ticker: value / total_value
            for ticker, value in position_values.items()
        }

    def get_position_values(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate position market values for tickers with valid prices."""
        position_values: Dict[str, float] = {}
        for ticker, shares in self.positions.items():
            if shares <= 0:
                continue
            price_raw = prices.get(ticker)
            if price_raw is None:
                continue
            try:
                price = float(price_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(price) or price <= 0:
                continue
            value = float(shares) * price
            if math.isfinite(value) and value > 0:
                position_values[ticker] = value
        return position_values

    def total_value(self, prices: Dict[str, float]) -> float:
        """Current portfolio value including cash."""
        return self.cash + sum(self.get_position_values(prices).values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "positions": self.positions,
            "cash": self.cash,
            "last_rebalance": self.last_rebalance.isoformat() if self.last_rebalance else None,
        }


class SignalGenerator:
    """
    Generates trading signals by comparing strategy allocation with current portfolio.

    This is the main class for live/forward signal generation.
    """

    def __init__(
        self,
        strategy: Strategy,
        portfolio: Optional[Portfolio] = None,
        skip_failed: bool = True,
        drift_tolerance: float = 0.005,
        exposure_policy: Optional[Dict[str, Any]] = None,
        external_features_provider: Any = None,
        analyst_dataset: Optional[str] = None,
        news_dataset: Optional[str] = None,
        ml_dataset: Optional[str] = None,
    ):
        """
        Initialize signal generator.

        Args:
            strategy: Strategy instance to generate signals from
            portfolio: Current portfolio (optional, for comparison)
            skip_failed: If True, skip tickers that fail to download
            drift_tolerance: Absolute weight tolerance for drift diagnostics
            external_features_provider: Optional Phase-B provider used to
                enrich signal reasons with analyst consensus. Accepts a
                singular ExternalFeaturesLoader or a
                MultiDatasetFeaturesProvider. ``None`` keeps the
                Phase-A reason text unchanged.
            analyst_dataset: Optional dataset_id to query on a
                MultiDatasetFeaturesProvider. ``None`` uses the
                provider's default dataset.
        """
        self.strategy = strategy
        self.portfolio = portfolio or Portfolio()
        self.skip_failed = skip_failed
        self.drift_tolerance = float(drift_tolerance)
        self.exposure_policy_engine = ExposurePolicyEngine.from_raw(exposure_policy)
        self._external_features_provider = external_features_provider
        self._analyst_dataset = analyst_dataset
        self._news_dataset = news_dataset
        self._ml_dataset = ml_dataset
        if self.drift_tolerance < 0:
            raise ValueError("drift_tolerance must be >= 0")

    def generate(
        self,
        as_of: Optional[date] = None,
        lookback_days: Optional[int] = None,
    ) -> SignalReport:
        """
        Generate trading signals for the given date.

        Args:
            as_of: Date to generate signals for (default: today)
            lookback_days: Calendar days of history to load. If None, auto-detect
                          from strategy (lookback_days, min_history, momentum_lookback).

        Returns:
            SignalReport with all signals
        """
        as_of = as_of or date.today()

        # Auto-detect required history from strategy
        if lookback_days is None:
            # Check various parameter names strategies might use
            strategy_lookback = getattr(self.strategy, 'lookback_days', None)
            strategy_min_hist = getattr(self.strategy, 'min_history', None)
            strategy_mom_lb = getattr(self.strategy, 'momentum_lookback', None)
            strategy_mom_lb2 = getattr(self.strategy, '_mom_lb', None)

            trading_days_needed = max(
                strategy_lookback or 252,
                strategy_min_hist or 252,
                strategy_mom_lb or 0,
                strategy_mom_lb2 or 0,
            )
            # Convert trading days to calendar days (~1.4x factor + buffer)
            lookback_days = int(trading_days_needed * 1.5) + 60

        # Load price data
        # We need enough history for momentum calculation
        from datetime import timedelta
        start_date = as_of - timedelta(days=lookback_days)

        held_tickers = [
            ticker
            for ticker, shares in self.portfolio.positions.items()
            if shares > 0
        ]
        tickers_to_load = sorted(set(self.strategy.assets) | set(held_tickers))
        if self.exposure_policy_engine.enabled:
            tickers_to_load = sorted(set(tickers_to_load) | set(self.exposure_policy_engine.config.required_assets))
        price_data = DataLoader.yahoo(
            tickers=tickers_to_load,
            start=start_date.isoformat(),
            end=as_of.isoformat(),
            currency="EUR",
            align="ffill",  # Use ffill to keep all tickers
            skip_failed=self.skip_failed,
        )
        prices_df = price_data.prices

        # Get current prices (last row)
        current_prices, price_warnings = self._current_prices_for_portfolio(price_data)

        # Get current portfolio weights
        current_weights = self.portfolio.get_weights(current_prices)

        # Get target allocation from strategy
        strategy_prices = prices_df[
            [ticker for ticker in self.strategy.assets if ticker in prices_df.columns]
        ]
        target_allocation = self.strategy.signal(as_of, strategy_prices)
        exposure_policy_payload = None
        if self.exposure_policy_engine.enabled:
            exposure_decision = self.exposure_policy_engine.apply(
                target=target_allocation,
                historical_prices=prices_df,
            )
            target_allocation = exposure_decision.allocation
            exposure_policy_payload = exposure_decision.to_dict()
            exposure_policy_payload["diagnostics"] = self.exposure_policy_engine.diagnostics()

        # Calculate momentum scores for context
        momentum_scores = self._calculate_momentum_scores(strategy_prices)

        # Resolve analyst-consensus suffixes per ticker (Phase B) +
        # news-consensus suffixes (Phase C).
        union_tickers = sorted(
            set(target_allocation.weights.keys()) | set(current_weights.keys())
        )
        analyst_suffixes = self._collect_analyst_suffixes(
            as_of=as_of,
            tickers=union_tickers,
        )
        news_suffixes = self._collect_news_suffixes(
            as_of=as_of,
            tickers=union_tickers,
        )
        ml_suffixes = self._collect_ml_suffixes(
            as_of=as_of,
            tickers=union_tickers,
        )

        # Generate signals by comparing current vs target
        signals = self._compare_allocations(
            current_weights=current_weights,
            target_allocation=target_allocation,
            momentum_scores=momentum_scores,
            analyst_suffixes=analyst_suffixes,
            news_suffixes=news_suffixes,
            ml_suffixes=ml_suffixes,
        )

        portfolio_value = self.portfolio.total_value(current_prices)
        self._enrich_signals_with_orders(signals, current_prices, portfolio_value)
        current_cash_weight = (
            self.portfolio.cash / portfolio_value if portfolio_value > 0 else None
        )
        target_cash_weight = target_allocation.cash
        tracked_tickers = set(target_allocation.weights.keys()) | set(held_tickers)
        missing_prices = sorted(ticker for ticker in tracked_tickers if ticker not in current_prices)

        # Calculate next rebalance date
        next_rebalance, days_until = self._calculate_next_rebalance(
            as_of,
            self.strategy.rebalance_frequency,
        )

        return SignalReport(
            as_of=as_of,
            strategy_name=self.strategy.display_name,
            strategy_params=getattr(self.strategy, 'params', {}),
            universe_size=len(self.strategy.assets),
            signals=signals,
            rebalance_frequency=self.strategy.rebalance_frequency,
            next_rebalance=next_rebalance,
            days_until_rebalance=days_until,
            portfolio_value=portfolio_value if portfolio_value > 0 else None,
            current_cash_weight=current_cash_weight,
            target_cash_weight=target_cash_weight,
            drift_tolerance=self.drift_tolerance,
            missing_prices=missing_prices,
            price_warnings=price_warnings,
            exposure_policy=exposure_policy_payload,
        )

    def _current_prices_for_portfolio(self, price_data: "PriceData") -> Tuple[Dict[str, float], List[str]]:
        """Return latest EUR prices for portfolio/order sizing.

        Strategy calculations keep using the raw historical frame. This method
        applies live-valuation conventions only to the current-price snapshot:
        manual broker-price overrides, LSE pence scaling, and FX conversion.
        """
        prices_df = price_data.prices
        if prices_df.empty:
            return {}, ["No live price rows returned."]

        valuation_prices = prices_df.copy()
        currency_map = dict(price_data.currency)

        for ticker in valuation_prices.columns:
            scale = self.portfolio.price_scale_for(ticker)
            if scale != 1.0:
                valuation_prices[ticker] = valuation_prices[ticker] * scale
            currency_override = self.portfolio.currency_override_for(ticker)
            if currency_override:
                currency_map[ticker] = currency_override

        if price_data.fx_rates is not None:
            try:
                valuation_prices = PriceData(
                    prices=valuation_prices,
                    currency=currency_map,
                    fx_rates=price_data.fx_rates,
                ).in_eur()
            except ValueError:
                # Keep raw prices if a non-supported FX override appears; the
                # warning below will surface missing/invalid current prices.
                pass

        current_prices: Dict[str, float] = {}
        latest_prices = valuation_prices.iloc[-1].to_dict()
        for ticker, raw_value in latest_prices.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                current_prices[str(ticker).upper()] = value

        manuell = set()
        for ticker in self.portfolio.positions:
            manual_price = self.portfolio.manual_price_eur(ticker)
            if manual_price is not None:
                key = str(ticker).upper()
                current_prices[key] = manual_price
                manuell.add(key)

        warnings = self.portfolio.price_validation_warnings(current_prices)
        # Freshness of the valuation prices. `align="ffill"` + `iloc[-1]`
        # silently carries a frozen ticker forward to the end of the frame
        # and makes it look like today's price -- that exact mechanism once
        # inverted a live signal. `last_print` keeps the real print date from
        # the loader; here it becomes a WARNING, which the CLI's refusal gate
        # then enforces (no order plan on a stale price).
        warnings.extend(self._stale_price_warnings(price_data, skip=manuell))
        return current_prices, warnings

    def _stale_price_warnings(self, price_data: "PriceData",
                              skip: Optional[Set[str]] = None) -> List[str]:
        """Warn where the last REAL print is too old.

        Only HELD positions are checked: a stale candidate we don't hold
        can't trigger an order. Manually set prices are excluded -- those
        don't come from the loader.
        """
        skip = skip or set()
        frame_end = price_data.prices.index.max() if not price_data.prices.empty else None
        if frame_end is None or not price_data.last_print:
            return []
        out: List[str] = []
        for ticker in sorted(self.portfolio.positions):
            key = str(ticker).upper()
            if key in skip:
                continue
            last = next((d for t, d in price_data.last_print.items()
                         if str(t).upper() == key), None)
            if last is None:
                continue
            lag = stale_lag_bdays(last, today=pd.Timestamp(frame_end).date())
            if lag >= STALE_BDAYS:
                out.append(
                    "%s: last real price print %s is %d trading days before the "
                    "Frame-Ende %s (ffill schreibt ihn fort) -> Bewertung nicht belegbar."
                    % (key, pd.Timestamp(last).date().isoformat(), lag,
                       pd.Timestamp(frame_end).date().isoformat()))
        return out

    def _enrich_signals_with_orders(
        self,
        signals: List[TradingSignal],
        current_prices: Dict[str, float],
        portfolio_value: float,
    ) -> None:
        """Attach order sizing and drift diagnostics to each signal."""
        for signal in signals:
            signal.drift_bps = signal.weight_change * 10_000.0
            signal.drift_in_tolerance = abs(signal.weight_change) <= self.drift_tolerance

            current_shares = float(self.portfolio.positions.get(signal.ticker, 0.0) or 0.0)
            signal.current_shares = current_shares

            if portfolio_value <= 0:
                continue

            price = current_prices.get(signal.ticker)
            if price is None or not math.isfinite(price) or price <= 0:
                continue

            current_value = current_shares * price
            target_value = signal.target_weight * portfolio_value
            target_shares = target_value / price
            shares_delta = target_shares - current_shares
            value_delta = target_value - current_value

            if abs(shares_delta) < 1e-9:
                shares_delta = 0.0
            if abs(value_delta) < 1e-9:
                value_delta = 0.0

            signal.current_value = current_value
            signal.target_value = target_value
            signal.value_delta = value_delta
            signal.target_shares = target_shares
            signal.shares_delta = shares_delta

    def _calculate_momentum_scores(
        self,
        prices: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate momentum scores for all tickers."""
        scores = {}

        # Try to use strategy's lookback/skip parameters if available
        lookback = getattr(self.strategy, 'lookback_days', 252)
        skip = getattr(self.strategy, 'skip_days', 21)

        for ticker in prices.columns:
            series = prices[ticker].dropna()
            if len(series) < lookback + skip + 1:
                continue

            try:
                end_idx = -skip - 1 if skip > 0 else -1
                start_idx = -lookback - skip

                end_price = series.iloc[end_idx]
                start_price = series.iloc[start_idx]

                if start_price > 0:
                    scores[ticker] = (end_price / start_price) - 1
            except (IndexError, KeyError):
                continue

        return scores

    def _compare_allocations(
        self,
        current_weights: Dict[str, float],
        target_allocation: Allocation,
        momentum_scores: Dict[str, float],
        analyst_suffixes: Optional[Dict[str, str]] = None,
        news_suffixes: Optional[Dict[str, str]] = None,
        ml_suffixes: Optional[Dict[str, str]] = None,
    ) -> List[TradingSignal]:
        """
        Compare current portfolio with target allocation.

        Returns list of TradingSignal objects.
        """
        signals = []
        analyst_suffixes = analyst_suffixes or {}
        news_suffixes = news_suffixes or {}
        ml_suffixes = ml_suffixes or {}

        # Sort momentum scores for ranking
        sorted_momentum = sorted(
            momentum_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        momentum_ranks = {
            ticker: rank + 1
            for rank, (ticker, _) in enumerate(sorted_momentum)
        }

        # All tickers to consider (union of current and target)
        all_tickers = set(current_weights.keys()) | set(target_allocation.weights.keys())

        for ticker in all_tickers:
            current = current_weights.get(ticker, 0.0)
            target = target_allocation.get(ticker, 0.0)
            momentum = momentum_scores.get(ticker)
            rank = momentum_ranks.get(ticker)

            # Determine action
            if target > 0 and current == 0:
                action = "BUY"
                reason = f"New position (rank {rank})" if rank else "New position"
            elif target == 0 and current > 0:
                action = "SELL"
                reason = f"Exit position (rank {rank})" if rank else "Exit position"
            elif target > 0 and current > 0:
                action = "HOLD"
                if abs(target - current) > 0.01:
                    reason = f"Rebalance: {current:.1%} → {target:.1%}"
                else:
                    reason = "On target"
            else:
                # Both zero - skip
                continue

            suffix = analyst_suffixes.get(ticker)
            if suffix:
                reason = f"{reason}; {suffix}"
            news_suffix = news_suffixes.get(ticker)
            if news_suffix:
                reason = f"{reason}; {news_suffix}"
            ml_suffix = ml_suffixes.get(ticker)
            if ml_suffix:
                reason = f"{reason}; {ml_suffix}"

            signals.append(TradingSignal(
                ticker=ticker,
                action=action,
                current_weight=current,
                target_weight=target,
                momentum_score=momentum,
                momentum_rank=rank,
                reason=reason,
            ))

        # Sort: BUYs first (by rank), then HOLDs, then SELLs
        def sort_key(s: TradingSignal):
            action_order = {"BUY": 0, "HOLD": 1, "SELL": 2}
            return (action_order[s.action], s.momentum_rank or 999)

        signals.sort(key=sort_key)

        return signals

    def _collect_analyst_suffixes(
        self,
        as_of: date,
        tickers: List[str],
    ) -> Dict[str, str]:
        """Build {ticker: 'analyst: BUY|HOLD|SELL (...)' } from the provider.

        Returns an empty dict when no provider is configured or the
        snapshot has no analyst rows for the requested tickers. The
        consensus mapping is deterministic:
        - score >= 0.33 -> BUY
        - score <= -0.33 -> SELL
        - else -> HOLD
        """

        provider = self._external_features_provider
        if provider is None or not tickers:
            return {}
        try:
            if self._analyst_dataset and hasattr(provider, "snapshot_dataset"):
                snap = provider.snapshot_dataset(self._analyst_dataset, as_of=as_of, tickers=tickers)
            else:
                snap = provider.snapshot(as_of=as_of, tickers=tickers)
        except Exception:
            return {}

        if snap is None or getattr(snap, "data", None) is None:
            return {}
        df = snap.data
        if df.empty or "feature_name" not in df.columns:
            return {}
        score_rows = df.loc[df["feature_name"] == "analyst_score"]
        if score_rows.empty:
            return {}

        suffixes: Dict[str, str] = {}
        for ticker, group in score_rows.groupby("ticker"):
            try:
                score = float(group["feature_value"].iloc[-1])
            except (TypeError, ValueError):
                continue
            if score >= 0.33:
                consensus = "BUY"
            elif score <= -0.33:
                consensus = "SELL"
            else:
                consensus = "HOLD"
            suffixes[str(ticker).upper()] = (
                f"analyst: {consensus} consensus (score={score:.2f})"
            )
        return suffixes

    def _collect_news_suffixes(
        self,
        as_of: date,
        tickers: List[str],
    ) -> Dict[str, str]:
        """Phase C: build {ticker: 'news: BULLISH|NEUTRAL|BEARISH (...)' }.

        Mirrors :meth:`_collect_analyst_suffixes` but reads the
        ``news_sentiment_score`` rows from the configured news dataset.
        Mapping:
        - score >= 0.33 -> BULLISH
        - score <= -0.33 -> BEARISH
        - else -> NEUTRAL
        """

        provider = self._external_features_provider
        if provider is None or not tickers or not self._news_dataset:
            return {}
        try:
            if hasattr(provider, "snapshot_dataset"):
                snap = provider.snapshot_dataset(
                    self._news_dataset, as_of=as_of, tickers=tickers
                )
            else:
                snap = provider.snapshot(as_of=as_of, tickers=tickers)
        except Exception:
            return {}
        if snap is None or getattr(snap, "data", None) is None:
            return {}
        df = snap.data
        if df.empty or "feature_name" not in df.columns:
            return {}
        score_rows = df.loc[df["feature_name"] == "news_sentiment_score"]
        if score_rows.empty:
            return {}
        suffixes: Dict[str, str] = {}
        for ticker, group in score_rows.groupby("ticker"):
            try:
                score = float(group["feature_value"].iloc[-1])
            except (TypeError, ValueError):
                continue
            if score >= 0.33:
                consensus = "BULLISH"
            elif score <= -0.33:
                consensus = "BEARISH"
            else:
                consensus = "NEUTRAL"
            suffixes[str(ticker).upper()] = (
                f"news: {consensus} sentiment (score={score:.2f})"
            )
        return suffixes

    def _collect_ml_suffixes(
        self,
        as_of: date,
        tickers: List[str],
    ) -> Dict[str, str]:
        """Phase D: build {ticker: 'ml: BULLISH|NEUTRAL|BEARISH (...)' }.

        Mapping mirrors :meth:`_collect_news_suffixes`:
        - score >= 0.33 -> BULLISH
        - score <= -0.33 -> BEARISH
        - else -> NEUTRAL
        """

        provider = self._external_features_provider
        if provider is None or not tickers or not self._ml_dataset:
            return {}
        try:
            if hasattr(provider, "snapshot_dataset"):
                snap = provider.snapshot_dataset(
                    self._ml_dataset, as_of=as_of, tickers=tickers
                )
            else:
                snap = provider.snapshot(as_of=as_of, tickers=tickers)
        except Exception:
            return {}
        if snap is None or getattr(snap, "data", None) is None:
            return {}
        df = snap.data
        if df.empty or "feature_name" not in df.columns:
            return {}
        score_rows = df.loc[df["feature_name"] == "ml_forecast_score"]
        if score_rows.empty:
            return {}
        suffixes: Dict[str, str] = {}
        for ticker, group in score_rows.groupby("ticker"):
            try:
                score = float(group["feature_value"].iloc[-1])
            except (TypeError, ValueError):
                continue
            if score >= 0.33:
                consensus = "BULLISH"
            elif score <= -0.33:
                consensus = "BEARISH"
            else:
                consensus = "NEUTRAL"
            suffixes[str(ticker).upper()] = (
                f"ml: {consensus} forecast (score={score:.2f})"
            )
        return suffixes

    def _calculate_next_rebalance(
        self,
        as_of: date,
        frequency: str,
    ) -> tuple[Optional[date], Optional[int]]:
        """
        Calculate next rebalance date based on frequency.

        Returns (next_date, days_until).
        """
        from calendar import monthrange

        year = as_of.year
        month = as_of.month

        if frequency == "daily":
            # Next trading day (simplified: next calendar day)
            from datetime import timedelta
            next_date = as_of + timedelta(days=1)
            # Skip weekends
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)

        elif frequency == "weekly":
            # Next Friday
            from datetime import timedelta
            days_until_friday = (4 - as_of.weekday()) % 7
            if days_until_friday == 0:
                days_until_friday = 7
            next_date = as_of + timedelta(days=days_until_friday)

        elif frequency == "monthly":
            # Last day of current or next month
            _, last_day = monthrange(year, month)
            if as_of.day >= last_day:
                # Move to next month
                if month == 12:
                    year += 1
                    month = 1
                else:
                    month += 1
                _, last_day = monthrange(year, month)
            next_date = date(year, month, last_day)

        elif frequency == "quarterly":
            # End of quarter
            quarter_end_months = [3, 6, 9, 12]
            for end_month in quarter_end_months:
                if month <= end_month:
                    _, last_day = monthrange(year, end_month)
                    next_date = date(year, end_month, last_day)
                    if next_date > as_of:
                        break
            else:
                # Next year Q1
                _, last_day = monthrange(year + 1, 3)
                next_date = date(year + 1, 3, last_day)

        elif frequency == "yearly":
            # End of year
            next_date = date(year, 12, 31)
            if next_date <= as_of:
                next_date = date(year + 1, 12, 31)

        else:
            return None, None

        days_until = (next_date - as_of).days
        return next_date, days_until


def format_signal_report(report: SignalReport, format: str = "table") -> str:
    """
    Format a SignalReport for display.

    Args:
        report: SignalReport to format
        format: Output format ("table", "compact")

    Returns:
        Formatted string
    """
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append(f"  {report.strategy_name}")
    lines.append(f"  Signals for {report.as_of.isoformat()}")
    lines.append("=" * 70)
    lines.append("")

    # Rebalance info
    if report.next_rebalance:
        lines.append(f"  Rebalance frequency: {report.rebalance_frequency}")
        lines.append(f"  Next rebalance: {report.next_rebalance.isoformat()} ({report.days_until_rebalance} days)")
        lines.append("")

    # Parameters
    if report.strategy_params:
        lines.append("  Parameters:")
        for key, value in report.strategy_params.items():
            if key not in ("warning", "point_in_time"):
                lines.append(f"    {key}: {value}")
        lines.append("")

    # Summary
    lines.append(f"  Universe: {report.universe_size} tickers")
    lines.append(f"  Signals: {len(report.buys)} BUY, {len(report.sells)} SELL, {len(report.holds)} HOLD")
    if report.portfolio_value is not None:
        lines.append(f"  Portfolio value: {report.portfolio_value:,.2f}")
        lines.append(
            f"  Orders: {len(report.order_buys)} BUY, {len(report.order_sells)} SELL, "
            f"{len(report.order_holds)} HOLD ({len(report.actionable_orders)} actionable)"
        )
        lines.append(
            f"  Drift: gross {report.gross_weight_drift:.1%}, max {report.max_abs_weight_drift:.1%}, "
            f"out-of-tolerance {report.drift_out_of_tolerance}/{len(report.signals)} "
            f"(tol {report.drift_tolerance:.1%})"
        )
    if report.current_cash_weight is not None and report.target_cash_weight is not None:
        cash_drift = report.target_cash_weight - report.current_cash_weight
        lines.append(
            f"  Cash: {report.current_cash_weight:.1%} -> {report.target_cash_weight:.1%} "
            f"({cash_drift:+.1%})"
        )
    if report.missing_prices:
        lines.append(f"  Missing prices: {', '.join(report.missing_prices)}")
    if report.price_warnings:
        lines.append("  Price warnings:")
        for warning in report.price_warnings:
            lines.append(f"    - {warning}")
    lines.append("")

    if report.exposure_policy:
        policy = report.exposure_policy
        lines.append("  EXPOSURE POLICY")
        lines.append("  " + "-" * 66)
        lines.append(f"  State:        {policy.get('exposure_state', '-')}")
        lines.append(f"  Raw target:   {policy.get('raw_strategy_target', {})}")
        lines.append(f"  Final target: {policy.get('policy_adjusted_target', {})}")
        lines.append(f"  Execution:    {policy.get('raw_signal_asset') or '-'} -> {policy.get('execution_asset') or '-'}")
        if policy.get("fallback_reason"):
            lines.append(f"  Reason:       {policy.get('fallback_reason')}")
        checks = policy.get("guard_checks") or []
        if checks:
            lines.append("  Guard checks:")
            for check in checks[:8]:
                lines.append(
                    f"    - {check.get('key', '-')}: {check.get('status', '-')} "
                    f"({check.get('detail', '-')})"
                )
        lines.append("")

    if report.meta_decision:
        meta = report.meta_decision
        lines.append("  META DECISION")
        lines.append("  " + "-" * 66)
        lines.append(f"  Current:      {meta.get('current_strategy', '-')}")
        lines.append(f"  Recommended:  {meta.get('recommended_target', '-')}")
        lines.append(f"  Score margin: {meta.get('score_margin', 0.0):+.4f}")
        if meta.get("performance_gap") is not None:
            lines.append(f"  Perf gap:     {meta.get('performance_gap', 0.0):+.4f}")
        if meta.get("decision_rule"):
            lines.append(f"  Rule:         {meta.get('decision_rule', '-')}")
        if meta.get("current_regime_bucket") or meta.get("target_regime_bucket"):
            lines.append(
                f"  Regime:       {meta.get('current_regime_bucket', '-')} -> {meta.get('target_regime_bucket', '-')}"
            )
        lines.append(
            f"  Evidence:     {str(meta.get('evidence_status', '-')).upper()} "
            f"(artifact: {meta.get('evidence_artifact_id') or '-'})"
        )
        if meta.get("conditioned_evidence_status") is not None:
            lines.append(
                f"  Cond. ev.:    {str(meta.get('conditioned_evidence_status', '-')).upper()} "
                f"(windows: {meta.get('conditioned_windows', '-')})"
            )
        lines.append(
            f"  Switch:       {'ALLOWED' if meta.get('switch_allowed') else 'BLOCKED'} "
            f"-> {meta.get('executed_action', '-')}"
        )
        evidence_reasons = meta.get("evidence_reasons") or []
        live_reasons = meta.get("live_reasons") or []
        if evidence_reasons or live_reasons:
            lines.append("  Reasons:")
            for reason in list(live_reasons) + list(evidence_reasons):
                lines.append(f"    - {reason}")
        lines.append("")

    # BUYS
    if report.buys:
        lines.append("  BUYS")
        lines.append("  " + "-" * 66)
        lines.append(f"  {'Ticker':<8} {'Target %':>10} {'Momentum':>12} {'Rank':>6}   Reason")
        lines.append("  " + "-" * 66)
        for s in report.buys:
            mom_str = f"{s.momentum_score:+.1%}" if s.momentum_score is not None else "N/A"
            rank_str = str(s.momentum_rank) if s.momentum_rank else "-"
            lines.append(
                f"  {s.ticker:<8} {s.target_weight:>10.1%} {mom_str:>12} {rank_str:>6}   {s.reason}"
            )
        lines.append("")

    # SELLS
    if report.sells:
        lines.append("  SELLS")
        lines.append("  " + "-" * 66)
        lines.append(f"  {'Ticker':<8} {'Current %':>10} {'Momentum':>12} {'Rank':>6}   Reason")
        lines.append("  " + "-" * 66)
        for s in report.sells:
            mom_str = f"{s.momentum_score:+.1%}" if s.momentum_score is not None else "N/A"
            rank_str = str(s.momentum_rank) if s.momentum_rank else "-"
            lines.append(
                f"  {s.ticker:<8} {s.current_weight:>10.1%} {mom_str:>12} {rank_str:>6}   {s.reason}"
            )
        lines.append("")

    # HOLDS
    if report.holds:
        lines.append("  HOLDS")
        lines.append("  " + "-" * 66)
        lines.append(f"  {'Ticker':<8} {'Weight':>10} {'Momentum':>12} {'Rank':>6}   Reason")
        lines.append("  " + "-" * 66)
        for s in report.holds:
            mom_str = f"{s.momentum_score:+.1%}" if s.momentum_score is not None else "N/A"
            rank_str = str(s.momentum_rank) if s.momentum_rank else "-"
            lines.append(
                f"  {s.ticker:<8} {s.target_weight:>10.1%} {mom_str:>12} {rank_str:>6}   {s.reason}"
            )
        lines.append("")

    # Order proposals
    if report.actionable_orders:
        lines.append("  ORDER PROPOSALS")
        lines.append("  " + "-" * 66)
        lines.append(f"  {'Ticker':<8} {'Order':<6} {'Delta Sh':>10} {'Delta Val':>12}   Current -> Target")
        lines.append("  " + "-" * 66)
        sorted_orders = sorted(
            report.actionable_orders,
            key=lambda s: abs(s.value_delta or 0.0),
            reverse=True,
        )
        for s in sorted_orders:
            shares_delta = f"{s.shares_delta:+.2f}" if s.shares_delta is not None else "N/A"
            value_delta = f"{s.value_delta:+.2f}" if s.value_delta is not None else "N/A"
            if s.current_value is not None and s.target_value is not None:
                current_target = f"{s.current_value:.2f} -> {s.target_value:.2f}"
            else:
                current_target = "N/A"
            lines.append(
                f"  {s.ticker:<8} {s.order_action:<6} {shares_delta:>10} {value_delta:>12}   {current_target}"
            )
        lines.append("")

    # Drift reconciliation
    drift_outliers = [s for s in report.signals if s.drift_in_tolerance is False]
    if drift_outliers:
        lines.append("  DRIFT RECONCILIATION")
        lines.append("  " + "-" * 66)
        lines.append(f"  {'Ticker':<8} {'Current %':>10} {'Target %':>10} {'Drift %':>10} {'Drift bps':>10}")
        lines.append("  " + "-" * 66)
        sorted_outliers = sorted(
            drift_outliers,
            key=lambda s: abs(s.weight_change),
            reverse=True,
        )
        for s in sorted_outliers:
            drift_bps = f"{s.drift_bps:+.0f}" if s.drift_bps is not None else "N/A"
            lines.append(
                f"  {s.ticker:<8} {s.current_weight:>10.1%} {s.target_weight:>10.1%} "
                f"{s.weight_change:>+10.1%} {drift_bps:>10}"
            )
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)
