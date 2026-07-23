"""
Validation module for backtest calculations.

This module provides comprehensive validation checks to ensure:
1. Data integrity (no NaN, valid prices, etc.)
2. Calculation correctness (invariants, consistency)
3. Configuration plausibility (reasonable parameters)
4. State consistency (portfolio, cash, positions)

Validation Categories:
- PRE-RUN: Checks before backtest execution
- POST-RUN: Checks after backtest completion
- INVARIANTS: Mathematical relationships that must always hold
- DATA QUALITY: Input data validation
"""

import os
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtest.backtester import BacktestConfig, BacktestResult, Portfolio, TaxSummary
    from backtest.strategy import Strategy
    from backtest.data import PriceData
    from backtest.metrics import Metrics


class ValidationLevel(Enum):
    """Severity level for validation issues."""
    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # Potential issue, proceed with caution
    ERROR = "error"  # Critical issue, results may be invalid
    FATAL = "fatal"  # Cannot proceed, must fix


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    level: ValidationLevel
    category: str
    message: str
    details: Optional[str] = None

    def __str__(self) -> str:
        prefix = f"[{self.level.value.upper()}] [{self.category}]"
        if self.details:
            return f"{prefix} {self.message}\n  Details: {self.details}"
        return f"{prefix} {self.message}"


@dataclass
class ValidationResult:
    """Collection of validation results."""
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """True if any ERROR or FATAL issues exist."""
        return any(i.level in (ValidationLevel.ERROR, ValidationLevel.FATAL) for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """True if any WARNING issues exist."""
        return any(i.level == ValidationLevel.WARNING for i in self.issues)

    @property
    def is_valid(self) -> bool:
        """True if no ERROR or FATAL issues exist."""
        return not self.has_errors

    def add(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)

    def add_info(self, category: str, message: str, details: str = None) -> None:
        """Add an INFO level issue."""
        self.add(ValidationIssue(ValidationLevel.INFO, category, message, details))

    def add_warning(self, category: str, message: str, details: str = None) -> None:
        """Add a WARNING level issue."""
        self.add(ValidationIssue(ValidationLevel.WARNING, category, message, details))

    def add_error(self, category: str, message: str, details: str = None) -> None:
        """Add an ERROR level issue."""
        self.add(ValidationIssue(ValidationLevel.ERROR, category, message, details))

    def add_fatal(self, category: str, message: str, details: str = None) -> None:
        """Add a FATAL level issue."""
        self.add(ValidationIssue(ValidationLevel.FATAL, category, message, details))

    def emit_warnings(self) -> None:
        """Emit all issues as Python warnings."""
        if not self.issues:
            return

        verbose = os.getenv("BACKTEST_VALIDATION_VERBOSE", "").lower() in {"1", "true", "yes"}
        max_warning_details = int(os.getenv("BACKTEST_VALIDATION_MAX_WARNINGS", "5"))
        warnings_only = [i for i in self.issues if i.level == ValidationLevel.WARNING]
        errors_only = [i for i in self.issues if i.level in (ValidationLevel.ERROR, ValidationLevel.FATAL)]

        for issue in errors_only:
            warnings.warn(str(issue), RuntimeWarning)

        if verbose or len(warnings_only) <= max_warning_details:
            for issue in warnings_only:
                warnings.warn(str(issue), UserWarning)
            return

        summary = self.summary()
        sample_count = max(0, max_warning_details)
        sample_lines = "\n".join(f"- {issue}" for issue in warnings_only[:sample_count])
        remaining = len(warnings_only) - sample_count
        message = (
            f"{summary}\n"
            f"Showing {sample_count} of {len(warnings_only)} warning(s):\n"
            f"{sample_lines}\n"
            f"{remaining} additional warning(s) suppressed. "
            "Set BACKTEST_VALIDATION_VERBOSE=1 to show all warnings."
        )
        warnings.warn(message, UserWarning)

    def raise_on_errors(self) -> None:
        """Raise ValueError if any ERROR or FATAL issues exist."""
        errors = [i for i in self.issues if i.level in (ValidationLevel.ERROR, ValidationLevel.FATAL)]
        if errors:
            error_messages = "\n".join(str(e) for e in errors)
            raise ValueError(f"Validation failed with {len(errors)} error(s):\n{error_messages}")

    def summary(self) -> str:
        """Generate a summary of validation results."""
        if not self.issues:
            return "Validation passed: No issues found"

        counts = {level: 0 for level in ValidationLevel}
        for issue in self.issues:
            counts[issue.level] += 1

        parts = []
        for level in [ValidationLevel.FATAL, ValidationLevel.ERROR, ValidationLevel.WARNING, ValidationLevel.INFO]:
            if counts[level] > 0:
                parts.append(f"{counts[level]} {level.value}")

        status = "FAILED" if self.has_errors else "PASSED"
        return f"Validation {status}: {', '.join(parts)}"

    def __str__(self) -> str:
        if not self.issues:
            return "Validation passed: No issues found"
        return "\n".join([self.summary(), ""] + [str(i) for i in self.issues])


# =============================================================================
# PRE-RUN VALIDATION: Checks before backtest execution
# =============================================================================

def validate_config(config: "BacktestConfig") -> ValidationResult:
    """
    Validate backtest configuration for plausibility.

    Checks:
    - Transaction costs are reasonable (< 5%)
    - Initial capital is positive
    - Tax parameters are valid
    - Rebalance frequency is valid
    """
    result = ValidationResult()

    # Initial capital
    if config.initial_capital <= 0:
        result.add_fatal("config", "Initial capital must be positive",
                        f"Got: {config.initial_capital}")

    # Transaction costs
    if config.costs_pct < 0:
        result.add_error("config", "Transaction costs cannot be negative",
                        f"Got: {config.costs_pct:.4f}")
    elif config.costs_pct > 0.05:
        result.add_warning("config", "Transaction costs seem unusually high (>5%)",
                          f"Got: {config.costs_pct:.2%}")

    # Slippage
    if config.slippage_pct < 0:
        result.add_error("config", "Slippage cannot be negative",
                        f"Got: {config.slippage_pct:.4f}")
    elif config.slippage_pct > 0.02:
        result.add_warning("config", "Slippage seems unusually high (>2%)",
                          f"Got: {config.slippage_pct:.2%}")

    # Optional cost profile structure
    if getattr(config, "cost_profile", None) is not None and not isinstance(config.cost_profile, dict):
        result.add_error("config", "cost_profile must be a dictionary when provided")

    # Execution lag
    execution_lag_days = getattr(config, "execution_lag_days", 0)
    if execution_lag_days < 0:
        result.add_error("config", "execution_lag_days must be >= 0", f"Got: {execution_lag_days}")
    elif execution_lag_days > 5:
        result.add_warning(
            "config",
            "execution_lag_days is unusually high (>5 trading days)",
            f"Got: {execution_lag_days}",
        )

    # Liquidity settings
    max_participation = getattr(config, "max_volume_participation", None)
    if max_participation is not None:
        if max_participation < 0:
            result.add_error(
                "config",
                "max_volume_participation must be >= 0 when provided",
                f"Got: {max_participation}",
            )
        elif max_participation > 1:
            result.add_warning(
                "config",
                "max_volume_participation > 1 means more than 100% daily volume participation",
                f"Got: {max_participation}",
            )

    min_daily_dollar_volume = getattr(config, "min_daily_dollar_volume", 0.0)
    if min_daily_dollar_volume < 0:
        result.add_error(
            "config",
            "min_daily_dollar_volume must be >= 0",
            f"Got: {min_daily_dollar_volume}",
        )

    liquidity_on_missing = getattr(config, "liquidity_on_missing_volume", "allow")
    if liquidity_on_missing not in {"allow", "skip"}:
        result.add_error(
            "config",
            "liquidity_on_missing_volume must be 'allow' or 'skip'",
            f"Got: {liquidity_on_missing}",
        )

    # Risk overlay settings
    risk_overlay = getattr(config, "risk_overlay", None)
    exposure_policy = getattr(config, "exposure_policy", None)
    if exposure_policy is not None:
        if not isinstance(exposure_policy, dict):
            result.add_error("config", "exposure_policy must be a dictionary when provided")

    if risk_overlay is not None:
        if not isinstance(risk_overlay, dict):
            result.add_error("config", "risk_overlay must be a dictionary when provided")
        else:
            max_position = risk_overlay.get("max_position")
            if max_position is not None:
                try:
                    max_position_val = float(max_position)
                    if max_position_val < 0:
                        result.add_error("config", "risk_overlay.max_position must be >= 0")
                    elif max_position_val > 1:
                        result.add_warning(
                            "config",
                            "risk_overlay.max_position > 1 allows allocations above 100%",
                            f"Got: {max_position_val}",
                        )
                except (TypeError, ValueError):
                    result.add_error("config", "risk_overlay.max_position must be numeric")

            turnover_budget = risk_overlay.get("turnover_budget")
            if turnover_budget is not None:
                try:
                    turnover_budget_val = float(turnover_budget)
                    if turnover_budget_val < 0:
                        result.add_error("config", "risk_overlay.turnover_budget must be >= 0")
                    elif turnover_budget_val > 1:
                        result.add_warning(
                            "config",
                            "risk_overlay.turnover_budget > 1 means more than 100% one-way turnover per rebalance",
                            f"Got: {turnover_budget_val}",
                        )
                except (TypeError, ValueError):
                    result.add_error("config", "risk_overlay.turnover_budget must be numeric")

            sector_caps = risk_overlay.get("sector_caps")
            if sector_caps is not None:
                if not isinstance(sector_caps, dict):
                    result.add_error("config", "risk_overlay.sector_caps must be a dictionary")
                else:
                    for sector, cap in sector_caps.items():
                        try:
                            cap_val = float(cap)
                            if cap_val < 0:
                                result.add_error(
                                    "config",
                                    "risk_overlay.sector_caps values must be >= 0",
                                    f"Sector {sector}: {cap_val}",
                                )
                            elif cap_val > 1:
                                result.add_warning(
                                    "config",
                                    "risk_overlay.sector_caps values > 1 allow allocations above 100%",
                                    f"Sector {sector}: {cap_val}",
                                )
                        except (TypeError, ValueError):
                            result.add_error(
                                "config",
                                "risk_overlay.sector_caps values must be numeric",
                                f"Sector {sector}: {cap}",
                            )

            ticker_sectors = risk_overlay.get("ticker_sectors")
            if ticker_sectors is not None and not isinstance(ticker_sectors, dict):
                result.add_error("config", "risk_overlay.ticker_sectors must be a dictionary")

            drawdown_brake = risk_overlay.get("drawdown_brake")
            if drawdown_brake is not None:
                if not isinstance(drawdown_brake, dict):
                    result.add_error("config", "risk_overlay.drawdown_brake must be a dictionary")
                else:
                    threshold = drawdown_brake.get("threshold", None)
                    cash_target = drawdown_brake.get("cash_target", None)
                    release_drawdown = drawdown_brake.get("release_drawdown", None)

                    threshold_val = None
                    if threshold is not None:
                        try:
                            threshold_val = float(threshold)
                            if threshold_val < 0 or threshold_val > 1:
                                result.add_error(
                                    "config",
                                    "risk_overlay.drawdown_brake.threshold must be between 0 and 1",
                                    f"Got: {threshold_val}",
                                )
                        except (TypeError, ValueError):
                            result.add_error(
                                "config",
                                "risk_overlay.drawdown_brake.threshold must be numeric",
                                f"Got: {threshold}",
                            )

                    if cash_target is not None:
                        try:
                            cash_target_val = float(cash_target)
                            if cash_target_val < 0 or cash_target_val > 1:
                                result.add_error(
                                    "config",
                                    "risk_overlay.drawdown_brake.cash_target must be between 0 and 1",
                                    f"Got: {cash_target_val}",
                                )
                        except (TypeError, ValueError):
                            result.add_error(
                                "config",
                                "risk_overlay.drawdown_brake.cash_target must be numeric",
                                f"Got: {cash_target}",
                            )

                    if release_drawdown is not None:
                        try:
                            release_val = float(release_drawdown)
                            if release_val < 0 or release_val > 1:
                                result.add_error(
                                    "config",
                                    "risk_overlay.drawdown_brake.release_drawdown must be between 0 and 1",
                                    f"Got: {release_val}",
                                )
                            elif threshold_val is not None and release_val >= threshold_val:
                                result.add_warning(
                                    "config",
                                    "drawdown brake release_drawdown should usually be below threshold",
                                    f"threshold={threshold_val}, release_drawdown={release_val}",
                                )
                        except (TypeError, ValueError):
                            result.add_error(
                                "config",
                                "risk_overlay.drawdown_brake.release_drawdown must be numeric",
                                f"Got: {release_drawdown}",
                            )

    # Tax rate
    if config.tax_enabled:
        if config.tax_rate < 0 or config.tax_rate > 1:
            result.add_error("config", "Tax rate must be between 0 and 1",
                            f"Got: {config.tax_rate:.4f}")
        if config.tax_partial_exemption < 0 or config.tax_partial_exemption > 1:
            result.add_error("config", "Partial exemption must be between 0 and 1",
                            f"Got: {config.tax_partial_exemption:.4f}")
        if config.tax_exemption_amount < 0:
            result.add_error("config", "Tax exemption amount cannot be negative",
                            f"Got: {config.tax_exemption_amount}")

    # Risk-free rate
    if config.risk_free_rate < -0.05 or config.risk_free_rate > 0.20:
        result.add_warning("config", "Risk-free rate seems unusual (not between -5% and 20%)",
                          f"Got: {config.risk_free_rate:.2%}")

    return result


def validate_strategy_assets(
    strategy: "Strategy",
    price_data: "PriceData"
) -> ValidationResult:
    """
    Validate that all assets required by strategy are available in price data.

    Checks:
    - All strategy.assets exist in price_data.prices columns
    - Assets have sufficient non-NaN data
    """
    result = ValidationResult()

    required_assets = set(getattr(strategy, "assets", []) or [])
    available_assets = set(price_data.prices.columns)

    # Check for missing assets
    missing = required_assets - available_assets
    if missing:
        result.add_error("assets", "Strategy requires assets not found in price data",
                        f"Missing: {sorted(missing)}")

    # Check data quality for available assets
    for asset in required_assets & available_assets:
        series = price_data.prices[asset]
        total_rows = len(series)
        non_nan_rows = series.notna().sum()
        coverage = non_nan_rows / total_rows if total_rows > 0 else 0

        if coverage < 0.5:
            result.add_warning("assets", f"Asset '{asset}' has low data coverage ({coverage:.1%})",
                              f"{non_nan_rows}/{total_rows} non-NaN values")
        elif coverage < 0.9:
            result.add_info("assets", f"Asset '{asset}' has {coverage:.1%} data coverage",
                           f"{non_nan_rows}/{total_rows} non-NaN values")

    return result


def validate_price_data(prices: pd.DataFrame) -> ValidationResult:
    """
    Validate price data for quality issues.

    Checks:
    - No negative prices
    - No extreme daily returns (>50%)
    - Sufficient data points
    - DatetimeIndex is sorted
    """
    result = ValidationResult()

    # Check index
    if not isinstance(prices.index, pd.DatetimeIndex):
        result.add_error("data", "Price data must have DatetimeIndex")
        return result

    if not prices.index.is_monotonic_increasing:
        result.add_warning("data", "Price data index is not sorted chronologically")

    # Check for sufficient data
    if len(prices) < 20:
        result.add_warning("data", f"Very few data points ({len(prices)}), results may be unreliable")

    # Check each column
    for col in prices.columns:
        series = prices[col].dropna()

        if len(series) == 0:
            result.add_error("data", f"Column '{col}' has no valid price data")
            continue

        # Negative prices
        negative_count = (series < 0).sum()
        if negative_count > 0:
            result.add_error("data", f"Column '{col}' has {negative_count} negative price(s)")

        # Zero prices
        zero_count = (series == 0).sum()
        if zero_count > 0:
            result.add_warning("data", f"Column '{col}' has {zero_count} zero price(s)")

        # Extreme returns
        returns = series.pct_change().dropna()
        extreme_returns = returns[returns.abs() > 0.5]
        if len(extreme_returns) > 0:
            result.add_warning("data",
                f"Column '{col}' has {len(extreme_returns)} extreme daily return(s) (>50%)",
                f"Max: {returns.max():.1%}, Min: {returns.min():.1%}")

    return result


def _to_datetime_index(values) -> pd.DatetimeIndex:
    """Best-effort conversion of common containers to DatetimeIndex."""
    if values is None:
        return pd.DatetimeIndex([])
    if isinstance(values, pd.DatetimeIndex):
        return values
    if isinstance(values, pd.Series):
        if isinstance(values.index, pd.DatetimeIndex):
            return values.index
        parsed = pd.to_datetime(values, errors="coerce")
        return pd.DatetimeIndex(parsed.dropna())
    if isinstance(values, pd.DataFrame):
        if isinstance(values.index, pd.DatetimeIndex):
            return values.index
        for col in ("release_date", "published_at", "date", "as_of"):
            if col in values.columns:
                parsed = pd.to_datetime(values[col], errors="coerce")
                return pd.DatetimeIndex(parsed.dropna())
        return pd.DatetimeIndex([])
    if isinstance(values, (list, tuple, set, np.ndarray, pd.Index)):
        parsed = pd.to_datetime(list(values), errors="coerce")
        parsed = parsed[~pd.isna(parsed)]
        return pd.DatetimeIndex(parsed)
    return pd.DatetimeIndex([])


def _validate_lag_attribute(
    result: ValidationResult,
    strategy: "Strategy",
    attr_name: str,
) -> None:
    """Validate optional strategy lag attributes used for leakage protection."""
    lag = getattr(strategy, attr_name, None)
    if lag is None:
        return
    try:
        lag_value = float(lag)
    except (TypeError, ValueError):
        result.add_error(
            "leakage",
            f"{attr_name} must be numeric",
            f"Got: {lag!r}",
        )
        return
    if lag_value < 1:
        result.add_error(
            "leakage",
            f"{attr_name} must be >= 1 to avoid look-ahead bias",
            f"Got: {lag_value}",
        )


def validate_temporal_leakage(
    strategy: "Strategy",
    data: "PriceData",
) -> ValidationResult:
    """
    Validate potential temporal leakage in feature/macro/fundamental inputs.

    This check is intentionally conservative. It catches hard errors
    (future timestamps, invalid lag settings) and warns on suspicious patterns
    (full same-day alignment without explicit lag metadata).
    """
    result = ValidationResult()

    if data is None or data.prices is None or data.prices.empty:
        return result

    price_index = data.prices.index
    if not isinstance(price_index, pd.DatetimeIndex):
        result.add_error("leakage", "Price data index must be DatetimeIndex for leakage checks")
        return result

    max_price_date = price_index.max()

    # Validate declared lag attributes (if provided by strategy).
    for lag_attr in (
        "feature_lag_days",
        "macro_lag_days",
        "fundamentals_lag_days",
        "fundamentals_release_lag_days",
    ):
        _validate_lag_attribute(result, strategy, lag_attr)

    # Macro checks.
    if data.macro is not None and not data.macro.empty:
        if not isinstance(data.macro.index, pd.DatetimeIndex):
            result.add_error("leakage", "Macro data must use DatetimeIndex")
        else:
            future_macro = data.macro.index[data.macro.index > max_price_date]
            if len(future_macro) > 0:
                result.add_error(
                    "leakage",
                    "Macro data extends beyond available price history",
                    f"Latest price date: {max_price_date.date()}, "
                    f"latest macro date: {future_macro.max().date()}",
                )
            overlap = price_index.intersection(data.macro.index)
            if len(price_index) > 0 and len(overlap) == len(price_index):
                result.add_warning(
                    "leakage",
                    "Macro data is fully same-day aligned with prices",
                    "Ensure macro inputs are lagged (e.g., macro_lag_days >= 1)",
                )

    # Feature checks from common optional strategy attributes.
    feature_sources = []
    for attr_name in ("feature_data", "features", "feature_frame", "feature_matrix"):
        feature_obj = getattr(strategy, attr_name, None)
        if isinstance(feature_obj, (pd.DataFrame, pd.Series)):
            feature_sources.append((attr_name, feature_obj))

    for source_name, feature_obj in feature_sources:
        feature_index = _to_datetime_index(feature_obj)
        if len(feature_index) == 0:
            result.add_warning(
                "leakage",
                f"Feature source '{source_name}' has no datetime information",
                "Temporal leakage could not be validated for this feature source",
            )
            continue
        future_features = feature_index[feature_index > max_price_date]
        if len(future_features) > 0:
            result.add_error(
                "leakage",
                f"Feature source '{source_name}' contains future rows",
                f"Latest feature date: {future_features.max().date()}, "
                f"latest price date: {max_price_date.date()}",
            )
        overlap = price_index.intersection(feature_index)
        if len(price_index) > 0 and len(overlap) == len(price_index):
            result.add_warning(
                "leakage",
                f"Feature source '{source_name}' is fully same-day aligned with prices",
                "Confirm a 1-bar lag in feature construction",
            )

        if isinstance(feature_obj, pd.DataFrame):
            suspicious_cols = [
                str(col) for col in feature_obj.columns
                if any(token in str(col).lower() for token in ("future", "lead", "target_next"))
            ]
            if suspicious_cols:
                sample = ", ".join(suspicious_cols[:5])
                result.add_warning(
                    "leakage",
                    f"Feature source '{source_name}' has potentially forward-looking column names",
                    f"Columns: {sample}",
                )

    # Fundamentals checks (strategy-declared metadata).
    uses_fundamentals = bool(getattr(strategy, "uses_fundamentals", False))
    fundamentals_pit = getattr(strategy, "fundamentals_point_in_time", None)
    if uses_fundamentals and fundamentals_pit is False:
        result.add_error(
            "leakage",
            "Strategy declares fundamentals but marks them as non point-in-time",
            "Use point-in-time fundamentals to avoid look-ahead bias",
        )

    release_dates = pd.DatetimeIndex([])
    for attr_name in (
        "fundamentals_release_dates",
        "fundamentals_dates",
        "fundamentals_as_of_dates",
    ):
        candidate = getattr(strategy, attr_name, None)
        release_dates = _to_datetime_index(candidate)
        if len(release_dates) > 0:
            break

    if len(release_dates) > 0:
        future_release_dates = release_dates[release_dates > max_price_date]
        if len(future_release_dates) > 0:
            result.add_error(
                "leakage",
                "Fundamentals release dates extend beyond available price history",
                f"Latest release date: {future_release_dates.max().date()}, "
                f"latest price date: {max_price_date.date()}",
            )
    elif uses_fundamentals and getattr(strategy, "fundamentals_release_lag_days", None) is None:
        result.add_warning(
            "leakage",
            "Strategy uses fundamentals but provides no release-lag metadata",
            "Set fundamentals_release_lag_days >= 1 or provide fundamentals_release_dates",
        )

    return result


# =============================================================================
# POST-RUN VALIDATION: Checks after backtest completion
# =============================================================================

def validate_equity_curve(equity_curve: pd.Series) -> ValidationResult:
    """
    Validate equity curve for integrity.

    Checks:
    - No NaN values
    - No negative values
    - No extreme single-period changes
    - Monotonic index
    """
    result = ValidationResult()

    if equity_curve is None or len(equity_curve) == 0:
        result.add_error("equity", "Equity curve is empty")
        return result

    # NaN values
    nan_count = equity_curve.isna().sum()
    if nan_count > 0:
        nan_pct = nan_count / len(equity_curve)
        result.add_error("equity", f"Equity curve contains {nan_count} NaN value(s) ({nan_pct:.1%})")

    # Clean for further checks
    clean_curve = equity_curve.dropna()

    if len(clean_curve) == 0:
        result.add_fatal("equity", "Equity curve is entirely NaN")
        return result

    # Negative values
    negative_count = (clean_curve < 0).sum()
    if negative_count > 0:
        result.add_error("equity", f"Equity curve has {negative_count} negative value(s)",
                        f"Min value: {clean_curve.min():.2f}")

    # Zero values (after start)
    zero_count = (clean_curve.iloc[1:] == 0).sum()
    if zero_count > 0:
        result.add_warning("equity", f"Equity curve has {zero_count} zero value(s) after start")

    # Extreme returns (>100% or < -90%)
    returns = clean_curve.pct_change().dropna()
    extreme_up = (returns > 1.0).sum()
    extreme_down = (returns < -0.9).sum()

    if extreme_up > 0:
        result.add_warning("equity", f"Equity curve has {extreme_up} extreme gain(s) (>100%)")
    if extreme_down > 0:
        result.add_warning("equity", f"Equity curve has {extreme_down} extreme loss(es) (>90%)")

    return result


def validate_portfolio_state(
    portfolio: "Portfolio",
    prices: pd.Series,
    allow_negative_cash: bool = False
) -> ValidationResult:
    """
    Validate portfolio state consistency.

    Checks:
    - Cash is non-negative (unless margin is allowed)
    - All positions are non-negative
    - Total value is positive
    - No NaN values in positions
    """
    result = ValidationResult()

    # Cash validation
    if not allow_negative_cash and portfolio.cash < -0.01:
        result.add_error("portfolio", f"Negative cash balance: {portfolio.cash:.2f}")

    # Position validation
    for ticker, shares in portfolio.positions.items():
        if np.isnan(shares):
            result.add_error("portfolio", f"NaN shares in position '{ticker}'")
        elif shares < -0.0001:
            result.add_error("portfolio", f"Negative position in '{ticker}': {shares:.4f} shares")

    # Total value
    total_value = portfolio.total_value(prices)
    if np.isnan(total_value):
        result.add_error("portfolio", "Portfolio total value is NaN")
    elif total_value < 0:
        result.add_error("portfolio", f"Portfolio total value is negative: {total_value:.2f}")

    return result


def validate_trades(trades: list) -> ValidationResult:
    """
    Validate trade list for consistency.

    Checks:
    - All trades have valid actions (BUY/SELL)
    - Shares and prices are positive
    - Costs are non-negative
    - No NaN values
    """
    result = ValidationResult()

    for i, trade in enumerate(trades):
        prefix = f"Trade {i} ({trade.ticker} @ {trade.date})"

        # Action
        if trade.action not in ("BUY", "SELL"):
            result.add_error("trades", f"{prefix}: Invalid action '{trade.action}'")

        # Shares
        if np.isnan(trade.shares):
            result.add_error("trades", f"{prefix}: NaN shares")
        elif trade.shares <= 0:
            result.add_warning("trades", f"{prefix}: Non-positive shares: {trade.shares}")

        # Price
        if np.isnan(trade.price):
            result.add_error("trades", f"{prefix}: NaN price")
        elif trade.price <= 0:
            result.add_warning("trades", f"{prefix}: Non-positive price: {trade.price}")

        # Costs
        if np.isnan(trade.costs):
            result.add_error("trades", f"{prefix}: NaN costs")
        elif trade.costs < 0:
            result.add_error("trades", f"{prefix}: Negative costs: {trade.costs}")

    return result


# =============================================================================
# INVARIANT VALIDATION: Mathematical relationships that must hold
# =============================================================================

def validate_tax_invariants(tax_summary: "TaxSummary", tolerance: float = 0.01) -> ValidationResult:
    """
    Validate tax-related invariants.

    Invariants:
    1. final_value_gross >= final_value_net_realized >= final_value_net_liquidation
    2. total_tax_paid >= 0
    3. tax_paid_liquidation >= 0 (when applicable)
    4. cagr_gross >= cagr_net_realized >= cagr_net_liquidation
    5. effective_tax_rate between 0 and tax_rate
    """
    result = ValidationResult()
    ts = tax_summary

    if not ts.tax_enabled:
        # When tax is disabled, verify all values are zero/equal
        if abs(ts.total_tax_paid) > tolerance:
            result.add_error("invariant",
                "Tax paid should be 0 when tax is disabled",
                f"Got: {ts.total_tax_paid:.2f}")
        if abs(ts.cagr_gross - ts.cagr_net_realized) > tolerance:
            result.add_error("invariant",
                "CAGR gross should equal net_realized when tax is disabled",
                f"Gross: {ts.cagr_gross:.4f}, Net: {ts.cagr_net_realized:.4f}")
        return result

    # Invariant 1: Value ordering
    if ts.final_value_gross + tolerance < ts.final_value_net_realized:
        result.add_error("invariant",
            "final_value_gross should be >= final_value_net_realized",
            f"Gross: {ts.final_value_gross:.2f}, Net Realized: {ts.final_value_net_realized:.2f}")

    if ts.final_value_net_realized + tolerance < ts.final_value_net_liquidation:
        result.add_error("invariant",
            "final_value_net_realized should be >= final_value_net_liquidation",
            f"Net Realized: {ts.final_value_net_realized:.2f}, Net Liq: {ts.final_value_net_liquidation:.2f}")

    # Invariant 2: Tax paid non-negative
    if ts.total_tax_paid < -tolerance:
        result.add_error("invariant",
            "total_tax_paid should be >= 0",
            f"Got: {ts.total_tax_paid:.2f}")

    # Invariant 3: Liquidation tax non-negative
    if ts.metric_basis == "net_liquidation" and ts.tax_paid_liquidation < -tolerance:
        result.add_error("invariant",
            "tax_paid_liquidation should be >= 0",
            f"Got: {ts.tax_paid_liquidation:.2f}")

    # Invariant 4: CAGR ordering
    if ts.cagr_gross + 0.0001 < ts.cagr_net_realized:
        result.add_warning("invariant",
            "cagr_gross should be >= cagr_net_realized",
            f"Gross: {ts.cagr_gross:.4f}, Net Realized: {ts.cagr_net_realized:.4f}")

    if ts.cagr_net_realized + 0.0001 < ts.cagr_net_liquidation:
        result.add_warning("invariant",
            "cagr_net_realized should be >= cagr_net_liquidation",
            f"Net Realized: {ts.cagr_net_realized:.4f}, Net Liq: {ts.cagr_net_liquidation:.4f}")

    # Invariant 5: Effective tax rate bounds
    if ts.effective_tax_rate < -tolerance:
        result.add_error("invariant",
            "effective_tax_rate should be >= 0",
            f"Got: {ts.effective_tax_rate:.4f}")

    if ts.effective_tax_rate > ts.tax_rate + tolerance and ts.net_realized_gain > 100:
        result.add_warning("invariant",
            "effective_tax_rate exceeds nominal tax rate (may be due to loss pot mechanics)",
            f"Effective: {ts.effective_tax_rate:.4f}, Nominal: {ts.tax_rate:.4f}")

    return result


def validate_metrics_consistency(
    equity_curve: pd.Series,
    metrics: "Metrics",
    tolerance: float = 0.01
) -> ValidationResult:
    """
    Validate that metrics are consistent with equity curve.

    Checks:
    - Total return matches equity curve
    - Max drawdown is negative or zero
    - Sharpe/Sortino ratios are finite
    """
    from backtest.metrics import MetricsCalculator

    result = ValidationResult()

    # Total return consistency
    expected_total_return = MetricsCalculator.total_return(equity_curve)
    if abs(metrics.total_return - expected_total_return) > tolerance:
        result.add_warning("metrics",
            "Total return inconsistent with equity curve",
            f"Metrics: {metrics.total_return:.4f}, Calculated: {expected_total_return:.4f}")

    # Max drawdown should be <= 0
    if metrics.max_drawdown > tolerance:
        result.add_error("metrics",
            "Max drawdown should be <= 0",
            f"Got: {metrics.max_drawdown:.4f}")

    # Finite ratios
    if not np.isfinite(metrics.sharpe_ratio) and metrics.sharpe_ratio != float('inf'):
        result.add_warning("metrics", "Sharpe ratio is not finite")

    if not np.isfinite(metrics.sortino_ratio) and metrics.sortino_ratio != float('inf'):
        result.add_warning("metrics", "Sortino ratio is not finite")

    # Win rate bounds
    if metrics.win_rate_monthly < 0 or metrics.win_rate_monthly > 1:
        result.add_error("metrics",
            "Win rate should be between 0 and 1",
            f"Got: {metrics.win_rate_monthly:.4f}")

    return result


# =============================================================================
# FULL VALIDATION: Run all checks on a BacktestResult
# =============================================================================

def validate_backtest_result(result: "BacktestResult") -> ValidationResult:
    """
    Run all validation checks on a completed backtest result.

    This is the main entry point for post-backtest validation.

    Args:
        result: Completed BacktestResult

    Returns:
        ValidationResult with all issues found
    """
    validation = ValidationResult()

    # Validate equity curves
    for curve_name, curve in [
        ("equity_curve", result.equity_curve),
        ("equity_curve_gross", result.equity_curve_gross),
        ("equity_curve_net", result.equity_curve_net),
    ]:
        if curve is not None:
            curve_validation = validate_equity_curve(curve)
            for issue in curve_validation.issues:
                issue.category = f"{curve_name}.{issue.category}"
                validation.add(issue)

    # Validate trades
    trades_validation = validate_trades(result.trades)
    validation.issues.extend(trades_validation.issues)

    # Validate tax invariants
    if result.tax_summary is not None:
        tax_validation = validate_tax_invariants(result.tax_summary)
        validation.issues.extend(tax_validation.issues)

    # Validate metrics consistency
    if result.metrics is not None:
        metrics_validation = validate_metrics_consistency(result.equity_curve, result.metrics)
        validation.issues.extend(metrics_validation.issues)

    if result.metrics_gross is not None and result.equity_curve_gross is not None:
        gross_validation = validate_metrics_consistency(result.equity_curve_gross, result.metrics_gross)
        for issue in gross_validation.issues:
            issue.category = f"gross.{issue.category}"
            validation.add(issue)

    return validation


def validate_before_run(
    strategy: "Strategy",
    data: "PriceData",
    config: "BacktestConfig"
) -> ValidationResult:
    """
    Run all pre-backtest validation checks.

    This is the main entry point for pre-backtest validation.

    Args:
        strategy: Strategy to be tested
        data: Price data to be used
        config: Backtest configuration

    Returns:
        ValidationResult with all issues found
    """
    validation = ValidationResult()

    # Validate config
    config_validation = validate_config(config)
    validation.issues.extend(config_validation.issues)

    # Validate strategy assets
    assets_validation = validate_strategy_assets(strategy, data)
    validation.issues.extend(assets_validation.issues)

    # Validate price data
    data_validation = validate_price_data(data.prices)
    validation.issues.extend(data_validation.issues)

    # Temporal leakage checks (features/macro/fundamentals)
    leakage_validation = validate_temporal_leakage(strategy, data)
    validation.issues.extend(leakage_validation.issues)

    return validation
