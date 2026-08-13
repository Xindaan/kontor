"""
Sweep module - Multi-window robustness analysis.

This module provides functionality to:
- Run backtests over multiple time windows (rolling or end-fixed)
- Measure strategy robustness across different start dates
- Generate summary statistics and identify sweet spots
- Compare multiple strategies using directory globs

The sweep analysis helps identify strategies that are robust across different
market conditions rather than optimized for a single time period.
"""

import glob
import copy
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from backtest.utils import MONTH_END_FREQ


@dataclass
class Window:
    """Represents a single time window for backtesting."""
    start: datetime
    end: datetime
    years: float

    def __repr__(self) -> str:
        return f"Window({self.start.strftime('%Y-%m-%d')} -> {self.end.strftime('%Y-%m-%d')}, {self.years:.1f}y)"


@dataclass
class WindowResult:
    """Results for a single strategy in a single window."""
    strategy_name: str
    strategy_file: str
    window_start: datetime
    window_end: datetime
    window_years: float
    status: Literal["ok", "skipped", "error"] = "ok"
    skip_reason: Optional[str] = None

    # Terminal valuation mode
    metric_basis: Literal["gross", "net_realized", "net_liquidation"] = "net_realized"

    # Metrics (primary - based on metric_basis mode)
    final_value: float = 0.0
    cagr: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    # Gross/Net/Liquidation final values (always populated)
    final_value_gross: float = 0.0
    final_value_net_realized: float = 0.0
    final_value_net_liquidation: float = 0.0

    # CAGR variants
    cagr_gross: float = 0.0
    cagr_net_realized: float = 0.0
    cagr_net_liquidation: float = 0.0

    # Trading
    trades: int = 0
    costs_total: float = 0.0
    tax_paid_realized: float = 0.0  # Renamed from tax_paid_total for clarity
    tax_paid_liquidation: float = 0.0  # Additional tax from virtual liquidation

    # Benchmark comparison
    benchmark_final_value: Optional[float] = None
    benchmark_cagr: Optional[float] = None
    benchmark_maxdd: Optional[float] = None
    benchmark_sharpe: Optional[float] = None
    benchmark_sortino: Optional[float] = None
    benchmark_vol: Optional[float] = None
    benchmark_calmar: Optional[float] = None
    excess_cagr: Optional[float] = None  # strategy_cagr - benchmark_cagr

    # Legacy alias
    @property
    def tax_paid_total(self) -> float:
        """Legacy: total realized tax paid."""
        return self.tax_paid_realized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "strategy_file": self.strategy_file,
            "window_start": self.window_start.strftime("%Y-%m-%d"),
            "window_end": self.window_end.strftime("%Y-%m-%d"),
            "window_years": round(self.window_years, 2),
            "status": self.status,
            "skip_reason": self.skip_reason,
            "metric_basis": self.metric_basis,
            "final_value": round(self.final_value, 2),
            "cagr": round(self.cagr, 4),
            "volatility": round(self.volatility, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "final_value_gross": round(self.final_value_gross, 2),
            "final_value_net_realized": round(self.final_value_net_realized, 2),
            "final_value_net_liquidation": round(self.final_value_net_liquidation, 2),
            "cagr_gross": round(self.cagr_gross, 4),
            "cagr_net_realized": round(self.cagr_net_realized, 4),
            "cagr_net_liquidation": round(self.cagr_net_liquidation, 4),
            "trades": self.trades,
            "costs_total": round(self.costs_total, 2),
            "tax_paid_realized": round(self.tax_paid_realized, 2),
            "tax_paid_liquidation": round(self.tax_paid_liquidation, 2),
            "benchmark_final_value": round(self.benchmark_final_value, 2) if self.benchmark_final_value is not None else None,
            "benchmark_cagr": round(self.benchmark_cagr, 4) if self.benchmark_cagr is not None else None,
            "benchmark_maxdd": round(self.benchmark_maxdd, 4) if self.benchmark_maxdd is not None else None,
            "benchmark_sharpe": round(self.benchmark_sharpe, 4) if self.benchmark_sharpe is not None else None,
            "benchmark_sortino": round(self.benchmark_sortino, 4) if self.benchmark_sortino is not None else None,
            "benchmark_vol": round(self.benchmark_vol, 4) if self.benchmark_vol is not None else None,
            "benchmark_calmar": round(self.benchmark_calmar, 4) if self.benchmark_calmar is not None else None,
            "excess_cagr": round(self.excess_cagr, 4) if self.excess_cagr is not None else None,
        }


@dataclass
class StrategySummary:
    """Aggregated statistics for a single strategy across all windows."""
    strategy_name: str
    strategy_file: str
    num_windows: int
    num_ok: int
    num_skipped: int

    # Return robustness
    median_cagr: float = 0.0
    p10_cagr: float = 0.0
    p90_cagr: float = 0.0
    worst_cagr: float = 0.0
    best_cagr: float = 0.0
    prob_negative_cagr: float = 0.0

    # Risk
    median_sharpe: float = 0.0
    p10_sharpe: float = 0.0
    median_sortino: float = 0.0
    median_maxdd: float = 0.0
    worst_maxdd: float = 0.0
    median_vol: float = 0.0
    median_calmar: float = 0.0

    # Friction
    median_costs_total: float = 0.0
    median_tax_paid_total: float = 0.0
    median_tax_paid_realized: float = 0.0
    median_tax_paid_liquidation: float = 0.0

    # Terminal valuation CAGR variants
    median_cagr_gross: float = 0.0
    median_cagr_net_realized: float = 0.0
    median_cagr_net_liquidation: float = 0.0
    p10_cagr_gross: float = 0.0
    p10_cagr_net_realized: float = 0.0
    p10_cagr_net_liquidation: float = 0.0

    # Benchmark comparison
    hit_rate_vs_benchmark: Optional[float] = None
    median_excess_cagr: Optional[float] = None
    p10_excess_cagr: Optional[float] = None
    prob_underperform_benchmark: Optional[float] = None

    # Ranking
    rank: int = 0
    pareto_front: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "strategy_file": self.strategy_file,
            "num_windows": self.num_windows,
            "num_ok": self.num_ok,
            "num_skipped": self.num_skipped,
            "median_cagr": round(self.median_cagr, 4),
            "p10_cagr": round(self.p10_cagr, 4),
            "p90_cagr": round(self.p90_cagr, 4),
            "worst_cagr": round(self.worst_cagr, 4),
            "best_cagr": round(self.best_cagr, 4),
            "prob_negative_cagr": round(self.prob_negative_cagr, 4),
            "median_sharpe": round(self.median_sharpe, 4),
            "p10_sharpe": round(self.p10_sharpe, 4),
            "median_sortino": round(self.median_sortino, 4),
            "median_maxdd": round(self.median_maxdd, 4),
            "worst_maxdd": round(self.worst_maxdd, 4),
            "median_vol": round(self.median_vol, 4),
            "median_calmar": round(self.median_calmar, 4),
            "median_costs_total": round(self.median_costs_total, 2),
            "median_tax_paid_total": round(self.median_tax_paid_total, 2),
            "median_tax_paid_realized": round(self.median_tax_paid_realized, 2),
            "median_tax_paid_liquidation": round(self.median_tax_paid_liquidation, 2),
            "median_cagr_gross": round(self.median_cagr_gross, 4),
            "median_cagr_net_realized": round(self.median_cagr_net_realized, 4),
            "median_cagr_net_liquidation": round(self.median_cagr_net_liquidation, 4),
            "p10_cagr_gross": round(self.p10_cagr_gross, 4),
            "p10_cagr_net_realized": round(self.p10_cagr_net_realized, 4),
            "p10_cagr_net_liquidation": round(self.p10_cagr_net_liquidation, 4),
            "hit_rate_vs_benchmark": round(self.hit_rate_vs_benchmark, 4) if self.hit_rate_vs_benchmark is not None else None,
            "median_excess_cagr": round(self.median_excess_cagr, 4) if self.median_excess_cagr is not None else None,
            "p10_excess_cagr": round(self.p10_excess_cagr, 4) if self.p10_excess_cagr is not None else None,
            "prob_underperform_benchmark": round(self.prob_underperform_benchmark, 4) if self.prob_underperform_benchmark is not None else None,
            "rank": self.rank,
            "pareto_front": self.pareto_front,
        }


from backtest.external_features.config import ExternalFeaturesConfig, build_loader_from_config


@dataclass
class SweepConfig:
    """Configuration for a sweep analysis."""
    mode: Literal["rolling", "end-fixed"] = "rolling"
    window_length: Optional[str] = "10y"  # e.g., "3y", "5y", "10y"
    end_date: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    start_grid: Literal["weekly", "monthly", "yearly"] = "monthly"
    step: int = 1
    warmup_days: int = 260

    # Backtest params
    initial_capital: float = 10000.0
    rebalance_frequency: str = "monthly"
    costs_enabled: bool = True
    costs_pct: float = 0.001
    cost_profile: Optional[Dict[str, Any]] = None
    execution_lag_days: int = 0
    max_volume_participation: Optional[float] = None
    min_daily_dollar_volume: float = 0.0
    liquidity_on_missing_volume: Literal["allow", "skip"] = "allow"
    risk_overlay: Optional[Dict[str, Any]] = None
    tax_enabled: bool = True
    tax_rate: float = 0.26375
    tax_exemption: float = 1000.0
    metric_basis: Literal["gross", "net_realized", "net_liquidation"] = "net_liquidation"

    # Data alignment
    align: Literal["intersection", "ffill"] = "ffill"

    # Skip failed downloads
    skip_failed: bool = True

    # Dividend reinvestment
    drip_enabled: bool = False

    # Validation
    validate: bool = True

    # Benchmark
    benchmark_ticker: str = "SPY"  # S&P 500 ETF

    # Universe look-ahead protection
    allow_universe_lookahead: bool = False  # Default: block non-PIT universes for historical backtests

    # Execution
    jobs: int = 1
    fail_fast: bool = False

    # External features pipeline (Phase A plumbing).
    external_features: ExternalFeaturesConfig = field(default_factory=ExternalFeaturesConfig)


@dataclass
class SweepResult:
    """Complete sweep results."""
    config: SweepConfig
    windows: List[Window]
    window_results: List[WindowResult]
    summaries: List[StrategySummary]
    metadata: Optional[Any] = None

    def to_windows_df(self) -> pd.DataFrame:
        """Convert window results to DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.window_results])

    def to_summary_df(self) -> pd.DataFrame:
        """Convert summaries to DataFrame."""
        return pd.DataFrame([s.to_dict() for s in self.summaries])


class SweepCancelled(Exception):
    """Raised when a sweep run is cancelled."""


def _clone_strategy_instance(strategy: "Strategy") -> "Strategy":
    """
    Create an isolated strategy instance for a single backtest run.

    Sweep runs many windows; reusing one instance can leak state between runs.
    """
    try:
        return copy.deepcopy(strategy)
    except Exception:
        cls = strategy.__class__
        try:
            return cls()
        except Exception:
            # Last resort: return original instance (may leak state)
            return strategy


def resolve_strategy_paths(patterns: List[str]) -> List[Path]:
    """
    Resolve strategy file paths from patterns including globs.

    Args:
        patterns: List of file paths or glob patterns
                  e.g., ["strategies/buy_and_hold.py", "strategies/[!_]*.py"]

    Returns:
        Sorted list of unique Path objects to strategy files

    Raises:
        ValueError: If no strategies match the patterns
    """
    paths = set()

    for pattern in patterns:
        # Check if it's a glob pattern
        if any(c in pattern for c in "*?["):
            # Use glob to expand
            matches = glob.glob(pattern, recursive=False)
            for match in matches:
                p = Path(match)
                if p.is_file() and p.suffix == ".py":
                    paths.add(p)
        else:
            # Direct file path
            p = Path(pattern)
            if p.exists() and p.is_file() and p.suffix == ".py":
                paths.add(p)
            elif not p.exists():
                raise ValueError(f"Strategy file not found: {pattern}")

    if not paths:
        raise ValueError(f"No strategies matched patterns: {patterns}")

    # Sort alphabetically for stable ordering
    return sorted(paths, key=lambda p: p.name)


def parse_window_length(length_str: str) -> timedelta:
    """
    Parse window length string to timedelta.

    Args:
        length_str: e.g., "3y", "5y", "10y", "6m", "52w"

    Returns:
        timedelta representing the window length
    """
    match = re.match(r"^(\d+)(y|m|w|d)$", length_str.lower())
    if not match:
        raise ValueError(f"Invalid window length format: {length_str}. Use e.g., '5y', '10y', '6m'")

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "y":
        return timedelta(days=value * 365)
    elif unit == "m":
        return timedelta(days=value * 30)
    elif unit == "w":
        return timedelta(days=value * 7)
    elif unit == "d":
        return timedelta(days=value)
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def generate_windows(
    data_index: pd.DatetimeIndex,
    mode: Literal["rolling", "end-fixed"],
    end_date: Optional[datetime],
    from_date: Optional[datetime],
    to_date: Optional[datetime],
    window_length: Optional[timedelta],
    start_grid: Literal["weekly", "monthly", "yearly"],
    step: int = 1,
) -> List[Window]:
    """
    Generate time windows for sweep analysis.

    Args:
        data_index: DatetimeIndex of available trading days
        mode: "rolling" (fixed window size) or "end-fixed" (all end at same date)
        end_date: End date for windows (defaults to last available date)
        from_date: Minimum start date to consider
        to_date: Maximum start date to consider
        window_length: Window size as timedelta (required for rolling mode)
        start_grid: Frequency of start date candidates
        step: Skip every N grid points

    Returns:
        List of Window objects
    """
    if mode == "rolling" and window_length is None:
        raise ValueError("Window length is required for rolling mode")

    # Normalize timezone to avoid tz-aware vs tz-naive comparisons
    if data_index.tz is not None:
        data_index = data_index.tz_convert(None)

    # Determine effective end date
    if end_date is None:
        effective_end = data_index[-1].to_pydatetime()
    else:
        # Snap to last trading day <= end_date
        mask = data_index <= pd.Timestamp(end_date)
        if not mask.any():
            raise ValueError(f"End date {end_date} is before data starts")
        effective_end = data_index[mask][-1].to_pydatetime()

    # Determine start candidate range
    if from_date is None:
        effective_from = data_index[0].to_pydatetime()
    else:
        # Snap to first trading day >= from_date
        mask = data_index >= pd.Timestamp(from_date)
        if not mask.any():
            raise ValueError(f"From date {from_date} is after data ends")
        effective_from = data_index[mask][0].to_pydatetime()

    if to_date is None:
        if mode == "rolling" and window_length is not None:
            # For rolling: max start is when window_end <= effective_end
            effective_to = effective_end - window_length
        else:
            effective_to = effective_end
    else:
        # to_date can be datetime or str depending on caller
        if isinstance(to_date, str):
            effective_to = datetime.strptime(to_date, "%Y-%m-%d")
        elif isinstance(to_date, datetime):
            effective_to = to_date
        else:
            # Handle date objects by converting to datetime
            effective_to = datetime.combine(to_date, datetime.min.time())

    # Check if date range is valid
    if effective_from > effective_to:
        import warnings
        warnings.warn(
            f"No valid window range: effective_from ({effective_from.strftime('%Y-%m-%d')}) > "
            f"effective_to ({effective_to.strftime('%Y-%m-%d')}). "
            f"Data may not cover enough history for {window_length.days if window_length else 'N/A'} day windows."
        )
        return []

    # Generate start date candidates based on grid
    if start_grid == "monthly":
        # Generate month-end dates within range (use compatible frequency)
        candidates = pd.date_range(
            start=effective_from,
            end=effective_to,
            freq=MONTH_END_FREQ,
        )
    elif start_grid == "yearly":
        # Use "A" for older pandas, "YE" for newer
        try:
            candidates = pd.date_range(
                start=effective_from,
                end=effective_to,
                freq="YE",
            )
        except ValueError:
            candidates = pd.date_range(
                start=effective_from,
                end=effective_to,
                freq="A",
            )
    elif start_grid == "weekly":
        candidates = pd.date_range(
            start=effective_from,
            end=effective_to,
            freq="W",
        )
    else:
        raise ValueError(f"Unknown start_grid: {start_grid}")

    # Apply step filter
    if step > 1:
        candidates = candidates[::step]

    # Snap candidates to actual trading days and create windows
    windows = []
    for candidate in candidates:
        # Snap start to next trading day >= candidate
        mask = data_index >= candidate
        if not mask.any():
            continue
        window_start = data_index[mask][0].to_pydatetime()

        # Determine window end
        if mode == "rolling":
            window_end_target = window_start + window_length
            # Snap to last trading day <= target
            mask = data_index <= pd.Timestamp(window_end_target)
            if not mask.any():
                continue
            window_end = data_index[mask][-1].to_pydatetime()

            # Skip if window would extend beyond effective_end
            if window_end > effective_end:
                continue
        else:  # end-fixed
            window_end = effective_end

        # Calculate actual years
        years = (window_end - window_start).days / 365.25

        windows.append(Window(
            start=window_start,
            end=window_end,
            years=years,
        ))

    return windows


def compute_benchmark_for_window(
    benchmark_ticker: str,
    data: "PriceData",
    window: Window,
    config: SweepConfig,
) -> Optional[Dict[str, float]]:
    """
    Compute benchmark metrics (Buy & Hold on benchmark_ticker) for a single window.

    Args:
        benchmark_ticker: Ticker symbol for benchmark asset
        data: PriceData object with all assets
        window: Window to run
        config: Sweep configuration

    Returns:
        Dict with benchmark metrics or None if benchmark fails
    """
    from backtest.backtester import Backtester, BacktestConfig
    from backtest.metrics import MetricsCalculator
    from backtest.strategy import Strategy

    try:
        # Check if benchmark ticker is in data
        if benchmark_ticker not in data.prices.columns:
            if config.validate:
                import warnings
                warnings.warn(f"Benchmark ticker {benchmark_ticker} not found in price data")
            return None

        # Slice data to warmup + window period (not entire dataset)
        warmup_start = window.start - pd.Timedelta(days=config.warmup_days)
        window_data = data.prices.loc[warmup_start:window.end].copy()

        # Create a new PriceData for this window
        from backtest.data import PriceData
        sliced_data = PriceData(
            prices=window_data,
            currency=data.currency,
            fx_rates=data.fx_rates.loc[warmup_start:window.end].copy() if data.fx_rates is not None else None,
            volumes=data.volumes.loc[warmup_start:window.end].copy() if data.volumes is not None else None,
            dividends=data.dividends.loc[warmup_start:window.end].copy() if data.dividends is not None else None,
        )

        # Fast path for no-friction/no-tax benchmark: compute directly from prices
        if not config.tax_enabled and not config.costs_enabled:
            bench_window_prices = sliced_data.prices[benchmark_ticker].loc[window.start:window.end].dropna()
            if len(bench_window_prices) >= 2 and bench_window_prices.iloc[0] > 0:
                bench_curve = config.initial_capital * (bench_window_prices / bench_window_prices.iloc[0])
                bench_monthly = MetricsCalculator.monthly_returns(bench_curve)
                bm_cagr = MetricsCalculator.cagr(bench_curve)
                bm_vol = MetricsCalculator.volatility(bench_monthly)
                bm_maxdd = MetricsCalculator.max_drawdown(bench_curve)
                bm_sharpe = MetricsCalculator.sharpe_ratio(bench_monthly, risk_free_rate=0.02)
                bm_sortino = MetricsCalculator.sortino_ratio(bench_monthly, risk_free_rate=0.02)
                bm_calmar = MetricsCalculator.calmar_ratio(bm_cagr, bm_maxdd)
                return {
                    "final_value": float(bench_curve.iloc[-1]),
                    "cagr": bm_cagr,
                    "volatility": bm_vol,
                    "sharpe_ratio": bm_sharpe,
                    "sortino_ratio": bm_sortino,
                    "max_drawdown": bm_maxdd,
                    "calmar_ratio": bm_calmar,
                }

        # Create Buy & Hold strategy for benchmark
        from backtest.strategy import Allocation

        class BenchmarkStrategy(Strategy):
            def __init__(self, ticker: str):
                self.name = f"Benchmark_{ticker}"
                self.assets = [ticker]
                self._ticker = ticker

            def signal(self, date, data):
                return Allocation({self._ticker: 1.0})

        benchmark_strategy = BenchmarkStrategy(benchmark_ticker)

        # Run with same costs/tax settings as strategies
        bt_config = BacktestConfig(
            initial_capital=config.initial_capital,
            costs_pct=config.costs_pct if config.costs_enabled else 0.0,
            slippage_pct=0.0005 if config.costs_enabled else 0.0,
            cost_profile=config.cost_profile,
            execution_lag_days=config.execution_lag_days,
            max_volume_participation=config.max_volume_participation,
            min_daily_dollar_volume=config.min_daily_dollar_volume,
            liquidity_on_missing_volume=config.liquidity_on_missing_volume,
            risk_overlay=config.risk_overlay,
            rebalance_frequency=config.rebalance_frequency,
            tax_enabled=config.tax_enabled,
            tax_rate=config.tax_rate,
            tax_exemption_amount=config.tax_exemption,
            metric_basis=config.metric_basis,
            allow_universe_lookahead=config.allow_universe_lookahead,
            validate=config.validate,
            drip_enabled=config.drip_enabled,
        )

        backtester = Backtester(benchmark_strategy, sliced_data, bt_config)
        bt_result = backtester.run()

        # Recompute metrics for window period only (excluding warmup)
        ec = bt_result.equity_curve
        window_ec = ec.loc[window.start:]
        if len(window_ec) >= 2:
            window_monthly = MetricsCalculator.monthly_returns(window_ec)
            bm_cagr = MetricsCalculator.cagr(window_ec)
            bm_vol = MetricsCalculator.volatility(window_monthly)
            bm_maxdd = MetricsCalculator.max_drawdown(window_ec)
            bm_sharpe = MetricsCalculator.sharpe_ratio(window_monthly, risk_free_rate=bt_config.risk_free_rate)
            bm_sortino = MetricsCalculator.sortino_ratio(window_monthly, risk_free_rate=bt_config.risk_free_rate)
            bm_calmar = MetricsCalculator.calmar_ratio(bm_cagr, bm_maxdd)
        else:
            m = bt_result.metrics
            bm_cagr = m.cagr
            bm_vol = m.volatility
            bm_maxdd = m.max_drawdown
            bm_sharpe = m.sharpe_ratio
            bm_sortino = m.sortino_ratio
            bm_calmar = m.calmar_ratio

        return {
            "final_value": bt_result.equity_curve.iloc[-1],
            "cagr": bm_cagr,
            "volatility": bm_vol,
            "sharpe_ratio": bm_sharpe,
            "sortino_ratio": bm_sortino,
            "max_drawdown": bm_maxdd,
            "calmar_ratio": bm_calmar,
        }

    except Exception as e:
        if config.validate:
            import warnings
            warnings.warn(f"Benchmark computation failed for window {window.start} - {window.end}: {e}")
        return None


def run_single_window(
    strategy,
    strategy_file: str,
    data: "PriceData",
    window: Window,
    config: SweepConfig,
    benchmark_result: Optional[Dict] = None,
) -> WindowResult:
    """
    Run a single strategy on a single window.

    Args:
        strategy: Strategy instance
        strategy_file: Path to strategy file
        data: PriceData object (already sliced to include warmup)
        window: Window to run
        config: Sweep configuration
        benchmark_result: Pre-computed benchmark metrics for this window

    Returns:
        WindowResult with metrics
    """
    from backtest.backtester import Backtester, BacktestConfig

    result = WindowResult(
        strategy_name=strategy.name,
        strategy_file=strategy_file,
        window_start=window.start,
        window_end=window.end,
        window_years=window.years,
    )

    try:
        # Slice data to warmup + window period only (not the entire dataset)
        # This ensures metrics are computed approximately for the window period
        warmup_start = window.start - pd.Timedelta(days=config.warmup_days)
        window_data = data.prices.loc[warmup_start:window.end]

        # Create a new PriceData for this window
        from backtest.data import PriceData
        sliced_data = PriceData(
            prices=window_data,
            currency=data.currency,
            fx_rates=data.fx_rates.loc[warmup_start:window.end] if data.fx_rates is not None else None,
            volumes=data.volumes.loc[warmup_start:window.end] if data.volumes is not None else None,
            dividends=data.dividends.loc[warmup_start:window.end] if data.dividends is not None else None,
        )

        # Create backtest config
        # Prefer per-strategy rebalance_frequency (from optimized params) over global config
        rebal_freq = getattr(strategy, 'rebalance_frequency', None) or config.rebalance_frequency
        bt_config = BacktestConfig(
            initial_capital=config.initial_capital,
            costs_pct=config.costs_pct if config.costs_enabled else 0.0,
            slippage_pct=0.0005 if config.costs_enabled else 0.0,
            cost_profile=config.cost_profile,
            execution_lag_days=config.execution_lag_days,
            max_volume_participation=config.max_volume_participation,
            min_daily_dollar_volume=config.min_daily_dollar_volume,
            liquidity_on_missing_volume=config.liquidity_on_missing_volume,
            risk_overlay=config.risk_overlay,
            rebalance_frequency=rebal_freq,
            tax_enabled=config.tax_enabled,
            tax_rate=config.tax_rate,
            tax_exemption_amount=config.tax_exemption,
            metric_basis=config.metric_basis,
            allow_universe_lookahead=config.allow_universe_lookahead,
            validate=config.validate,
            drip_enabled=config.drip_enabled,
            external_features_loader=build_loader_from_config(config.external_features),
        )

        # Run backtest
        backtester = Backtester(strategy, sliced_data, bt_config)
        bt_result = backtester.run()

        # Recompute metrics for the window period only (excluding warmup)
        # The backtester runs on [warmup_start, window.end] but we want
        # metrics for [window.start, window.end] to get accurate window-specific results.
        from backtest.metrics import MetricsCalculator
        ec = bt_result.equity_curve
        window_ec = ec.loc[window.start:]
        if len(window_ec) >= 2:
            window_monthly = MetricsCalculator.monthly_returns(window_ec)
            window_cagr = MetricsCalculator.cagr(window_ec)
            window_vol = MetricsCalculator.volatility(window_monthly)
            window_maxdd = MetricsCalculator.max_drawdown(window_ec)
            window_sharpe = MetricsCalculator.sharpe_ratio(window_monthly, risk_free_rate=bt_config.risk_free_rate)
            window_sortino = MetricsCalculator.sortino_ratio(window_monthly, risk_free_rate=bt_config.risk_free_rate)
            window_calmar = MetricsCalculator.calmar_ratio(window_cagr, window_maxdd)
        else:
            # Fallback to full-period metrics if window trimming yields too few points
            m = bt_result.metrics
            window_cagr = m.cagr
            window_vol = m.volatility
            window_maxdd = m.max_drawdown
            window_sharpe = m.sharpe_ratio
            window_sortino = m.sortino_ratio
            window_calmar = m.calmar_ratio

        # Extract metrics (using window-specific values)
        m = bt_result.metrics
        result.metric_basis = config.metric_basis
        result.volatility = window_vol
        result.sharpe_ratio = window_sharpe
        result.sortino_ratio = window_sortino
        result.max_drawdown = window_maxdd
        result.calmar_ratio = window_calmar
        result.trades = m.num_trades
        result.costs_total = m.total_costs

        # Tax info and final values
        if bt_result.tax_summary is not None:
            ts = bt_result.tax_summary
            result.tax_paid_realized = ts.total_tax_paid
            result.tax_paid_liquidation = ts.tax_paid_liquidation
            result.final_value_gross = ts.final_value_gross
            result.final_value_net_realized = ts.final_value_net_realized
            result.final_value_net_liquidation = ts.final_value_net_liquidation

            # Recompute CAGR values for window period only (not including warmup)
            if bt_result.equity_curve_gross is not None and len(bt_result.equity_curve_gross.loc[window.start:]) >= 2:
                result.cagr_gross = MetricsCalculator.cagr(bt_result.equity_curve_gross.loc[window.start:])
            else:
                result.cagr_gross = ts.cagr_gross

            if bt_result.equity_curve_net is not None and len(bt_result.equity_curve_net.loc[window.start:]) >= 2:
                window_net_ec = bt_result.equity_curve_net.loc[window.start:]
                result.cagr_net_realized = MetricsCalculator.cagr(window_net_ec)
                # Net liquidation: adjust for virtual liquidation tax relative to net curve
                if ts.final_value_net_realized > 0:
                    liq_ratio = ts.final_value_net_liquidation / ts.final_value_net_realized
                    window_years = (window_net_ec.index[-1] - window_net_ec.index[0]).days / 365.25
                    if window_years > 0:
                        adjusted_final = window_net_ec.iloc[-1] * liq_ratio
                        result.cagr_net_liquidation = (adjusted_final / window_net_ec.iloc[0]) ** (1 / window_years) - 1
                    else:
                        result.cagr_net_liquidation = ts.cagr_net_liquidation
                else:
                    result.cagr_net_liquidation = ts.cagr_net_liquidation
            else:
                result.cagr_net_realized = ts.cagr_net_realized
                result.cagr_net_liquidation = ts.cagr_net_liquidation

            # Set primary metrics based on metric_basis mode
            if config.metric_basis == "gross":
                result.final_value = ts.final_value_gross
                result.cagr = result.cagr_gross
            elif config.metric_basis == "net_liquidation":
                result.final_value = ts.final_value_net_liquidation
                result.cagr = result.cagr_net_liquidation
            else:  # net_realized (default)
                result.final_value = ts.final_value_net_realized
                result.cagr = result.cagr_net_realized
        else:
            # No tax model - use window-specific CAGR
            result.final_value = bt_result.equity_curve.iloc[-1]
            result.cagr = window_cagr
            result.final_value_gross = result.final_value
            result.final_value_net_realized = result.final_value
            result.final_value_net_liquidation = result.final_value
            result.cagr_gross = window_cagr
            result.cagr_net_realized = window_cagr
            result.cagr_net_liquidation = window_cagr

        # Benchmark comparison
        if benchmark_result is not None:
            result.benchmark_final_value = benchmark_result.get("final_value")
            result.benchmark_cagr = benchmark_result.get("cagr")
            result.benchmark_maxdd = benchmark_result.get("max_drawdown")
            result.benchmark_sharpe = benchmark_result.get("sharpe_ratio")
            result.benchmark_sortino = benchmark_result.get("sortino_ratio")
            result.benchmark_vol = benchmark_result.get("volatility")
            result.benchmark_calmar = benchmark_result.get("calmar_ratio")
            # Compute excess CAGR
            if result.benchmark_cagr is not None:
                result.excess_cagr = result.cagr - result.benchmark_cagr

        result.status = "ok"

    except Exception as e:
        result.status = "error"
        result.skip_reason = str(e)

    return result


def aggregate_results(
    window_results: List[WindowResult],
    strategies: List[Tuple[str, str]],  # (name, file) tuples
) -> List[StrategySummary]:
    """
    Aggregate window results into strategy summaries.

    Args:
        window_results: List of all window results
        strategies: List of (strategy_name, strategy_file) tuples

    Returns:
        List of StrategySummary objects, ranked by median_cagr
    """
    summaries = []

    for strategy_name, strategy_file in strategies:
        # Filter results for this strategy
        results = [r for r in window_results if r.strategy_name == strategy_name and r.status == "ok"]
        all_results = [r for r in window_results if r.strategy_name == strategy_name]

        num_ok = len(results)
        num_skipped = len([r for r in all_results if r.status != "ok"])

        if num_ok == 0:
            # No successful runs
            summaries.append(StrategySummary(
                strategy_name=strategy_name,
                strategy_file=strategy_file,
                num_windows=len(all_results),
                num_ok=0,
                num_skipped=num_skipped,
            ))
            continue

        # Extract metrics arrays
        cagrs = np.array([r.cagr for r in results])
        sharpes = np.array([r.sharpe_ratio for r in results])
        sortinos = np.array([r.sortino_ratio for r in results])
        maxdds = np.array([r.max_drawdown for r in results])
        vols = np.array([r.volatility for r in results])
        calmars = np.array([r.calmar_ratio for r in results])
        costs = np.array([r.costs_total for r in results])
        taxes = np.array([r.tax_paid_total for r in results])

        # Terminal valuation metrics
        taxes_realized = np.array([r.tax_paid_realized for r in results])
        taxes_liquidation = np.array([r.tax_paid_liquidation for r in results])
        cagrs_gross = np.array([r.cagr_gross for r in results])
        cagrs_net_realized = np.array([r.cagr_net_realized for r in results])
        cagrs_net_liquidation = np.array([r.cagr_net_liquidation for r in results])

        # Benchmark comparison
        excess_cagrs = [r.excess_cagr for r in results if r.excess_cagr is not None]
        if excess_cagrs:
            hit_rate = sum(1 for e in excess_cagrs if e > 0) / len(excess_cagrs)
            median_excess = float(np.median(excess_cagrs))
            p10_excess = float(np.percentile(excess_cagrs, 10))
            prob_underperform = float(np.sum(np.array(excess_cagrs) < 0) / len(excess_cagrs))
        else:
            hit_rate = None
            median_excess = None
            p10_excess = None
            prob_underperform = None

        summary = StrategySummary(
            strategy_name=strategy_name,
            strategy_file=strategy_file,
            num_windows=len(all_results),
            num_ok=num_ok,
            num_skipped=num_skipped,
            # Return robustness
            median_cagr=float(np.median(cagrs)),
            p10_cagr=float(np.percentile(cagrs, 10)),
            p90_cagr=float(np.percentile(cagrs, 90)),
            worst_cagr=float(np.min(cagrs)),
            best_cagr=float(np.max(cagrs)),
            prob_negative_cagr=float(np.sum(cagrs < 0) / len(cagrs)),
            # Risk
            median_sharpe=float(np.median(sharpes)),
            p10_sharpe=float(np.percentile(sharpes, 10)),
            median_sortino=float(np.median(sortinos)),
            median_maxdd=float(np.median(maxdds)),
            worst_maxdd=float(np.min(maxdds)),  # Most negative
            median_vol=float(np.median(vols)),
            median_calmar=float(np.median(calmars)),
            # Friction
            median_costs_total=float(np.median(costs)),
            median_tax_paid_total=float(np.median(taxes)),
            median_tax_paid_realized=float(np.median(taxes_realized)),
            median_tax_paid_liquidation=float(np.median(taxes_liquidation)),
            # Terminal valuation CAGR variants
            median_cagr_gross=float(np.median(cagrs_gross)),
            median_cagr_net_realized=float(np.median(cagrs_net_realized)),
            median_cagr_net_liquidation=float(np.median(cagrs_net_liquidation)),
            p10_cagr_gross=float(np.percentile(cagrs_gross, 10)),
            p10_cagr_net_realized=float(np.percentile(cagrs_net_realized, 10)),
            p10_cagr_net_liquidation=float(np.percentile(cagrs_net_liquidation, 10)),
            # Benchmark
            hit_rate_vs_benchmark=hit_rate,
            median_excess_cagr=median_excess,
            p10_excess_cagr=p10_excess,
            prob_underperform_benchmark=prob_underperform,
        )
        summaries.append(summary)

    # Rank by median_cagr (descending)
    summaries.sort(key=lambda s: s.median_cagr, reverse=True)
    for i, s in enumerate(summaries):
        s.rank = i + 1

    # Identify Pareto front (non-dominated on median_cagr, p10_cagr, worst_maxdd)
    for s in summaries:
        is_dominated = False
        for other in summaries:
            if other is s:
                continue
            # Other dominates s if other is >= on all objectives and > on at least one
            # Objectives: max median_cagr, max p10_cagr, max worst_maxdd (less negative)
            if (other.median_cagr >= s.median_cagr and
                other.p10_cagr >= s.p10_cagr and
                other.worst_maxdd >= s.worst_maxdd):
                if (other.median_cagr > s.median_cagr or
                    other.p10_cagr > s.p10_cagr or
                    other.worst_maxdd > s.worst_maxdd):
                    is_dominated = True
                    break
        s.pareto_front = not is_dominated

    return summaries


def run_sweep(
    strategy_paths: List[Path],
    config: SweepConfig,
    progress: bool = True,
    progress_callback: Optional[callable] = None,
    cancel_check: Optional[callable] = None,
) -> SweepResult:
    """
    Run sweep analysis on multiple strategies.

    Args:
        strategy_paths: List of paths to strategy files
        config: Sweep configuration
        progress: Whether to print progress messages

    Returns:
        SweepResult with all window results and summaries
    """
    import warnings
    import time

    # Suppress validation warnings during sweep (they flood the output)
    warnings.filterwarnings("ignore", module="backtest.validation")
    warnings.filterwarnings("ignore", category=FutureWarning)

    if progress:
        print(f"\n{'=' * 60}")
        print(" SWEEP ANALYSIS")
        print(f"{'=' * 60}\n")

    # Load strategies
    if progress:
        print(f"Loading {len(strategy_paths)} strategies...")

    strategies = []
    strategy_files = {}

    if progress_callback is not None:
        progress_callback(
            status="loading_strategies",
            run_count=0,
            total_runs=0,
            strategy_index=0,
            strategy_total=len(strategy_paths),
            strategy_name=None,
            iteration_index=0,
            iteration_total=0,
        )

    for path in strategy_paths:
        try:
            from backtest.cli import load_strategy_from_file
            strategy = load_strategy_from_file(str(path))
            strategies.append(strategy)
            strategy_files[strategy.name] = str(path)
            if progress_callback is not None:
                progress_callback(
                    status="loading_strategies",
                    run_count=0,
                    total_runs=0,
                    strategy_index=len(strategies),
                    strategy_total=len(strategy_paths),
                    strategy_name=strategy.name,
                    iteration_index=0,
                    iteration_total=0,
                )
            if progress:
                print(f"  + {strategy.name}")
        except Exception as e:
            if progress:
                print(f"  - {path}: {e}")
            if config.fail_fast:
                raise

    return run_sweep_with_strategies(
        strategies=strategies,
        strategy_files=strategy_files,
        config=config,
        progress=progress,
        timer=time.perf_counter,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )


def run_sweep_with_strategies(
    strategies: List["Strategy"],
    strategy_files: Dict[str, str],
    config: SweepConfig,
    progress: bool = True,
    timer: Optional[callable] = None,
    progress_callback: Optional[callable] = None,
    cancel_check: Optional[callable] = None,
) -> SweepResult:
    """
    Run sweep analysis with pre-loaded strategy instances.

    Args:
        strategies: List of Strategy instances
        strategy_files: Mapping of strategy name -> file path
        config: Sweep configuration
        progress: Whether to print progress messages
        timer: Optional timing function for progress reporting

    Returns:
        SweepResult with all window results and summaries
    """
    import warnings
    from backtest.data import DataLoader

    # Suppress validation warnings during sweep (they flood the output)
    warnings.filterwarnings("ignore", module="backtest.validation")
    warnings.filterwarnings("ignore", category=FutureWarning)

    if not strategies:
        raise ValueError("No valid strategies loaded")

    all_assets = set()
    for strategy in strategies:
        all_assets.update(strategy.assets)

    # Add benchmark asset
    all_assets.add(config.benchmark_ticker)

    # Calculate date range for data download
    window_td = parse_window_length(config.window_length) if config.window_length else None

    # Determine download range
    from_date = config.from_date
    end_date = config.end_date

    if progress:
        print(f"\nLoading price data for {len(all_assets)} assets...")

    # Load data with warmup
    # We need to start earlier to allow for warmup
    download_start = from_date
    if download_start and config.warmup_days > 0:
        download_start_dt = datetime.strptime(download_start, "%Y-%m-%d") - timedelta(days=config.warmup_days + 30)
        download_start = download_start_dt.strftime("%Y-%m-%d")

    if progress_callback is not None:
        progress_callback(
            status="loading_data",
            run_count=0,
            total_runs=0,
            strategy_index=0,
            strategy_total=len(strategies),
            strategy_name=None,
            iteration_index=0,
            iteration_total=0,
        )

    load_volumes = (
        (config.max_volume_participation is not None and config.max_volume_participation > 0)
        or config.min_daily_dollar_volume > 0
    )
    data = DataLoader.yahoo(
        tickers=list(all_assets),
        start=download_start or "2000-01-01",
        end=end_date,
        currency="EUR",
        align=config.align,
        skip_failed=config.skip_failed,
        load_dividends=config.drip_enabled or config.tax_enabled,
        load_volumes=load_volumes,
        validate=config.validate,
    )

    if progress:
        print(f"Loaded {len(data.prices)} trading days")
        print(f"Date range: {data.start_date.strftime('%Y-%m-%d')} -> {data.end_date.strftime('%Y-%m-%d')}")

    # Generate windows
    if progress:
        print(f"\nGenerating windows (mode={config.mode}, grid={config.start_grid})...")

    if progress_callback is not None:
        progress_callback(
            status="generating_windows",
            run_count=0,
            total_runs=0,
            strategy_index=0,
            strategy_total=len(strategies),
            strategy_name=None,
            iteration_index=0,
            iteration_total=0,
        )

    from_dt = datetime.strptime(config.from_date, "%Y-%m-%d") if config.from_date else None
    to_dt = datetime.strptime(config.to_date, "%Y-%m-%d") if config.to_date else None
    end_dt = datetime.strptime(config.end_date, "%Y-%m-%d") if config.end_date else None

    windows = generate_windows(
        data_index=data.prices.index,
        mode=config.mode,
        end_date=end_dt,
        from_date=from_dt,
        to_date=to_dt,
        window_length=window_td,
        start_grid=config.start_grid,
        step=config.step,
    )

    if not windows:
        # Provide helpful error message
        data_start = data.prices.index[0].strftime('%Y-%m-%d')
        data_end = data.prices.index[-1].strftime('%Y-%m-%d')
        msg = f"No valid windows generated.\n"
        msg += f"  Data available: {data_start} to {data_end}\n"
        msg += f"  Requested --from: {config.from_date or 'not set'}\n"
        msg += f"  Window length: {config.window_length}\n"
        if from_dt and from_dt < data.prices.index[0]:
            msg += f"\n  HINT: Data starts at {data_start}, but you requested --from {config.from_date}.\n"
            msg += f"        Try --from {data_start} or a later date."
        raise ValueError(msg)

    total_runs = len(windows) * len(strategies)
    if progress_callback is not None:
        progress_callback(
            status="running",
            run_count=0,
            total_runs=total_runs,
            strategy_index=0,
            strategy_total=len(strategies),
            strategy_name=None,
            iteration_index=0,
            iteration_total=len(windows),
        )

    if progress:
        print(f"Generated {len(windows)} windows")
        if windows:
            print(f"  First: {windows[0]}")
            print(f"  Last:  {windows[-1]}")

    def format_duration(seconds: float) -> str:
        if seconds < 0:
            return "0s"
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    if progress:
        print(f"\nRunning {total_runs} backtests ({len(strategies)} strategies x {len(windows)} windows)...")

    window_results = []
    run_count = 0
    start_time = timer() if timer is not None else None

    max_workers = max(1, int(config.jobs or 1))

    for window_idx, window in enumerate(windows, 1):
        if cancel_check is not None and cancel_check():
            raise SweepCancelled("Sweep cancelled by user.")
        # Pre-compute benchmark for this window (once, shared by all strategies)
        benchmark_result = compute_benchmark_for_window(
            benchmark_ticker=config.benchmark_ticker,
            data=data,
            window=window,
            config=config,
        )

        if max_workers == 1 or len(strategies) == 1:
            for strategy_idx, strategy in enumerate(strategies, 1):
                if cancel_check is not None and cancel_check():
                    raise SweepCancelled("Sweep cancelled by user.")

                strategy_run = _clone_strategy_instance(strategy)
                run_count += 1
                if progress_callback is not None:
                    progress_callback(
                        run_count=run_count,
                        total_runs=total_runs,
                        strategy_index=strategy_idx,
                        strategy_total=len(strategies),
                        strategy_name=strategy.name,
                        iteration_index=window_idx,
                        iteration_total=len(windows),
                    )
                if progress and run_count % 10 == 0 and start_time is not None:
                    elapsed = timer() - start_time
                    avg_time = elapsed / run_count if run_count else 0
                    eta = avg_time * (total_runs - run_count)
                    pct = run_count / total_runs * 100
                    print(
                        f"  Progress: {run_count}/{total_runs} ({pct:.0f}%) | "
                        f"Strategy: {strategy_idx}/{len(strategies)} ({strategy.name}) | "
                        f"Window: {window_idx}/{len(windows)} | "
                        f"Elapsed: {format_duration(elapsed)} | ETA: {format_duration(eta)}"
                    )

                try:
                    result = run_single_window(
                        strategy=strategy_run,
                        strategy_file=strategy_files.get(strategy.name, ""),
                        data=data,
                        window=window,
                        config=config,
                        benchmark_result=benchmark_result,
                    )
                    window_results.append(result)
                except Exception as e:
                    if config.fail_fast:
                        raise
                    window_results.append(WindowResult(
                        strategy_name=strategy.name,
                        strategy_file=strategy_files.get(strategy.name, ""),
                        window_start=window.start,
                        window_end=window.end,
                        window_years=window.years,
                        status="error",
                        skip_reason=str(e),
                    ))
        else:
            future_to_meta = {}
            with ThreadPoolExecutor(max_workers=min(max_workers, len(strategies))) as executor:
                for strategy_idx, strategy in enumerate(strategies, 1):
                    if cancel_check is not None and cancel_check():
                        raise SweepCancelled("Sweep cancelled by user.")
                    strategy_run = _clone_strategy_instance(strategy)
                    future = executor.submit(
                        run_single_window,
                        strategy_run,
                        strategy_files.get(strategy.name, ""),
                        data,
                        window,
                        config,
                        benchmark_result,
                    )
                    future_to_meta[future] = (strategy_idx, strategy.name)

                ordered_results: Dict[int, WindowResult] = {}
                for future in as_completed(future_to_meta):
                    strategy_idx, strategy_name = future_to_meta[future]
                    run_count += 1
                    if progress_callback is not None:
                        progress_callback(
                            run_count=run_count,
                            total_runs=total_runs,
                            strategy_index=strategy_idx,
                            strategy_total=len(strategies),
                            strategy_name=strategy_name,
                            iteration_index=window_idx,
                            iteration_total=len(windows),
                        )
                    if progress and run_count % 10 == 0 and start_time is not None:
                        elapsed = timer() - start_time
                        avg_time = elapsed / run_count if run_count else 0
                        eta = avg_time * (total_runs - run_count)
                        pct = run_count / total_runs * 100
                        print(
                            f"  Progress: {run_count}/{total_runs} ({pct:.0f}%) | "
                            f"Strategy: {strategy_idx}/{len(strategies)} ({strategy_name}) | "
                            f"Window: {window_idx}/{len(windows)} | "
                            f"Elapsed: {format_duration(elapsed)} | ETA: {format_duration(eta)}"
                        )
                    try:
                        ordered_results[strategy_idx] = future.result()
                    except Exception as e:
                        if config.fail_fast:
                            raise
                        ordered_results[strategy_idx] = WindowResult(
                            strategy_name=strategy_name,
                            strategy_file=strategy_files.get(strategy_name, ""),
                            window_start=window.start,
                            window_end=window.end,
                            window_years=window.years,
                            status="error",
                            skip_reason=str(e),
                        )

                for strategy_idx in sorted(ordered_results):
                    window_results.append(ordered_results[strategy_idx])

    if progress:
        print(f"  Completed {run_count} runs")

    # Aggregate results
    if progress:
        print("\nAggregating results...")

    strategy_tuples = [(s.name, strategy_files.get(s.name, "")) for s in strategies]
    summaries = aggregate_results(window_results, strategy_tuples)

    # Print summary
    if progress:
        print("\n" + "=" * 100)
        print(" SWEEP SUMMARY")
        print(f" Metric Basis: {config.metric_basis.replace('_', ' ').title()}")
        print("=" * 100)
        print(f"\n{'Rank':<5} {'Strategy':<30} {'CAGR Gross':>11} {'CAGR Net':>10} {'Tax Drag':>9} {'P10 Net':>9} {'Worst DD':>9} {'Pareto':>7}")
        print("-" * 100)
        for s in summaries:
            pareto_mark = "✓" if s.pareto_front else ""
            if config.metric_basis == "net_liquidation":
                cagr_net = s.median_cagr_net_liquidation
                p10_net = s.p10_cagr_net_liquidation
            elif config.metric_basis == "net_realized":
                cagr_net = s.median_cagr_net_realized
                p10_net = s.p10_cagr_net_realized
            else:
                cagr_net = s.median_cagr_gross
                p10_net = s.p10_cagr_gross
            tax_drag_pp = (s.median_cagr_gross - cagr_net) * 100
            print(f"{s.rank:<5} {s.strategy_name:<30} {s.median_cagr_gross:>10.1%} {cagr_net:>9.1%} {tax_drag_pp:>8.1f}pp {p10_net:>8.1%} {s.worst_maxdd:>8.1%} {pareto_mark:>7}")
        print("=" * 100)

    return SweepResult(
        config=config,
        windows=windows,
        window_results=window_results,
        summaries=summaries,
    )


def save_sweep_results(result: SweepResult, output_dir: Path) -> None:
    """
    Save sweep results to files.

    Args:
        result: SweepResult to save
        output_dir: Directory to save files to
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save windows.csv
    windows_df = result.to_windows_df()
    windows_df.to_csv(output_dir / "windows.csv", index=False)

    # Save summary.csv
    summary_df = result.to_summary_df()
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    # Save metadata JSON with complete information
    config = result.config
    metadata = {
        # Sweep configuration
        "mode": config.mode,
        "window_length": config.window_length,
        "start_grid": config.start_grid,
        "step": config.step,
        "from_date": config.from_date,
        "to_date": config.to_date,
        "end_date": config.end_date,
        "warmup_days": config.warmup_days,
        # Backtest settings
        "initial_capital": config.initial_capital,
        "rebalance_frequency": config.rebalance_frequency,
        "costs_enabled": config.costs_enabled,
        "costs_pct": config.costs_pct,
        "tax_enabled": config.tax_enabled,
        "tax_rate": config.tax_rate,
        "tax_exemption": config.tax_exemption,
        "metric_basis": config.metric_basis,
        "headline_metric_basis": "net" if config.tax_enabled else "gross",
        # Benchmark
        "benchmark_ticker": config.benchmark_ticker,
        # Summary stats
        "num_windows": len(result.windows),
        "num_strategies": len(result.summaries),
        "first_window_start": result.windows[0].start.strftime("%Y-%m-%d") if result.windows else None,
        "last_window_end": result.windows[-1].end.strftime("%Y-%m-%d") if result.windows else None,
        # Execution info
        "generated_at": datetime.now().isoformat(),
        "framework_version": "0.1",
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - windows.csv ({len(windows_df)} rows)")
    print(f"  - summary.csv ({len(summary_df)} rows)")
    print(f"  - run_metadata.json")


def render_sweep_html(result: SweepResult, output_path: Path) -> None:
    """
    Render sweep results as an HTML report.

    Args:
        result: SweepResult to render
        output_path: Path for output HTML file
    """
    import json
    from datetime import datetime

    config = result.config
    summaries = result.summaries
    windows_df = result.to_windows_df()

    # Determine headline metric basis
    headline_basis = "NET" if config.tax_enabled else "GROSS"
    tax_status = "Enabled" if config.tax_enabled else "Disabled"
    metric_basis_display = config.metric_basis.replace("_", " ").title()

    # Build summary table rows
    summary_rows = ""
    for s in summaries:
        pareto_badge = '<span class="badge pareto">Pareto</span>' if s.pareto_front else ""
        hit_rate = f"{s.hit_rate_vs_benchmark:.0%}" if s.hit_rate_vs_benchmark is not None else "N/A"
        median_excess = f"{s.median_excess_cagr:+.1%}" if s.median_excess_cagr is not None else "N/A"
        excess_class = "positive" if s.median_excess_cagr and s.median_excess_cagr > 0 else "negative" if s.median_excess_cagr else ""
        calmar_display = f"{s.median_calmar:.2f}" if abs(s.median_calmar) < 100 else "∞"

        # Determine which net CAGR to use based on metric_basis
        if config.metric_basis == "net_liquidation":
            cagr_net = s.median_cagr_net_liquidation
            p10_net = s.p10_cagr_net_liquidation
        elif config.metric_basis == "net_realized":
            cagr_net = s.median_cagr_net_realized
            p10_net = s.p10_cagr_net_realized
        else:
            cagr_net = s.median_cagr_gross
            p10_net = s.p10_cagr_gross

        # Tax drag in percentage points
        tax_drag_pp = (s.median_cagr_gross - cagr_net) * 100

        summary_rows += f"""
        <tr>
            <td>{s.rank}</td>
            <td><strong>{s.strategy_name}</strong> {pareto_badge}</td>
            <td class="{'positive' if s.median_cagr_gross > 0 else 'negative'}">{s.median_cagr_gross:.1%}</td>
            <td class="{'positive' if cagr_net > 0 else 'negative'}">{cagr_net:.1%}</td>
            <td>{tax_drag_pp:.1f}pp</td>
            <td class="{'positive' if p10_net > 0 else 'negative'}">{p10_net:.1%}</td>
            <td>{s.median_sharpe:.2f}</td>
            <td>{calmar_display}</td>
            <td class="negative">{s.worst_maxdd:.1%}</td>
            <td class="{excess_class}">{median_excess}</td>
            <td>{hit_rate}</td>
            <td>{s.num_ok}/{s.num_windows}</td>
        </tr>
        """

    # Collect git and environment metadata
    from backtest.metadata import collect_git_metadata, collect_environment_metadata
    git_info = collect_git_metadata()
    env_info = collect_environment_metadata()

    # Build metadata JSON for embedding
    metadata_json = json.dumps({
        # Run identification
        "run_mode": "sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat() if hasattr(datetime, 'now') else datetime.now().isoformat(),
        "framework_version": "0.1",
        # Git info
        "git": git_info.to_dict(),
        # Environment
        "environment": env_info.to_dict(),
        # Sweep configuration (windowing)
        "windowing": {
            "mode": config.mode,
            "window_length": config.window_length,
            "start_grid": config.start_grid,
            "step": config.step,
            "from_date": config.from_date,
            "to_date": config.to_date,
            "end_date": config.end_date,
            "warmup_days": config.warmup_days,
            "num_windows": len(result.windows),
            "first_window_start": result.windows[0].start.strftime("%Y-%m-%d") if result.windows else None,
            "last_window_end": result.windows[-1].end.strftime("%Y-%m-%d") if result.windows else None,
        },
        # Backtest settings
        "backtest": {
            "initial_capital": config.initial_capital,
            "rebalance_frequency": config.rebalance_frequency,
        },
        # Costs
        "costs": {
            "enabled": config.costs_enabled,
            "costs_pct": config.costs_pct,
        },
        # Tax
        "tax": {
            "enabled": config.tax_enabled,
            "tax_rate": config.tax_rate,
            "exemption_amount": config.tax_exemption,
            "metric_basis": config.metric_basis,
            "headline_metric_basis": "net" if config.tax_enabled else "gross",
        },
        # Benchmark
        "benchmark": {
            "ticker": config.benchmark_ticker,
            "enabled": True,
        },
        # Summary stats
        "num_strategies": len(summaries),
        # Detailed results
        "summaries": [s.to_dict() for s in summaries],
    }, indent=2)

    # Window range info
    if result.windows:
        first_window = result.windows[0]
        last_window = result.windows[-1]
        window_info = f"{first_window.start.strftime('%Y-%m-%d')} → {last_window.end.strftime('%Y-%m-%d')}"
    else:
        window_info = "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sweep Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #1f2937;
            background: #f9fafb;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        header {{
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }}
        header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        header p {{
            opacity: 0.9;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .config-card {{
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .config-card h3 {{
            font-size: 0.75rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }}
        .config-card .value {{
            font-size: 1.25rem;
            font-weight: 600;
        }}
        section {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        section h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: #1f2937;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background: #f9fafb;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        tr:hover {{
            background: #f9fafb;
        }}
        .positive {{
            color: #16a34a;
        }}
        .negative {{
            color: #dc2626;
        }}
        .badge {{
            display: inline-block;
            padding: 0.125rem 0.5rem;
            border-radius: 9999px;
            font-size: 0.625rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .badge.pareto {{
            background: #ddd6fe;
            color: #7c3aed;
        }}
        .chart-container {{
            width: 100%;
            min-height: 400px;
        }}
        footer {{
            text-align: center;
            padding: 2rem;
            color: #6b7280;
            font-size: 0.875rem;
        }}
        .section-title {{
            font-size: 1rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e5e7eb;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Sweep Analysis Report</h1>
            <p>{len(summaries)} strategies | {len(result.windows)} windows | {window_info}</p>
            <p style="margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.9;">
                <strong>Metric Basis:</strong> {metric_basis_display} |
                <strong>Tax:</strong> {tax_status}{f' ({config.tax_rate:.1%})' if config.tax_enabled else ''} |
                <strong>Benchmark:</strong> {config.benchmark_ticker}
            </p>
        </header>

        <p class="section-title">Sweep Configuration</p>
        <div class="config-grid">
            <div class="config-card">
                <h3>Mode</h3>
                <div class="value">{config.mode}</div>
            </div>
            <div class="config-card">
                <h3>Window Length</h3>
                <div class="value">{config.window_length or 'N/A'}</div>
            </div>
            <div class="config-card">
                <h3>Start Grid</h3>
                <div class="value">{config.start_grid} (step={config.step})</div>
            </div>
            <div class="config-card">
                <h3>Windows</h3>
                <div class="value">{len(result.windows)}</div>
            </div>
            <div class="config-card">
                <h3>Rebalance</h3>
                <div class="value">{config.rebalance_frequency}</div>
            </div>
            <div class="config-card">
                <h3>Tax</h3>
                <div class="value">{'Enabled' if config.tax_enabled else 'Disabled'}{f' ({config.tax_rate:.1%})' if config.tax_enabled else ''}</div>
            </div>
            <div class="config-card">
                <h3>Costs</h3>
                <div class="value">{'Enabled' if config.costs_enabled else 'Disabled'}</div>
            </div>
            <div class="config-card">
                <h3>Metric Basis</h3>
                <div class="value" style="color: {'#7c3aed' if config.metric_basis == 'net_liquidation' else '#6b7280'};">{metric_basis_display}</div>
            </div>
            <div class="config-card">
                <h3>Benchmark</h3>
                <div class="value">{config.benchmark_ticker}</div>
            </div>
        </div>

        <section>
            <h2>Strategy Summary (Ranked by Median CAGR, {headline_basis})</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Strategy</th>
                        <th>CAGR Gross</th>
                        <th>CAGR Net</th>
                        <th>Tax Drag</th>
                        <th>P10 CAGR</th>
                        <th>Median Sharpe</th>
                        <th>Median Calmar</th>
                        <th>Worst DD</th>
                        <th>Median Excess</th>
                        <th>Hit Rate</th>
                        <th>OK/Total</th>
                    </tr>
                </thead>
                <tbody>
                    {summary_rows}
                </tbody>
            </table>
            <p style="font-size: 0.85rem; color: #6b7280; margin-top: 0.5rem;">
                <strong>Tax Drag</strong> = CAGR Gross - CAGR Net (in percentage points).
                <strong>Hit Rate</strong> = % of windows where strategy beat benchmark.
                <strong>Median Excess</strong> = Median outperformance vs benchmark.
            </p>
        </section>

        <section>
            <h2>CAGR Distribution by Strategy</h2>
            <div class="chart-container" id="cagr-chart"></div>
        </section>

        <section>
            <h2>Max Drawdown Distribution by Strategy</h2>
            <div class="chart-container" id="maxdd-chart"></div>
        </section>

        <section>
            <h2>Definitions (Glossary)</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem;">
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #7c3aed;">CAGR</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Compound Annual Growth Rate. The annualized return that would have been required to grow the initial investment to its final value over the period.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #7c3aed;">Volatility</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Annualized standard deviation of returns, measuring the dispersion of returns around the mean. Higher volatility indicates higher risk.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #7c3aed;">Sharpe Ratio</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Risk-adjusted return: (Return - Risk-Free Rate) / Volatility. Higher is better. Assumes risk-free rate of 2%.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #7c3aed;">Sortino Ratio</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Like Sharpe, but uses only downside deviation (negative returns) instead of total volatility. Penalizes only harmful volatility.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #7c3aed;">Max Drawdown</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Maximum peak-to-trough decline during the investment period. Measures worst-case loss scenario.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #7c3aed;">Calmar Ratio</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">CAGR divided by absolute Max Drawdown. Measures return per unit of drawdown risk. Higher is better.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #7c3aed;">Tax Drag</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">The reduction in returns due to taxation. Calculated as CAGR_gross - CAGR_net.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #7c3aed;">Pareto Front</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Strategies that are not dominated by any other strategy on all key metrics (CAGR, risk). Non-dominated means there's no other strategy that is strictly better in all dimensions.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #7c3aed;">Metric Basis</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;"><strong>Gross:</strong> Before tax. <strong>Net Realized:</strong> After realized taxes only. <strong>Net Liquidation:</strong> After all taxes including virtual end-of-period liquidation tax on unrealized gains.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #7c3aed;">P10 / Median / P90</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Percentile statistics across all windows. P10 = 10th percentile (worst 10%), Median = 50th percentile, P90 = 90th percentile (best 10%).</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #7c3aed;">Hit Rate vs Benchmark</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Percentage of windows where the strategy outperformed the benchmark (CAGR_strategy > CAGR_benchmark).</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #7c3aed;">Benchmark</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">A Buy-and-Hold strategy on {config.benchmark_ticker} used as reference for comparison. All strategies are compared against this baseline.</p>
                </div>
            </div>
        </section>

        <footer>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Framework: backtest v0.1</p>
        </footer>
    </div>

    <script type="application/json" id="run-metadata">
{metadata_json}
    </script>

    <script>
        // CAGR box plot
        const summaries = {json.dumps([s.to_dict() for s in summaries])};

        const cagrTraces = summaries.map((s, i) => ({{
            type: 'box',
            name: s.strategy_name.substring(0, 20),
            y: [s.worst_cagr, s.p10_cagr, s.median_cagr, s.p90_cagr, s.best_cagr],
            boxpoints: false,
            marker: {{ color: s.pareto_front ? '#7c3aed' : '#6b7280' }},
        }}));

        Plotly.newPlot('cagr-chart', cagrTraces, {{
            title: 'CAGR Range (P10 / Median / P90)',
            yaxis: {{ title: 'CAGR', tickformat: '.0%' }},
            showlegend: false,
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
        }});

        // MaxDD box plot
        const maxddData = summaries.map((s, i) => ({{
            type: 'bar',
            name: s.strategy_name.substring(0, 20),
            x: [s.strategy_name.substring(0, 20)],
            y: [s.worst_maxdd],
            marker: {{ color: s.pareto_front ? '#7c3aed' : '#dc2626' }},
        }}));

        Plotly.newPlot('maxdd-chart', maxddData, {{
            title: 'Worst Max Drawdown by Strategy',
            yaxis: {{ title: 'Max Drawdown', tickformat: '.0%' }},
            showlegend: false,
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
        }});
    </script>
</body>
</html>"""

    output_path.write_text(html)
    print(f"Report saved to: {output_path}")
