"""
Utility functions for the backtest framework.

Includes:
- Date parsing and formatting
- Currency and percentage formatting
- Frequency inference and annualization utilities
- Centralized metrics calculations
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """
    Parse a date string in YYYY-MM-DD format.

    Args:
        date_str: Date string or None

    Returns:
        datetime object or None
    """
    if date_str is None:
        return None
    return datetime.strptime(date_str, "%Y-%m-%d")


def format_date(date: datetime) -> str:
    """
    Format a datetime as YYYY-MM-DD string.

    Args:
        date: datetime object

    Returns:
        Formatted date string
    """
    return date.strftime("%Y-%m-%d")


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_currency(value: float, currency: str = "EUR") -> str:
    """
    Format a value as currency.

    Args:
        value: Numeric value
        currency: Currency code

    Returns:
        Formatted currency string
    """
    symbols = {
        "EUR": "€",
        "USD": "$",
        "GBP": "£",
    }
    symbol = symbols.get(currency, currency + " ")
    return f"{symbol}{value:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a decimal as percentage.

    Args:
        value: Decimal value (e.g., 0.05 for 5%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


# =============================================================================
# Frequency and Annualization Utilities
# =============================================================================

def month_end_freq() -> str:
    """
    Return the correct month-end frequency string for pandas resample.

    Returns 'ME' for pandas >= 2.2, falls back to 'M' for older versions.
    This handles the deprecation of 'M' in favor of 'ME'.
    """
    try:
        # Prefer explicit month-end alias on newer pandas.
        to_offset("ME")
        return "ME"
    except ValueError:
        # Fall back to deprecated 'M' for older pandas.
        return "M"


# Cache the result to avoid repeated checks
MONTH_END_FREQ = month_end_freq()


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """
    Infer the number of periods per year from a DatetimeIndex.

    Detects whether data is daily, weekly, monthly, quarterly, or yearly.

    Args:
        index: DatetimeIndex to analyze

    Returns:
        Approximate number of periods per year:
        - Daily: 252 (trading days)
        - Weekly: 52
        - Monthly: 12
        - Quarterly: 4
        - Yearly: 1

    Raises:
        ValueError: If frequency cannot be inferred
    """
    if len(index) < 2:
        return 12  # Default to monthly

    # Calculate median days between observations
    diffs = pd.Series(index).diff().dropna()
    median_days = diffs.dt.days.median()

    if median_days <= 1.5:
        return 252  # Daily (trading days)
    elif median_days <= 8:
        return 52  # Weekly
    elif median_days <= 35:
        return 12  # Monthly
    elif median_days <= 100:
        return 4  # Quarterly
    else:
        return 1  # Yearly


def infer_frequency_name(index: pd.DatetimeIndex) -> str:
    """
    Infer frequency name from DatetimeIndex.

    Args:
        index: DatetimeIndex to analyze

    Returns:
        Frequency name: 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
    """
    periods = infer_periods_per_year(index)
    if periods >= 200:
        return "daily"
    elif periods >= 40:
        return "weekly"
    elif periods >= 10:
        return "monthly"
    elif periods >= 3:
        return "quarterly"
    else:
        return "yearly"


# =============================================================================
# Centralized Metrics Calculations
# =============================================================================

def annualized_return(returns: pd.Series, periods_per_year: Optional[int] = None) -> float:
    """
    Calculate annualized return from a series of periodic returns.

    Args:
        returns: Series of periodic returns (e.g., daily or monthly)
        periods_per_year: Number of periods per year. If None, inferred from index.

    Returns:
        Annualized return as decimal (e.g., 0.08 = 8%)
    """
    if len(returns) < 1:
        return 0.0

    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(returns.index)

    # Compound the returns
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year

    if years <= 0:
        return 0.0

    # Annualize: (1 + total)^(1/years) - 1
    if total_return <= -1:
        return -1.0
    return (1 + total_return) ** (1 / years) - 1


def annualized_vol(returns: pd.Series, periods_per_year: Optional[int] = None) -> float:
    """
    Calculate annualized volatility from a series of returns.

    Formula: vol * sqrt(periods_per_year)

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year. If None, inferred from index.

    Returns:
        Annualized volatility as decimal
    """
    if len(returns) < 2:
        return 0.0

    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(returns.index)

    return returns.std() * np.sqrt(periods_per_year)


def cagr(equity_curve: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate from equity curve.

    Formula: (V_end / V_start)^(1/years) - 1

    Args:
        equity_curve: Series of portfolio values with DatetimeIndex

    Returns:
        CAGR as decimal (e.g., 0.08 = 8%)
    """
    if len(equity_curve) < 2:
        return 0.0

    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]

    if start_value <= 0:
        return 0.0

    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0:
        return 0.0

    total_return = end_value / start_value
    if total_return <= 0:
        return -1.0

    return total_return ** (1 / years) - 1


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int] = None
) -> float:
    """
    Calculate Sharpe Ratio.

    Formula: (annualized_return - rf) / annualized_volatility

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (e.g., 0.02 = 2%)
        periods_per_year: Number of periods per year. If None, inferred from index.

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(returns.index)

    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = annualized_vol(returns, periods_per_year)

    if ann_vol == 0:
        return 0.0

    return (ann_return - risk_free_rate) / ann_vol


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int] = None
) -> float:
    """
    Calculate Sortino Ratio (using only downside volatility).

    Formula: (annualized_return - rf) / downside_volatility

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year. If None, inferred from index.

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(returns.index)

    ann_return = annualized_return(returns, periods_per_year)

    # Downside deviation (only negative returns)
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return float("inf")

    downside_vol = negative_returns.std() * np.sqrt(periods_per_year)
    if downside_vol == 0:
        return float("inf")

    return (ann_return - risk_free_rate) / downside_vol


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Formula: min((V_t - peak_t) / peak_t)

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Maximum drawdown as negative decimal (e.g., -0.20 = -20%)
    """
    if len(equity_curve) < 2:
        return 0.0

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    """
    Calculate the duration of the longest drawdown period in days.

    Uses vectorized pandas operations for better performance on large datasets.

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Number of days in the longest drawdown period
    """
    if len(equity_curve) < 2:
        return 0

    rolling_max = equity_curve.cummax()
    is_drawdown = equity_curve < rolling_max

    if not is_drawdown.any():
        return 0

    # Identify drawdown period groups using cumsum on state changes
    # Each time we exit/enter a drawdown, the group number increments
    drawdown_groups = (~is_drawdown).cumsum()

    # Filter to only drawdown periods
    drawdown_periods = drawdown_groups[is_drawdown]

    if len(drawdown_periods) == 0:
        return 0

    # Calculate duration for each drawdown group
    # Group by the period identifier and compute (last_date - first_date).days
    def group_duration(group):
        if len(group) < 1:
            return 0
        return (group.index[-1] - group.index[0]).days

    durations = drawdown_periods.groupby(drawdown_periods).apply(group_duration)

    return int(durations.max()) if len(durations) > 0 else 0


def calmar_ratio(cagr_value: float, max_dd: float) -> float:
    """
    Calculate Calmar Ratio.

    Formula: CAGR / |MaxDD|

    Args:
        cagr_value: Compound annual growth rate
        max_dd: Maximum drawdown (as negative value)

    Returns:
        Calmar ratio
    """
    if max_dd >= 0:
        return float("inf") if cagr_value > 0 else 0.0

    return cagr_value / abs(max_dd)


def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate period returns from equity curve.

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Series of period returns
    """
    return equity_curve.pct_change().dropna()


def monthly_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate monthly returns from equity curve.

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Series of monthly returns
    """
    # Forward-fill to daily first to handle quarterly/sparse data correctly
    # This ensures months between rebalance dates have values
    daily = equity_curve.resample('D').last().ffill()
    monthly = daily.resample(MONTH_END_FREQ).last()
    return monthly.pct_change().dropna()


def win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        returns: Series of returns

    Returns:
        Win rate as decimal (e.g., 0.6 = 60%)
    """
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


def tracking_difference(
    strategy_curve: pd.Series,
    benchmark_curve: pd.Series
) -> float:
    """
    Calculate annualized tracking difference vs benchmark.

    Args:
        strategy_curve: Strategy equity curve
        benchmark_curve: Benchmark equity curve

    Returns:
        Annualized tracking difference (strategy CAGR - benchmark CAGR)
    """
    strategy_cagr = cagr(strategy_curve)
    benchmark_cagr = cagr(benchmark_curve)
    return strategy_cagr - benchmark_cagr


def tracking_error(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: Optional[int] = None
) -> float:
    """
    Calculate annualized tracking error vs benchmark.

    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year. If None, inferred.

    Returns:
        Annualized tracking error (std of return differences)
    """
    # Align returns
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0

    aligned.columns = ["strategy", "benchmark"]
    diff = aligned["strategy"] - aligned["benchmark"]

    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(diff.index)

    return diff.std() * np.sqrt(periods_per_year)


def information_ratio(
    strategy_curve: pd.Series,
    benchmark_curve: pd.Series,
    periods_per_year: Optional[int] = None
) -> float:
    """
    Calculate Information Ratio.

    Formula: tracking_difference / tracking_error

    Args:
        strategy_curve: Strategy equity curve
        benchmark_curve: Benchmark equity curve
        periods_per_year: Number of periods per year. If None, inferred.

    Returns:
        Information ratio
    """
    strategy_returns = calculate_returns(strategy_curve)
    benchmark_returns = calculate_returns(benchmark_curve)

    td = tracking_difference(strategy_curve, benchmark_curve)
    te = tracking_error(strategy_returns, benchmark_returns, periods_per_year)

    if te == 0:
        return 0.0

    return td / te
