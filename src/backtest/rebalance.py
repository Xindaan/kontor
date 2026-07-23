"""
Rebalance date generator module.

This module provides functionality to generate rebalance dates from a daily price index.
The key insight is that rebalance frequency only controls WHEN signal() is called,
not the data frequency - strategies always receive daily historical data.

Supported frequencies:
- daily: Every trading day
- weekly: Last trading day of each week (Friday or nearest)
- monthly: Last trading day of each month
- quarterly: Last trading day of each quarter
- yearly: Last trading day of each year
"""

from typing import Literal
import pandas as pd


RebalanceFrequency = Literal["daily", "weekly", "monthly", "quarterly", "yearly"]


def generate_rebalance_dates(
    index: pd.DatetimeIndex,
    frequency: RebalanceFrequency,
) -> pd.DatetimeIndex:
    """
    Generate rebalance dates from a daily price index.

    The rebalance dates are a subset of the input index, representing
    when signal() should be called and trades should be executed.

    For non-daily frequencies, the LAST trading day of each period is used.
    This ensures we use actual trading days, not calendar dates.

    Args:
        index: DatetimeIndex of daily trading dates (from price data)
        frequency: Rebalancing frequency

    Returns:
        DatetimeIndex of rebalance dates (subset of input index)

    Examples:
        >>> prices = pd.DataFrame({'SPY': [100, 101, 102]},
        ...                       index=pd.date_range('2024-01-01', periods=3))
        >>> rebalance_dates = generate_rebalance_dates(prices.index, 'daily')
        >>> len(rebalance_dates) == len(prices.index)
        True

        >>> rebalance_dates = generate_rebalance_dates(prices.index, 'monthly')
        >>> # Returns last trading day of each month present in index
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    if len(index) == 0:
        return index

    freq = frequency.lower()

    if freq == "daily":
        # All trading days
        return index

    # Create a Series for groupby operations (value doesn't matter, just need index)
    series = pd.Series(range(len(index)), index=index)

    if freq == "weekly":
        # Last trading day of each ISO week
        # Group by (year, week) and take the last date
        grouped = series.groupby([index.isocalendar().year, index.isocalendar().week])
        last_dates = grouped.apply(lambda x: x.index[-1])
        return pd.DatetimeIndex(last_dates.values)

    elif freq == "monthly":
        # Last trading day of each month
        grouped = series.groupby([index.year, index.month])
        last_dates = grouped.apply(lambda x: x.index[-1])
        return pd.DatetimeIndex(last_dates.values)

    elif freq == "quarterly":
        # Last trading day of each quarter
        grouped = series.groupby([index.year, index.quarter])
        last_dates = grouped.apply(lambda x: x.index[-1])
        return pd.DatetimeIndex(last_dates.values)

    elif freq == "yearly":
        # Last trading day of each year
        grouped = series.groupby(index.year)
        last_dates = grouped.apply(lambda x: x.index[-1])
        return pd.DatetimeIndex(last_dates.values)

    else:
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")


def get_warmup_start_date(
    start_date: pd.Timestamp,
    warmup_days: int = 260,
    calendar: pd.DatetimeIndex = None,
) -> pd.Timestamp:
    """
    Calculate the data start date needed for warmup before the backtest start.

    Many strategies need historical data before the first rebalance date
    (e.g., 126-day momentum lookback). This function calculates how far back
    we need to load data.

    Args:
        start_date: The desired backtest start date
        warmup_days: Number of trading days to look back (default: 260 ≈ 1 year)
        calendar: Optional trading calendar to use for counting days

    Returns:
        The date from which data should be loaded

    Note:
        The warmup_days is in trading days, not calendar days.
        260 trading days ≈ 1 calendar year.
    """
    if calendar is not None and len(calendar) > 0:
        # Find position of start_date in calendar and go back
        try:
            idx = calendar.get_loc(start_date)
            warmup_idx = max(0, idx - warmup_days)
            return calendar[warmup_idx]
        except KeyError:
            # start_date not in calendar, use approximation
            pass

    # Approximate: trading days ≈ calendar days * 252/365
    calendar_days = int(warmup_days * 365 / 252)
    return start_date - pd.Timedelta(days=calendar_days)
