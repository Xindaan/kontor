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

from typing import Optional, Literal
import pandas as pd


RebalanceFrequency = Literal["daily", "weekly", "monthly", "quarterly", "yearly"]


def generate_rebalance_dates(
    index: pd.DatetimeIndex,
    frequency: RebalanceFrequency,
    weekday: Optional[int] = None,
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
        weekday: "weekly" ONLY -- pin the weekly rebalance to a fixed weekday
            (0=Mon .. 4=Fri). None (default) keeps the legacy behaviour: the last
            trading day of the ISO week (effectively Friday). If the anchor day is
            a holiday, the next open day in the SAME week is used; if none, the
            last open day before it. This mirrors reality -- when the trading day
            slips, you trade on the next open day, you do not skip the week.

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
        iso = index.isocalendar()
        grouped = series.groupby([iso.year, iso.week])
        if weekday is None:
            # Legacy behaviour: last trading day of each ISO week (= usually Friday).
            last_dates = grouped.apply(lambda x: x.index[-1])
            return pd.DatetimeIndex(last_dates.values)

        if not 0 <= int(weekday) <= 4:
            raise ValueError("weekday must be 0 (Mon) to 4 (Fri), was %r" % (weekday,))
        target_dow = int(weekday)

        def _anchor(group: pd.Series) -> pd.Timestamp:
            days = group.index
            on_day = days[days.dayofweek == target_dow]
            if len(on_day):
                return on_day[0]
            later = days[days.dayofweek > target_dow]
            if len(later):
                return later[0]          # anchor day was a holiday -> next open day
            return days[-1]              # only earlier days open -> last one before it

        anchor = grouped.apply(_anchor)
        return pd.DatetimeIndex(anchor.values)

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
