"""
Momentum Utilities - Helper functions for momentum strategies.

This module provides common functions used across momentum-based strategies:
- compute_momentum: Calculate momentum score from price series
- pick_top: Select top N assets by score
- inv_vol_weights: Compute inverse-volatility weights
- sma: Simple Moving Average calculation

Note: This file starts with underscore to be excluded from glob patterns
when loading strategies automatically.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


def compute_momentum(
    prices: pd.Series,
    lookback_days: int = 252,
    skip_days: int = 21,
) -> Optional[float]:
    """
    Compute momentum (return) over a lookback period, optionally skipping recent days.

    The classic 12-1 momentum: 12-month lookback, skip the most recent month.
    This avoids short-term mean reversion effects.

    Args:
        prices: Price series (must have at least lookback_days of data)
        lookback_days: Number of days to look back (e.g., 252 for ~12 months)
        skip_days: Number of recent days to skip (e.g., 21 for ~1 month)

    Returns:
        Momentum as decimal return (e.g., 0.15 for 15%), or None if insufficient data

    Example:
        >>> prices = pd.Series([100, 110, 120, 115])  # simplified
        >>> compute_momentum(prices, lookback_days=3, skip_days=0)
        0.15  # (115 / 100) - 1
    """
    clean_prices = prices.dropna()

    # Need at least lookback_days of data
    min_required = lookback_days
    if skip_days > 0:
        min_required = lookback_days + skip_days

    if len(clean_prices) < min_required:
        return None

    # Calculate momentum
    if skip_days > 0:
        # End price is skip_days before the last price
        end_price = clean_prices.iloc[-skip_days - 1]
        start_price = clean_prices.iloc[-lookback_days - skip_days]
    else:
        end_price = clean_prices.iloc[-1]
        start_price = clean_prices.iloc[-lookback_days]

    if start_price <= 0:
        return None

    momentum = (end_price / start_price) - 1
    return momentum


def pick_top(
    scores: Dict[str, float],
    n: int,
    ascending: bool = False,
) -> List[str]:
    """
    Select top N assets by score.

    Args:
        scores: Dictionary mapping ticker to score
        n: Number of top assets to select
        ascending: If True, pick lowest scores (for volatility etc.)

    Returns:
        List of top N ticker symbols

    Example:
        >>> scores = {"AAPL": 0.25, "MSFT": 0.30, "GOOGL": 0.15}
        >>> pick_top(scores, n=2)
        ['MSFT', 'AAPL']
    """
    if not scores:
        return []

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)
    return [ticker for ticker, _ in sorted_items[:n]]


def inv_vol_weights(
    data: pd.DataFrame,
    tickers: List[str],
    vol_lookback: int = 63,
    cap: float = 0.25,
    annualize: bool = True,
) -> Dict[str, float]:
    """
    Compute inverse-volatility weights for a set of tickers.

    Weight proportional to 1/volatility, then capped and normalized.

    Args:
        data: DataFrame with price data (DatetimeIndex, ticker columns)
        tickers: List of tickers to weight
        vol_lookback: Days to use for volatility calculation (e.g., 63 for ~3 months)
        cap: Maximum weight per asset (e.g., 0.25 for 25%)
        annualize: Whether to annualize volatility (doesn't affect weights)

    Returns:
        Dictionary mapping ticker to weight (sums to <= 1.0)

    Example:
        >>> weights = inv_vol_weights(data, ["AAPL", "MSFT"], vol_lookback=63, cap=0.4)
        >>> # Returns something like {"AAPL": 0.4, "MSFT": 0.35}
    """
    if not tickers:
        return {}

    inv_vols = {}

    for ticker in tickers:
        if ticker not in data.columns:
            continue

        prices = data[ticker].dropna()
        if len(prices) < vol_lookback + 1:
            continue

        returns = prices.pct_change(fill_method=None).dropna().iloc[-vol_lookback:]
        if len(returns) < vol_lookback:
            continue

        vol = returns.std()
        if vol > 0:
            inv_vols[ticker] = 1.0 / vol

    if not inv_vols:
        # Fallback to equal weights if no volatility data
        return {t: 1.0 / len(tickers) for t in tickers if t in data.columns}

    # Normalize to sum to 1
    total = sum(inv_vols.values())
    weights = {t: v / total for t, v in inv_vols.items()}

    # Apply cap
    needs_redistribution = True
    while needs_redistribution:
        needs_redistribution = False
        excess = 0.0
        uncapped_weight = 0.0

        for t, w in weights.items():
            if w > cap:
                excess += w - cap
                weights[t] = cap
                needs_redistribution = True
            else:
                uncapped_weight += w

        if excess > 0 and uncapped_weight > 0:
            # Redistribute excess to uncapped assets
            for t in weights:
                if weights[t] < cap:
                    additional = excess * (weights[t] / uncapped_weight)
                    weights[t] += additional

    return weights


def sma(series: pd.Series, period: int) -> Optional[float]:
    """
    Calculate Simple Moving Average of a price series.

    Args:
        series: Price series
        period: Number of periods for SMA

    Returns:
        SMA value, or None if insufficient data

    Example:
        >>> prices = pd.Series([100, 102, 104, 103, 105])
        >>> sma(prices, period=3)
        104.0  # (104 + 103 + 105) / 3
    """
    clean = series.dropna()
    if len(clean) < period:
        return None
    return clean.iloc[-period:].mean()


def compute_volatility(
    prices: pd.Series,
    lookback_days: int = 63,
    annualize: bool = True,
) -> Optional[float]:
    """
    Compute historical volatility from price series.

    Args:
        prices: Price series
        lookback_days: Number of days to use
        annualize: Whether to annualize (multiply by sqrt(252))

    Returns:
        Volatility as decimal, or None if insufficient data
    """
    clean = prices.dropna()
    if len(clean) < lookback_days + 1:
        return None

    returns = clean.pct_change(fill_method=None).dropna().iloc[-lookback_days:]
    vol = returns.std()

    if annualize:
        vol *= np.sqrt(252)

    return vol


def compute_hhi(weights: Dict[str, float]) -> float:
    """
    Compute Herfindahl-Hirschman Index (concentration measure).

    HHI ranges from 1/n (equal weights) to 1.0 (single asset).

    Args:
        weights: Dictionary of asset weights

    Returns:
        HHI value between 0 and 1
    """
    if not weights:
        return 1.0
    return sum(w ** 2 for w in weights.values())


def compute_turnover(
    old_weights: Dict[str, float],
    new_weights: Dict[str, float],
) -> float:
    """
    Compute one-way turnover between two allocations.

    Args:
        old_weights: Previous allocation weights
        new_weights: New allocation weights

    Returns:
        One-way turnover (sum of absolute weight changes / 2)
    """
    all_tickers = set(old_weights.keys()) | set(new_weights.keys())

    total_change = 0.0
    for ticker in all_tickers:
        old_w = old_weights.get(ticker, 0.0)
        new_w = new_weights.get(ticker, 0.0)
        total_change += abs(new_w - old_w)

    return total_change / 2  # One-way turnover


def ensemble_momentum_score(
    prices: pd.Series,
    weights: List[Tuple[int, int, float]] = None,
) -> Optional[float]:
    """
    Compute ensemble momentum score combining multiple lookback periods.

    Default: 50% 12-1, 30% 6-1, 20% 3-0

    Args:
        prices: Price series
        weights: List of (lookback_days, skip_days, weight) tuples

    Returns:
        Weighted average momentum score, or None if insufficient data
    """
    if weights is None:
        weights = [
            (252, 21, 0.5),   # 12-1 momentum
            (126, 21, 0.3),   # 6-1 momentum
            (63, 0, 0.2),     # 3-0 momentum
        ]

    total_weight = 0.0
    weighted_score = 0.0

    for lookback, skip, weight in weights:
        mom = compute_momentum(prices, lookback_days=lookback, skip_days=skip)
        if mom is not None:
            weighted_score += mom * weight
            total_weight += weight

    if total_weight == 0:
        return None

    return weighted_score / total_weight
