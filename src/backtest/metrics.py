"""
Metrics module - Performance metrics calculation.

This module provides functions to calculate various investment performance
metrics including returns, risk measures, and risk-adjusted ratios.

Note: This module uses centralized utilities from backtest.utils for
frequency inference and annualization to ensure consistency across
all calculations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from backtest.utils import (
    MONTH_END_FREQ,
    infer_periods_per_year,
    annualized_vol,
    cagr as utils_cagr,
    sharpe_ratio as utils_sharpe,
    sortino_ratio as utils_sortino,
    max_drawdown as utils_max_dd,
    max_drawdown_duration as utils_max_dd_duration,
    calmar_ratio as utils_calmar,
    monthly_returns as utils_monthly_returns,
    win_rate as utils_win_rate,
)

if TYPE_CHECKING:
    from backtest.backtester import Trade, BacktestConfig


@dataclass
class Metrics:
    """
    Collection of performance metrics for a backtest.

    Attributes:
        total_return: Total return over the period
        cagr: Compound Annual Growth Rate
        volatility: Annualized volatility
        max_drawdown: Maximum drawdown (as negative percentage)
        max_drawdown_duration: Days until recovery from max drawdown
        sharpe_ratio: Risk-adjusted return (return - rf) / volatility
        sortino_ratio: Downside risk-adjusted return
        calmar_ratio: CAGR / |MaxDD|
        win_rate_monthly: Percentage of positive months
        best_month: Best monthly return
        worst_month: Worst monthly return
        num_trades: Total number of trades
        turnover_annual: Average annual portfolio turnover
        total_costs: Total transaction costs paid
        alpha: Excess return vs benchmark (optional)
        beta: Sensitivity to benchmark (optional)
        correlation: Correlation with benchmark (optional)
        information_ratio: Risk-adjusted excess return vs benchmark (optional)
    """
    # Returns
    total_return: float = 0.0
    cagr: float = 0.0

    # Risk
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    max_drawdown_daily: Optional[float] = None
    max_drawdown_duration_daily: Optional[int] = None
    underwater_days_daily: Optional[int] = None

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Statistics
    win_rate_monthly: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0

    # Trading
    num_trades: int = 0
    turnover_annual: float = 0.0
    total_costs: float = 0.0
    costs_per_year: float = 0.0
    costs_pct_of_final: float = 0.0

    # Concentration & Holdings (P1 features)
    avg_holdings_count: float = 0.0  # Average number of holdings per rebalance
    max_weight: float = 0.0  # Maximum single-asset weight observed
    avg_hhi: float = 0.0  # Average Herfindahl-Hirschman Index (concentration)

    # Benchmark comparison (optional)
    alpha: Optional[float] = None
    beta: Optional[float] = None
    correlation: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_difference: Optional[float] = None
    tracking_error: Optional[float] = None


class MetricsCalculator:
    """
    Calculator for investment performance metrics.

    All methods are static and can be used independently.
    """

    @staticmethod
    def total_return(equity_curve: pd.Series) -> float:
        """
        Calculate total return over the period.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Total return as decimal (e.g., 0.5 = 50%)
        """
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    @staticmethod
    def cagr(equity_curve: pd.Series) -> float:
        """
        Calculate Compound Annual Growth Rate.

        Formula: (V_end / V_start)^(1/years) - 1

        Args:
            equity_curve: Series of portfolio values with DatetimeIndex

        Returns:
            CAGR as decimal (e.g., 0.08 = 8%)
        """
        if len(equity_curve) < 2:
            return 0.0

        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        if years <= 0:
            return 0.0

        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
        if total_return <= 0:
            return -1.0

        return total_return ** (1 / years) - 1

    @staticmethod
    def volatility(returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility (standard deviation of returns).

        Args:
            returns: Series of periodic returns
            annualize: If True, annualize the volatility (assumes monthly returns)

        Returns:
            Volatility as decimal
        """
        if len(returns) < 2:
            return 0.0

        vol = returns.std()
        if annualize:
            vol *= np.sqrt(12)  # Assuming monthly returns
        return vol

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Formula: min((V_t - max(V_s for s <= t)) / max(V_s for s <= t))

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

    @staticmethod
    def max_drawdown_duration(equity_curve: pd.Series) -> int:
        """
        Calculate the duration of the maximum drawdown (days until recovery).

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
        drawdown_groups = (~is_drawdown).cumsum()

        # Filter to only drawdown periods
        drawdown_periods = drawdown_groups[is_drawdown]

        if len(drawdown_periods) == 0:
            return 0

        # Calculate duration for each drawdown group
        def group_duration(group):
            if len(group) < 1:
                return 0
            return (group.index[-1] - group.index[0]).days

        durations = drawdown_periods.groupby(drawdown_periods).apply(group_duration)

        return int(durations.max()) if len(durations) > 0 else 0

    @staticmethod
    def underwater_days(equity_curve: pd.Series) -> int:
        """
        Calculate the longest underwater streak in observed samples.

        For daily mark-to-market curves this is trading days below the prior
        high. For lower-frequency curves it is the same sample-count concept.
        """
        if len(equity_curve) < 2:
            return 0

        rolling_max = equity_curve.cummax()
        is_underwater = equity_curve < rolling_max
        if not is_underwater.any():
            return 0

        groups = (is_underwater != is_underwater.shift(fill_value=False)).cumsum()
        streaks = is_underwater[is_underwater].groupby(groups[is_underwater]).sum()
        return int(streaks.max()) if len(streaks) > 0 else 0

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe Ratio.

        Formula: (R_p - R_f) / sigma_p

        Args:
            returns: Series of periodic returns (assumed monthly)
            risk_free_rate: Annual risk-free rate
            annualize: If True, annualize the ratio

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        # Convert annual risk-free rate to monthly
        rf_monthly = risk_free_rate / 12

        excess_returns = returns - rf_monthly
        mean_excess = excess_returns.mean()
        std_returns = returns.std()

        if std_returns == 0:
            return 0.0

        sharpe = mean_excess / std_returns

        if annualize:
            sharpe *= np.sqrt(12)

        return sharpe

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sortino Ratio (using only downside volatility).

        Formula: (R_p - R_f) / sigma_downside

        Args:
            returns: Series of periodic returns (assumed monthly)
            risk_free_rate: Annual risk-free rate
            annualize: If True, annualize the ratio

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        rf_monthly = risk_free_rate / 12
        excess_returns = returns - rf_monthly

        # Calculate downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float("inf")  # No negative returns

        downside_std = negative_returns.std()
        if downside_std == 0:
            return float("inf")

        mean_excess = excess_returns.mean()
        sortino = mean_excess / downside_std

        if annualize:
            sortino *= np.sqrt(12)

        return sortino

    @staticmethod
    def calmar_ratio(cagr: float, max_drawdown: float) -> float:
        """
        Calculate Calmar Ratio.

        Formula: CAGR / |MaxDD|

        Args:
            cagr: Compound annual growth rate
            max_drawdown: Maximum drawdown (as negative value)

        Returns:
            Calmar ratio
        """
        if max_drawdown >= 0:
            return float("inf") if cagr > 0 else 0.0

        return cagr / abs(max_drawdown)

    @staticmethod
    def monthly_returns(equity_curve: pd.Series) -> pd.Series:
        """
        Calculate monthly returns from equity curve.

        Args:
            equity_curve: Series of portfolio values

        Returns:
            Series of monthly returns
        """
        return utils_monthly_returns(equity_curve)

    @staticmethod
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

    @staticmethod
    def turnover(trades: List["Trade"], equity_curve: pd.Series) -> float:
        """
        Calculate average annual portfolio turnover.

        Turnover = Total traded value / Average portfolio value

        Args:
            trades: List of trades
            equity_curve: Series of portfolio values

        Returns:
            Annual turnover as decimal
        """
        if len(trades) == 0 or len(equity_curve) < 2:
            return 0.0

        total_traded = sum(t.value for t in trades)
        avg_value = equity_curve.mean()
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25

        if avg_value == 0 or years == 0:
            return 0.0

        return (total_traded / avg_value) / years

    @staticmethod
    def total_costs(trades: List["Trade"]) -> float:
        """
        Calculate total trading costs.

        Args:
            trades: List of trades

        Returns:
            Total costs (transaction + slippage)
        """
        return sum(t.total_costs for t in trades)

    @staticmethod
    def concentration_metrics(allocations: pd.DataFrame) -> dict:
        """
        Calculate concentration and holdings metrics from allocation history.

        Args:
            allocations: DataFrame with columns for each asset and 'cash',
                        indexed by date

        Returns:
            Dictionary with:
                - avg_holdings_count: Average number of non-zero positions
                - max_weight: Maximum single-asset weight observed
                - avg_hhi: Average Herfindahl-Hirschman Index
        """
        if allocations is None or len(allocations) == 0:
            return {
                "avg_holdings_count": 0.0,
                "max_weight": 0.0,
                "avg_hhi": 0.0,
            }

        # Exclude 'cash' column for holdings count
        asset_cols = [c for c in allocations.columns if c != "cash"]

        holdings_counts = []
        max_weights = []
        hhis = []

        for _, row in allocations.iterrows():
            # Get non-zero weights for this row
            weights = {col: row[col] for col in asset_cols if col in row and row[col] > 0.001}

            # Holdings count (non-zero positions)
            holdings_counts.append(len(weights))

            if weights:
                # Max weight
                max_w = max(weights.values())
                max_weights.append(max_w)

                # HHI = sum of squared weights
                hhi = sum(w ** 2 for w in weights.values())
                hhis.append(hhi)

        return {
            "avg_holdings_count": np.mean(holdings_counts) if holdings_counts else 0.0,
            "max_weight": max(max_weights) if max_weights else 0.0,
            "avg_hhi": np.mean(hhis) if hhis else 0.0,
        }

    @staticmethod
    def segment_exposure(
        allocations: pd.DataFrame,
        segment_map: Dict[str, str],
        *,
        unknown_segment: str = "other",
        include_cash: bool = False,
    ) -> pd.DataFrame:
        """Per-rebalance segment-weight time series.

        Joins an allocation history with a ticker -> segment map to track how
        much portfolio weight each value-chain segment carried over time. Used
        to verify that a segment cap (e.g. compute/semi <= X%) actually held:
        ``segment_exposure(...).max()`` gives the peak weight per segment.

        Args:
            allocations: BacktestResult.allocations (index=dates,
                columns=tickers + optional 'cash', values=weights).
            segment_map: ticker -> segment label (e.g. {"NVDA": "compute"}).
            unknown_segment: bucket for tickers absent from segment_map.
            include_cash: keep a 'cash' segment column when True.

        Returns:
            DataFrame index=allocations.index, columns=segment labels,
            values=summed weights per segment (not renormalized; weights are
            read as-is, mirroring concentration_metrics).
        """
        if allocations is None or len(allocations) == 0:
            return pd.DataFrame()

        asset_cols = [c for c in allocations.columns if c != "cash"]
        col_segment = {c: segment_map.get(c, unknown_segment) for c in asset_cols}
        segments = sorted(set(col_segment.values()))

        rows = []
        for _, row in allocations.iterrows():
            seg_w = {s: 0.0 for s in segments}
            for c in asset_cols:
                w = row.get(c, 0.0)
                if pd.notna(w):
                    seg_w[col_segment[c]] += float(w)
            if include_cash and "cash" in allocations.columns:
                cash_w = row.get("cash", 0.0)
                seg_w["cash"] = float(cash_w) if pd.notna(cash_w) else 0.0
            rows.append(seg_w)

        return pd.DataFrame(rows, index=allocations.index)

    @classmethod
    def calculate_all(
        cls,
        equity_curve: pd.Series,
        trades: List["Trade"],
        config: "BacktestConfig",
        benchmark_curve: Optional[pd.Series] = None,
        allocations: Optional[pd.DataFrame] = None,
    ) -> Metrics:
        """
        Calculate all metrics for a backtest.

        Args:
            equity_curve: Series of portfolio values
            trades: List of trades executed
            config: Backtest configuration
            benchmark_curve: Optional benchmark equity curve
            allocations: Optional allocation history DataFrame

        Returns:
            Metrics object with all calculated values
        """
        monthly_rets = cls.monthly_returns(equity_curve)
        cagr_val = cls.cagr(equity_curve)
        max_dd = cls.max_drawdown(equity_curve)

        # Calculate cost metrics
        total_costs_val = cls.total_costs(trades)
        years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        final_value = equity_curve.iloc[-1] if len(equity_curve) > 0 else 0

        costs_per_year = total_costs_val / years if years > 0 else 0.0
        costs_pct = (total_costs_val / final_value * 100) if final_value > 0 else 0.0

        # Calculate concentration metrics if allocations provided
        concentration = cls.concentration_metrics(allocations)

        metrics = Metrics(
            total_return=cls.total_return(equity_curve),
            cagr=cagr_val,
            volatility=cls.volatility(monthly_rets),
            max_drawdown=max_dd,
            max_drawdown_duration=cls.max_drawdown_duration(equity_curve),
            sharpe_ratio=cls.sharpe_ratio(monthly_rets, config.risk_free_rate),
            sortino_ratio=cls.sortino_ratio(monthly_rets, config.risk_free_rate),
            calmar_ratio=cls.calmar_ratio(cagr_val, max_dd),
            win_rate_monthly=cls.win_rate(monthly_rets),
            best_month=monthly_rets.max() if len(monthly_rets) > 0 else 0.0,
            worst_month=monthly_rets.min() if len(monthly_rets) > 0 else 0.0,
            num_trades=len(trades),
            turnover_annual=cls.turnover(trades, equity_curve),
            total_costs=total_costs_val,
            costs_per_year=costs_per_year,
            costs_pct_of_final=costs_pct,
            avg_holdings_count=concentration["avg_holdings_count"],
            max_weight=concentration["max_weight"],
            avg_hhi=concentration["avg_hhi"],
        )

        # Calculate benchmark comparison metrics if available
        if benchmark_curve is not None and len(benchmark_curve) > 0:
            bench_rets = cls.monthly_returns(benchmark_curve)
            bench_cagr = cls.cagr(benchmark_curve)

            # Tracking Difference (annualized return difference)
            metrics.tracking_difference = cagr_val - bench_cagr

            # Align returns for correlation-based metrics
            aligned = pd.concat([monthly_rets, bench_rets], axis=1).dropna()
            if len(aligned) > 1:
                aligned.columns = ["strategy", "benchmark"]

                # Correlation
                metrics.correlation = aligned["strategy"].corr(aligned["benchmark"])

                # Beta and Alpha (linear regression)
                cov = aligned.cov()
                if cov.loc["benchmark", "benchmark"] != 0:
                    metrics.beta = cov.loc["strategy", "benchmark"] / cov.loc["benchmark", "benchmark"]
                    metrics.alpha = cagr_val - (config.risk_free_rate + metrics.beta * (bench_cagr - config.risk_free_rate))

                # Tracking Error and Information Ratio
                tracking_error = (aligned["strategy"] - aligned["benchmark"]).std() * np.sqrt(12)
                metrics.tracking_error = tracking_error
                if tracking_error != 0:
                    metrics.information_ratio = metrics.tracking_difference / tracking_error

        return metrics
