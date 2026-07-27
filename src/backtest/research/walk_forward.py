"""
Walk-Forward Analysis - Test strategy out-of-sample performance.

This module provides functionality to:
- Split data into train/test windows
- Optimize on training data, validate on test data
- Detect overfitting by comparing in-sample vs out-of-sample performance

Usage:
    from backtest.research import WalkForwardAnalysis

    wfa = WalkForwardAnalysis(
        strategy_class=DualMomentum,
        param_grid={"lookback_months": [6, 9, 12, 15, 18]},
        train_months=36,
        test_months=12,
        anchored=False,  # Rolling or anchored window
    )

    results = wfa.run(data, config)
    results.plot_windows()
    print(f"Out-of-sample Sharpe: {results.oos_sharpe:.2f}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta

import pandas as pd

if TYPE_CHECKING:
    from backtest.strategy import Strategy
    from backtest.data import PriceData
    from backtest.config.run_config import RunConfig


@dataclass
class WalkForwardWindow:
    """
    A single train/test window in walk-forward analysis.

    Attributes:
        train_start: Start of training period
        train_end: End of training period
        test_start: Start of test period
        test_end: End of test period
        best_params: Best parameters from training
        train_metrics: Metrics on training data
        test_metrics: Metrics on test data
    """
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: Dict[str, Any] = field(default_factory=dict)
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def degradation(self) -> float:
        """
        Calculate performance degradation from train to test.

        Returns:
            (train_sharpe - test_sharpe) / train_sharpe
        """
        train_sharpe = self.train_metrics.get("sharpe_ratio", 0)
        test_sharpe = self.test_metrics.get("sharpe_ratio", 0)

        if train_sharpe == 0:
            return 0.0

        return (train_sharpe - test_sharpe) / abs(train_sharpe)


@dataclass
class WalkForwardResult:
    """
    Results of walk-forward analysis.

    Attributes:
        windows: List of train/test windows
        overall_is_sharpe: Average in-sample Sharpe ratio
        overall_oos_sharpe: Average out-of-sample Sharpe ratio
        degradation_ratio: Average performance degradation
        overfitting_score: Measure of overfitting (higher = more overfit)
    """
    windows: List[WalkForwardWindow] = field(default_factory=list)

    @property
    def overall_is_sharpe(self) -> float:
        """Average in-sample (training) Sharpe ratio."""
        if not self.windows:
            return 0.0
        sharpes = [w.train_metrics.get("sharpe_ratio", 0) for w in self.windows]
        return sum(sharpes) / len(sharpes)

    @property
    def overall_oos_sharpe(self) -> float:
        """Average out-of-sample (test) Sharpe ratio."""
        if not self.windows:
            return 0.0
        sharpes = [w.test_metrics.get("sharpe_ratio", 0) for w in self.windows]
        return sum(sharpes) / len(sharpes)

    @property
    def degradation_ratio(self) -> float:
        """Average performance degradation from train to test."""
        if not self.windows:
            return 0.0
        degradations = [w.degradation for w in self.windows]
        return sum(degradations) / len(degradations)

    @property
    def overfitting_score(self) -> float:
        """
        Overfitting score (0 = no overfitting, 1 = complete overfit).

        Based on degradation ratio, capped at 1.0.
        """
        return min(1.0, max(0.0, self.degradation_ratio))

    def summary(self) -> str:
        """Generate text summary of results."""
        lines = [
            "Walk-Forward Analysis Results",
            "=" * 40,
            f"Windows: {len(self.windows)}",
            f"In-Sample Sharpe: {self.overall_is_sharpe:.2f}",
            f"Out-of-Sample Sharpe: {self.overall_oos_sharpe:.2f}",
            f"Degradation: {self.degradation_ratio:.1%}",
            f"Overfitting Score: {self.overfitting_score:.2f}",
        ]
        return "\n".join(lines)

    def plot_windows(self):
        """Plot train/test windows with performance."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if not self.windows:
            print("No windows to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

        # Plot 1: Timeline with windows
        for i, w in enumerate(self.windows):
            # Training period (blue)
            ax1.barh(i, (w.train_end - w.train_start).days, left=w.train_start.toordinal(),
                     color='#3b82f6', alpha=0.7, height=0.4)
            # Test period (green/red based on performance)
            color = '#16a34a' if w.test_metrics.get("sharpe_ratio", 0) > 0 else '#dc2626'
            ax1.barh(i, (w.test_end - w.test_start).days, left=w.test_start.toordinal(),
                     color=color, alpha=0.7, height=0.4)

        ax1.set_yticks(range(len(self.windows)))
        ax1.set_yticklabels([f"Window {i+1}" for i in range(len(self.windows))])
        ax1.set_xlabel("Date")
        ax1.set_title("Walk-Forward Windows")

        # Legend
        train_patch = mpatches.Patch(color='#3b82f6', alpha=0.7, label='Training')
        test_pos_patch = mpatches.Patch(color='#16a34a', alpha=0.7, label='Test (Positive)')
        test_neg_patch = mpatches.Patch(color='#dc2626', alpha=0.7, label='Test (Negative)')
        ax1.legend(handles=[train_patch, test_pos_patch, test_neg_patch], loc='upper right')

        # Plot 2: Performance comparison
        windows_idx = range(1, len(self.windows) + 1)
        train_sharpes = [w.train_metrics.get("sharpe_ratio", 0) for w in self.windows]
        test_sharpes = [w.test_metrics.get("sharpe_ratio", 0) for w in self.windows]

        x = list(windows_idx)
        width = 0.35
        ax2.bar([xi - width/2 for xi in x], train_sharpes, width, label='In-Sample', color='#3b82f6', alpha=0.7)
        ax2.bar([xi + width/2 for xi in x], test_sharpes, width, label='Out-of-Sample', color='#16a34a', alpha=0.7)

        ax2.set_xlabel("Window")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title("In-Sample vs Out-of-Sample Performance")
        ax2.legend()
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig

    def to_dataframe(self) -> pd.DataFrame:
        """Convert windows to DataFrame."""
        rows = []
        for i, w in enumerate(self.windows):
            row = {
                "window": i + 1,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "train_sharpe": w.train_metrics.get("sharpe_ratio", 0),
                "test_sharpe": w.test_metrics.get("sharpe_ratio", 0),
                "degradation": w.degradation,
            }
            row.update({f"best_{k}": v for k, v in w.best_params.items()})
            rows.append(row)
        return pd.DataFrame(rows)


class WalkForwardAnalysis:
    """
    Walk-forward analysis for detecting overfitting.

    Splits data into rolling or anchored train/test windows,
    optimizes on training data, and validates on test data.

    Key concepts:
    - Rolling: Training window moves forward with test window
    - Anchored: Training window always starts from the beginning

    Example (planned):
        wfa = WalkForwardAnalysis(
            strategy_class=VolatilityTargeting,
            param_grid={"target_vol": [0.10, 0.15, 0.20]},
            train_months=36,
            test_months=12,
        )

        results = wfa.run(data, config)

        if results.overfitting_score > 0.5:
            print("Warning: Strategy may be overfit!")
    """

    def __init__(
        self,
        strategy_class: Type["Strategy"],
        param_grid: Dict[str, List[Any]],
        train_months: int = 36,
        test_months: int = 12,
        step_months: int = 12,
        anchored: bool = False,
        metric: str = "sharpe_ratio",
    ):
        """
        Initialize walk-forward analysis.

        Args:
            strategy_class: Strategy class to test
            param_grid: Parameters to optimize over
            train_months: Training window size in months
            test_months: Test window size in months
            step_months: Step size between windows in months
            anchored: If True, training always starts from beginning
            metric: Metric to optimize (default: sharpe_ratio)
        """
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = max(1, int(step_months))
        self.anchored = anchored
        self.metric = metric

    def run(
        self,
        data: "PriceData",
        config: "RunConfig",
        progress: bool = True,
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.

        Args:
            data: Price data for backtesting
            config: Run configuration
            progress: Show progress bar

        Returns:
            WalkForwardResult with all windows and metrics
        """
        from backtest.backtester import Backtester
        from backtest.config.run_config import config_to_backtest_config
        from backtest.research.sweep import ParameterSweep

        # Get date range from data
        start_date = data.prices.index.min().to_pydatetime()
        end_date = data.prices.index.max().to_pydatetime()

        # Generate windows
        window_boundaries = self._generate_windows(start_date, end_date)

        if not window_boundaries:
            raise ValueError(
                f"No valid windows can be generated. Data range ({start_date} to {end_date}) "
                f"is too short for train={self.train_months}m + test={self.test_months}m windows."
            )

        windows: List[WalkForwardWindow] = []
        backtest_config = config_to_backtest_config(config)

        iterator = window_boundaries
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(window_boundaries, desc="Walk-Forward Analysis")
            except ImportError:
                pass

        for train_start, train_end, test_start, test_end in iterator:
            # Filter data for training period
            train_data = data.filter_dates(train_start, train_end)

            # Run parameter sweep on training data
            sweep = ParameterSweep(
                strategy_class=self.strategy_class,
                param_grid=self.param_grid,
                metric=self.metric,
                higher_is_better=True,
            )

            try:
                sweep_result = sweep.run(train_data, config, progress=False)
                best_params = sweep_result.best_params

                # Get training metrics with best params
                train_metrics = {
                    "sharpe_ratio": sweep_result.best_metric,
                }
                if sweep_result.ranking:
                    best_tuple = sweep_result.ranking[0][0]
                    if best_tuple in sweep_result.results:
                        train_metrics = sweep_result.results[best_tuple]
            except Exception:
                best_params = {}
                train_metrics = {"sharpe_ratio": 0.0}

            # Test on out-of-sample data with best parameters
            test_data = data.filter_dates(test_start, test_end)

            try:
                if best_params:
                    try:
                        strategy = self.strategy_class(**best_params)
                    except TypeError:
                        strategy = self.strategy_class()
                        for key, value in best_params.items():
                            setattr(strategy, key, value)
                else:
                    strategy = self.strategy_class()

                backtester = Backtester(strategy, test_data, backtest_config)
                result = backtester.run()
                m = result.metrics

                test_metrics = {
                    "sharpe_ratio": m.sharpe_ratio,
                    "sortino_ratio": m.sortino_ratio,
                    "cagr": m.cagr,
                    "volatility": m.volatility,
                    "max_drawdown": m.max_drawdown,
                    "calmar_ratio": m.calmar_ratio,
                    "total_return": m.total_return,
                }
            except Exception:
                test_metrics = {
                    "sharpe_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "cagr": 0.0,
                    "volatility": 0.0,
                    "max_drawdown": 0.0,
                    "calmar_ratio": 0.0,
                    "total_return": 0.0,
                }

            window = WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
            windows.append(window)

        return WalkForwardResult(windows=windows)

    def _generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate train/test window boundaries.

        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        windows = []
        current_test_end = start_date + timedelta(days=30 * (self.train_months + self.test_months))

        while current_test_end <= end_date:
            test_start = current_test_end - timedelta(days=30 * self.test_months)

            if self.anchored:
                train_start = start_date
            else:
                train_start = test_start - timedelta(days=30 * self.train_months)

            train_end = test_start

            windows.append((train_start, train_end, test_start, current_test_end))

            # Move to next window
            current_test_end += timedelta(days=30 * self.step_months)

        return windows

    @property
    def num_windows(self) -> int:
        """Estimate number of windows (depends on data length)."""
        # This is just an estimate without actual data
        return -1  # Unknown without data
