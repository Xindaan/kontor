"""
Parameter Sweep - Test strategy robustness across parameter ranges.

This module provides functionality to:
- Run backtests across multiple parameter combinations
- Generate heatmaps and rankings
- Identify robust parameter regions

Usage:
    from backtest.research import ParameterSweep

    sweep = ParameterSweep(
        strategy_class=DualMomentum,
        param_grid={
            "lookback_months": [6, 9, 12, 15, 18],
            "safe_asset": ["BND", "SHY"],
        },
        metric="sharpe_ratio",
    )

    results = sweep.run(data, config)
    sweep.plot_heatmap(results)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type, Callable, TYPE_CHECKING
import itertools

import pandas as pd

if TYPE_CHECKING:
    from backtest.strategy import Strategy
    from backtest.data import PriceData
    from backtest.config.run_config import RunConfig


@dataclass
class SweepResult:
    """
    Result of a parameter sweep.

    Attributes:
        param_grid: Parameter combinations tested
        results: Metrics for each combination
        best_params: Best performing parameters
        best_metric: Best metric value
        ranking: Sorted list of (params, metric) tuples
    """
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    results: Dict[tuple, Dict[str, float]] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_metric: float = 0.0
    ranking: List[tuple] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.

        Returns:
            DataFrame with columns for each parameter and metric
        """
        if not self.results:
            return pd.DataFrame()

        rows = []
        for params_tuple, metrics in self.results.items():
            row = dict(zip(self.param_grid.keys(), params_tuple))
            row.update(metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    def plot_heatmap(
        self,
        param_x: str,
        param_y: str,
        metric: str = "sharpe_ratio",
        cmap: str = "RdYlGn",
    ):
        """
        Plot heatmap of results for two parameters.

        Args:
            param_x: Parameter for x-axis
            param_y: Parameter for y-axis
            metric: Metric to display
            cmap: Colormap for heatmap
        """
        import matplotlib.pyplot as plt

        df = self.to_dataframe()
        if df.empty:
            print("No results to plot")
            return

        # Pivot to create heatmap matrix
        pivot = df.pivot(index=param_y, columns=param_x, values=metric)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")

        # Set ticks
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)

        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label(metric)

        # Add value labels
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_title(f"Parameter Sweep: {metric}")

        plt.tight_layout()
        return fig


class ParameterSweep:
    """
    Parameter sweep for strategy robustness testing.

    Runs backtests across all combinations of specified parameters
    and ranks results by a target metric.

    Example (planned):
        sweep = ParameterSweep(
            strategy_class=VolatilityTargeting,
            param_grid={
                "target_vol": [0.10, 0.12, 0.15, 0.18, 0.20],
                "lookback": [20, 40, 60],
            },
        )

        results = sweep.run(data, config)
        print(results.best_params)
    """

    def __init__(
        self,
        strategy_class: Type["Strategy"],
        param_grid: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
        higher_is_better: bool = True,
    ):
        """
        Initialize parameter sweep.

        Args:
            strategy_class: Strategy class to instantiate
            param_grid: Dict of parameter name -> list of values to test
            metric: Metric to optimize (default: sharpe_ratio)
            higher_is_better: True if higher metric is better
        """
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.metric = metric
        self.higher_is_better = higher_is_better

    def run(
        self,
        data: "PriceData",
        config: "RunConfig",
        progress: bool = True,
    ) -> SweepResult:
        """
        Run parameter sweep.

        Args:
            data: Price data for backtesting
            config: Run configuration
            progress: Show progress bar

        Returns:
            SweepResult with all metrics and rankings
        """
        from backtest.backtester import Backtester
        from backtest.config.run_config import config_to_backtest_config

        combinations = self._generate_combinations()
        results: Dict[tuple, Dict[str, float]] = {}

        # Convert RunConfig to BacktestConfig
        backtest_config = config_to_backtest_config(config)

        iterator = combinations
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(combinations, desc="Parameter Sweep")
            except ImportError:
                pass

        for params in iterator:
            # Create strategy instance with these parameters
            try:
                strategy = self.strategy_class(**params)
            except TypeError:
                # Some strategies may need different initialization
                strategy = self.strategy_class()
                for key, value in params.items():
                    setattr(strategy, key, value)

            # Run backtest
            try:
                backtester = Backtester(strategy, data, backtest_config)
                result = backtester.run()
                m = result.metrics

                # Store metrics
                params_tuple = tuple(params[k] for k in self.param_grid.keys())
                results[params_tuple] = {
                    "sharpe_ratio": m.sharpe_ratio,
                    "sortino_ratio": m.sortino_ratio,
                    "cagr": m.cagr,
                    "volatility": m.volatility,
                    "max_drawdown": m.max_drawdown,
                    "calmar_ratio": m.calmar_ratio,
                    "total_return": m.total_return,
                }
            except Exception as e:
                # Record failed run
                params_tuple = tuple(params[k] for k in self.param_grid.keys())
                results[params_tuple] = {
                    "sharpe_ratio": float("nan"),
                    "error": str(e),
                }

        # Create ranking
        valid_results = [
            (params, metrics[self.metric])
            for params, metrics in results.items()
            if not pd.isna(metrics.get(self.metric, float("nan")))
        ]
        ranking = sorted(valid_results, key=lambda x: x[1], reverse=self.higher_is_better)

        # Determine best params
        best_params = {}
        best_metric = float("-inf") if self.higher_is_better else float("inf")
        if ranking:
            best_tuple, best_metric = ranking[0]
            best_params = dict(zip(self.param_grid.keys(), best_tuple))

        return SweepResult(
            param_grid=self.param_grid,
            results=results,
            best_params=best_params,
            best_metric=best_metric,
            ranking=ranking,
        )

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    @property
    def num_combinations(self) -> int:
        """Total number of parameter combinations."""
        n = 1
        for values in self.param_grid.values():
            n *= len(values)
        return n
