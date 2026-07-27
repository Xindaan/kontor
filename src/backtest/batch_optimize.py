"""
Batch optimization module for optimizing parameters across all strategies.

Provides:
- Parameter grid definitions for each strategy
- Batch optimization runner
- Results aggregation and ranking
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import copy
import importlib.util
import inspect
import itertools
import math
import sys

import pandas as pd

from backtest.backtester import Backtester, BacktestConfig
from backtest.data import DataLoader, PriceData
from backtest.external_features.config import (
    ExternalFeaturesConfig,
    build_loader_from_config,
)
from backtest.strategy import Strategy


# =============================================================================
# Parameter Grid Definitions
# =============================================================================

# Default parameter grids for each strategy type
# Key: Strategy class name, Value: dict of param_name -> list of values
STRATEGY_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    # Dual Momentum
    "DualMomentum": {
        "lookback_months": [6, 9, 12, 15, 18],
    },

    # Volatility Targeting
    "VolatilityTargeting": {
        "target_vol": [0.08, 0.10, 0.12, 0.15, 0.18],
        "lookback_days": [21, 42, 63, 126],
    },

    # SMA Trend Filter
    "SMATrendFilter": {
        "sma_window": [50, 100, 150, 200, 252],
    },

    # Momentum Basket Regime
    "MomentumBasketRegime": {
        "momentum_lookback_days": [126, 189, 252],
        "top_k": [3, 5, 7],
        "trend_window_days": [100, 150, 200],
        "weight_mode": ["inv_vol", "equal"],
    },

    # Top N Momentum Universe
    "TopNMomentumUniverse": {
        "lookback_days": [126, 189, 252],
        "top_n": [3, 5, 10],
    },

    # Momentum Top N Basket Regime
    "MomentumTopNBasketRegime": {
        "momentum_lookback_days": [126, 189, 252],
        "top_k": [3, 5, 7],
        "trend_window_days": [100, 150, 200],
    },

    # Drawdown Brake
    "DrawdownBrake": {
        "lookback_days": [63, 126, 189, 252],
        "drawdown_threshold": [0.05, 0.10, 0.15, 0.20],
    },

    # Trend Filtered Risk Parity
    "TrendFilteredRiskParity": {
        "sma_window": [100, 150, 200, 252],
        "vol_lookback": [21, 42, 63],
    },

    # Vol Scaled Dual Momentum
    "VolScaledDualMomentum": {
        "lookback_months": [6, 9, 12],
        "target_vol": [0.10, 0.12, 0.15],
        "vol_lookback_days": [21, 42, 63],
    },

    # Gearbox Vol Target Trend
    "GearboxVolTargetTrend": {
        "target_vol": [0.10, 0.12, 0.15],
        "trend_sma": [100, 150, 200],
        "vol_lookback": [21, 42, 63],
    },

    # Vol Target Trend Filter
    "VolTargetTrendFilter": {
        "target_vol": [0.10, 0.12, 0.15],
        "sma_window": [100, 150, 200],
        "vol_lookback_days": [21, 42, 63],
    },

    # Vol Target 1x vs 3x
    "VolTarget1xVs3x": {
        "target_vol": [0.10, 0.15, 0.20],
        "vol_lookback_days": [21, 42, 63],
    },

    # Relative Momentum Switch
    "RelativeMomentumSwitch": {
        "lookback_days": [63, 126, 189, 252],
    },

    # Levered Trend Filter
    "LeveredTrendFilter": {
        "sma_window": [100, 150, 200],
        "leverage": [1.5, 2.0, 3.0],
    },

    # Levered Drawdown Brake
    "LeveredDrawdownBrake": {
        "lookback_days": [63, 126, 189],
        "drawdown_threshold": [0.05, 0.10, 0.15],
        "leverage": [1.5, 2.0, 3.0],
    },

    # Top N Momentum variants
    "TopNMomentumTrendFilter": {
        "lookback_days": [126, 189, 252],
        "top_n": [3, 5, 10],
        "sma_window": [100, 150, 200],
    },

    "TopNMomentumInvVol": {
        "lookback_days": [126, 189, 252],
        "top_n": [3, 5, 10],
        "vol_lookback_days": [21, 42, 63],
    },

    "TopNMomentumVolTarget": {
        "lookback_days": [126, 189, 252],
        "top_n": [3, 5, 10],
        "target_vol": [0.10, 0.12, 0.15],
    },

    "TopNMomentumAbsGate": {
        "lookback_days": [126, 189, 252],
        "top_n": [3, 5, 10],
    },

    "TopNMomentumEnsemble": {
        "top_n": [3, 5, 10],
    },

    # Strategies with no tunable parameters (or simple defaults)
    "BuyAndHold": {},
    "Classic6040": {},
    "InverseVolRiskParity": {
        "vol_lookback": [21, 42, 63],
    },
}

# Default rebalance frequencies to test
DEFAULT_REBALANCE_FREQUENCIES = ["monthly", "quarterly"]


@dataclass
class WalkForwardResultSummary:
    """Summary of walk-forward analysis for a single strategy."""
    num_windows: int
    avg_is_sharpe: float  # Average In-Sample Sharpe
    avg_oos_sharpe: float  # Average Out-of-Sample Sharpe
    degradation_ratio: float  # (IS - OOS) / IS
    overfitting_score: float  # 0-1 score
    best_oos_params: Dict[str, Any]  # Parameters from best OOS window
    best_oos_sharpe: float
    mode: str = "standard"
    nested: bool = False
    inner_train_years: Optional[float] = None
    inner_test_years: Optional[float] = None
    inner_step_months: Optional[int] = None
    inner_anchored: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_windows": self.num_windows,
            "avg_is_sharpe": self.avg_is_sharpe,
            "avg_oos_sharpe": self.avg_oos_sharpe,
            "degradation_ratio": self.degradation_ratio,
            "overfitting_score": self.overfitting_score,
            "best_oos_params": self.best_oos_params,
            "best_oos_sharpe": self.best_oos_sharpe,
            "mode": self.mode,
            "nested": self.nested,
            "inner_train_years": self.inner_train_years,
            "inner_test_years": self.inner_test_years,
            "inner_step_months": self.inner_step_months,
            "inner_anchored": self.inner_anchored,
        }


@dataclass
class OptimizationResult:
    """Result of optimizing a single strategy."""
    strategy_name: str
    strategy_class: str
    strategy_file: str
    best_params: Dict[str, Any]
    best_rebalance_frequency: str
    best_metric_value: float
    metric_name: str
    all_results: List[Dict[str, Any]]
    # Walk-forward results (optional)
    walk_forward: Optional[WalkForwardResultSummary] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "strategy_name": self.strategy_name,
            "strategy_class": self.strategy_class,
            "strategy_file": self.strategy_file,
            "best_params": self.best_params,
            "best_rebalance_frequency": self.best_rebalance_frequency,
            "best_metric_value": self.best_metric_value,
            "metric_name": self.metric_name,
        }
        if self.walk_forward:
            result["walk_forward"] = self.walk_forward.to_dict()
        return result


@dataclass
class BatchOptimizationResult:
    """Result of batch optimization across all strategies."""
    results: List[OptimizationResult] = field(default_factory=list)
    metric_name: str = "sharpe_ratio"
    start_date: str = ""
    end_date: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame for analysis."""
        rows = []
        for r in self.results:
            row = {
                "strategy_name": r.strategy_name,
                "strategy_class": r.strategy_class,
                "strategy_file": r.strategy_file,
                "best_rebalance": r.best_rebalance_frequency,
                f"best_{r.metric_name}": r.best_metric_value,
            }
            # Add best params
            for k, v in r.best_params.items():
                row[f"param_{k}"] = v
            # Add walk-forward metrics if available
            if r.walk_forward:
                row["wf_is_sharpe"] = r.walk_forward.avg_is_sharpe
                row["wf_oos_sharpe"] = r.walk_forward.avg_oos_sharpe
                row["wf_degradation"] = r.walk_forward.degradation_ratio
                row["wf_overfitting"] = r.walk_forward.overfitting_score
            rows.append(row)

        df = pd.DataFrame(rows)
        # Sort by metric (use OOS sharpe if walk-forward, otherwise best metric)
        if "wf_oos_sharpe" in df.columns and df["wf_oos_sharpe"].notna().any():
            df = df.sort_values("wf_oos_sharpe", ascending=False)
        else:
            metric_col = f"best_{self.metric_name}"
            if metric_col in df.columns:
                df = df.sort_values(metric_col, ascending=False)
        return df

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            "=" * 70,
            "BATCH OPTIMIZATION RESULTS",
            "=" * 70,
            f"Metric: {self.metric_name}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Strategies optimized: {len(self.results)}",
            "",
            "TOP 10 STRATEGIES (by {})".format(self.metric_name),
            "-" * 70,
        ]

        # Sort by metric value
        sorted_results = sorted(
            self.results,
            key=lambda x: x.best_metric_value if x.best_metric_value == x.best_metric_value else float('-inf'),
            reverse=True
        )

        for i, r in enumerate(sorted_results[:10], 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r.best_params.items())
            if params_str:
                params_str = f" ({params_str})"
            lines.append(
                f"{i:2d}. {r.strategy_name:<35} "
                f"{r.metric_name}={r.best_metric_value:.4f} "
                f"[{r.best_rebalance_frequency}]{params_str}"
            )

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


def load_strategy_from_file(path: str) -> Tuple[Strategy, type, str]:
    """
    Load a strategy from a Python file.

    Returns:
        Tuple of (strategy_instance, strategy_class, strategy_name)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    spec = importlib.util.spec_from_file_location("strategy_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_module"] = module
    spec.loader.exec_module(module)

    # Find strategy
    strategy_instance = None
    strategy_class = None

    # First check for pre-instantiated 'strategy' variable
    if hasattr(module, 'strategy'):
        obj = getattr(module, 'strategy')
        if isinstance(obj, Strategy):
            strategy_instance = obj
            strategy_class = type(obj)

    # Otherwise find Strategy subclass
    if strategy_class is None:
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                strategy_class = obj
                break

    if strategy_class is None:
        raise ValueError(f"No Strategy class found in {path}")

    if strategy_instance is None:
        try:
            strategy_instance = strategy_class()
        except Exception:
            strategy_instance = None

    strategy_name = strategy_instance.name if strategy_instance else strategy_class.__name__

    return strategy_instance, strategy_class, strategy_name


def get_param_grid_for_strategy(
    strategy_class: type,
    strategy_instance: Optional[Strategy] = None,
    custom_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None
) -> Dict[str, List[Any]]:
    """
    Get the parameter grid for a strategy class.

    Priority order:
    1. Custom grids passed as argument
    2. Strategy's get_param_grid() class method (auto-discovery)
    3. Global STRATEGY_PARAM_GRIDS registry (hardcoded fallback)
    4. Auto-inferred from __init__ signature
    """
    class_name = strategy_class.__name__

    # Check custom grids first
    if custom_grids and class_name in custom_grids:
        return custom_grids[class_name]

    # Check strategy's get_param_grid() method (auto-discovery)
    if hasattr(strategy_class, 'get_param_grid'):
        strategy_grid = strategy_class.get_param_grid()
        if strategy_grid is not None:
            return strategy_grid

    # Check global registry (hardcoded fallback)
    if class_name in STRATEGY_PARAM_GRIDS:
        return STRATEGY_PARAM_GRIDS[class_name]

    # Try to infer from __init__ signature
    param_grid = {}
    try:
        sig = inspect.signature(strategy_class.__init__)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            # Only include parameters with numeric defaults
            if param.default is not inspect.Parameter.empty:
                default = param.default
                if isinstance(default, (int, float)) and not isinstance(default, bool):
                    # Create a grid around the default value
                    if isinstance(default, int):
                        if default > 0:
                            param_grid[param_name] = [
                                max(1, int(default * 0.5)),
                                int(default * 0.75),
                                default,
                                int(default * 1.25),
                                int(default * 1.5),
                            ]
                    elif isinstance(default, float):
                        if 0 < default < 1:  # Likely a ratio/percentage
                            param_grid[param_name] = [
                                default * 0.5,
                                default * 0.75,
                                default,
                                default * 1.25,
                                default * 1.5,
                            ]
    except Exception:
        pass

    return param_grid


def optimize_single_strategy(
    strategy_file: str,
    data: PriceData,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    rebalance_frequencies: List[str] = None,
    metric: str = "sharpe_ratio",
    minimize: bool = False,
    initial_capital: float = 10000.0,
    costs_pct: float = 0.001,
    cost_profile: Optional[Dict[str, Any]] = None,
    execution_lag_days: int = 0,
    max_volume_participation: Optional[float] = None,
    min_daily_dollar_volume: float = 0.0,
    liquidity_on_missing_volume: str = "allow",
    risk_overlay: Optional[Dict[str, Any]] = None,
    tax_enabled: bool = True,
    metric_basis: str = "net_liquidation",
    progress_callback: Optional[callable] = None,
    validate: bool = True,
    drip_enabled: bool = False,
    external_features: Optional[ExternalFeaturesConfig] = None,
) -> OptimizationResult:
    """
    Optimize parameters for a single strategy.

    Args:
        strategy_file: Path to the strategy file
        data: Price data to use
        param_grid: Optional custom parameter grid (uses defaults if None)
        rebalance_frequencies: Rebalance frequencies to test
        metric: Metric to optimize
        minimize: If True, minimize metric (for max_drawdown)
        initial_capital: Initial capital
        costs_pct: Transaction costs
        cost_profile: Optional per-ticker/asset-class cost profile
        execution_lag_days: Delay between signal and execution in trading days
        max_volume_participation: Optional max traded share as fraction of daily volume
        min_daily_dollar_volume: Optional minimum daily notional floor
        liquidity_on_missing_volume: "allow" or "skip" when volume is missing
        risk_overlay: Optional portfolio-level risk overlay settings
        tax_enabled: Whether to enable German tax model
        metric_basis: Metric basis (gross, net_realized, net_liquidation)
        progress_callback: Optional callback(run_num, total_runs, params, metric_value)

    Returns:
        OptimizationResult with best parameters
    """
    if rebalance_frequencies is None:
        rebalance_frequencies = DEFAULT_REBALANCE_FREQUENCIES

    # Load strategy
    strategy_instance, strategy_class, strategy_name = load_strategy_from_file(strategy_file)

    # Get parameter grid
    if param_grid is None:
        param_grid = get_param_grid_for_strategy(strategy_class, strategy_instance)

    # Build all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    if param_values:
        param_combinations = list(itertools.product(*param_values))
    else:
        param_combinations = [()]

    total_runs = len(param_combinations) * len(rebalance_frequencies)
    all_results = []
    run_count = 0

    best_metric = float('-inf') if not minimize else float('inf')
    best_params = {}
    best_rebal = rebalance_frequencies[0]

    for rebal_freq in rebalance_frequencies:
        for param_combo in param_combinations:
            run_count += 1
            params = dict(zip(param_names, param_combo))

            # Create strategy with these parameters
            try:
                if strategy_instance:
                    # Get __init__ parameters
                    init_sig = inspect.signature(strategy_class.__init__)
                    init_params = [p for p in init_sig.parameters.keys() if p != 'self']

                    # Build kwargs from instance attributes + overrides
                    kwargs = {}
                    for p in init_params:
                        if p in params:
                            kwargs[p] = params[p]
                        elif hasattr(strategy_instance, p):
                            kwargs[p] = getattr(strategy_instance, p)
                        elif hasattr(strategy_instance, f"_{p}"):
                            kwargs[p] = getattr(strategy_instance, f"_{p}")

                    strategy = strategy_class(**kwargs)
                else:
                    strategy = strategy_class(**params)
            except Exception as e:
                # Fallback: copy instance and override attributes
                if strategy_instance:
                    strategy = copy.deepcopy(strategy_instance)
                    for k, v in params.items():
                        setattr(strategy, k, v)
                        if hasattr(strategy, f"_{k}"):
                            setattr(strategy, f"_{k}", v)
                else:
                    continue

            # Configure backtest
            config = BacktestConfig(
                initial_capital=initial_capital,
                costs_pct=costs_pct,
                cost_profile=cost_profile,
                execution_lag_days=execution_lag_days,
                max_volume_participation=max_volume_participation,
                min_daily_dollar_volume=min_daily_dollar_volume,
                liquidity_on_missing_volume=liquidity_on_missing_volume,
                risk_overlay=risk_overlay,
                rebalance_frequency=rebal_freq,
                tax_enabled=tax_enabled,
                metric_basis=metric_basis,
                validate=validate,
                drip_enabled=drip_enabled,
                external_features_loader=build_loader_from_config(
                    external_features or ExternalFeaturesConfig()
                ),
            )

            # Run backtest
            try:
                backtester = Backtester(strategy, data, config)
                result = backtester.run()

                # Always use headline metrics selected by metric_basis in Backtester
                m = result.metrics

                metric_value = getattr(m, metric, float('nan'))

                result_entry = {
                    'rebalance_frequency': rebal_freq,
                    **params,
                    'sharpe_ratio': m.sharpe_ratio,
                    'sortino_ratio': m.sortino_ratio,
                    'cagr': m.cagr,
                    'volatility': m.volatility,
                    'max_drawdown': m.max_drawdown,
                    'calmar_ratio': m.calmar_ratio,
                    '_metric_value': metric_value,
                }
                all_results.append(result_entry)

                # Check if best
                if not pd.isna(metric_value):
                    is_better = (metric_value > best_metric) if not minimize else (metric_value < best_metric)
                    if is_better:
                        best_metric = metric_value
                        best_params = params.copy()
                        best_rebal = rebal_freq

                if progress_callback:
                    progress_callback(run_count, total_runs, params, metric_value)

            except Exception as e:
                all_results.append({
                    'rebalance_frequency': rebal_freq,
                    **params,
                    'error': str(e),
                    '_metric_value': float('nan'),
                })

    return OptimizationResult(
        strategy_name=strategy_name,
        strategy_class=strategy_class.__name__,
        strategy_file=strategy_file,
        best_params=best_params,
        best_rebalance_frequency=best_rebal,
        best_metric_value=best_metric,
        metric_name=metric,
        all_results=all_results,
    )


def walk_forward_single_strategy(
    strategy_file: str,
    data: PriceData,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    rebalance_frequencies: Optional[List[str]] = None,
    train_years: float = 5.0,
    test_years: float = 1.0,
    step_months: int = 12,
    anchored: bool = False,
    nested: bool = False,
    inner_train_years: float = 3.0,
    inner_test_years: float = 1.0,
    inner_step_months: int = 6,
    inner_anchored: bool = False,
    metric: str = "sharpe_ratio",
    minimize: bool = False,
    initial_capital: float = 10000.0,
    costs_pct: float = 0.001,
    cost_profile: Optional[Dict[str, Any]] = None,
    execution_lag_days: int = 0,
    max_volume_participation: Optional[float] = None,
    min_daily_dollar_volume: float = 0.0,
    liquidity_on_missing_volume: str = "allow",
    risk_overlay: Optional[Dict[str, Any]] = None,
    tax_enabled: bool = True,
    metric_basis: str = "net_liquidation",
    progress_callback: Optional[callable] = None,
    validate: bool = True,
    drip_enabled: bool = False,
    external_features: Optional[ExternalFeaturesConfig] = None,
) -> OptimizationResult:
    """
    Run walk-forward analysis for a single strategy.

    Args:
        strategy_file: Path to the strategy file
        data: Price data to use
        param_grid: Optional custom parameter grid (uses defaults if None)
        rebalance_frequencies: Rebalance frequencies to evaluate
        train_years: Training window in years
        test_years: Test window in years
        step_months: Step size between windows in months
        anchored: Use anchored (expanding) training windows
        nested: Use nested walk-forward
        inner_train_years: Inner train window for nested mode
        inner_test_years: Inner test window for nested mode
        inner_step_months: Inner step size for nested mode
        inner_anchored: Anchored inner windows for nested mode
        metric: Metric to optimize
        minimize: Minimize metric (instead of maximize)
        initial_capital: Initial capital
        costs_pct: Transaction costs
        cost_profile: Optional per-ticker/asset-class cost profile
        execution_lag_days: Delay between signal and execution in trading days
        max_volume_participation: Optional max traded share as fraction of daily volume
        min_daily_dollar_volume: Optional minimum daily notional floor
        liquidity_on_missing_volume: "allow" or "skip" when volume is missing
        risk_overlay: Optional portfolio-level risk overlay settings
        tax_enabled: Whether to enable German tax model
        metric_basis: Metric basis (gross, net_realized, net_liquidation)
        progress_callback: Optional callback(window_num, total_windows, params, oos_sharpe)

    Returns:
        OptimizationResult with walk-forward results
    """
    from backtest.research.walk_forward import WalkForwardAnalysis
    from backtest.config.run_config import RunConfig, CostConfig, TaxConfig
    from backtest.cli import _build_walk_forward_windows, _instantiate_strategy_for_params

    # Load strategy
    strategy_instance, strategy_class, strategy_name = load_strategy_from_file(strategy_file)

    # Get parameter grid
    if param_grid is None:
        param_grid = get_param_grid_for_strategy(strategy_class, strategy_instance)

    frequencies = list(rebalance_frequencies or ["monthly"])
    if not frequencies:
        frequencies = ["monthly"]

    # Convert years to months
    train_months = int(train_years * 12)
    test_months = int(test_years * 12)

    def run_metric_backtest(eval_data: PriceData, rebalance_frequency: str, params: Dict[str, Any]):
        strategy = _instantiate_strategy_for_params(strategy_class, strategy_instance, params)
        config = BacktestConfig(
            initial_capital=initial_capital,
            costs_pct=costs_pct,
            cost_profile=cost_profile,
            execution_lag_days=execution_lag_days,
            max_volume_participation=max_volume_participation,
            min_daily_dollar_volume=min_daily_dollar_volume,
            liquidity_on_missing_volume=liquidity_on_missing_volume,
            risk_overlay=risk_overlay,
            rebalance_frequency=rebalance_frequency,
            tax_enabled=tax_enabled,
            metric_basis=metric_basis,
            validate=validate,
            drip_enabled=drip_enabled,
            external_features_loader=build_loader_from_config(
                external_features or ExternalFeaturesConfig()
            ),
        )
        backtester = Backtester(strategy, eval_data, config)
        result = backtester.run()
        metrics = result.metrics
        metric_value = getattr(metrics, metric, float("nan"))
        if pd.isna(metric_value):
            raise ValueError(f"Metric '{metric}' returned NaN")
        return float(metric_value), metrics

    if not nested:
        # Build RunConfig for standard walk-forward.
        run_config = RunConfig(
            start_date=data.start_date.strftime("%Y-%m-%d"),
            end_date=data.end_date.strftime("%Y-%m-%d"),
            initial_capital=initial_capital,
            rebalance_frequency=frequencies[0],
            costs=CostConfig(commission_pct=costs_pct),
            cost_profile=cost_profile,
            execution_lag_days=execution_lag_days,
            max_volume_participation=max_volume_participation,
            min_daily_dollar_volume=min_daily_dollar_volume,
            liquidity_on_missing_volume=liquidity_on_missing_volume,
            risk_overlay=risk_overlay,
            tax=TaxConfig(
                enabled=tax_enabled,
            ),
            external_features=external_features or ExternalFeaturesConfig(),
        )

        wfa = WalkForwardAnalysis(
            strategy_class=strategy_class,
            param_grid=param_grid,
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            anchored=anchored,
            metric=metric,
        )

        wf_result = wfa.run(data, run_config, progress=False)

        best_oos_window = None
        best_oos_sharpe = float("-inf")
        for w in wf_result.windows:
            oos_sharpe = w.test_metrics.get("sharpe_ratio", float("-inf"))
            if oos_sharpe > best_oos_sharpe:
                best_oos_sharpe = oos_sharpe
                best_oos_window = w

        wf_summary = WalkForwardResultSummary(
            num_windows=len(wf_result.windows),
            avg_is_sharpe=wf_result.overall_is_sharpe,
            avg_oos_sharpe=wf_result.overall_oos_sharpe,
            degradation_ratio=wf_result.degradation_ratio,
            overfitting_score=wf_result.overfitting_score,
            best_oos_params=best_oos_window.best_params if best_oos_window else {},
            best_oos_sharpe=best_oos_sharpe if best_oos_sharpe != float("-inf") else 0.0,
            mode="standard",
            nested=False,
        )

        return OptimizationResult(
            strategy_name=strategy_name,
            strategy_class=strategy_class.__name__,
            strategy_file=strategy_file,
            best_params=wf_summary.best_oos_params,
            best_rebalance_frequency=frequencies[0],
            best_metric_value=wf_summary.avg_oos_sharpe,  # Use OOS metric as best value
            metric_name=metric,
            all_results=[],  # Walk-forward doesn't use grid results
            walk_forward=wf_summary,
        )

    # Nested walk-forward path.
    outer_train_days = int(train_years * 252)
    outer_test_days = int(test_years * 252)
    outer_step_days = int(step_months * 21)
    inner_train_days = int(inner_train_years * 252)
    inner_test_days = int(inner_test_years * 252)
    inner_step_days = int(inner_step_months * 21)

    if outer_train_days <= 0 or outer_test_days <= 0 or outer_step_days <= 0:
        raise ValueError("Outer walk-forward settings must be positive.")
    if inner_train_days <= 0 or inner_test_days <= 0 or inner_step_days <= 0:
        raise ValueError("Nested inner walk-forward settings must be positive.")
    if outer_train_days < inner_train_days + inner_test_days:
        raise ValueError(
            "Outer training window too short for nested walk-forward: "
            f"need at least {inner_train_years}y + {inner_test_years}y."
        )

    outer_windows = _build_walk_forward_windows(
        dates=data.prices.index,
        train_days=outer_train_days,
        test_days=outer_test_days,
        step_days=outer_step_days,
        anchored=anchored,
    )
    if not outer_windows:
        raise ValueError("No valid outer walk-forward windows could be generated.")

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values)) if param_values else [()]

    window_rows: List[Dict[str, Any]] = []

    for outer_idx, outer_window in enumerate(outer_windows, start=1):
        outer_train_data = data.filter_dates(outer_window["train_start"], outer_window["train_end"])
        outer_test_data = data.filter_dates(outer_window["test_start"], outer_window["test_end"])

        inner_windows = _build_walk_forward_windows(
            dates=outer_train_data.prices.index,
            train_days=inner_train_days,
            test_days=inner_test_days,
            step_days=inner_step_days,
            anchored=inner_anchored,
        )
        if not inner_windows:
            continue

        candidate_scores = []
        for rebal_freq in frequencies:
            for param_combo in param_combinations:
                params = dict(zip(param_names, param_combo))
                inner_scores: List[float] = []
                for inner_window in inner_windows:
                    inner_eval = outer_train_data.filter_dates(
                        inner_window["test_start"],
                        inner_window["test_end"],
                    )
                    try:
                        metric_value, _ = run_metric_backtest(
                            eval_data=inner_eval,
                            rebalance_frequency=rebal_freq,
                            params=params,
                        )
                        inner_scores.append(metric_value)
                    except Exception:
                        continue

                if inner_scores:
                    mean_score = float(sum(inner_scores) / len(inner_scores))
                    variance = float(sum((x - mean_score) ** 2 for x in inner_scores) / len(inner_scores))
                    candidate_scores.append(
                        (rebal_freq, params, mean_score, math.sqrt(variance), len(inner_scores))
                    )

        if not candidate_scores:
            continue

        if minimize:
            candidate_scores.sort(key=lambda row: (row[2], row[3]))
        else:
            candidate_scores.sort(key=lambda row: (-row[2], row[3]))
        best_rebal, best_params, inner_score_mean, inner_score_std, inner_windows_used = candidate_scores[0]

        try:
            train_metric, train_metrics = run_metric_backtest(
                eval_data=outer_train_data,
                rebalance_frequency=best_rebal,
                params=best_params,
            )
            test_metric, test_metrics = run_metric_backtest(
                eval_data=outer_test_data,
                rebalance_frequency=best_rebal,
                params=best_params,
            )
        except Exception:
            continue

        if train_metric == 0:
            degradation = 0.0
        elif minimize:
            degradation = (test_metric - train_metric) / abs(train_metric)
        else:
            degradation = (train_metric - test_metric) / abs(train_metric)

        window_rows.append(
            {
                "window": outer_idx,
                "best_params": best_params,
                "best_rebalance": best_rebal,
                "train_sharpe": float(getattr(train_metrics, "sharpe_ratio", 0.0) or 0.0),
                "test_sharpe": float(getattr(test_metrics, "sharpe_ratio", 0.0) or 0.0),
                "degradation": degradation,
                "inner_score_mean": inner_score_mean,
                "inner_score_std": inner_score_std,
                "inner_windows": inner_windows_used,
            }
        )

    if not window_rows:
        raise ValueError("No valid nested walk-forward windows produced results.")

    avg_is_sharpe = float(sum(w["train_sharpe"] for w in window_rows) / len(window_rows))
    avg_oos_sharpe = float(sum(w["test_sharpe"] for w in window_rows) / len(window_rows))
    avg_degradation = float(sum(w["degradation"] for w in window_rows) / len(window_rows))
    overfitting_score = float(min(1.0, max(0.0, avg_degradation)))
    best_window = max(window_rows, key=lambda row: row["test_sharpe"])

    wf_summary = WalkForwardResultSummary(
        num_windows=len(window_rows),
        avg_is_sharpe=avg_is_sharpe,
        avg_oos_sharpe=avg_oos_sharpe,
        degradation_ratio=avg_degradation,
        overfitting_score=overfitting_score,
        best_oos_params=best_window["best_params"],
        best_oos_sharpe=best_window["test_sharpe"],
        mode="nested",
        nested=True,
        inner_train_years=inner_train_years,
        inner_test_years=inner_test_years,
        inner_step_months=inner_step_months,
        inner_anchored=inner_anchored,
    )

    return OptimizationResult(
        strategy_name=strategy_name,
        strategy_class=strategy_class.__name__,
        strategy_file=strategy_file,
        best_params=wf_summary.best_oos_params,
        best_rebalance_frequency=best_window["best_rebalance"],
        best_metric_value=wf_summary.avg_oos_sharpe,
        metric_name=metric,
        all_results=[],
        walk_forward=wf_summary,
    )


def run_batch_optimization(
    strategy_files: List[str],
    start: str = "2010-01-01",
    end: Optional[str] = None,
    metric: str = "sharpe_ratio",
    minimize: bool = False,
    rebalance_frequencies: Optional[List[str]] = None,
    initial_capital: float = 10000.0,
    costs_pct: float = 0.001,
    cost_profile: Optional[Dict[str, Any]] = None,
    execution_lag_days: int = 0,
    max_volume_participation: Optional[float] = None,
    min_daily_dollar_volume: float = 0.0,
    liquidity_on_missing_volume: str = "allow",
    risk_overlay: Optional[Dict[str, Any]] = None,
    tax_enabled: bool = True,
    metric_basis: str = "net_liquidation",
    custom_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    progress: bool = True,
    fail_fast: bool = False,
    align: str = "ffill",
    skip_failed: bool = True,
    validate: bool = True,
    drip_enabled: bool = False,
    # Walk-Forward options
    walk_forward: bool = False,
    walk_forward_nested: bool = False,
    train_years: float = 5.0,
    test_years: float = 1.0,
    step_months: int = 12,
    anchored: bool = False,
    inner_train_years: float = 3.0,
    inner_test_years: float = 1.0,
    inner_step_months: int = 6,
    inner_anchored: bool = False,
    progress_callback: Optional[callable] = None,
    cancel_check: Optional[callable] = None,
    external_features: Optional[ExternalFeaturesConfig] = None,
) -> BatchOptimizationResult:
    """
    Run batch optimization across multiple strategies.

    Args:
        strategy_files: List of paths to strategy files
        start: Start date for backtest
        end: End date for backtest (None = latest)
        metric: Metric to optimize
        minimize: If True, minimize metric
        rebalance_frequencies: Rebalance frequencies to test
        initial_capital: Initial capital
        costs_pct: Transaction costs
        cost_profile: Optional per-ticker/asset-class cost profile
        execution_lag_days: Delay between signal and execution in trading days
        max_volume_participation: Optional max traded share as fraction of daily volume
        min_daily_dollar_volume: Optional minimum daily notional floor
        liquidity_on_missing_volume: "allow" or "skip" when volume is missing
        risk_overlay: Optional portfolio-level risk overlay settings
        tax_enabled: Enable German tax model
        metric_basis: Metric basis
        custom_grids: Custom parameter grids (overrides defaults)
        progress: Show progress output
        fail_fast: Stop on first error
        align: Data alignment method
        skip_failed: Skip failed ticker downloads
        validate: Enable data validation
        drip_enabled: Enable dividend reinvestment
        walk_forward: Use walk-forward analysis instead of grid search
        walk_forward_nested: Use nested walk-forward
        train_years: Walk-forward training window in years
        test_years: Walk-forward test window in years
        step_months: Walk-forward step size in months
        anchored: Use anchored (expanding) training windows
        inner_train_years: Nested mode inner train window in years
        inner_test_years: Nested mode inner test window in years
        inner_step_months: Nested mode inner step size in months
        inner_anchored: Nested mode inner anchored windows

    Returns:
        BatchOptimizationResult with all results
    """
    import warnings
    import time

    # Suppress validation warnings during batch optimization (they flood the output)
    warnings.filterwarnings("ignore", module="backtest.validation")
    warnings.filterwarnings("ignore", category=FutureWarning)

    if rebalance_frequencies is None:
        rebalance_frequencies = DEFAULT_REBALANCE_FREQUENCIES

    batch_result = BatchOptimizationResult(
        metric_name=metric,
        start_date=start,
        end_date=end or "latest",
        config={
            "initial_capital": initial_capital,
            "costs_pct": costs_pct,
            "cost_profile": cost_profile,
            "execution_lag_days": execution_lag_days,
            "max_volume_participation": max_volume_participation,
            "min_daily_dollar_volume": min_daily_dollar_volume,
            "liquidity_on_missing_volume": liquidity_on_missing_volume,
            "risk_overlay": risk_overlay,
            "tax_enabled": tax_enabled,
            "metric_basis": metric_basis,
            "rebalance_frequencies": rebalance_frequencies,
            "walk_forward": walk_forward,
            "walk_forward_nested": walk_forward_nested if walk_forward else None,
            "wf_train_years": train_years if walk_forward else None,
            "wf_test_years": test_years if walk_forward else None,
            "wf_step_months": step_months if walk_forward else None,
            "wf_anchored": anchored if walk_forward else None,
            "wf_inner_train_years": inner_train_years if walk_forward and walk_forward_nested else None,
            "wf_inner_test_years": inner_test_years if walk_forward and walk_forward_nested else None,
            "wf_inner_step_months": inner_step_months if walk_forward and walk_forward_nested else None,
            "wf_inner_anchored": inner_anchored if walk_forward and walk_forward_nested else None,
        }
    )

    # Collect all required assets
    all_assets = set()
    strategy_info = []

    if progress:
        print("\nLoading strategies...")

    for path in strategy_files:
        try:
            instance, cls, name = load_strategy_from_file(path)
            if instance:
                all_assets.update(instance.assets)
            strategy_info.append((path, instance, cls, name))
            if progress_callback:
                progress_callback(
                    status="loading_strategies",
                    strategy_index=len(strategy_info),
                    strategy_total=len(strategy_files),
                    strategy_name=name,
                    run_count=0,
                    total_runs=0,
                )
            if progress:
                print(f"  + {name} ({cls.__name__})")
        except Exception as e:
            if fail_fast:
                raise
            if progress:
                print(f"  - {path}: {e}")

    if not strategy_info:
        raise ValueError("No valid strategies loaded")

    # Load data
    if progress:
        print(f"\nLoading price data for {len(all_assets)} assets...")

    if progress_callback:
        progress_callback(
            status="loading_data",
            strategy_index=0,
            strategy_total=len(strategy_files),
            strategy_name=None,
            run_count=0,
            total_runs=0,
        )

    load_volumes = (
        (max_volume_participation is not None and max_volume_participation > 0)
        or min_daily_dollar_volume > 0
    )
    data = DataLoader.yahoo(
        tickers=list(all_assets),
        start=start,
        end=end,
        currency="EUR",
        align=align,
        skip_failed=skip_failed,
        load_dividends=drip_enabled or tax_enabled,
        load_volumes=load_volumes,
        validate=validate,
    )

    if progress:
        print(f"Loaded {len(data.prices)} days of data")
        print(f"Date range: {data.start_date.strftime('%Y-%m-%d')} → {data.end_date.strftime('%Y-%m-%d')}")

    batch_result.end_date = data.end_date.strftime('%Y-%m-%d')

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

    # Optimize each strategy
    total_strategies = len(strategy_info)

    for idx, (path, instance, cls, name) in enumerate(strategy_info, 1):
        if cancel_check and cancel_check():
            raise BatchOptimizeCancelled("Batch optimization cancelled by user.")
        if progress:
            if walk_forward:
                mode_str = "Nested Walk-Forward" if walk_forward_nested else "Walk-Forward"
            else:
                mode_str = "Grid Search"
            print(f"\n[{idx}/{total_strategies}] {mode_str} optimizing {name}...")

        if progress_callback:
            progress_callback(
                status="running",
                strategy_index=idx,
                strategy_total=total_strategies,
                strategy_name=name,
                run_count=0,
                total_runs=0,
            )

        try:
            # Get param grid
            param_grid = get_param_grid_for_strategy(cls, instance, custom_grids)

            if progress and param_grid:
                print(f"  Parameters: {list(param_grid.keys())}")
                if walk_forward:
                    mode = "anchored" if anchored else "rolling"
                    print(
                        f"  Walk-Forward: train={train_years}y, test={test_years}y, "
                        f"step={step_months}m ({mode})"
                    )
                    if walk_forward_nested:
                        inner_mode = "anchored" if inner_anchored else "rolling"
                        print(
                            f"  Nested inner: train={inner_train_years}y, test={inner_test_years}y, "
                            f"step={inner_step_months}m ({inner_mode})"
                        )
                else:
                    total_combos = 1
                    for values in param_grid.values():
                        total_combos *= len(values)
                    total_runs = total_combos * len(rebalance_frequencies)
                    print(f"  Combinations: {total_runs}")

            if walk_forward:
                # Use walk-forward analysis
                result = walk_forward_single_strategy(
                    strategy_file=path,
                    data=data,
                    param_grid=param_grid,
                    rebalance_frequencies=rebalance_frequencies,
                    train_years=train_years,
                    test_years=test_years,
                    step_months=step_months,
                    anchored=anchored,
                    nested=walk_forward_nested,
                    inner_train_years=inner_train_years,
                    inner_test_years=inner_test_years,
                    inner_step_months=inner_step_months,
                    inner_anchored=inner_anchored,
                    metric=metric,
                    minimize=minimize,
                    initial_capital=initial_capital,
                    costs_pct=costs_pct,
                    cost_profile=cost_profile,
                    execution_lag_days=execution_lag_days,
                    max_volume_participation=max_volume_participation,
                    min_daily_dollar_volume=min_daily_dollar_volume,
                    liquidity_on_missing_volume=liquidity_on_missing_volume,
                    risk_overlay=risk_overlay,
                    tax_enabled=tax_enabled,
                    metric_basis=metric_basis,
                    validate=validate,
                    drip_enabled=drip_enabled,
                    external_features=external_features,
                )

                batch_result.results.append(result)

                if progress and result.walk_forward:
                    wf = result.walk_forward
                    params_str = ", ".join(f"{k}={v}" for k, v in result.best_params.items())
                    print(f"  Windows: {wf.num_windows}")
                    print(f"  IS Sharpe: {wf.avg_is_sharpe:.3f}, OOS Sharpe: {wf.avg_oos_sharpe:.3f}")
                    print(f"  Degradation: {wf.degradation_ratio:.1%}, Overfitting: {wf.overfitting_score:.2f}")
                    print(f"  Best OOS params: {params_str}")

            else:
                # Use standard grid search
                # Progress callback
                start_time = time.perf_counter()

                def progress_cb(run, total, params, metric_val):
                    if cancel_check and cancel_check():
                        raise BatchOptimizeCancelled("Batch optimization cancelled by user.")
                    if progress_callback:
                        progress_callback(
                            status="running",
                            strategy_index=idx,
                            strategy_total=total_strategies,
                            strategy_name=name,
                            run_count=run,
                            total_runs=total,
                        )
                    if progress and run % 10 == 0:
                        elapsed = time.perf_counter() - start_time
                        avg_time = elapsed / run if run else 0
                        eta = avg_time * (total - run)
                        print(
                            f"    [{run}/{total}] {metric}={metric_val:.4f} | "
                            f"Elapsed: {format_duration(elapsed)} | ETA: {format_duration(eta)}"
                        )

                result = optimize_single_strategy(
                    strategy_file=path,
                    data=data,
                    param_grid=param_grid,
                    rebalance_frequencies=rebalance_frequencies,
                    metric=metric,
                    minimize=minimize,
                    initial_capital=initial_capital,
                    costs_pct=costs_pct,
                    cost_profile=cost_profile,
                    execution_lag_days=execution_lag_days,
                    max_volume_participation=max_volume_participation,
                    min_daily_dollar_volume=min_daily_dollar_volume,
                    liquidity_on_missing_volume=liquidity_on_missing_volume,
                    risk_overlay=risk_overlay,
                    tax_enabled=tax_enabled,
                    metric_basis=metric_basis,
                    progress_callback=progress_cb if progress else None,
                    validate=validate,
                    drip_enabled=drip_enabled,
                    external_features=external_features,
                )

                batch_result.results.append(result)

                if progress:
                    params_str = ", ".join(f"{k}={v}" for k, v in result.best_params.items())
                    print(f"  Best: {result.best_rebalance_frequency}, {params_str}")
                    print(f"  {metric}={result.best_metric_value:.4f}")

        except Exception as e:
            if fail_fast:
                raise
            if progress:
                print(f"  ERROR: {e}")

    return batch_result


class BatchOptimizeCancelled(Exception):
    """Raised when batch optimization is cancelled."""


def save_batch_results(
    result: BatchOptimizationResult,
    output_dir: Path,
) -> None:
    """Save batch optimization results to files."""
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary CSV
    df = result.to_dataframe()
    df.to_csv(output_dir / "optimization_results.csv", index=False)
    print(f"  - optimization_results.csv")

    # Save detailed results per strategy
    details_dir = output_dir / "details"
    details_dir.mkdir(exist_ok=True)

    has_walk_forward = any(r.walk_forward for r in result.results)

    for r in result.results:
        if r.all_results:
            df_detail = pd.DataFrame(r.all_results)
            safe_name = r.strategy_name.lower().replace(' ', '_').replace('/', '_')
            df_detail.to_csv(details_dir / f"{safe_name}.csv", index=False)

    print(f"  - details/ (per-strategy CSVs)")

    # Save best params as Python file for reuse
    with open(output_dir / "optimized_params.py", "w") as f:
        f.write('"""Auto-generated optimized parameters from batch optimization."""\n\n')
        if has_walk_forward:
            f.write("# Parameters were validated with Walk-Forward Analysis\n")
            f.write("# OOS = Out-of-Sample (test data), IS = In-Sample (training data)\n\n")
        f.write("OPTIMIZED_PARAMS = {\n")
        for r in result.results:
            f.write(f'    "{r.strategy_class}": {{\n')
            f.write(f'        "rebalance_frequency": "{r.best_rebalance_frequency}",\n')
            for k, v in r.best_params.items():
                if isinstance(v, str):
                    f.write(f'        "{k}": "{v}",\n')
                else:
                    f.write(f'        "{k}": {v},\n')
            # Add walk-forward comment if available
            if r.walk_forward:
                f.write(f'        # OOS Sharpe: {r.walk_forward.avg_oos_sharpe:.3f}, Overfitting: {r.walk_forward.overfitting_score:.2f}\n')
            f.write(f'    }},\n')
        f.write("}\n")

    print(f"  - optimized_params.py")

    # Save as JSON for web UI
    params_json = {}
    for r in result.results:
        params_json[r.strategy_class] = {
            "rebalance_frequency": r.best_rebalance_frequency,
            **r.best_params,
        }
        if r.walk_forward:
            params_json[r.strategy_class]["_walk_forward"] = {
                "avg_oos_sharpe": r.walk_forward.avg_oos_sharpe,
                "overfitting_score": r.walk_forward.overfitting_score,
                "degradation_ratio": r.walk_forward.degradation_ratio,
                "mode": r.walk_forward.mode,
                "nested": r.walk_forward.nested,
                "inner_train_years": r.walk_forward.inner_train_years,
                "inner_test_years": r.walk_forward.inner_test_years,
                "inner_step_months": r.walk_forward.inner_step_months,
                "inner_anchored": r.walk_forward.inner_anchored,
            }

    with open(output_dir / "optimized_params.json", "w") as f:
        json.dump(params_json, f, indent=2)

    print(f"  - optimized_params.json")

    # Save walk-forward summary if available
    if has_walk_forward:
        wf_rows = []
        for r in result.results:
            if r.walk_forward:
                wf = r.walk_forward
                wf_rows.append({
                    "strategy": r.strategy_name,
                    "num_windows": wf.num_windows,
                    "avg_is_sharpe": wf.avg_is_sharpe,
                    "avg_oos_sharpe": wf.avg_oos_sharpe,
                    "degradation": wf.degradation_ratio,
                    "overfitting_score": wf.overfitting_score,
                    "best_oos_sharpe": wf.best_oos_sharpe,
                    "mode": wf.mode,
                    "nested": wf.nested,
                    "inner_train_years": wf.inner_train_years,
                    "inner_test_years": wf.inner_test_years,
                    "inner_step_months": wf.inner_step_months,
                    "inner_anchored": wf.inner_anchored,
                })
        if wf_rows:
            df_wf = pd.DataFrame(wf_rows)
            df_wf = df_wf.sort_values("avg_oos_sharpe", ascending=False)
            df_wf.to_csv(output_dir / "walk_forward_summary.csv", index=False)
            print(f"  - walk_forward_summary.csv")
