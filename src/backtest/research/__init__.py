"""
Research tools for Kontor.

Provides tools for strategy robustness testing:
- Parameter sweeps
- Walk-forward analysis
- Out-of-sample testing
"""

from backtest.research.sweep import ParameterSweep, SweepResult
from backtest.research.walk_forward import WalkForwardAnalysis, WalkForwardResult
from backtest.research.segments import (
    build_ai_infra_pit_universe,
    build_ai_infra_ub_universe,
    equity_fund_map,
    load_cap_groups,
    load_segment_map,
    segment_cap_report,
    segment_tickers,
)

__all__ = [
    "ParameterSweep",
    "SweepResult",
    "WalkForwardAnalysis",
    "WalkForwardResult",
    "build_ai_infra_pit_universe",
    "build_ai_infra_ub_universe",
    "equity_fund_map",
    "load_cap_groups",
    "load_segment_map",
    "segment_cap_report",
    "segment_tickers",
]
