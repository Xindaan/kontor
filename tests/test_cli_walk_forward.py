import types

import pandas as pd

from backtest.cli import (
    _build_walk_forward_windows,
    _calculate_parameter_drift,
    _compute_degradation_pct,
    _select_metrics_for_optimization,
)


def test_build_walk_forward_windows_rolling():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    windows = _build_walk_forward_windows(
        dates=dates,
        train_days=10,
        test_days=5,
        step_days=5,
        anchored=False,
    )

    assert len(windows) == 2
    assert windows[0]["train_start"] == dates[0]
    assert windows[0]["train_end"] == dates[9]
    assert windows[0]["test_start"] == dates[10]
    assert windows[0]["test_end"] == dates[14]
    assert windows[1]["train_start"] == dates[5]
    assert windows[1]["test_end"] == dates[19]


def test_build_walk_forward_windows_anchored():
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    windows = _build_walk_forward_windows(
        dates=dates,
        train_days=10,
        test_days=5,
        step_days=5,
        anchored=True,
    )

    assert len(windows) == 2
    assert windows[0]["train_start"] == dates[0]
    assert windows[0]["train_end"] == dates[9]
    assert windows[1]["train_start"] == dates[0]
    assert windows[1]["train_end"] == dates[14]
    assert windows[1]["test_start"] == dates[15]
    assert windows[1]["test_end"] == dates[19]


def test_compute_degradation_pct_for_maximize_and_minimize():
    assert _compute_degradation_pct(2.0, 1.0, minimize=False) == 50.0
    assert _compute_degradation_pct(0.1, 0.2, minimize=True) == 100.0


def test_calculate_parameter_drift():
    window_results = [
        {"best_rebalance": "monthly", "best_lookback": 126, "best_top_n": 5},
        {"best_rebalance": "monthly", "best_lookback": 126, "best_top_n": 10},
        {"best_rebalance": "quarterly", "best_lookback": 189, "best_top_n": 10},
    ]

    avg_drift, drift_by_key = _calculate_parameter_drift(window_results, ["lookback", "top_n"])

    assert round(avg_drift, 4) == 0.5
    assert round(drift_by_key["best_rebalance"], 4) == 0.5
    assert round(drift_by_key["best_lookback"], 4) == 0.5
    assert round(drift_by_key["best_top_n"], 4) == 0.5


def test_select_metrics_for_optimization_prefers_primary_metrics():
    result = types.SimpleNamespace(metrics="primary", metrics_net="net", metrics_gross="gross")
    assert _select_metrics_for_optimization(result) == "primary"


def test_select_metrics_for_optimization_fallbacks():
    result = types.SimpleNamespace(metrics=None, metrics_net="net", metrics_gross="gross")
    assert _select_metrics_for_optimization(result) == "net"

    result = types.SimpleNamespace(metrics=None, metrics_net=None, metrics_gross="gross")
    assert _select_metrics_for_optimization(result) == "gross"
