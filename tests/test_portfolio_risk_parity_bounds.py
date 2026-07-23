"""Phase E4 — Bounds-Feasibility-Tests (T-0323, Codex R3.9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.portfolio.risk_parity import erc_weights


def _make_returns(n_assets: int, n_periods: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    data = {c: rng.normal(0.0005, 0.01 + 0.001 * i, n_periods) for i, c in enumerate(cols)}
    return pd.DataFrame(data, index=pd.bdate_range("2024-01-01", periods=n_periods))


def test_raises_when_max_weight_too_tight():
    """n=5 × max_weight=0.1 = 0.5 < target_sum=1.0 -> ValueError."""
    rets = _make_returns(5)
    with pytest.raises(ValueError, match="max_weight"):
        erc_weights(rets, target_sum=1.0, max_weight=0.1)


def test_raises_when_min_weight_too_loose():
    """n=3 × min_weight=0.5 = 1.5 > target_sum=1.0 -> ValueError."""
    rets = _make_returns(3)
    with pytest.raises(ValueError, match="min_weight"):
        erc_weights(rets, target_sum=1.0, min_weight=0.5)


def test_accepts_feasible_bounds():
    """n=5 × max_weight=0.3 = 1.5 >= target_sum=1.0 -> OK."""
    rets = _make_returns(5)
    weights = erc_weights(rets, target_sum=1.0, max_weight=0.3)
    assert abs(sum(weights.values()) - 1.0) < 1e-5
    assert max(weights.values()) <= 0.3 + 1e-6


def test_target_sum_partial():
    """n=5 × max_weight=0.2 = 1.0 >= target_sum=0.8 -> OK."""
    rets = _make_returns(5)
    weights = erc_weights(rets, target_sum=0.8, max_weight=0.2)
    assert abs(sum(weights.values()) - 0.8) < 1e-5
