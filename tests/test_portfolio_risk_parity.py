"""Phase E4 — ERC solver tests (T-0321, T-0324, T-0327..T-0331)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest.portfolio.risk_parity import (
    _build_covariance_matrix,
    erc_weights,
    inverse_vol_weights,
)


def _toy_3_asset_returns(seed: int = 0, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = ["AAA", "BBB", "CCC"]
    # Three assets with different vol levels.
    data = {
        "AAA": rng.normal(0.0005, 0.01, n),
        "BBB": rng.normal(0.0005, 0.02, n),
        "CCC": rng.normal(0.0005, 0.03, n),
    }
    return pd.DataFrame(data, index=pd.bdate_range("2022-01-01", periods=n))


def test_inverse_vol_weights_sum_to_target_sum():
    rets = _toy_3_asset_returns()
    weights = inverse_vol_weights(rets, target_sum=1.0)
    assert abs(sum(weights.values()) - 1.0) < 1e-8


def test_erc_converges_on_toy_3_asset():
    rets = _toy_3_asset_returns()
    weights = erc_weights(rets, target_sum=1.0, max_iter=200, tol=1e-8)
    assert abs(sum(weights.values()) - 1.0) < 1e-5
    # Risk contribution differences < 5% (tolerance for 3-asset toy).
    cov = rets.cov().to_numpy()
    w_array = np.array([weights[c] for c in rets.columns])
    sigma_w = cov @ w_array
    total_risk = float(np.sqrt(w_array @ sigma_w))
    rc = w_array * sigma_w / max(total_risk, 1e-12)
    rc_share = rc / rc.sum()
    target_share = 1.0 / 3
    assert np.max(np.abs(rc_share - target_share)) < 0.05


def test_erc_higher_vol_gets_lower_weight():
    """Higher vol = lower ERC weight."""
    rets = _toy_3_asset_returns()
    weights = erc_weights(rets)
    # AAA (vol 0.01) should have higher weight than CCC (vol 0.03).
    assert weights["AAA"] > weights["CCC"]


def test_erc_different_from_inverse_vol():
    """ERC and inverse vol produce different allocations."""
    rets = _toy_3_asset_returns()
    erc = erc_weights(rets)
    inv = inverse_vol_weights(rets)
    # At least one asset has a difference > 1e-4.
    diffs = [abs(erc[c] - inv[c]) for c in rets.columns]
    assert max(diffs) > 1e-4


def test_erc_target_sum_other_than_one():
    """Codex R4.16: target_sum=0.8 -> output sums to 0.8."""
    rets = _toy_3_asset_returns()
    weights = erc_weights(rets, target_sum=0.8)
    assert abs(sum(weights.values()) - 0.8) < 1e-5


def test_erc_max_weight_cap():
    rets = _toy_3_asset_returns()
    weights = erc_weights(rets, target_sum=1.0, max_weight=0.4)
    assert max(weights.values()) <= 0.4 + 1e-6


def test_build_covariance_diagonal_above_shrinkage():
    prices = (1.0 + _toy_3_asset_returns()).cumprod() * 100.0
    sigma = _build_covariance_matrix(prices, as_of=prices.index[-1], window=252)
    diag = np.diag(sigma.to_numpy())
    assert (diag > 0).all()
