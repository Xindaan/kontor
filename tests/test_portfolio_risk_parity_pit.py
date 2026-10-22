"""Phase E4 — PIT window tests (T-0325)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from backtest.portfolio.risk_parity import _build_covariance_matrix


def _make_prices(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = ["AAA", "BBB", "CCC"]
    rets = rng.normal(0.0005, 0.01, (n, len(cols)))
    return (1.0 + pd.DataFrame(rets, columns=cols, index=pd.bdate_range("2022-01-01", periods=n))).cumprod() * 100.0


def test_covariance_uses_window_before_as_of_only():
    """Correlation/vol only from data UP TO as_of (PIT)."""
    prices = _make_prices()
    cutoff = prices.index[200]
    sigma_at_cutoff = _build_covariance_matrix(prices, as_of=cutoff.date(), window=252)
    # Changing later prices after cutoff -> Sigma unchanged.
    prices_modified = prices.copy()
    prices_modified.iloc[201:] = prices_modified.iloc[201:] * 100.0  # absurd
    sigma_after_change = _build_covariance_matrix(prices_modified, as_of=cutoff.date(), window=252)
    np.testing.assert_allclose(
        sigma_at_cutoff.to_numpy(),
        sigma_after_change.to_numpy(),
        rtol=1e-10,
    )


def test_covariance_diagonal_strictly_positive():
    prices = _make_prices()
    sigma = _build_covariance_matrix(prices, window=252)
    diag = np.diag(sigma.to_numpy())
    assert (diag > 0).all()
