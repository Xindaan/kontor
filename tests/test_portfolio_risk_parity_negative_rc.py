"""Phase E4 — negative RC guard test (T-0332, Codex R3.8)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.portfolio.risk_parity import erc_weights


def test_erc_handles_negatively_correlated_assets():
    """Codex R3.8: two negatively correlated assets, solver must not
    crash or produce NaN/inf."""

    rng = np.random.default_rng(42)
    n = 500
    base = rng.normal(0.0, 0.01, n)
    rets = pd.DataFrame(
        {
            "A": base,  # correlated with base
            "B": -base + rng.normal(0.0, 0.003, n),  # ANTI-correlated with A
            "C": rng.normal(0.0005, 0.02, n),  # independent
        },
        index=pd.bdate_range("2022-01-01", periods=n),
    )
    weights = erc_weights(rets, target_sum=1.0)
    # Solver converges, all weights finite and positive.
    values = list(weights.values())
    assert all(np.isfinite(v) for v in values)
    assert all(v >= 0 for v in values)
    assert abs(sum(values) - 1.0) < 1e-4
