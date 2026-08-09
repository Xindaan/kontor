"""Phase E4 — pure-numpy Equal-Risk-Contribution solver (T-0321..T-0326).

Implements the Spinu-style multiplicative-update solver with a
negative-RC guard (Codex R3.8), a bounds-feasibility check (Codex R3.9),
and an inverse-vol fallback on non-convergence.

No SciPy dependency. Only numpy + pandas (for
`_build_covariance_matrix`).

Example usage:

```
import numpy as np
import pandas as pd
from backtest.portfolio.risk_parity import erc_weights, _build_covariance_matrix

prices = pd.DataFrame(...)
Σ = _build_covariance_matrix(prices, as_of=date(2024, 6, 30))
returns_df = prices.pct_change().dropna()
w = erc_weights(returns_df, target_sum=0.8, max_weight=0.4)
# w sums to 0.8, max cap 0.4 per asset.
```
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# Numerical constants (Codex R3.8).
_RC_FLOOR = 1e-12  # actual_rc[i] floor to avoid 1/0.
_DIAG_SHRINKAGE = 1e-8  # covariance diagonal shrinkage.


def inverse_vol_weights(
    returns: pd.DataFrame,
    target_sum: float = 1.0,
) -> Dict[str, float]:
    """Inverse-volatility weights scaled to ``target_sum``.

    Used both as the initial guess for the ERC solver and as a
    fallback on non-convergence.
    """

    if returns is None or returns.empty:
        return {}
    vol = returns.std(ddof=0)
    safe_vol = vol.where(vol > 0, np.nan)
    inv = 1.0 / safe_vol
    inv = inv.fillna(0.0)
    total = float(inv.sum())
    if total <= 0:
        # All vols 0 or NaN -> equal-weight.
        equal = target_sum / len(returns.columns)
        return {str(c): equal for c in returns.columns}
    weights = inv / total * float(target_sum)
    return {str(c): float(v) for c, v in weights.items()}


def _build_covariance_matrix(
    prices: pd.DataFrame,
    as_of: Optional[date] = None,
    window: int = 252,
    min_periods: int = 200,
) -> pd.DataFrame:
    """Codex R2.9 + R3.8: Σ = D × C × D with diagonal shrinkage.

    Args:
        prices: DataFrame with a DatetimeIndex and asset columns.
        as_of: PIT cutoff (inclusive). If ``None``, the entire frame
            is used.
        window: number of trading days for the estimation window.
        min_periods: minimum periods for the correlation estimation.

    Returns:
        Pandas DataFrame of the covariance matrix. Diagonal never
        below ``_DIAG_SHRINKAGE * trace(Σ) / n``.
    """

    if prices is None or prices.empty:
        return pd.DataFrame()

    if as_of is not None:
        sliced = prices.loc[: pd.Timestamp(as_of)]
    else:
        sliced = prices
    if len(sliced) > window:
        sliced = sliced.tail(window)
    returns = sliced.pct_change().dropna(how="all")
    if returns.empty:
        return pd.DataFrame(index=prices.columns, columns=prices.columns, data=0.0)

    # D = diag(vol_i), annualized.
    vol = returns.std(ddof=0) * math.sqrt(252.0)
    vol = vol.fillna(0.0)

    # C = corr (Pearson). NaN cells -> 0.0.
    corr = returns.corr(method="pearson", min_periods=min_periods).fillna(0.0)

    # Σ = D × C × D
    diag = np.diag(vol.to_numpy())
    sigma_np = diag @ corr.to_numpy() @ diag
    sigma = pd.DataFrame(sigma_np, index=corr.index, columns=corr.columns)

    # Diagonal shrinkage (Codex R3.8).
    n = len(sigma)
    if n > 0:
        trace_sigma = float(np.trace(sigma.to_numpy()))
        shrinkage = max(_DIAG_SHRINKAGE * trace_sigma / max(1, n), _DIAG_SHRINKAGE)
        for i, asset in enumerate(sigma.index):
            current = float(sigma.iloc[i, i])
            sigma.iloc[i, i] = max(current + shrinkage, _DIAG_SHRINKAGE)
    return sigma


def _validate_bounds(
    n: int,
    target_sum: float,
    max_weight: Optional[float],
    min_weight: float,
) -> None:
    """Codex R3.9: bounds feasibility check BEFORE the solver."""

    if max_weight is not None and n * float(max_weight) < float(target_sum):
        raise ValueError(
            f"max_weight={max_weight} × n={n} = {n*float(max_weight):.6g} "
            f"< target_sum={target_sum:.6g}; constraints infeasible"
        )
    if float(min_weight) > 0 and n * float(min_weight) > float(target_sum):
        raise ValueError(
            f"min_weight={min_weight} × n={n} = {n*float(min_weight):.6g} "
            f"> target_sum={target_sum:.6g}; constraints infeasible"
        )


def _apply_max_weight_cap(
    w: np.ndarray,
    max_weight: Optional[float],
    target_sum: float,
    inner_iter: int = 50,
) -> np.ndarray:
    """Iteratively apply the cap + renormalize until no asset exceeds the cap."""

    if max_weight is None:
        return w
    cap = float(max_weight)
    out = np.array(w, dtype=float)
    for _ in range(inner_iter):
        # If all assets are already capped, there's nowhere further to go.
        over = out > cap + 1e-12
        if not over.any():
            break
        # Set the exceeding values exactly to the cap.
        out = np.minimum(out, cap)
        # Renormalize the non-capped components onto the remaining
        # share.
        capped_total = float(np.sum(out[over])) if over.any() else 0.0
        remaining_target = float(target_sum) - capped_total
        free_mask = ~over
        free_sum = float(np.sum(out[free_mask]))
        if free_sum <= 0 or remaining_target <= 0:
            # All assets capped, or the cap distributes below target_sum/n.
            s = float(out.sum())
            if s > 0:
                out = out / s * float(target_sum)
            break
        scale = remaining_target / free_sum
        out[free_mask] = out[free_mask] * scale
    return out


def erc_weights(
    returns: pd.DataFrame,
    target_sum: float = 1.0,
    max_weight: Optional[float] = None,
    min_weight: float = 0.0,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """Spinu-style Equal-Risk-Contribution weights.

    Args:
        returns: DataFrame with asset returns (one column per asset).
        target_sum: target sum of the weights (Codex R2.10). Cash =
            1 - target_sum.
        max_weight: optional per-asset cap (e.g. 0.4).
        min_weight: optional per-asset floor (default 0.0).
        max_iter: max iterations (default 200).
        tol: convergence tolerance (default 1e-8).

    Returns:
        Dict ``{ticker: weight}`` with ``sum(weights) ≈ target_sum``.

    Raises:
        ValueError: if bounds are infeasible (Codex R3.9).
    """

    if returns is None or returns.empty:
        return {}
    columns = list(returns.columns)
    n = len(columns)

    _validate_bounds(n, target_sum, max_weight, min_weight)

    # min_weight > 0: pre-filter assets below the floor — not relevant
    # for the inverse-vol path in Phase E, so not implemented.
    if min_weight > 0:
        # Not needed in Phase E; the plan reserves the feature, but
        # erc_weights is only ever called with min_weight=0.0.
        logger.warning("min_weight>0 not yet implemented; using min_weight=0")

    # Estimate Σ from the returns themselves (annualized, without
    # sqrt(252) because we work consistently in the same scale here).
    cov_np = returns.cov(min_periods=max(2, len(returns) // 2)).fillna(0.0).to_numpy()
    # Diagonal shrinkage (Codex R3.8).
    trace_cov = float(np.trace(cov_np))
    shrinkage = max(_DIAG_SHRINKAGE * trace_cov / max(1, n), _DIAG_SHRINKAGE)
    for i in range(n):
        cov_np[i, i] = max(cov_np[i, i] + shrinkage, _DIAG_SHRINKAGE)

    # Initial guess: inverse-vol scaled to target_sum (Codex answer).
    inv_vol = inverse_vol_weights(returns, target_sum=target_sum)
    w = np.array([inv_vol.get(str(c), target_sum / n) for c in columns], dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.full(n, target_sum / n, dtype=float)

    converged = False
    target_rc_share = 1.0 / n  # equal contribution share

    for iteration in range(max_iter):
        sigma_w = cov_np @ w
        total_risk = float(math.sqrt(max(w @ sigma_w, _RC_FLOOR)))
        # Negative-RC guard (Codex R3.8): floor on actual_rc.
        actual_rc = w * sigma_w / max(total_risk, _RC_FLOOR)
        actual_rc = np.maximum(actual_rc, _RC_FLOOR * total_risk)
        target_rc = total_risk * target_rc_share

        # Spinu update.
        ratio = target_rc / actual_rc
        if not np.isfinite(ratio).all() or (ratio <= 0).any():
            logger.warning(
                "ERC solver hit non-finite ratio at iter %d; falling back to inverse-vol",
                iteration,
            )
            return _fallback_inverse_vol(returns, target_sum, max_weight)

        w_new = w * np.sqrt(ratio)
        s = float(w_new.sum())
        if s <= 0 or not math.isfinite(s):
            return _fallback_inverse_vol(returns, target_sum, max_weight)
        w_new = w_new / s * float(target_sum)

        if max_weight is not None:
            w_new = _apply_max_weight_cap(w_new, max_weight, target_sum)

        max_step = float(np.max(np.abs(w_new - w)))
        w = w_new
        if max_step < tol:
            converged = True
            break

    if not converged:
        logger.warning(
            "ERC solver did not converge after %d iterations; falling back to inverse-vol",
            max_iter,
        )
        return _fallback_inverse_vol(returns, target_sum, max_weight)

    return {str(c): float(weight) for c, weight in zip(columns, w)}


def _fallback_inverse_vol(
    returns: pd.DataFrame,
    target_sum: float,
    max_weight: Optional[float],
) -> Dict[str, float]:
    weights = inverse_vol_weights(returns, target_sum=target_sum)
    if max_weight is None:
        return weights
    columns = list(weights.keys())
    w = np.array([weights[c] for c in columns], dtype=float)
    w = _apply_max_weight_cap(w, max_weight, target_sum)
    return {c: float(v) for c, v in zip(columns, w)}


__all__ = [
    "erc_weights",
    "inverse_vol_weights",
    "_build_covariance_matrix",
]
