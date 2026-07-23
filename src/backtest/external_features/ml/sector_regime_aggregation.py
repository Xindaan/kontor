"""Cross-section sector regime aggregation (ML Round 3, Phase 3A).

Aggregates per-sector `P(stressed)` probabilities (from a
:mod:`regime_classifier_training` bundle, trained on US sector ETFs)
into a **scalar market-wide stress score** per ``as_of``.

Three modes:

- ``"std"`` — standard deviation of the per-sector probabilities.
  High when sectors diverge (typical risk-off, rotation).
  *This is the backtest-validated best variant* (Sharpe +0.06,
  MaxDD improvement +6.4pp at alpha=2.0 vs pure VolTarget,
  robust across all 4 sub-periods 2019-2024).
- ``"mean"`` — average. High when the whole market is under stress.
- ``"quantile_75"`` — upper quartile. High when the weakest
  sectors are under heavy stress.
- ``"quantile_95"`` / ``"max"`` — even more tail-focused.

PIT contract: the function is purist — it takes a snapshot of the
per-sector probabilities and returns a scalar. No past/future data.
The ``ts`` value is only passed through (for recording). Forward
leakage risk lies upstream at the bundle lookup (`select_bundle_for_as_of`
resp. the PIT-strict `available_from <= as_of` filter).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

AGGREGATION_MODES = ("mean", "std", "quantile_75", "quantile_95", "max", "min")

DEFAULT_MODE: str = "std"


def aggregate_sector_stress(
    per_sector_probs: Sequence[float] | Mapping[str, float] | pd.Series,
    mode: str = DEFAULT_MODE,
    *,
    min_sectors: int = 5,
) -> Optional[float]:
    """Cross-section aggregation of a collection of ``P(stressed)``.

    Args:
        per_sector_probs: per-sector stress probabilities
            (list, dict or Series). Values outside ``[0, 1]``
            are clipped.
        mode: aggregation mode, see :data:`AGGREGATION_MODES`.
        min_sectors: required minimum number of valid sectors.
            Otherwise the function returns ``None`` (caller decides
            the fallback).

    Returns:
        Scalar stress score (>= 0). ``None`` if too few
        valid inputs.

    Raises:
        ValueError: for an unknown ``mode``.
    """

    if mode not in AGGREGATION_MODES:
        raise ValueError(
            f"unknown aggregation mode '{mode}', expected {AGGREGATION_MODES}"
        )

    if isinstance(per_sector_probs, Mapping):
        values = list(per_sector_probs.values())
    elif isinstance(per_sector_probs, pd.Series):
        values = per_sector_probs.tolist()
    else:
        values = list(per_sector_probs)

    arr = np.asarray(
        [float(v) for v in values if v is not None and np.isfinite(v)],
        dtype=float,
    )
    if arr.size < min_sectors:
        return None
    # Clamp into valid probability range.
    arr = np.clip(arr, 0.0, 1.0)

    if mode == "mean":
        return float(np.mean(arr))
    if mode == "std":
        return float(np.std(arr, ddof=0))
    if mode == "quantile_75":
        return float(np.quantile(arr, 0.75))
    if mode == "quantile_95":
        return float(np.quantile(arr, 0.95))
    if mode == "max":
        return float(np.max(arr))
    if mode == "min":
        return float(np.min(arr))
    raise AssertionError(f"unreachable: {mode}")  # pragma: no cover


__all__ = ["aggregate_sector_stress", "AGGREGATION_MODES", "DEFAULT_MODE"]
