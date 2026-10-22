"""Analyst-specific snapshot validation (Phase B T-0054).

Kept SEPARATE from the generic ``schema.validate_schema()`` so non-analyst
adapters (mock, news, ML, ...) are not forced into the analyst contract.

The required rows per ticker are:
- ``feature_name == "analyst_score"`` with ``feature_value`` in [-1, 1].

The optional rows must obey type/range constraints:
- ``analyst_buy_count``, ``analyst_hold_count``, ``analyst_sell_count``:
  integer-like, >= 0.
- ``price_target_mean``, ``price_target_median``: numeric, >= 0.
- ``price_target_age_days``: integer-like, >= 0. Strings (ISO dates etc.)
  are intentionally NOT allowed — ``feature_value`` is numeric per the
  Phase-A schema.
"""

from __future__ import annotations

from typing import Iterable, Set

import pandas as pd

REQUIRED_ANALYST_FEATURE = "analyst_score"
ANALYST_SCORE_MIN = -1.0
ANALYST_SCORE_MAX = 1.0

NON_NEGATIVE_INT_FEATURES: tuple[str, ...] = (
    "analyst_buy_count",
    "analyst_hold_count",
    "analyst_sell_count",
    "price_target_age_days",
)
NON_NEGATIVE_FLOAT_FEATURES: tuple[str, ...] = (
    "price_target_mean",
    "price_target_median",
)


def validate_analyst_snapshot(df: pd.DataFrame) -> None:
    """Validate a long-form analyst snapshot.

    Empty frames are allowed (no rows to validate). Otherwise:
    - Every distinct ticker in the frame must have at least one
      ``feature_name == "analyst_score"`` row.
    - All ``analyst_score`` values are numeric and in [-1, 1].
    - Optional features obey their type/range constraints.

    Raises ``ValueError`` with a descriptive message on the first
    violation.
    """

    if df is None or df.empty:
        return

    for required in ("ticker", "feature_name", "feature_value"):
        if required not in df.columns:
            raise ValueError(f"analyst snapshot missing required column '{required}'")

    score_rows = df.loc[df["feature_name"] == REQUIRED_ANALYST_FEATURE]
    tickers_with_score: Set[str] = set(score_rows["ticker"].astype(str).str.upper())
    all_tickers: Set[str] = set(df["ticker"].astype(str).str.upper())
    missing = sorted(all_tickers - tickers_with_score)
    if missing:
        raise ValueError(
            "analyst snapshot missing 'analyst_score' rows for tickers: "
            + ", ".join(missing)
        )

    score_values = pd.to_numeric(score_rows["feature_value"], errors="coerce")
    if score_values.isna().any():
        raise ValueError("'analyst_score' values must be numeric")
    if (score_values < ANALYST_SCORE_MIN).any() or (score_values > ANALYST_SCORE_MAX).any():
        raise ValueError(
            f"'analyst_score' values must be within [{ANALYST_SCORE_MIN}, "
            f"{ANALYST_SCORE_MAX}]"
        )

    for feature in NON_NEGATIVE_INT_FEATURES:
        _check_non_negative_int(df, feature)
    for feature in NON_NEGATIVE_FLOAT_FEATURES:
        _check_non_negative_float(df, feature)


def _check_non_negative_int(df: pd.DataFrame, feature: str) -> None:
    rows = df.loc[df["feature_name"] == feature]
    if rows.empty:
        return
    values = pd.to_numeric(rows["feature_value"], errors="coerce")
    if values.isna().any():
        raise ValueError(f"'{feature}' values must be numeric")
    if (values < 0).any():
        raise ValueError(f"'{feature}' values must be >= 0")
    # Integer-like check: reject fractional values.
    diffs = (values - values.round()).abs()
    if (diffs > 1e-9).any():
        raise ValueError(f"'{feature}' values must be integer-like (>= 0)")


def _check_non_negative_float(df: pd.DataFrame, feature: str) -> None:
    rows = df.loc[df["feature_name"] == feature]
    if rows.empty:
        return
    values = pd.to_numeric(rows["feature_value"], errors="coerce")
    if values.isna().any():
        raise ValueError(f"'{feature}' values must be numeric")
    if (values < 0).any():
        raise ValueError(f"'{feature}' values must be >= 0")


def analyst_feature_names() -> Iterable[str]:
    """Stable list of analyst feature names recognized by this helper."""
    return (
        REQUIRED_ANALYST_FEATURE,
        *NON_NEGATIVE_INT_FEATURES,
        *NON_NEGATIVE_FLOAT_FEATURES,
    )
