"""Phase D long-form ML-snapshot schema validation (T-0213).

Mirrors :mod:`backtest.external_features.news_schema`. Required rows
include the row-level forecast score plus the two ordinal metadata
features that let :func:`assess_ml_evidence` filter out lookahead
snapshots without touching provenance (Codex D14/D18).
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

REQUIRED_FEATURES: tuple[str, ...] = (
    "ml_forecast_score",
    "ml_available_from_ordinal",
    "ml_feature_trained_through_ordinal",
)


def validate_ml_snapshot(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    for col in ("ticker", "feature_name", "feature_value"):
        if col not in df.columns:
            raise ValueError(f"ml snapshot is missing required column '{col}'")

    tickers_in_frame = set(df["ticker"].astype(str).str.upper().unique())
    for feature in REQUIRED_FEATURES:
        rows = df.loc[df["feature_name"] == feature]
        if rows.empty:
            raise ValueError(f"ml snapshot must contain '{feature}' rows")
        tickers_with = set(rows["ticker"].astype(str).str.upper().unique())
        missing = sorted(tickers_in_frame - tickers_with)
        if missing:
            raise ValueError(
                f"ml snapshot missing '{feature}' rows for tickers: "
                + ", ".join(missing)
            )

    score_rows = df.loc[df["feature_name"] == "ml_forecast_score"]
    scores = pd.to_numeric(score_rows["feature_value"], errors="coerce")
    if scores.isna().any():
        raise ValueError("ml_forecast_score values must be numeric")
    if (scores < -1.0).any() or (scores > 1.0).any():
        raise ValueError("ml_forecast_score values must lie in [-1, 1]")


__all__ = ["REQUIRED_FEATURES", "validate_ml_snapshot"]
