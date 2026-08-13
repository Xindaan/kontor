"""Forward-looking regime labels (ML Round 2, Phase 1).

PIT-strict classification of "which regime do the next H trading
days fall into" per ``(as_of, ticker)``:

- ``normal`` (0): low forward vol, no deeper drawdown
- ``fragile`` (1): elevated forward vol OR moderate drawdown
- ``stressed`` (2): high forward vol OR deep drawdown

Forward values are computed from the H trading days AFTER ``as_of``
(``label_end = as_of + H trading days``). Thresholds are asset-specific
**trailing** percentiles (only data up to and including ``as_of``) plus
absolute drawdown cutoffs. Schema and convention are parallel to
:mod:`backtest.external_features.ml.targets`, so that the existing
:class:`PurgedDateSplit` (Codex D16/D27) can purge as-of dates that
become available via ``label_end_{H}d``.

Output schema (long-form, one row per ``(as_of, ticker)``):

- ``as_of``, ``ticker``, ``label_end_{H}d``
- ``next_vol_{H}d``           — annualized forward realized vol
- ``next_max_dd_{H}d``        — forward max drawdown from the ``as_of`` price
- ``vol_threshold_fragile``   — trailing p90 of the rolling H-day vol
- ``vol_threshold_stressed``  — trailing p95
- ``regime_label``            — int (0/1/2)
- ``regime_name``             — str ("normal"/"fragile"/"stressed")

Design goal: a simple EWMA/GARCH estimator should be tested as a
baseline against a LightGBM classifier — if ML delivers no clear AUC
advantage over EWMA vol persistence, the classifier has no edge.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_LABEL_HORIZON: int = 21
DEFAULT_VOL_PERCENTILE_FRAGILE: float = 0.90
DEFAULT_VOL_PERCENTILE_STRESSED: float = 0.95
DEFAULT_VOL_PERCENTILE_WINDOW: int = 252
DEFAULT_DD_THRESHOLD_FRAGILE: float = -0.10
DEFAULT_DD_THRESHOLD_STRESSED: float = -0.20
MIN_TRAILING_HISTORY: int = 63

REGIME_NAMES = {0: "normal", 1: "fragile", 2: "stressed"}
REGIME_LABELS = {v: k for k, v in REGIME_NAMES.items()}


def _label_end_column(horizon: int) -> str:
    return f"label_end_{horizon}d"


def _next_vol_column(horizon: int) -> str:
    return f"next_vol_{horizon}d"


def _next_dd_column(horizon: int) -> str:
    return f"next_max_dd_{horizon}d"


def _output_columns(horizon: int) -> List[str]:
    return [
        "as_of",
        "ticker",
        _label_end_column(horizon),
        _next_vol_column(horizon),
        _next_dd_column(horizon),
        "vol_threshold_fragile",
        "vol_threshold_stressed",
        "regime_label",
        "regime_name",
    ]


def compute_regime_labels(
    prices: pd.DataFrame,
    label_horizon_days: int = DEFAULT_LABEL_HORIZON,
    vol_percentile_window: int = DEFAULT_VOL_PERCENTILE_WINDOW,
    vol_percentile_fragile: float = DEFAULT_VOL_PERCENTILE_FRAGILE,
    vol_percentile_stressed: float = DEFAULT_VOL_PERCENTILE_STRESSED,
    dd_threshold_fragile: float = DEFAULT_DD_THRESHOLD_FRAGILE,
    dd_threshold_stressed: float = DEFAULT_DD_THRESHOLD_STRESSED,
    as_of_range: Optional[Iterable[pd.Timestamp]] = None,
) -> pd.DataFrame:
    """PIT-strikte Forward-Regime-Labels (normal/fragile/stressed).

    Args:
        prices: DataFrame indexed by trading days, columns are tickers.
        label_horizon_days: Forward window length (H, in trading days).
        vol_percentile_window: Trailing window from which per-ticker
            vol percentile thresholds are computed (in trading days).
        vol_percentile_fragile / _stressed: Percentile cutoffs.
        dd_threshold_fragile / _stressed: Absolute drawdown cutoffs
            (negative numbers; e.g. -0.10 means -10%).
        as_of_range: Optional iterable of evaluation timestamps. Each
            is snapped to the most recent trading day. Defaults to
            every index in ``prices``.

    Returns:
        Long-form DataFrame. Rows whose forward window extends past
        ``prices.index`` or whose trailing history is shorter than
        :data:`MIN_TRAILING_HISTORY` are dropped.

    Raises:
        ValueError: if ``label_horizon_days < 1`` or the requested
            percentiles are not in ``(0, 1)``.
    """

    if prices is None or prices.empty:
        return pd.DataFrame(columns=_output_columns(int(label_horizon_days)))
    horizon = int(label_horizon_days)
    if horizon < 1:
        raise ValueError(f"label_horizon_days must be >=1, got {horizon}")
    for q in (vol_percentile_fragile, vol_percentile_stressed):
        if not (0.0 < q < 1.0):
            raise ValueError(f"percentile must be in (0,1), got {q}")
    if vol_percentile_stressed <= vol_percentile_fragile:
        raise ValueError(
            "vol_percentile_stressed must be > vol_percentile_fragile"
        )
    if dd_threshold_stressed > dd_threshold_fragile:
        raise ValueError(
            "dd_threshold_stressed must be <= dd_threshold_fragile "
            "(both negative, stressed more extreme)"
        )

    index = pd.DatetimeIndex(prices.index)
    if as_of_range is None:
        as_of_index = index
    else:
        wanted = pd.DatetimeIndex([pd.Timestamp(t) for t in as_of_range])
        snapped = index.searchsorted(wanted, side="right") - 1
        valid = snapped >= 0
        as_of_index = index[snapped[valid]] if valid.any() else index[:0]

    if as_of_index.empty:
        return pd.DataFrame(columns=_output_columns(horizon))

    rows: List[dict] = []
    label_end_col = _label_end_column(horizon)
    next_vol_col = _next_vol_column(horizon)
    next_dd_col = _next_dd_column(horizon)

    for ticker in prices.columns:
        series = pd.to_numeric(prices[ticker], errors="coerce")
        rets = series.pct_change()
        # H-day rolling realized vol (annualized). rolling_vol.iloc[i]
        # uses rets.iloc[i-H+1 : i+1] (H daily returns ending at i).
        rolling_vol = rets.rolling(window=horizon).std(ddof=0) * np.sqrt(252.0)

        for as_of_ts in as_of_index:
            pos_arr = index.searchsorted(as_of_ts, side="right") - 1
            pos = int(pos_arr)
            if pos < 0 or pos >= len(series):
                continue
            base_price = float(series.iloc[pos])
            if not np.isfinite(base_price) or base_price <= 0:
                continue
            end_pos = pos + horizon
            if end_pos >= len(series):
                continue  # Forward window extends past the available data.

            forward_prices = series.iloc[pos + 1 : end_pos + 1]
            if forward_prices.isna().any():
                continue
            forward_min = float(forward_prices.min())
            if not np.isfinite(forward_min) or forward_min <= 0:
                continue
            forward_max_dd = float(forward_min / base_price - 1.0)
            forward_vol = float(rolling_vol.iloc[end_pos])
            if not np.isfinite(forward_vol):
                continue

            # Trailing distribution ONLY from rolling_vol at positions <= pos.
            # This avoids forward leak: rolling_vol.iloc[i<=pos] only uses
            # rets.iloc[i-H+1 : i+1] with i+1 <= pos+1, i.e. data up to as_of.
            tail_lo = max(0, pos - vol_percentile_window + 1)
            trailing = rolling_vol.iloc[tail_lo : pos + 1].dropna()
            if len(trailing) < MIN_TRAILING_HISTORY:
                continue
            vol_thresh_fragile = float(trailing.quantile(vol_percentile_fragile))
            vol_thresh_stressed = float(trailing.quantile(vol_percentile_stressed))

            is_stressed = (
                forward_vol > vol_thresh_stressed
                or forward_max_dd < dd_threshold_stressed
            )
            is_fragile = (
                forward_vol > vol_thresh_fragile
                or forward_max_dd < dd_threshold_fragile
            )
            if is_stressed:
                label = 2
            elif is_fragile:
                label = 1
            else:
                label = 0

            rows.append(
                {
                    "as_of": as_of_ts,
                    "ticker": ticker,
                    label_end_col: index[end_pos],
                    next_vol_col: forward_vol,
                    next_dd_col: forward_max_dd,
                    "vol_threshold_fragile": vol_thresh_fragile,
                    "vol_threshold_stressed": vol_thresh_stressed,
                    "regime_label": int(label),
                    "regime_name": REGIME_NAMES[label],
                }
            )

    if not rows:
        return pd.DataFrame(columns=_output_columns(horizon))
    frame = pd.DataFrame(rows)
    return frame[_output_columns(horizon)].sort_values(
        ["as_of", "ticker"]
    ).reset_index(drop=True)


__all__ = [
    "compute_regime_labels",
    "REGIME_NAMES",
    "REGIME_LABELS",
    "DEFAULT_LABEL_HORIZON",
    "DEFAULT_VOL_PERCENTILE_FRAGILE",
    "DEFAULT_VOL_PERCENTILE_STRESSED",
    "DEFAULT_VOL_PERCENTILE_WINDOW",
    "DEFAULT_DD_THRESHOLD_FRAGILE",
    "DEFAULT_DD_THRESHOLD_STRESSED",
    "MIN_TRAILING_HISTORY",
]
