"""Phase D forward-return target helpers (T-0202).

Produces a long-form DataFrame with one row per ``(as_of, ticker)`` plus
forward-return columns *and* their concrete calendar end-dates
``label_end_h``. The end-dates are critical for the leakage-safe
:class:`PurgedDateSplit` (Codex D16/D27) — they pin down when a target
is fully observable rather than relying on a pauschal ``+253d`` rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

DEFAULT_HORIZONS: tuple[int, ...] = (21, 63, 252)


@dataclass(frozen=True)
class TargetSpec:
    """Resolved target columns for a horizon list."""

    horizons: tuple[int, ...]

    @property
    def horizon_return_columns(self) -> List[str]:
        return [f"horizon_{h}d" for h in self.horizons]

    @property
    def horizon_label_end_columns(self) -> List[str]:
        return [f"label_end_{h}d" for h in self.horizons]


def compute_forward_returns(
    prices: pd.DataFrame,
    horizons_days: Sequence[int] = DEFAULT_HORIZONS,
    as_of_range: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build the forward-return panel.

    Args:
        prices: DataFrame indexed by trading days, columns are tickers.
        horizons_days: list of forward windows expressed in trading
            days. ``21d`` means "shift index by 21 business rows".
        as_of_range: optional iterable of ``as_of`` timestamps to
            evaluate. Defaults to every index in ``prices``. Each
            element is snapped down to the most recent trading day in
            the index.

    Returns:
        DataFrame with columns ``[as_of, ticker, horizon_{h}d,
        label_end_{h}d]`` for every requested horizon. NaNs appear for
        rows whose forward window extends past ``prices.index``.
    """

    if prices is None or prices.empty:
        return pd.DataFrame(columns=["as_of", "ticker"])
    if not horizons_days:
        raise ValueError("horizons_days must not be empty")

    spec = TargetSpec(horizons=tuple(int(h) for h in horizons_days))
    index = pd.DatetimeIndex(prices.index)
    if as_of_range is None:
        as_of_index = index
    else:
        wanted = pd.DatetimeIndex([pd.Timestamp(t) for t in as_of_range])
        snapped = index.searchsorted(wanted, side="right") - 1
        valid = snapped >= 0
        as_of_index = index[snapped[valid]] if valid.any() else index[:0]

    if as_of_index.empty:
        return pd.DataFrame(
            columns=["as_of", "ticker", *spec.horizon_return_columns, *spec.horizon_label_end_columns]
        )

    pos_lookup = pd.Series(np.arange(len(index)), index=index)
    base_positions = pos_lookup.loc[as_of_index].to_numpy()

    rows: List[dict] = []
    tickers = list(prices.columns)
    for ticker in tickers:
        series = pd.to_numeric(prices[ticker], errors="coerce")
        values = series.to_numpy()
        for pos, as_of_ts in zip(base_positions, as_of_index):
            base_price = values[pos] if 0 <= pos < len(values) else np.nan
            if not np.isfinite(base_price) or base_price <= 0:
                continue
            row: dict = {"as_of": as_of_ts, "ticker": ticker}
            keep = True
            for h in spec.horizons:
                end_pos = pos + h
                if end_pos >= len(values):
                    row[f"horizon_{h}d"] = np.nan
                    row[f"label_end_{h}d"] = pd.NaT
                    continue
                end_price = values[end_pos]
                if not np.isfinite(end_price) or end_price <= 0:
                    row[f"horizon_{h}d"] = np.nan
                    row[f"label_end_{h}d"] = pd.NaT
                    continue
                row[f"horizon_{h}d"] = float(end_price / base_price - 1.0)
                row[f"label_end_{h}d"] = index[end_pos]
            if keep:
                rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["as_of", "ticker", *spec.horizon_return_columns, *spec.horizon_label_end_columns]
        )
    frame = pd.DataFrame(rows)
    # Stable column ordering.
    ordered_columns = ["as_of", "ticker"]
    for h in spec.horizons:
        ordered_columns.append(f"horizon_{h}d")
        ordered_columns.append(f"label_end_{h}d")
    return frame[ordered_columns].sort_values(["as_of", "ticker"]).reset_index(drop=True)


__all__ = ["DEFAULT_HORIZONS", "TargetSpec", "compute_forward_returns"]
