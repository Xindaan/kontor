"""Phase D leakage-safe split iterator (T-0203b).

``PurgedDateSplit`` is a date-grouped, label-aware cross validator. It
addresses two failure modes that scikit-learn's `TimeSeriesSplit`
exposes on panel data:

1. Same ``as_of`` showing up in train and test (group leakage).
2. Pauschal embargo on calendar days that does not match the actual
   trading-day-based forward window of the targets.

Splits use the label-end calendar dates produced by
:func:`compute_forward_returns`. A train row is admitted only if its
``label_end`` is strictly before the first ``as_of`` in the test fold —
guaranteeing that the row's target is fully realised before the fold
boundary.

Stages 1/2 use horizon-specific label-end columns; Stage 3 uses the
``max`` over all horizons because its target depends on all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd


def _as_datetime_series(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values), errors="coerce")


@dataclass
class PurgedDateSplit:
    """Date-grouped CV splitter with label-based purging.

    Args:
        n_splits: number of folds. Splits are taken on the sorted list
            of unique ``as_of`` dates.
        horizon_key: which label-end column drives the purge. ``None``
            (default) means "take the per-row maximum across the
            provided label-end columns" — appropriate for Stage 3.
            ``"21d"``/``"63d"``/``"252d"`` pick a single horizon for
            Stage 1/2.

    Usage:
        ``for train_idx, test_idx in splitter.split(as_of_dates,
        label_end_columns)``. ``label_end_columns`` is a DataFrame
        whose columns are ``label_end_{h}d``.
    """

    n_splits: int = 5
    horizon_key: Optional[str] = None

    def split(
        self,
        as_of_dates: Sequence,
        label_ends: pd.DataFrame,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        as_of_series = _as_datetime_series(as_of_dates)
        if len(as_of_series) != len(label_ends):
            raise ValueError("as_of_dates and label_ends must have equal length")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        unique_dates = (
            as_of_series.dropna().drop_duplicates().sort_values().reset_index(drop=True)
        )
        if len(unique_dates) < self.n_splits + 1:
            raise ValueError(
                f"need at least {self.n_splits + 1} unique as_of dates, "
                f"got {len(unique_dates)}"
            )

        purge_series = self._resolve_purge_series(label_ends)

        # Equal-sized contiguous test partitions in the date order.
        folds = np.array_split(unique_dates.to_numpy(), self.n_splits + 1)
        # First fold is the warm-up train segment — never used as test.
        for fold_idx in range(1, self.n_splits + 1):
            test_dates = pd.DatetimeIndex(folds[fold_idx])
            if test_dates.empty:
                continue
            test_start = test_dates.min()
            test_end = test_dates.max()
            test_mask = as_of_series.isin(test_dates).to_numpy()
            train_mask = (
                (as_of_series < test_start).to_numpy()
                & (purge_series < test_start).to_numpy()
            )
            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            yield train_idx, test_idx

    def _resolve_purge_series(self, label_ends: pd.DataFrame) -> pd.Series:
        if label_ends is None or label_ends.empty:
            raise ValueError("label_ends frame is empty")
        if self.horizon_key is None:
            cols = label_ends.columns
            converted = label_ends[cols].apply(pd.to_datetime, errors="coerce")
            return converted.max(axis=1)
        column = f"label_end_{self.horizon_key}"
        if column not in label_ends.columns:
            raise ValueError(f"label_ends is missing column {column!r}")
        return pd.to_datetime(label_ends[column], errors="coerce")


__all__ = ["PurgedDateSplit"]
