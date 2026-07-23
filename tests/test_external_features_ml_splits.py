"""Phase D: PurgedDateSplit leakage tests (T-0230)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.external_features.ml.splits import PurgedDateSplit


def _make_panel(num_days: int = 100, horizons=(21, 63, 252)) -> pd.DataFrame:
    """Synthetic (as_of, ticker, label_end_*) panel."""

    days = pd.bdate_range("2024-01-01", periods=num_days)
    rows = []
    for day in days:
        for ticker in ["AAA", "BBB"]:
            row = {"as_of": day, "ticker": ticker}
            for h in horizons:
                row[f"label_end_{h}d"] = day + pd.tseries.offsets.BusinessDay(h)
            rows.append(row)
    return pd.DataFrame(rows)


def test_split_disjoint_as_of_in_train_test():
    """Codex D1: same as_of must never appear in both train and test."""

    panel = _make_panel()
    label_cols = panel[[c for c in panel.columns if c.startswith("label_end_")]]
    splitter = PurgedDateSplit(n_splits=3)
    for train_idx, test_idx in splitter.split(panel["as_of"], label_cols):
        train_dates = set(panel.iloc[train_idx]["as_of"])
        test_dates = set(panel.iloc[test_idx]["as_of"])
        assert train_dates.isdisjoint(test_dates), (
            "PurgedDateSplit must not share as_of between train and test"
        )


def test_label_end_strictly_less_than_test_start():
    """Codex D27: train rows must satisfy label_end < test_start."""

    panel = _make_panel(num_days=120)
    label_cols = panel[[c for c in panel.columns if c.startswith("label_end_")]]
    splitter = PurgedDateSplit(n_splits=3, horizon_key=None)  # max across horizons
    for train_idx, test_idx in splitter.split(panel["as_of"], label_cols):
        test_start = panel.iloc[test_idx]["as_of"].min()
        train_labels = label_cols.iloc[train_idx].max(axis=1)
        assert (train_labels < test_start).all(), (
            f"Found train rows with label_end >= test_start={test_start}"
        )


def test_horizon_specific_purge_admits_more_rows_than_stage3():
    """Codex D26: 21d horizon must admit at least as many train rows as
    the Stage 3 max-horizon purge for the same fold."""

    panel = _make_panel(num_days=400)
    label_cols = panel[[c for c in panel.columns if c.startswith("label_end_")]]
    splitter_21 = PurgedDateSplit(n_splits=3, horizon_key="21d")
    splitter_all = PurgedDateSplit(n_splits=3, horizon_key=None)

    folds_21 = list(splitter_21.split(panel["as_of"], label_cols))
    folds_all = list(splitter_all.split(panel["as_of"], label_cols))
    # The looser 21d purge admits at least as many valid folds (and never
    # fewer rows) as the conservative max-horizon purge.
    assert len(folds_21) >= len(folds_all)
    # When both produce a fold for the same test window, Stage-1's train
    # set is at least as large as Stage-3's.
    if folds_21 and folds_all:
        # Compare the LAST fold of each — they target the same date range.
        train_21 = folds_21[-1][0]
        train_all = folds_all[-1][0]
        assert train_21.size >= train_all.size
