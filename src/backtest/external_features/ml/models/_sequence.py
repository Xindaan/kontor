"""Phase E3 — sequence tensor builder (T-0343, Codex R3.12).

Converts long-form feature frames (Phase D `(as_of, ticker)` rows)
into 3D tensors `(N, seq_len, n_features)` for LSTM/Transformer.

**PIT contract** (Codex R3.12): per `(as_of, ticker)` the tensor
contains ONLY trading days UP TO AND INCLUDING `as_of`. Targets lie
AFTER `as_of` and are not part of the sequence tensor construction.

framework-free: uses only numpy + pandas.
"""

from __future__ import annotations

from datetime import date as _date
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def build_sequence_tensor(
    feature_frame: pd.DataFrame,
    seq_len: int = 63,
    *,
    feature_columns: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[Tuple[pd.Timestamp, str]]]:
    """Produces a ``(N, seq_len, n_features)`` tensor + meta index.

    Args:
        feature_frame: long-form DataFrame with at least the columns
            ``as_of`` and ``ticker``, plus one column per feature
            (see Phase D ``FeatureMatrixBuilder``).
        seq_len: desired sequence length (number of trading days).
            Default 63 (Phase D convention for a short window).
        feature_columns: selection of feature columns to convert.
            Default = all columns except ``as_of`` and ``ticker``.

    Returns:
        - ``tensor``: shape ``(N, seq_len, n_features)`` with float values.
            ``N`` = number of `(as_of, ticker)` rows with at least
            ``seq_len`` historical observations.
        - ``meta``: list of ``(as_of_timestamp, ticker)`` tuples,
            parallel to the batch.

    PIT contract: for row ``(as_of=T, ticker=X)`` the sub-window
    contains the last ``seq_len`` entries up to and including ``T``
    from ``feature_frame`` for the same ``ticker``. NO forward look.
    """

    if feature_frame is None or feature_frame.empty:
        return np.zeros((0, seq_len, 0), dtype=float), []

    if "as_of" not in feature_frame.columns or "ticker" not in feature_frame.columns:
        raise ValueError("feature_frame muss 'as_of' und 'ticker' enthalten")

    if feature_columns is None:
        feature_columns = [
            c for c in feature_frame.columns if c not in {"as_of", "ticker"}
        ]
    feature_columns = list(feature_columns)
    if not feature_columns:
        return np.zeros((0, seq_len, 0), dtype=float), []

    frame = feature_frame.copy()
    frame["as_of"] = pd.to_datetime(frame["as_of"])
    frame = frame.sort_values(["ticker", "as_of"]).reset_index(drop=True)

    tensors: List[np.ndarray] = []
    meta: List[Tuple[pd.Timestamp, str]] = []

    for ticker, group in frame.groupby("ticker", sort=False):
        values = group[feature_columns].to_numpy(dtype=float, copy=True)
        as_of_series = group["as_of"].to_numpy()
        # For each position i in the group we take the sub-window
        # `values[i - seq_len + 1 : i + 1]`. Only rows with i >= seq_len-1
        # qualify (Codex R3.12: up to and including as_of, no future).
        n = len(values)
        for i in range(seq_len - 1, n):
            window = values[i - seq_len + 1 : i + 1, :]
            if window.shape[0] != seq_len:
                continue
            tensors.append(window)
            meta.append((pd.Timestamp(as_of_series[i]), str(ticker)))

    if not tensors:
        return (
            np.zeros((0, seq_len, len(feature_columns)), dtype=float),
            [],
        )
    return np.stack(tensors, axis=0), meta


__all__ = ["build_sequence_tensor"]
