"""
Compatibility helpers for pandas version differences.

Currently handled:
- Pandas versions that do not support "ME" in pd.date_range
"""

from __future__ import annotations

from functools import wraps

import pandas as pd


def _normalize_freq(freq):
    """Map unsupported frequency aliases to backward-compatible ones."""
    if isinstance(freq, str) and freq.upper() == "ME":
        return "M"
    return freq


def enable_month_end_alias() -> bool:
    """
    Enable ME compatibility for pandas versions where pd.date_range(..., freq="ME") fails.

    Returns:
        True if a compatibility patch was installed, False otherwise.
    """
    # Modern pandas supports "ME" natively.
    try:
        pd.date_range("2020-01-01", periods=2, freq="ME")
        return False
    except Exception:
        pass

    if getattr(pd, "_backtest_me_alias_enabled", False):
        return True

    original_date_range = pd.date_range

    @wraps(original_date_range)
    def _date_range_with_me_alias(*args, **kwargs):
        if "freq" in kwargs:
            kwargs["freq"] = _normalize_freq(kwargs["freq"])
            return original_date_range(*args, **kwargs)

        # date_range(start, end, periods, freq, ...)
        if len(args) >= 4:
            mutable_args = list(args)
            mutable_args[3] = _normalize_freq(mutable_args[3])
            args = tuple(mutable_args)
        return original_date_range(*args, **kwargs)

    pd.date_range = _date_range_with_me_alias
    pd._backtest_me_alias_enabled = True
    return True
