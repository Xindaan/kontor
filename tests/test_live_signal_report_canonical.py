"""Phase E2 — SignalReport canonical hash tests (T-0362, Codex R3.3+R4.1)."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from backtest.live.signal_report_io import (
    _normalize_for_hash,
    canonical_signal_report_hash,
)


def test_canonical_hash_deterministic():
    a = canonical_signal_report_hash({"x": 1.234567890, "y": [1, 2, 3]})
    b = canonical_signal_report_hash({"y": [1, 2, 3], "x": 1.234567890})
    assert a == b  # sort_keys makes order irrelevant


def test_canonical_hash_changes_on_float():
    """Codex R4.1: smallest float change -> different hash."""
    a = canonical_signal_report_hash({"x": 1.0})
    b = canonical_signal_report_hash({"x": 1.0000001})
    # With %.10g, 1.0 and 1.0000001 are slightly different,
    # but both representations print differently.
    assert a != b


def test_canonical_hash_handles_datetime():
    a = canonical_signal_report_hash(
        {"as_of": datetime(2024, 5, 13, 12, 0, 0)}
    )
    b = canonical_signal_report_hash(
        {"as_of": datetime(2024, 5, 13, 12, 0, 0)}
    )
    assert a == b


def test_canonical_hash_handles_date():
    a = canonical_signal_report_hash({"as_of": date(2024, 5, 13)})
    b = canonical_signal_report_hash({"as_of": date(2024, 5, 13)})
    assert a == b


def test_canonical_hash_accepts_to_dict_object():
    class _Mock:
        def to_dict(self):
            return {"x": 1.0}

    h = canonical_signal_report_hash(_Mock())
    assert h == canonical_signal_report_hash({"x": 1.0})


def test_normalize_handles_nested():
    payload = {
        "outer": {"inner": [1.5, {"deep": 2.0}]},
        "stamp": datetime(2024, 1, 1, tzinfo=None),
    }
    normalized = _normalize_for_hash(payload)
    assert normalized["outer"]["inner"][0] == "1.5"
    assert normalized["outer"]["inner"][1]["deep"] == "2"
