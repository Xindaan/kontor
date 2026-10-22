"""Cache/manifest integrity.

Unit tests prove the checker fires on prepared poison fixtures; the
integration test keeps the REAL data/ cache clean. Finding at
introduction: 4 dividend entries with start > end (tickers whose first
dividend fell after the window end, e.g. PYPL) -- root cause fixed in
data.py, entries healed.
"""

import datetime
from pathlib import Path

from backtest.cache_integrity import DEFAULT_DATA_DIR, MANIFEST_NAME, check_cache, manifest_problems

TODAY = datetime.date(2026, 7, 14)


def _touch(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("Date,X\n2024-01-02,1.0\n")


def test_clean_manifest_has_no_problems(tmp_path):
    _touch(tmp_path, "SPY.csv")
    manifest = {"SPY.csv": {"start": "2020-01-01", "end": "2026-07-15"}}
    assert manifest_problems(manifest, tmp_path, TODAY) == []


def test_future_end_is_flagged(tmp_path):
    _touch(tmp_path, "SOXL.csv")
    manifest = {"SOXL.csv": {"start": "2020-01-01", "end": "2026-08-30"}}
    problems = manifest_problems(manifest, tmp_path, TODAY)
    assert len(problems) == 1
    assert "in the future" in problems[0]


def test_start_after_end_is_flagged(tmp_path):
    _touch(tmp_path, "PYPL_dividends.csv")
    manifest = {"PYPL_dividends.csv": {"start": "2025-11-19", "end": "2024-12-31"}}
    problems = manifest_problems(manifest, tmp_path, TODAY)
    assert len(problems) == 1
    assert "start" in problems[0]


def test_missing_file_is_flagged(tmp_path):
    manifest = {"WEG.csv": {"start": "2020-01-01", "end": "2024-12-31"}}
    problems = manifest_problems(manifest, tmp_path, TODAY)
    assert len(problems) == 1
    assert "without a cache file" in problems[0]


def test_unreadable_range_is_flagged(tmp_path):
    _touch(tmp_path, "X.csv")
    manifest = {"X.csv": {"start": None, "end": "kaputt"}}
    problems = manifest_problems(manifest, tmp_path, TODAY)
    assert len(problems) == 1
    assert "unreadable range" in problems[0]


def test_real_cache_is_clean():
    """Integration: the real data/ cache must not carry poison entries.
    If this test fires, that's a genuine finding -- don't fix the test,
    fix the cache (cf. incident 2026-06-14)."""
    import pytest

    if not (DEFAULT_DATA_DIR / MANIFEST_NAME).exists():
        pytest.skip("no local data/ cache (fresh checkout)")
    assert check_cache() == []
