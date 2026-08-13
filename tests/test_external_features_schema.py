"""Tests for external_features.schema."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from backtest.external_features.schema import (
    REQUIRED_COLUMNS,
    iter_snapshot_files,
    read_snapshot_csv,
    snapshot_path,
    validate_schema,
    write_snapshot_csv,
)


def _build_frame() -> pd.DataFrame:
    snapshot_ts = datetime(2026, 5, 12, 12, 0, 0, tzinfo=timezone.utc)
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "release_date": pd.Timestamp("2026-05-01"),
                "snapshot_ts": pd.Timestamp(snapshot_ts),
                "feature_name": "score",
                "feature_value": 0.5,
                "source": "MockSource",
                "dataset": "test_ds",
                "confidence": 0.9,
            },
            {
                "ticker": "BBB",
                "release_date": pd.Timestamp("2026-05-02"),
                "snapshot_ts": pd.Timestamp(snapshot_ts),
                "feature_name": "score",
                "feature_value": -0.25,
                "source": "MockSource",
                "dataset": "test_ds",
                "confidence": 0.6,
            },
        ]
    )


def test_validate_schema_accepts_well_formed_frame():
    validate_schema(_build_frame())


def test_validate_schema_rejects_missing_required_column():
    df = _build_frame().drop(columns=["release_date"])
    with pytest.raises(ValueError, match="release_date"):
        validate_schema(df)


def test_validate_schema_rejects_non_numeric_feature_value():
    df = _build_frame()
    df["feature_value"] = ["a", "b"]
    with pytest.raises(ValueError, match="feature_value"):
        validate_schema(df)


def test_write_snapshot_csv_is_byte_stable(tmp_path: Path):
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    df = _build_frame()
    write_snapshot_csv(df, path_a)
    write_snapshot_csv(df.iloc[::-1].copy(), path_b)  # different row order
    assert path_a.read_bytes() == path_b.read_bytes()


def test_read_after_write_roundtrip(tmp_path: Path):
    path = tmp_path / "snap.csv"
    write_snapshot_csv(_build_frame(), path)
    parsed = read_snapshot_csv(path)
    for col in REQUIRED_COLUMNS:
        assert col in parsed.columns


def test_snapshot_path_uses_root_and_iso(tmp_path: Path):
    expected = tmp_path / "ds1" / "2026-05-01.csv"
    assert snapshot_path("ds1", date(2026, 5, 1), root=tmp_path) == expected


def test_iter_snapshot_files_returns_sorted(tmp_path: Path):
    (tmp_path / "ds1").mkdir()
    (tmp_path / "ds1" / "2026-05-01.csv").write_text("a")
    (tmp_path / "ds1" / "2026-05-02.csv").write_text("b")
    (tmp_path / "ds2").mkdir()
    (tmp_path / "ds2" / "2026-05-03.csv").write_text("c")
    all_paths = list(iter_snapshot_files(root=tmp_path))
    assert [p.name for p in all_paths] == [
        "2026-05-01.csv",
        "2026-05-02.csv",
        "2026-05-03.csv",
    ]
    ds1_only = list(iter_snapshot_files("ds1", root=tmp_path))
    assert {p.parent.name for p in ds1_only} == {"ds1"}
