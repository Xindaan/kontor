"""Tests for ExternalFeaturesLoader (PIT cutoff, dedup, provenance modes)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from backtest.external_features.loader import ExternalFeaturesLoader
from backtest.external_features.schema import write_snapshot_csv
from backtest.provenance import ManualDataProvenanceRegistry


def _row(ticker: str, release: str, feature: str, value: float, snapshot: str = "2026-05-12T12:00:00Z", source: str = "MockSource", dataset: str = "test_ds"):
    return {
        "ticker": ticker,
        "release_date": pd.Timestamp(release),
        "snapshot_ts": pd.Timestamp(snapshot),
        "feature_name": feature,
        "feature_value": value,
        "source": source,
        "dataset": dataset,
        "confidence": 1.0,
    }


def _write(tmp_path: Path, dataset: str, as_of: str, rows):
    df = pd.DataFrame(rows)
    target_dir = tmp_path / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    return write_snapshot_csv(df, target_dir / f"{as_of}.csv")


def test_load_requires_dataset(tmp_path: Path):
    loader = ExternalFeaturesLoader(root=tmp_path)
    with pytest.raises(ValueError):
        loader.load()


def test_snapshot_pit_cutoff_includes_equal_excludes_future(tmp_path: Path):
    _write(
        tmp_path,
        "test_ds",
        "2026-04-30",
        [_row("AAA", "2026-04-30", "score", 0.1)],
    )
    _write(
        tmp_path,
        "test_ds",
        "2026-05-01",
        [_row("AAA", "2026-05-01", "score", 0.2)],
    )
    _write(
        tmp_path,
        "test_ds",
        "2026-05-02",
        [_row("AAA", "2026-05-02", "score", 0.3)],
    )

    loader = ExternalFeaturesLoader(root=tmp_path, dataset="test_ds")
    loader.load()
    snap = loader.snapshot(as_of=date(2026, 5, 1))

    values = snap.data["feature_value"].tolist()
    assert 0.3 not in values
    assert 0.2 in values  # release_date == as_of must be visible


def test_snapshot_dedup_keeps_latest_per_group(tmp_path: Path):
    _write(
        tmp_path,
        "test_ds",
        "2026-05-01",
        [
            _row("AAA", "2026-04-30", "score", 0.1, snapshot="2026-04-30T10:00:00Z"),
            _row("AAA", "2026-04-30", "score", 0.15, snapshot="2026-04-30T11:00:00Z"),
            _row("AAA", "2026-05-01", "score", 0.2),
        ],
    )
    loader = ExternalFeaturesLoader(root=tmp_path, dataset="test_ds")
    loader.load()
    snap = loader.snapshot(as_of=date(2026, 5, 1), tickers=["AAA"])
    rows = snap.data
    score_rows = rows[rows["feature_name"] == "score"]
    assert len(score_rows) == 1
    assert score_rows.iloc[0]["feature_value"] == 0.2


def test_snapshot_filter_by_ticker(tmp_path: Path):
    _write(
        tmp_path,
        "test_ds",
        "2026-05-01",
        [
            _row("AAA", "2026-05-01", "score", 0.1),
            _row("BBB", "2026-05-01", "score", 0.2),
            _row("CCC", "2026-05-01", "score", 0.3),
        ],
    )
    loader = ExternalFeaturesLoader(root=tmp_path, dataset="test_ds")
    loader.load()
    snap = loader.snapshot(as_of=date(2026, 5, 1), tickers=["AAA", "BBB"])
    assert sorted(snap.data["ticker"].tolist()) == ["AAA", "BBB"]


def test_provenance_strict_raises_for_missing_entry(tmp_path: Path):
    _write(
        tmp_path,
        "test_ds",
        "2026-05-01",
        [_row("AAA", "2026-05-01", "score", 0.1)],
    )
    registry_path = tmp_path / "provenance.json"
    registry_path.write_text(json.dumps({"schema_version": 1, "entries": []}))

    loader = ExternalFeaturesLoader(
        root=tmp_path,
        dataset="test_ds",
        provenance_mode="strict",
        provenance_registry_path=registry_path,
    )
    with pytest.raises(ValueError):
        loader.load()


def test_provenance_warn_emits_warning_but_loads(tmp_path: Path):
    _write(
        tmp_path,
        "test_ds",
        "2026-05-01",
        [_row("AAA", "2026-05-01", "score", 0.1)],
    )
    registry_path = tmp_path / "provenance.json"
    registry_path.write_text(json.dumps({"schema_version": 1, "entries": []}))

    loader = ExternalFeaturesLoader(
        root=tmp_path,
        dataset="test_ds",
        provenance_mode="warn",
        provenance_registry_path=registry_path,
    )
    with pytest.warns(UserWarning):
        loader.load()
    assert loader.provenance_issues  # populated


def test_provenance_off_is_silent(tmp_path: Path, recwarn):
    _write(
        tmp_path,
        "test_ds",
        "2026-05-01",
        [_row("AAA", "2026-05-01", "score", 0.1)],
    )
    loader = ExternalFeaturesLoader(root=tmp_path, dataset="test_ds", provenance_mode="off")
    loader.load()
    feature_warnings = [
        w
        for w in recwarn
        if "provenance" in str(w.message).lower() or "external feature" in str(w.message).lower()
    ]
    assert not feature_warnings


def test_provenance_strict_passes_when_entry_matches(tmp_path: Path):
    path = _write(
        tmp_path,
        "test_ds",
        "2026-05-01",
        [_row("AAA", "2026-05-01", "score", 0.1)],
    )
    registry_path = tmp_path / "provenance.json"
    registry = ManualDataProvenanceRegistry(path=registry_path)
    registry.register_entry(
        file_path=path,
        dataset="test_ds",
        source="MockSource",
        quality_tag="proxy",
        as_of_date="2026-05-01",
        import_method="api_pull",
        license_tos_note="synthetic",
        entry_id="test-ds-2026-05-01",
    )

    loader = ExternalFeaturesLoader(
        root=tmp_path,
        dataset="test_ds",
        provenance_mode="strict",
        provenance_registry_path=registry_path,
    )
    loader.load()
    assert not loader.provenance_issues
