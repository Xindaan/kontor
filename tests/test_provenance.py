from pathlib import Path

import pytest

from backtest.provenance import ManualDataProvenanceRegistry


def _create_manual_csv(path: Path, rows: str = "ticker,score\nAAPL,0.9\nMSFT,0.8\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows)
    return path


def test_register_and_get_manual_provenance_entry(tmp_path):
    registry_path = tmp_path / "provenance.json"
    manual_file = _create_manual_csv(tmp_path / "data/manual/seekingalpha/export.csv")
    registry = ManualDataProvenanceRegistry(path=registry_path)

    entry = registry.register_entry(
        file_path=manual_file,
        dataset="fundamentals_sp500",
        source="SeekingAlpha",
        quality_tag="manual",
        as_of_date="2026-02-06",
        import_method="manual_csv_export",
        license_tos_note="Manual export from personal subscription.",
        source_url="https://seekingalpha.com/",
    )

    loaded = registry.get_entry(entry.entry_id)
    assert loaded is not None
    assert loaded.dataset == "fundamentals_sp500"
    assert loaded.source == "SeekingAlpha"
    assert loaded.row_count == 2
    assert loaded.column_count == 2
    assert loaded.checksum_sha256


def test_list_filter_and_verify_detects_checksum_mismatch(tmp_path):
    registry_path = tmp_path / "provenance.json"
    manual_file = _create_manual_csv(tmp_path / "data/manual/export.csv")
    registry = ManualDataProvenanceRegistry(path=registry_path)

    entry = registry.register_entry(
        file_path=manual_file,
        dataset="sentiment_batch",
        source="SeekingAlpha",
        quality_tag="manual",
        import_method="manual_csv_export",
        license_tos_note="ToS compliant manual export.",
    )

    filtered = registry.list_entries(dataset="sentiment_batch", source="SeekingAlpha")
    assert len(filtered) == 1
    assert filtered[0].entry_id == entry.entry_id

    verified = registry.verify_entries()
    assert verified["total_entries"] == 1
    assert verified["ok_entries"] == 1
    assert verified["issue_count"] == 0

    manual_file.write_text("ticker,score\nAAPL,1.0\nMSFT,0.7\n")
    verified_after_change = registry.verify_entries()
    assert verified_after_change["total_entries"] == 1
    assert verified_after_change["ok_entries"] == 0
    assert verified_after_change["issue_count"] == 1
    assert verified_after_change["issues"][0]["entry_id"] == entry.entry_id
    assert verified_after_change["issues"][0]["status"] == "checksum_mismatch"


def test_register_validates_quality_tag_and_date(tmp_path):
    registry_path = tmp_path / "provenance.json"
    manual_file = _create_manual_csv(tmp_path / "data/manual/export.csv")
    registry = ManualDataProvenanceRegistry(path=registry_path)

    with pytest.raises(ValueError):
        registry.register_entry(
            file_path=manual_file,
            dataset="fundamentals",
            source="manual",
            quality_tag="invalid",
        )

    with pytest.raises(ValueError):
        registry.register_entry(
            file_path=manual_file,
            dataset="fundamentals",
            source="manual",
            quality_tag="manual",
            as_of_date="2026-13-99",
        )
