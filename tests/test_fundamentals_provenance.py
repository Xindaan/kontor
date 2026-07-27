import pandas as pd
import pytest

from backtest.fundamentals import FundamentalsLoader
from backtest.provenance import ManualDataProvenanceRegistry


def _write_fundamentals_csv(path):
    frame = pd.DataFrame(
        {
            "date": ["2020-03-31", "2020-06-30"],
            "release_date": ["2020-04-30", "2020-07-31"],
            "pe": [10.0, 20.0],
            "ev_ebitda": [8.0, 9.0],
            "pb": [1.2, 1.4],
            "roe": [0.12, 0.14],
            "gross_margin": [0.40, 0.42],
            "debt_to_equity": [0.8, 0.7],
        }
    )
    frame.to_csv(path, index=False)


def test_fundamentals_loader_strict_requires_provenance_entry(tmp_path):
    root = tmp_path / "fundamentals"
    root.mkdir(parents=True, exist_ok=True)
    _write_fundamentals_csv(root / "AAA.csv")

    loader = FundamentalsLoader(
        root=root,
        provenance_mode="strict",
        provenance_registry_path=tmp_path / "provenance.json",
    )

    with pytest.raises(ValueError, match="Missing provenance entry"):
        loader.load()


def test_fundamentals_loader_warns_without_provenance_entry(tmp_path):
    root = tmp_path / "fundamentals"
    root.mkdir(parents=True, exist_ok=True)
    _write_fundamentals_csv(root / "AAA.csv")

    loader = FundamentalsLoader(
        root=root,
        provenance_mode="warn",
        provenance_registry_path=tmp_path / "provenance.json",
    )

    with pytest.warns(UserWarning, match="Missing provenance entry"):
        loader.load()

    snap = loader.snapshot(as_of=pd.Timestamp("2020-08-01").date(), tickers=["AAA"])
    assert snap.data.loc["AAA", "pe"] == 20.0


def test_fundamentals_loader_strict_accepts_matching_registry_entry(tmp_path):
    root = tmp_path / "fundamentals"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "AAA.csv"
    _write_fundamentals_csv(csv_path)

    registry = ManualDataProvenanceRegistry(path=tmp_path / "provenance.json")
    registry.register_entry(
        file_path=csv_path,
        dataset="fundamentals_sp500",
        source="SeekingAlpha",
        quality_tag="manual",
        as_of_date="2026-02-06",
        import_method="manual_csv_export",
        license_tos_note="Manual export from personal subscription.",
    )

    loader = FundamentalsLoader(
        root=root,
        provenance_mode="strict",
        provenance_registry_path=tmp_path / "provenance.json",
        provenance_dataset="fundamentals_sp500",
        provenance_source="SeekingAlpha",
    )
    loader.load()

    snap = loader.snapshot(as_of=pd.Timestamp("2020-08-01").date(), tickers=["AAA"])
    assert snap.data.loc["AAA", "pe"] == 20.0
    assert loader.provenance_issues == []


def test_fundamentals_loader_strict_detects_checksum_mismatch(tmp_path):
    root = tmp_path / "fundamentals"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "AAA.csv"
    _write_fundamentals_csv(csv_path)

    registry = ManualDataProvenanceRegistry(path=tmp_path / "provenance.json")
    registry.register_entry(
        file_path=csv_path,
        dataset="fundamentals_sp500",
        source="SeekingAlpha",
        quality_tag="manual",
        import_method="manual_csv_export",
        license_tos_note="Manual export from personal subscription.",
    )

    # Modify file after registration to force hash mismatch.
    csv_path.write_text(
        "date,release_date,pe,ev_ebitda,pb,roe,gross_margin,debt_to_equity\n"
        "2020-03-31,2020-04-30,11,8,1.2,0.12,0.40,0.8\n"
    )

    loader = FundamentalsLoader(
        root=root,
        provenance_mode="strict",
        provenance_registry_path=tmp_path / "provenance.json",
        provenance_dataset="fundamentals_sp500",
        provenance_source="SeekingAlpha",
    )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        loader.load()
