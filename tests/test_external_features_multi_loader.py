"""Tests for MultiDatasetFeaturesProvider (Phase B T-0051) and the
multi-dataset path of ExternalFeaturesConfig.build_loader_from_config
(T-0052).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from backtest.external_features import (
    ExternalFeaturesConfig,
    ExternalFeaturesLoader,
    MultiDatasetFeaturesProvider,
    build_loader_from_config,
)
from backtest.external_features.schema import write_snapshot_csv


def _write(tmp_path: Path, dataset: str, as_of: str, ticker: str, value: float) -> Path:
    df = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "release_date": pd.Timestamp(as_of),
                "snapshot_ts": pd.Timestamp("2026-05-13T12:00:00Z"),
                "feature_name": "analyst_score",
                "feature_value": value,
                "source": "TestSource",
                "dataset": dataset,
            }
        ]
    )
    target_dir = tmp_path / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    return write_snapshot_csv(df, target_dir / f"{as_of}.csv")


def _loader(tmp_path: Path, dataset: str) -> ExternalFeaturesLoader:
    loader = ExternalFeaturesLoader(root=tmp_path, dataset=dataset, provenance_mode="off")
    loader.load()
    return loader


def test_multi_provider_requires_at_least_one_loader():
    with pytest.raises(ValueError):
        MultiDatasetFeaturesProvider({})


def test_multi_provider_default_dataset_first_inserted(tmp_path: Path):
    _write(tmp_path, "ds_a", "2026-05-01", "AAA", 0.5)
    _write(tmp_path, "ds_b", "2026-05-01", "BBB", -0.5)
    provider = MultiDatasetFeaturesProvider(
        {
            "ds_a": _loader(tmp_path, "ds_a"),
            "ds_b": _loader(tmp_path, "ds_b"),
        }
    )
    assert provider.default_dataset == "ds_a"


def test_multi_provider_explicit_default(tmp_path: Path):
    _write(tmp_path, "ds_a", "2026-05-01", "AAA", 0.5)
    _write(tmp_path, "ds_b", "2026-05-01", "BBB", -0.5)
    provider = MultiDatasetFeaturesProvider(
        {
            "ds_a": _loader(tmp_path, "ds_a"),
            "ds_b": _loader(tmp_path, "ds_b"),
        },
        default_dataset="ds_b",
    )
    assert provider.default_dataset == "ds_b"


def test_multi_provider_unknown_default_raises(tmp_path: Path):
    _write(tmp_path, "ds_a", "2026-05-01", "AAA", 0.5)
    with pytest.raises(KeyError):
        MultiDatasetFeaturesProvider(
            {"ds_a": _loader(tmp_path, "ds_a")},
            default_dataset="not_present",
        )


def test_multi_provider_snapshot_delegates_to_default(tmp_path: Path):
    _write(tmp_path, "ds_a", "2026-05-01", "AAA", 0.5)
    _write(tmp_path, "ds_b", "2026-05-01", "BBB", -0.5)
    provider = MultiDatasetFeaturesProvider(
        {
            "ds_a": _loader(tmp_path, "ds_a"),
            "ds_b": _loader(tmp_path, "ds_b"),
        },
        default_dataset="ds_a",
    )
    snap = provider.snapshot(as_of=date(2026, 5, 1))
    tickers = set(snap.data["ticker"].tolist())
    assert tickers == {"AAA"}  # default dataset only


def test_multi_provider_snapshot_dataset_explicit(tmp_path: Path):
    _write(tmp_path, "ds_a", "2026-05-01", "AAA", 0.5)
    _write(tmp_path, "ds_b", "2026-05-01", "BBB", -0.5)
    provider = MultiDatasetFeaturesProvider(
        {
            "ds_a": _loader(tmp_path, "ds_a"),
            "ds_b": _loader(tmp_path, "ds_b"),
        }
    )
    snap = provider.snapshot_dataset("ds_b", as_of=date(2026, 5, 1))
    assert set(snap.data["ticker"].tolist()) == {"BBB"}


def test_multi_provider_has_and_available(tmp_path: Path):
    _write(tmp_path, "ds_a", "2026-05-01", "AAA", 0.5)
    provider = MultiDatasetFeaturesProvider({"ds_a": _loader(tmp_path, "ds_a")})
    assert provider.has("ds_a")
    assert not provider.has("ds_b")
    assert list(provider.available_datasets()) == ["ds_a"]


def test_multi_provider_get_unknown_raises(tmp_path: Path):
    _write(tmp_path, "ds_a", "2026-05-01", "AAA", 0.5)
    provider = MultiDatasetFeaturesProvider({"ds_a": _loader(tmp_path, "ds_a")})
    with pytest.raises(KeyError):
        provider.get("does_not_exist")


def test_config_singular_still_returns_loader(tmp_path: Path):
    # Phase-A compatibility: only `dataset` set.
    _write(tmp_path, "ds_solo", "2026-05-01", "AAA", 0.1)
    cfg = ExternalFeaturesConfig(
        enabled=True,
        dataset="ds_solo",
        root=str(tmp_path.parent),
        provenance_mode="off",
    )
    # The loader builds from <root>/snapshots; tweak with a custom layout.
    cfg2 = ExternalFeaturesConfig(
        enabled=True,
        dataset="ds_solo",
        root=str(tmp_path.parent / "external_features_root"),
        provenance_mode="off",
    )
    # We need the snapshot to actually live under <root>/snapshots/<dataset>/
    ef_root = tmp_path.parent / "external_features_root"
    snaps = ef_root / "snapshots"
    (snaps / "ds_solo").mkdir(parents=True, exist_ok=True)
    (snaps / "ds_solo" / "2026-05-01.csv").write_bytes(
        (tmp_path / "ds_solo" / "2026-05-01.csv").read_bytes()
    )
    obj = build_loader_from_config(cfg2)
    assert isinstance(obj, ExternalFeaturesLoader)


def test_config_multi_returns_provider(tmp_path: Path):
    ef_root = tmp_path / "external_features"
    snaps = ef_root / "snapshots"
    for ds, val in (("ds_a", 0.2), ("ds_b", -0.4)):
        d = snaps / ds
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "release_date": pd.Timestamp("2026-05-01"),
                    "snapshot_ts": pd.Timestamp("2026-05-13T12:00:00Z"),
                    "feature_name": "analyst_score",
                    "feature_value": val,
                    "source": "TestSource",
                    "dataset": ds,
                }
            ]
        )
        write_snapshot_csv(df, d / "2026-05-01.csv")
    cfg = ExternalFeaturesConfig(
        enabled=True,
        datasets=("ds_a", "ds_b"),
        root=str(ef_root),
        provenance_mode="off",
    )
    obj = build_loader_from_config(cfg)
    assert isinstance(obj, MultiDatasetFeaturesProvider)
    assert obj.default_dataset == "ds_a"
    assert obj.has("ds_a") and obj.has("ds_b")


def test_config_disabled_returns_none():
    cfg = ExternalFeaturesConfig(enabled=False)
    assert build_loader_from_config(cfg) is None


def test_config_to_from_dict_roundtrip_with_datasets():
    cfg = ExternalFeaturesConfig(
        enabled=True,
        dataset="ds_main",
        datasets=("ds_main", "ds_extra"),
        provenance_mode="strict",
    )
    restored = ExternalFeaturesConfig.from_dict(cfg.to_dict())
    assert restored.dataset == "ds_main"
    assert restored.datasets == ("ds_main", "ds_extra")
    assert restored.provenance_mode == "strict"
    assert restored.default_dataset == "ds_main"
    assert restored.effective_datasets == ("ds_main", "ds_extra")


def test_config_effective_datasets_dedup_and_order():
    cfg = ExternalFeaturesConfig(
        enabled=True,
        dataset="ds_first",
        datasets=("ds_first", "ds_second"),
    )
    assert cfg.effective_datasets == ("ds_first", "ds_second")


def test_config_enabled_requires_dataset_or_datasets():
    with pytest.raises(ValueError):
        ExternalFeaturesConfig(enabled=True)
