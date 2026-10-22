"""Phase D bundle manifest tests (T-0211 + Codex D30)."""

from __future__ import annotations

import json
from pathlib import Path

from backtest.external_features.ml.manifest import (
    MLBundleManifest,
    feature_schema_hash,
    stable_model_entry_id,
)


def _make_manifest(**overrides):
    payload = dict(
        horizons=[21, 63, 252],
        feature_trained_through="2024-06-30",
        labels_known_through="2025-06-30",
        available_from="2025-07-01",
        lib_versions={"lightgbm": "4.5"},
        seed=42,
        training_run_id="run-001",
        config_hash="cfg-deadbeef",
        feature_schema_hash=feature_schema_hash(["ret_21d", "vol_63d"]),
        stage_paths={"stage1_21d": "stage1_21d.pkl"},
        zscore_stats_path="zscore_stats.json",
        imputer_state_path="imputer_state.json",
        model_family="lightgbm",
    )
    payload.update(overrides)
    return MLBundleManifest(**payload)


def test_bundle_hash_is_deterministic():
    m1 = _make_manifest().with_bundle_hash()
    m2 = _make_manifest().with_bundle_hash()
    assert m1.bundle_hash == m2.bundle_hash
    assert len(m1.bundle_hash) == 64  # sha256


def test_stable_model_entry_id_is_deterministic():
    bundle_hash = _make_manifest().with_bundle_hash().bundle_hash
    entry_a = stable_model_entry_id("run-001", "cfg-deadbeef", bundle_hash)
    entry_b = stable_model_entry_id("run-001", "cfg-deadbeef", bundle_hash)
    assert entry_a == entry_b
    assert entry_a.startswith("ml-bundle-")


def test_stable_model_entry_id_changes_with_config():
    bundle_hash = _make_manifest().with_bundle_hash().bundle_hash
    a = stable_model_entry_id("run-001", "cfg-one", bundle_hash)
    b = stable_model_entry_id("run-001", "cfg-two", bundle_hash)
    assert a != b


def test_manifest_roundtrip(tmp_path: Path):
    manifest = _make_manifest()
    path = tmp_path / "manifest.json"
    manifest.write(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["available_from"] == "2025-07-01"
    assert payload["bundle_hash"]  # populated on write
    re_read = MLBundleManifest.read(path)
    assert re_read.bundle_hash == payload["bundle_hash"]
