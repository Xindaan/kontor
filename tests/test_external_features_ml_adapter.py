"""Phase D ML adapter tests (T-0217/T-0218 + Codex D15/D20)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from backtest.external_features.adapters import _REGISTRY, get_adapter
from backtest.external_features.adapters.base import (
    ExternalFeatureFetchResult,
    SidecarBlob,
)
from backtest.external_features.adapters.synthetic_ml_forecast import (
    SyntheticMLForecastAdapter,
)
from backtest.external_features.ml.ml_schema import validate_ml_snapshot
from backtest.provenance import ManualDataProvenanceRegistry


def test_synthetic_adapter_registered():
    assert "synthetic_ml_forecast" in _REGISTRY
    assert "lightgbm_forecast" in _REGISTRY
    assert "xgboost_forecast" in _REGISTRY
    assert "ml_forecast_ensemble" in _REGISTRY


def test_synthetic_adapter_emits_required_features():
    adapter = get_adapter("synthetic_ml_forecast")
    result = adapter.fetch_remote(["AAPL", "MSFT"], date(2026, 5, 13))
    assert isinstance(result, ExternalFeatureFetchResult)
    validate_ml_snapshot(result.frame)


def test_synthetic_adapter_score_is_deterministic():
    adapter = get_adapter("synthetic_ml_forecast")
    a = adapter.fetch_remote(["AAPL"], date(2026, 5, 13))
    b = adapter.fetch_remote(["AAPL"], date(2026, 5, 13))
    score_a = (
        a.frame.loc[a.frame.feature_name == "ml_forecast_score", "feature_value"]
        .iloc[0]
    )
    score_b = (
        b.frame.loc[b.frame.feature_name == "ml_forecast_score", "feature_value"]
        .iloc[0]
    )
    assert score_a == score_b


def test_synthetic_adapter_writes_sidecar():
    adapter = get_adapter("synthetic_ml_forecast")
    result = adapter.fetch_remote(["AAPL"], date(2026, 5, 13))
    assert len(result.sidecars) == 1
    sidecar = result.sidecars[0]
    assert sidecar.kind == "ml_attribution_ndjson"
    assert sidecar.rows, "synthetic sidecar must carry deterministic rows"
    sample = sidecar.rows[0]
    assert sample["engine_code"] == "synthetic_ml@1.0"
    assert "contributions_unavailable_reason" in sample


def test_with_options_does_not_mutate_singleton():
    """Codex D20: with_options must return a fresh instance."""

    base = get_adapter("lightgbm_forecast")
    other = base.with_options(bundle_root=Path("/tmp/other"))
    assert base is not other
    # Registry singleton stays unchanged.
    again = get_adapter("lightgbm_forecast")
    assert again is base
    # Underlying option must differ.
    assert other.options.bundle_root == Path("/tmp/other")
    assert base.options.bundle_root != Path("/tmp/other")


def test_pull_snapshot_idempotent(tmp_path: Path):
    adapter = SyntheticMLForecastAdapter()
    registry = ManualDataProvenanceRegistry(path=str(tmp_path / "prov.json"))
    root = tmp_path / "snapshots"
    path_a = adapter.pull_snapshot(
        ["AAPL", "MSFT"], date(2026, 5, 13), registry=registry, root=root
    )
    path_b = adapter.pull_snapshot(
        ["AAPL", "MSFT"], date(2026, 5, 13), registry=registry, root=root
    )
    assert path_a == path_b
    entries = registry.list_entries(dataset="synthetic_ml_forecast")
    assert len(entries) == 1, "duplicate registry entry on second pull"
    # Verify sidecar hash anchor in CSV.
    csv = pd.read_csv(path_a)
    raw_hashes = csv["raw_payload_hash"].dropna().unique().tolist()
    assert len(raw_hashes) == 1
