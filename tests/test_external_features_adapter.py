"""Tests for ExternalFeatureAdapter template-method and Mock-Adapter.

Verifies cache idempotency and provenance idempotency: a second pull of
the same snapshot must NOT trigger a new fetch_remote call AND must NOT
create a duplicate registry entry.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from backtest.external_features.adapters import get_adapter
from backtest.external_features.adapters.mock import MockAnalystAdapter
from backtest.provenance import ManualDataProvenanceRegistry


@pytest.fixture(autouse=True)
def _reset_mock_counter():
    MockAnalystAdapter.fetch_call_count = 0


def test_get_adapter_returns_mock():
    adapter = get_adapter("mock_analyst")
    assert isinstance(adapter, MockAnalystAdapter)


def test_get_adapter_unknown_raises():
    with pytest.raises(KeyError):
        get_adapter("does_not_exist")


def test_pull_snapshot_writes_csv_and_registers_provenance(tmp_path: Path):
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    adapter = MockAnalystAdapter()
    path = adapter.pull_snapshot(
        ["AAA", "BBB"], date(2026, 5, 1), registry=registry, root=tmp_path / "snap"
    )
    assert path.exists()
    entries = registry.list_entries(dataset="mock_analyst")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.dataset == "mock_analyst"
    assert entry.source == "MockAnalyst"
    assert entry.checksum_sha256 is not None


def test_pull_snapshot_is_idempotent_no_extra_fetch(tmp_path: Path):
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    adapter = MockAnalystAdapter()
    adapter.pull_snapshot(
        ["AAA", "BBB"], date(2026, 5, 1), registry=registry, root=tmp_path / "snap"
    )
    assert MockAnalystAdapter.fetch_call_count == 1

    adapter.pull_snapshot(
        ["AAA", "BBB"], date(2026, 5, 1), registry=registry, root=tmp_path / "snap"
    )
    # Cache hit: no new fetch
    assert MockAnalystAdapter.fetch_call_count == 1

    # Registry: still exactly one entry for that snapshot
    entries = registry.list_entries(dataset="mock_analyst")
    assert len(entries) == 1


def test_pull_snapshot_force_refetches_but_keeps_single_entry(tmp_path: Path):
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    adapter = MockAnalystAdapter()
    adapter.pull_snapshot(
        ["AAA"], date(2026, 5, 1), registry=registry, root=tmp_path / "snap"
    )
    adapter.pull_snapshot(
        ["AAA"], date(2026, 5, 1), registry=registry, root=tmp_path / "snap", force=True
    )
    assert MockAnalystAdapter.fetch_call_count == 2
    entries = registry.list_entries(dataset="mock_analyst")
    # Force re-fetch generates the same CSV bytes (deterministic mock),
    # so the dedup hash matches and we still have exactly one entry.
    assert len(entries) == 1


def test_snapshot_cache_path_under_root(tmp_path: Path):
    adapter = MockAnalystAdapter()
    p = adapter.snapshot_cache_path(date(2026, 5, 1), root=tmp_path / "snap")
    assert p == tmp_path / "snap" / "mock_analyst" / "2026-05-01.csv"


def test_adapter_abc_cannot_instantiate():
    from backtest.external_features.adapters.base import ExternalFeatureAdapter

    with pytest.raises(TypeError):
        ExternalFeatureAdapter()  # type: ignore[abstract]
