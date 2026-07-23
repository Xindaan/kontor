"""Tests for SyntheticAnalystPITAdapter (Phase B T-0058)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from backtest.external_features.adapters.synthetic_analyst_pit import (
    DATASET_ID,
    SyntheticAnalystPITAdapter,
)
from backtest.external_features.analyst_schema import validate_analyst_snapshot
from backtest.provenance import ManualDataProvenanceRegistry

pytestmark = pytest.mark.no_network


def test_empty_tickers_raises():
    adapter = SyntheticAnalystPITAdapter()
    with pytest.raises(ValueError, match="explicit tickers"):
        adapter.fetch_remote([], date(2015, 5, 1))


def test_deterministic_output_without_price_provider():
    adapter = SyntheticAnalystPITAdapter()
    out_a = adapter.fetch_remote(["AAA", "BBB"], date(2015, 5, 1))
    out_b = adapter.fetch_remote(["AAA", "BBB"], date(2015, 5, 1))
    # Same call -> same feature_value bytes.
    pd.testing.assert_series_equal(
        out_a["feature_value"].reset_index(drop=True),
        out_b["feature_value"].reset_index(drop=True),
    )


def test_different_as_of_changes_score():
    adapter = SyntheticAnalystPITAdapter()
    a = adapter.fetch_remote(["AAA"], date(2015, 5, 1))
    b = adapter.fetch_remote(["AAA"], date(2015, 5, 2))
    assert not a.equals(b)


def test_score_in_range_and_validates_against_schema():
    adapter = SyntheticAnalystPITAdapter()
    out = adapter.fetch_remote(["AAA", "BBB", "CCC"], date(2015, 5, 1))
    assert ((out["feature_value"] >= -1.0) & (out["feature_value"] <= 1.0)).all()
    validate_analyst_snapshot(out)


def test_price_provider_used_when_available():
    def provider(ticker, as_of):
        # Strong positive returns -> score > 0.
        idx = pd.bdate_range(end=pd.Timestamp(as_of), periods=63)
        return pd.Series([0.005] * 63, index=idx)

    adapter = SyntheticAnalystPITAdapter(price_provider=provider)
    out = adapter.fetch_remote(["AAA"], date(2015, 5, 1))
    score = float(out.loc[out["feature_name"] == "analyst_score", "feature_value"].iloc[0])
    assert score > 0
    confidence = float(out.loc[out["feature_name"] == "analyst_score", "confidence"].iloc[0])
    assert confidence == 1.0  # real prices -> full confidence


def test_pull_snapshot_writes_and_registers(tmp_path: Path):
    adapter = SyntheticAnalystPITAdapter()
    registry = ManualDataProvenanceRegistry(path=tmp_path / "prov.json")
    path = adapter.pull_snapshot(
        ["AAA"], date(2015, 5, 1), registry=registry, root=tmp_path / "snap"
    )
    assert path.exists()
    entries = registry.list_entries(dataset=DATASET_ID)
    assert len(entries) == 1
    # Idempotency: second pull should not add another entry.
    adapter.pull_snapshot(["AAA"], date(2015, 5, 1), registry=registry, root=tmp_path / "snap")
    assert len(registry.list_entries(dataset=DATASET_ID)) == 1
