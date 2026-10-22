"""Phase B propagation: MultiDatasetFeaturesProvider through Backtester
(T-0067) and PIT-strictness across signals/strategy (T-0069).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from backtest.backtester import BacktestConfig, Backtester
from backtest.data import PriceData
from backtest.external_features.loader import (
    ExternalFeatureSnapshot,
    ExternalFeaturesLoader,
)
from backtest.external_features.multi_loader import MultiDatasetFeaturesProvider
from backtest.external_features.schema import write_snapshot_csv
from backtest.strategy import Allocation, Strategy


class _SeenProviderStrategy(Strategy):
    name = "SeenProvider"
    assets = ["SPY"]
    rebalance_frequency = "monthly"
    captured: list = []

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        self.__class__.captured.append(self._external_features_provider)
        return Allocation({"SPY": 1.0})


@pytest.fixture(autouse=True)
def _reset_capture():
    _SeenProviderStrategy.captured = []


@pytest.fixture
def sample_data() -> PriceData:
    dates = pd.date_range("2020-01-01", periods=36, freq="ME")
    prices = pd.DataFrame({"SPY": np.linspace(100, 180, 36)}, index=dates)
    return PriceData(prices=prices, currency={"SPY": "USD"}, fx_rates=None)


def _build_multi_provider(tmp_path: Path) -> MultiDatasetFeaturesProvider:
    loaders = {}
    for ds, value in (("ds_a", 0.3), ("ds_b", -0.1)):
        df = pd.DataFrame(
            [
                {
                    "ticker": "SPY",
                    "release_date": pd.Timestamp("2020-01-31"),
                    "snapshot_ts": pd.Timestamp("2020-01-31T12:00:00Z"),
                    "feature_name": "analyst_score",
                    "feature_value": value,
                    "source": "Test",
                    "dataset": ds,
                }
            ]
        )
        target_dir = tmp_path / ds
        target_dir.mkdir(parents=True, exist_ok=True)
        write_snapshot_csv(df, target_dir / "2020-01-31.csv")
        loader = ExternalFeaturesLoader(root=tmp_path, dataset=ds, provenance_mode="off")
        loader.load()
        loaders[ds] = loader
    return MultiDatasetFeaturesProvider(loaders, default_dataset="ds_a")


def test_multi_provider_reaches_strategy_in_backtester(tmp_path: Path, sample_data):
    provider = _build_multi_provider(tmp_path)
    strategy = _SeenProviderStrategy()
    config = BacktestConfig(benchmark=None, external_features_loader=provider)
    Backtester(strategy, sample_data, config).run()
    # Provider seen at least once during run.
    assert any(c is provider for c in _SeenProviderStrategy.captured)
    # After run: slot restored to class default (None).
    assert strategy._external_features_provider is None


def test_multi_provider_explicit_dataset_snapshot(tmp_path: Path):
    provider = _build_multi_provider(tmp_path)
    snap_a = provider.snapshot_dataset("ds_a", as_of=date(2020, 1, 31))
    snap_b = provider.snapshot_dataset("ds_b", as_of=date(2020, 1, 31))
    assert float(snap_a.data["feature_value"].iloc[0]) == 0.3
    assert float(snap_b.data["feature_value"].iloc[0]) == -0.1


# ---------------------------------------------------------------------------
# T-0069: PIT strictness — release_date > as_of must not reach strategy /
# signals / regime measurements.
# ---------------------------------------------------------------------------


class _PITSnapshotProvider:
    """Synthetic provider that always returns at least one row with
    release_date strictly AFTER as_of plus one valid row at as_of."""

    def snapshot(self, as_of, tickers=None):
        # We deliberately include a row in the future — the loader's
        # PIT-cutoff is applied upstream by ExternalFeaturesLoader; but
        # this in-memory provider does NOT do filtering. The contract is
        # that strategy code reads only rows the loader returns; the
        # PIT-test below uses the loader path with real CSV writes.
        rows = [
            {
                "ticker": "AAA",
                "release_date": pd.Timestamp(as_of),
                "snapshot_ts": pd.Timestamp(as_of),
                "feature_name": "analyst_score",
                "feature_value": 0.5,
                "source": "Test",
                "dataset": "pit",
            },
            {
                "ticker": "AAA",
                "release_date": pd.Timestamp(as_of) + pd.Timedelta(days=10),
                "snapshot_ts": pd.Timestamp(as_of) + pd.Timedelta(days=10),
                "feature_name": "analyst_score",
                "feature_value": 0.99,  # future-only row
                "source": "Test",
                "dataset": "pit",
            },
        ]
        return ExternalFeatureSnapshot(
            as_of=as_of, dataset="pit", data=pd.DataFrame(rows)
        )


def test_pit_loader_filters_future_release_date(tmp_path: Path):
    """The PIT cutoff lives in ExternalFeaturesLoader.snapshot — verify
    that an artificially advanced release_date is filtered out."""
    df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "release_date": pd.Timestamp("2026-04-30"),
                "snapshot_ts": pd.Timestamp("2026-04-30T12:00:00Z"),
                "feature_name": "analyst_score",
                "feature_value": 0.2,
                "source": "Test",
                "dataset": "pit",
            },
            {
                "ticker": "AAA",
                "release_date": pd.Timestamp("2026-05-15"),  # FUTURE vs as_of
                "snapshot_ts": pd.Timestamp("2026-05-15T12:00:00Z"),
                "feature_name": "analyst_score",
                "feature_value": 0.99,
                "source": "Test",
                "dataset": "pit",
            },
        ]
    )
    target_dir = tmp_path / "pit"
    target_dir.mkdir(parents=True, exist_ok=True)
    write_snapshot_csv(df, target_dir / "2026-04-30.csv")
    loader = ExternalFeaturesLoader(root=tmp_path, dataset="pit", provenance_mode="off")
    loader.load()
    snap = loader.snapshot(as_of=date(2026, 5, 1), tickers=["AAA"])
    values = snap.data["feature_value"].tolist()
    assert 0.99 not in values  # future row stripped
    assert 0.2 in values
