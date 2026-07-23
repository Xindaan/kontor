"""Multi-dataset feature provider.

Wraps multiple ExternalFeaturesLoader instances keyed by dataset_id.
``snapshot(as_of, tickers=None)`` delegates to the default dataset so the
provider is a drop-in replacement for the singular loader; explicit
multi-dataset queries go through ``snapshot_dataset(dataset_id, ...)``.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, Optional

from backtest.external_features.loader import (
    ExternalFeatureSnapshot,
    ExternalFeaturesLoader,
)


class MultiDatasetFeaturesProvider:
    """Container for multiple loaders, one per dataset_id.

    Backward-compat: a caller that holds a MultiProvider instance can
    call ``provider.snapshot(as_of, tickers)`` and will receive the
    default dataset's snapshot. The default is the first dataset added
    unless ``default_dataset`` is set explicitly.
    """

    def __init__(
        self,
        loaders: Dict[str, ExternalFeaturesLoader],
        default_dataset: Optional[str] = None,
    ) -> None:
        if not loaders:
            raise ValueError("MultiDatasetFeaturesProvider requires at least one loader")
        self._loaders: Dict[str, ExternalFeaturesLoader] = dict(loaders)
        if default_dataset is not None and default_dataset not in self._loaders:
            raise KeyError(
                f"default_dataset '{default_dataset}' not in loaders: "
                f"{sorted(self._loaders)}"
            )
        self.default_dataset: str = default_dataset or next(iter(self._loaders))

    def has(self, dataset_id: str) -> bool:
        return dataset_id in self._loaders

    def available_datasets(self) -> Iterable[str]:
        return sorted(self._loaders)

    def get(self, dataset_id: str) -> ExternalFeaturesLoader:
        if dataset_id not in self._loaders:
            raise KeyError(
                f"no loader for dataset '{dataset_id}'. "
                f"Available: {sorted(self._loaders)}"
            )
        return self._loaders[dataset_id]

    def snapshot(
        self,
        as_of: date,
        tickers: Optional[Iterable[str]] = None,
    ) -> ExternalFeatureSnapshot:
        """Default-dataset snapshot. Drop-in for singular loader."""

        return self._loaders[self.default_dataset].snapshot(as_of, tickers=tickers)

    def snapshot_dataset(
        self,
        dataset_id: str,
        as_of: date,
        tickers: Optional[Iterable[str]] = None,
    ) -> ExternalFeatureSnapshot:
        """Explicit multi-dataset snapshot."""

        return self.get(dataset_id).snapshot(as_of, tickers=tickers)
