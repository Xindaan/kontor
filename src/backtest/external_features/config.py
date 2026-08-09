"""Serializable configuration for the external features pipeline.

The loader is deliberately NOT stored in RunConfig — RunConfig must remain
JSON-serializable so config_hash stays stable across runs. Instead, we keep
a small immutable configuration and build the loader on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

PROVENANCE_MODES = ("off", "warn", "strict")


@dataclass(frozen=True)
class ExternalFeaturesConfig:
    """Configuration for the external features pipeline.

    Set ``enabled=True`` plus a ``dataset`` (singular) for the Phase-A
    behaviour, or ``datasets`` (tuple) for Phase-B multi-dataset usage.
    Defaults leave external features inactive — backward compatible.

    The effective default-dataset (when callers go through the
    singular ``snapshot()`` of a MultiProvider) is ``dataset`` if set,
    otherwise the first element of ``datasets``.
    """

    enabled: bool = False
    dataset: Optional[str] = None
    datasets: tuple = ()
    root: str = "data/external_features"
    provenance_mode: str = "warn"
    registry_path: str = "data/manual/provenance.json"

    def __post_init__(self) -> None:
        if self.provenance_mode not in PROVENANCE_MODES:
            raise ValueError(
                "provenance_mode must be one of: " + ", ".join(PROVENANCE_MODES)
            )
        # Normalise datasets to a tuple of non-empty strings.
        if self.datasets and not isinstance(self.datasets, tuple):
            object.__setattr__(self, "datasets", tuple(self.datasets))
        if self.enabled and not self.dataset and not self.datasets:
            raise ValueError(
                "dataset or datasets must be set when external features are enabled"
            )

    @property
    def default_dataset(self) -> Optional[str]:
        if self.dataset:
            return self.dataset
        if self.datasets:
            return self.datasets[0]
        return None

    @property
    def effective_datasets(self) -> tuple:
        """Datasets that should be loaded.

        Singular ``dataset`` is included as the first element if set.
        ``datasets`` follows. Duplicates are removed while preserving
        order so the default-dataset stays in front.
        """
        seen: list[str] = []
        if self.dataset:
            seen.append(self.dataset)
        for ds in self.datasets:
            if ds and ds not in seen:
                seen.append(ds)
        return tuple(seen)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "dataset": self.dataset,
            "datasets": list(self.datasets),
            "root": str(self.root),
            "provenance_mode": self.provenance_mode,
            "registry_path": str(self.registry_path),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExternalFeaturesConfig":
        if payload is None:
            return cls()
        raw_datasets = payload.get("datasets") or ()
        if isinstance(raw_datasets, (list, tuple)):
            datasets = tuple(str(d) for d in raw_datasets if d)
        else:
            datasets = ()
        return cls(
            enabled=bool(payload.get("enabled", False)),
            dataset=payload.get("dataset") or None,
            datasets=datasets,
            root=str(payload.get("root") or "data/external_features"),
            provenance_mode=str(payload.get("provenance_mode") or "warn"),
            registry_path=str(payload.get("registry_path") or "data/manual/provenance.json"),
        )

    def with_overrides(self, **changes: Any) -> "ExternalFeaturesConfig":
        return replace(self, **changes)


def build_loader_from_config(cfg: ExternalFeaturesConfig):
    """Instantiate the loader (or multi-provider) if enabled.

    Returns:
        - ``None`` if disabled.
        - ``ExternalFeaturesLoader`` when only the singular ``dataset``
          is set (Phase-A backward compatible).
        - ``MultiDatasetFeaturesProvider`` when ``datasets`` is set
          (Phase-B multi-dataset). The default dataset of the provider
          is ``cfg.default_dataset``.
    """

    if cfg is None or not cfg.enabled:
        return None

    from backtest.external_features.loader import ExternalFeaturesLoader

    datasets = cfg.effective_datasets
    if not datasets:
        return None

    snapshots_root = Path(cfg.root) / "snapshots"

    if len(datasets) == 1 and not cfg.datasets:
        # Pure Phase-A singular path.
        loader = ExternalFeaturesLoader(
            root=snapshots_root,
            dataset=datasets[0],
            provenance_mode=cfg.provenance_mode,
            provenance_registry_path=cfg.registry_path,
        )
        loader.load()
        return loader

    from backtest.external_features.multi_loader import MultiDatasetFeaturesProvider

    loaders = {}
    for ds in datasets:
        loader = ExternalFeaturesLoader(
            root=snapshots_root,
            dataset=ds,
            provenance_mode=cfg.provenance_mode,
            provenance_registry_path=cfg.registry_path,
        )
        loader.load()
        loaders[ds] = loader
    return MultiDatasetFeaturesProvider(loaders, default_dataset=cfg.default_dataset)
