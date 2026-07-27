"""External features pipeline: PIT-safe news, analyst and ML signal layer.

Phase A ships the foundation only: a long-form CSV snapshot format, a
PIT loader, a serializable config, a template-method adapter base and a
deterministic mock adapter. Real upstream adapters arrive in later phases.
"""

from __future__ import annotations

from backtest.external_features.adapters import (
    available_datasets,
    datasets_allowing_empty_tickers,
    get_adapter,
    register_adapter,
)
from backtest.external_features.adapters.base import (
    ExternalFeatureAdapter,
    stable_entry_id,
)
from backtest.external_features.config import (
    ExternalFeaturesConfig,
    build_loader_from_config,
)
from backtest.external_features.loader import (
    ExternalFeatureSnapshot,
    ExternalFeaturesLoader,
)
from backtest.external_features.multi_loader import MultiDatasetFeaturesProvider
from backtest.external_features.schema import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SNAPSHOT_DIR,
    iter_snapshot_files,
    read_snapshot_csv,
    snapshot_path,
    validate_schema,
    write_snapshot_csv,
)

__all__ = [
    "ExternalFeatureAdapter",
    "ExternalFeatureSnapshot",
    "ExternalFeaturesConfig",
    "ExternalFeaturesLoader",
    "MultiDatasetFeaturesProvider",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "SNAPSHOT_DIR",
    "available_datasets",
    "build_loader_from_config",
    "datasets_allowing_empty_tickers",
    "get_adapter",
    "iter_snapshot_files",
    "read_snapshot_csv",
    "register_adapter",
    "snapshot_path",
    "stable_entry_id",
    "validate_schema",
    "write_snapshot_csv",
]
