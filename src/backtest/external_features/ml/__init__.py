"""Phase D ML-Forecast package."""

from backtest.external_features.ml.config import (
    MLInferenceConfig,
    MLTrainingConfig,
)
from backtest.external_features.ml.features import (
    FeatureMatrixBuilder,
    FeatureMatrixState,
)
from backtest.external_features.ml.manifest import (
    MLBundleManifest,
    feature_schema_hash,
    stable_model_entry_id,
)
from backtest.external_features.ml.ml_schema import (
    REQUIRED_FEATURES,
    validate_ml_snapshot,
)
from backtest.external_features.ml.splits import PurgedDateSplit
from backtest.external_features.ml.targets import (
    DEFAULT_HORIZONS,
    TargetSpec,
    compute_forward_returns,
)

__all__ = [
    "DEFAULT_HORIZONS",
    "FeatureMatrixBuilder",
    "FeatureMatrixState",
    "MLBundleManifest",
    "MLInferenceConfig",
    "MLTrainingConfig",
    "PurgedDateSplit",
    "REQUIRED_FEATURES",
    "TargetSpec",
    "compute_forward_returns",
    "feature_schema_hash",
    "stable_model_entry_id",
    "validate_ml_snapshot",
]
