"""Walk-forward training for the regime classifier (ML Round 2, Phase 1).

Trains one LightGBM classifier per walk-forward outer window
(3 classes: normal/fragile/stressed). Features come from
:class:`FeatureMatrixBuilder` (5 standard price features), labels from
:func:`compute_regime_labels`. Bundle layout per window::

  <output_dir>/<holdout_start>/lightgbm/
      manifest.json          — regime-specific BundleManifest (see below)
      classifier.pkl         — pickled LGBMClassifier
      imputer_state.json     — FeatureMatrixState.to_dict()
      label_distribution.json — class shares in training (diagnostics)

Manifest contains:

- ``available_from``: ISO date of the holdout start (PIT lookup anchor).
- ``feature_trained_through``: last training ``as_of``.
- ``labels_known_through``: latest ``label_end_{H}d`` in training.
- ``bundle_hash``: SHA256 over manifest + classifier.pkl + state.
- ``framework``: ``"lightgbm_classifier"``.
- ``label_horizon_days``, ``feature_columns``, ``classes``,
  ``class_names``, ``label_distribution``.

Reuse: `_build_outer_windows` from :mod:`training` for identical
walk-forward logic as the forecast bundles.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backtest.external_features.ml.features import (
    FeatureMatrixBuilder,
    FeatureMatrixState,
)
from backtest.external_features.ml.regime_labels import (
    DEFAULT_LABEL_HORIZON,
    REGIME_NAMES,
    compute_regime_labels,
)
from backtest.external_features.ml.training import (
    WalkForwardWindow,
    _build_outer_windows,
)


@dataclass(frozen=True)
class RegimeClassifierConfig:
    """Configuration for walk-forward training of a regime classifier."""

    tickers: Tuple[str, ...]
    label_horizon_days: int = DEFAULT_LABEL_HORIZON
    outer_train_years: float = 4.0
    outer_holdout_months: int = 6
    seed: int = 42
    # LightGBM hyperparameters (conservative defaults).
    num_leaves: int = 31
    n_estimators: int = 200
    learning_rate: float = 0.05
    min_child_samples: int = 20


@dataclass
class RegimeTrainingResult:
    manifest_paths: List[Path] = field(default_factory=list)
    windows: List[WalkForwardWindow] = field(default_factory=list)


def _filter_train_rows(labels: pd.DataFrame, holdout_start: pd.Timestamp) -> pd.DataFrame:
    """Keeps only rows whose label is fully observable before ``holdout_start``."""
    if labels.empty:
        return labels
    label_end_col = next(c for c in labels.columns if c.startswith("label_end_"))
    return labels.loc[labels[label_end_col] < holdout_start].copy()


def _bundle_hash(manifest_payload: dict, classifier_bytes: bytes, state_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(manifest_payload, sort_keys=True).encode("utf-8"))
    h.update(classifier_bytes)
    h.update(state_bytes)
    return h.hexdigest()


def _lightgbm_lib_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    try:
        import lightgbm  # type: ignore

        versions["lightgbm"] = getattr(lightgbm, "__version__", "unknown")
    except Exception:
        versions["lightgbm"] = "unknown"
    versions["numpy"] = np.__version__
    versions["pandas"] = pd.__version__
    return versions


def _train_one_bundle(
    *,
    window: WalkForwardWindow,
    feature_frame: pd.DataFrame,
    feature_state: FeatureMatrixState,
    label_frame: pd.DataFrame,
    config: RegimeClassifierConfig,
    output_dir: Path,
) -> Path:
    """Trains + persists one bundle for ONE holdout window."""

    from lightgbm import LGBMClassifier

    bundle_dir = output_dir / window.holdout_start.isoformat() / "lightgbm"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    merged = feature_frame.merge(
        label_frame[["as_of", "ticker", "regime_label", *_label_end_cols(label_frame)]],
        on=["as_of", "ticker"],
        how="inner",
    )
    if merged.empty:
        raise RuntimeError(
            f"No training rows after merge for window {window.holdout_start}"
        )

    feature_cols = feature_state.feature_columns
    X = merged[feature_cols].to_numpy(dtype=float)
    y = merged["regime_label"].to_numpy(dtype=int)

    classifier = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        num_leaves=config.num_leaves,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        min_child_samples=config.min_child_samples,
        random_state=config.seed,
        n_jobs=1,
        verbose=-1,
    )
    classifier.fit(X, y)

    classifier_bytes = pickle.dumps(classifier)
    state_payload = feature_state.to_dict()
    state_bytes = json.dumps(state_payload, sort_keys=True).encode("utf-8")

    label_dist = {
        REGIME_NAMES[int(k)]: float(v)
        for k, v in (pd.Series(y).value_counts(normalize=True).sort_index().items())
    }
    # ensure all 3 classes are present (even if 0%)
    for cls_idx, name in REGIME_NAMES.items():
        label_dist.setdefault(name, 0.0)

    feature_trained_through = pd.Timestamp(merged["as_of"].max()).date()
    label_end_col = _label_end_cols(label_frame)[0]
    labels_known_through = pd.Timestamp(merged[label_end_col].max()).date()

    manifest_payload = {
        "available_from": window.holdout_start.isoformat(),
        "feature_trained_through": feature_trained_through.isoformat(),
        "labels_known_through": labels_known_through.isoformat(),
        "framework": "lightgbm_classifier",
        "label_horizon_days": int(config.label_horizon_days),
        "feature_columns": list(feature_cols),
        "classes": [0, 1, 2],
        "class_names": REGIME_NAMES,
        "label_distribution": label_dist,
        "lib_versions": _lightgbm_lib_versions(),
        "seed": int(config.seed),
        "notes": None,
        "n_training_rows": int(len(merged)),
    }
    bundle_hash = _bundle_hash(manifest_payload, classifier_bytes, state_bytes)
    manifest_payload["bundle_hash"] = bundle_hash

    (bundle_dir / "classifier.pkl").write_bytes(classifier_bytes)
    (bundle_dir / "imputer_state.json").write_bytes(state_bytes)
    (bundle_dir / "label_distribution.json").write_text(
        json.dumps(label_dist, indent=2, sort_keys=True)
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True))
    return manifest_path


def _label_end_cols(label_frame: pd.DataFrame) -> List[str]:
    return [c for c in label_frame.columns if c.startswith("label_end_")]


def run_walk_forward_regime_training(
    prices: pd.DataFrame,
    config: RegimeClassifierConfig,
    *,
    output_dir: Path = Path("data/external_features/ml/models_regime/default"),
    feature_builder: Optional[FeatureMatrixBuilder] = None,
) -> RegimeTrainingResult:
    """Trains one classifier per outer holdout window.

    Walk-forward logic identical to the forecast pipeline (reuse of
    ``_build_outer_windows``): each outer window has a `train_start`,
    a `holdout_start` and a `holdout_end`. Label PIT is enforced via
    `label_end < holdout_start`; features are fit with
    ``FeatureMatrixBuilder.fit_transform`` (imputer medians from the
    training window).
    """

    builder = feature_builder or FeatureMatrixBuilder()
    result = RegimeTrainingResult()
    tickers = list(config.tickers) or list(prices.columns)
    labels = compute_regime_labels(prices, label_horizon_days=config.label_horizon_days)
    if labels.empty:
        return result

    windows = _build_outer_windows(
        pd.DatetimeIndex(labels["as_of"].unique()),
        train_years=config.outer_train_years,
        holdout_months=config.outer_holdout_months,
    )
    if not windows:
        return result

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for window in windows:
        holdout_start_ts = pd.Timestamp(window.holdout_start)
        train_rows = _filter_train_rows(labels, holdout_start_ts)
        if train_rows.empty:
            continue
        train_as_of = pd.DatetimeIndex(train_rows["as_of"].unique())
        feature_frame, feature_state = builder.fit_transform(
            prices, train_as_of, tickers=tickers
        )
        if feature_frame.empty:
            continue
        try:
            manifest_path = _train_one_bundle(
                window=window,
                feature_frame=feature_frame,
                feature_state=feature_state,
                label_frame=train_rows,
                config=config,
                output_dir=output_dir,
            )
        except RuntimeError:
            continue
        result.manifest_paths.append(manifest_path)
        result.windows.append(window)
    return result


__all__ = [
    "RegimeClassifierConfig",
    "RegimeTrainingResult",
    "run_walk_forward_regime_training",
]
