"""Phase D walk-forward training pipeline (T-0211).

Trains one bundle per outer holdout window. Each bundle contains:

- 3 Stage-1 cross-sectional models (one per horizon)
- 3 Stage-2 dicts (per-ticker residual models)
- 1 Stage-3 stacking meta-model
- imputer state + z-score stats
- canonical ``manifest.json``

Training is intentionally a separate path from the inference adapter.
Real LightGBM/XGBoost runs need ``poetry install --with ml``. Unit
tests drive the loop with :class:`MockForecastModel`.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd

from backtest.external_features.ml.config import MLTrainingConfig
from backtest.external_features.ml.features import (
    FeatureMatrixBuilder,
    FeatureMatrixState,
)
from backtest.external_features.ml.manifest import (
    MLBundleManifest,
    feature_schema_hash,
)
from backtest.external_features.ml.models.base import ForecastModel, MockForecastModel
from backtest.external_features.ml.stage1_cross_sectional import (
    fit_cross_sectional,
    predict_oof,
)
from backtest.external_features.ml.stage2_residual import (
    Stage2Result,
    fit_residual_models,
)
from backtest.external_features.ml.stage3_stacking import (
    HORIZON_KEYS,
    Stage3Result,
    ZScoreStats,
    fit_stacking_model,
)
from backtest.external_features.ml.targets import compute_forward_returns
from backtest.provenance import ManualDataProvenanceRegistry


@dataclass
class WalkForwardWindow:
    train_start: date
    holdout_start: date
    holdout_end: date


@dataclass
class TrainingResult:
    manifest_paths: List[Path] = field(default_factory=list)
    windows: List[WalkForwardWindow] = field(default_factory=list)


def _model_class_from_family(family: str) -> Type[ForecastModel]:
    family_lower = (family or "").lower()
    if family_lower == "lightgbm":
        from backtest.external_features.ml.models.lightgbm_impl import LightGBMModel

        return LightGBMModel
    if family_lower == "xgboost":
        from backtest.external_features.ml.models.xgboost_impl import XGBoostModel

        return XGBoostModel
    if family_lower in {"ridge", "linear"}:
        from backtest.external_features.ml.models.linear import RidgeModel

        return RidgeModel
    if family_lower in {"mock", "synthetic"}:
        return MockForecastModel
    # Phase E3 (T-0346, Codex R2.11): DL families via factory strings,
    # so `import` doesn't crash without torch.
    if family_lower == "lstm":
        from backtest.external_features.ml.models import (
            LSTM_FACTORY_PATH,
            resolve_factory,
        )

        return resolve_factory(LSTM_FACTORY_PATH)
    if family_lower == "transformer":
        from backtest.external_features.ml.models import (
            TRANSFORMER_FACTORY_PATH,
            resolve_factory,
        )

        return resolve_factory(TRANSFORMER_FACTORY_PATH)
    raise ValueError(f"unknown model family '{family}'")


def _resolve_lib_versions(family: str) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    try:
        import sys

        versions["python"] = sys.version.split(" ")[0]
    except Exception:
        pass
    try:
        import numpy as _np

        versions["numpy"] = _np.__version__
    except Exception:
        pass
    try:
        import pandas as _pd

        versions["pandas"] = _pd.__version__
    except Exception:
        pass
    family_lower = (family or "").lower()
    if family_lower == "lightgbm":
        try:
            import lightgbm as _lgb  # type: ignore

            versions["lightgbm"] = _lgb.__version__
        except ImportError:
            pass
    elif family_lower == "xgboost":
        try:
            import xgboost as _xgb  # type: ignore

            versions["xgboost"] = _xgb.__version__
        except ImportError:
            pass
    elif family_lower in {"lstm", "transformer"}:
        # Phase E3: only register torch if installed.
        try:
            import torch as _torch  # type: ignore

            versions["torch"] = _torch.__version__
        except ImportError:
            pass
    return versions


def _next_business_day(base: pd.Timestamp) -> pd.Timestamp:
    return (base + pd.tseries.offsets.BusinessDay()).normalize()


def _next_after(values: pd.Series, anchor: pd.Timestamp) -> pd.Timestamp:
    future = values[values > anchor]
    if future.empty:
        return _next_business_day(anchor)
    return pd.Timestamp(future.min()).normalize()


def _build_outer_windows(
    as_of_dates: pd.DatetimeIndex,
    *,
    train_years: float,
    holdout_months: int,
) -> List[WalkForwardWindow]:
    if as_of_dates.empty:
        return []
    sorted_dates = as_of_dates.sort_values()
    start = sorted_dates.min()
    end = sorted_dates.max()
    train_delta = pd.Timedelta(days=int(train_years * 365))
    holdout_delta = pd.DateOffset(months=int(holdout_months))
    windows: List[WalkForwardWindow] = []
    holdout_start = start + train_delta
    while holdout_start <= end:
        holdout_end = pd.Timestamp(holdout_start) + holdout_delta
        if holdout_end > end:
            break
        windows.append(
            WalkForwardWindow(
                train_start=start.date(),
                holdout_start=pd.Timestamp(holdout_start).date(),
                holdout_end=pd.Timestamp(holdout_end).date(),
            )
        )
        holdout_start = pd.Timestamp(holdout_end)
    return windows


def _filter_train_rows(
    target_frame: pd.DataFrame,
    holdout_start: pd.Timestamp,
    horizon_label_columns: List[str],
) -> pd.DataFrame:
    """Drop rows whose label_end touches the holdout — Codex D16/D27."""

    if target_frame.empty:
        return target_frame
    purge_dates = target_frame[horizon_label_columns].apply(pd.to_datetime, errors="coerce")
    max_label = purge_dates.max(axis=1)
    mask = (target_frame["as_of"] < holdout_start) & (max_label < holdout_start)
    return target_frame.loc[mask]


@dataclass
class _ModelClasses:
    cross_sectional: Type[ForecastModel]
    residual: Type[ForecastModel]
    stacking: Type[ForecastModel]


def run_walk_forward_training(
    prices: pd.DataFrame,
    config: MLTrainingConfig,
    *,
    feature_builder: Optional[FeatureMatrixBuilder] = None,
    output_dir: Path = Path("data/external_features/ml/models"),
    registry: Optional[ManualDataProvenanceRegistry] = None,
    model_classes: Optional[Mapping[str, _ModelClasses]] = None,
) -> TrainingResult:
    """Train one bundle per outer holdout window.

    Real workloads use LightGBM/XGBoost; ``model_classes`` lets unit
    tests inject :class:`MockForecastModel` to keep the loop fast.
    """

    if registry is None:
        registry = ManualDataProvenanceRegistry()
    builder = feature_builder or FeatureMatrixBuilder()
    result = TrainingResult()
    tickers = list(config.tickers) or list(prices.columns)
    forward = compute_forward_returns(prices, list(config.horizons))
    if forward.empty:
        return result

    horizon_label_columns = [f"label_end_{h}d" for h in config.horizons]
    horizon_return_columns = [f"horizon_{h}d" for h in config.horizons]

    windows = _build_outer_windows(
        pd.DatetimeIndex(forward["as_of"].unique()),
        train_years=config.outer_train_years,
        holdout_months=config.outer_holdout_months,
    )

    for window in windows:
        holdout_start_ts = pd.Timestamp(window.holdout_start)
        train_rows = _filter_train_rows(forward, holdout_start_ts, horizon_label_columns)
        if train_rows.empty:
            continue
        train_as_of = pd.DatetimeIndex(train_rows["as_of"].unique())
        feature_frame, feature_state = builder.fit_transform(
            prices, train_as_of, tickers=tickers
        )
        feature_cols = feature_state.feature_columns
        merged = feature_frame.merge(
            train_rows[["as_of", "ticker", *horizon_return_columns, *horizon_label_columns]],
            on=["as_of", "ticker"],
            how="inner",
        )
        if merged.empty:
            continue

        for family in config.model_families:
            classes = (
                model_classes.get(family)
                if model_classes and family in model_classes
                else _ModelClasses(
                    cross_sectional=_model_class_from_family(family),
                    residual=_model_class_from_family(family),
                    stacking=_model_class_from_family(family),
                )
            )
            bundle_path = _train_one_bundle(
                window=window,
                merged=merged,
                feature_state=feature_state,
                feature_cols=feature_cols,
                family=family,
                classes=classes,
                config=config,
                output_dir=output_dir,
                registry=registry,
            )
            result.manifest_paths.append(bundle_path)
            result.windows.append(window)
    return result


def _train_one_bundle(
    *,
    window: WalkForwardWindow,
    merged: pd.DataFrame,
    feature_state: FeatureMatrixState,
    feature_cols: List[str],
    family: str,
    classes: _ModelClasses,
    config: MLTrainingConfig,
    output_dir: Path,
    registry: ManualDataProvenanceRegistry,
) -> Path:
    bundle_dir = (
        Path(output_dir) / window.holdout_start.isoformat() / family
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    stage1_models: Dict[int, ForecastModel] = {}
    stage2_results: Dict[int, Stage2Result] = {}
    oof_predictions: Dict[int, pd.Series] = {}
    combined_oof = pd.DataFrame(index=merged.index)
    label_ends = merged[[f"label_end_{h}d" for h in config.horizons]]
    for horizon in config.horizons:
        target = pd.to_numeric(merged[f"horizon_{horizon}d"], errors="coerce")
        mask = target.notna()
        X = merged.loc[mask, feature_cols]
        y = target.loc[mask]
        local_label_ends = label_ends.loc[mask]
        local_as_of = merged.loc[mask, "as_of"]
        # OOF predictions per horizon-specific purge.
        stage1_result = predict_oof(
            X=X,
            y=y,
            as_of=local_as_of,
            label_ends=local_label_ends,
            n_splits=3,
            horizon_key=f"{horizon}d",
            model_class=classes.cross_sectional,
        )
        # Final stage-1 model on the full available train.
        stage1_final = fit_cross_sectional(
            X, y, model_class=classes.cross_sectional
        )
        stage1_models[horizon] = stage1_final

        oof = pd.Series(np.nan, index=merged.index, dtype=float)
        oof.loc[mask] = stage1_result.oof_predictions.values
        oof_predictions[horizon] = oof

        # Stage 2: per-ticker residual on OOF residuals (Codex D6).
        stage2 = fit_residual_models(
            feature_frame=merged.loc[mask, ["as_of", "ticker", *feature_cols]],
            oof_stage1_predictions=stage1_result.oof_predictions,
            actuals=y,
            min_ticker_history_days=int(config.min_ticker_history_days),
            model_class=classes.residual,
        )
        stage2_results[horizon] = stage2

        # Combined OOF (Stage 1 + Stage 2) per row.
        residual_pred = pd.Series(0.0, index=merged.index, dtype=float)
        for ticker, group in merged.loc[mask].groupby("ticker"):
            model = stage2.residual_models.get(str(ticker))
            if model is None:
                continue
            try:
                residual_pred.loc[group.index] = model.predict(group[feature_cols])
            except Exception:
                continue
        combined = oof.fillna(0.0) + residual_pred.fillna(0.0)
        combined_oof[f"combined_{horizon}d"] = combined

    # Stage 3: stacking only on rows with all three horizons.
    stage3 = fit_stacking_model(
        combined_oof=combined_oof,
        forward_returns=merged[[f"horizon_{h}d" for h in config.horizons]],
        model_class=classes.stacking,
    )

    # Persist artefacts.
    stage_paths: Dict[str, str] = {}
    for horizon, model in stage1_models.items():
        rel = f"stage1_{horizon}d.pkl"
        model.save(bundle_dir / rel)
        stage_paths[f"stage1_{horizon}d"] = rel
    for horizon, stage2 in stage2_results.items():
        rel = f"stage2_{horizon}d.pkl"
        import pickle as _pickle

        with (bundle_dir / rel).open("wb") as handle:
            _pickle.dump(stage2.residual_models, handle, protocol=_pickle.HIGHEST_PROTOCOL)
        stage_paths[f"stage2_{horizon}d"] = rel
    stage3.model.save(bundle_dir / "stage3.pkl")
    stage_paths["stage3"] = "stage3.pkl"
    (bundle_dir / "imputer_state.json").write_text(
        json.dumps(feature_state.to_dict(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    (bundle_dir / "zscore_stats.json").write_text(
        json.dumps(stage3.zscore_stats.to_dict(), sort_keys=True, indent=2),
        encoding="utf-8",
    )

    # Manifest dates: feature_trained_through = max as_of; labels_known_through =
    # max label_end over all horizons; available_from = next business day.
    feature_through = pd.Timestamp(merged["as_of"].max()).normalize()
    label_through = pd.Timestamp(label_ends.max().max()).normalize()
    available_from = _next_business_day(label_through)

    manifest = MLBundleManifest(
        horizons=list(config.horizons),
        feature_trained_through=feature_through.date().isoformat(),
        labels_known_through=label_through.date().isoformat(),
        available_from=available_from.date().isoformat(),
        lib_versions=_resolve_lib_versions(family),
        seed=int(config.seed),
        training_run_id=str(uuid.uuid4()),
        config_hash=config.config_hash,
        feature_schema_hash=feature_schema_hash(feature_cols),
        stage_paths=stage_paths,
        model_family=family,
    )
    manifest_path = manifest.write(bundle_dir / "manifest.json")
    return manifest_path


__all__ = [
    "TrainingResult",
    "WalkForwardWindow",
    "run_walk_forward_training",
]
