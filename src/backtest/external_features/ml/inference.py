"""Phase D inference helpers (T-0212).

The inference engine answers two questions:

1. Which manifest applies for ``inference_as_of=X``? Picks the bundle
   with the highest ``available_from <= X`` (Codex D3/D17). Raises
   when no eligible bundle exists.
2. Given an aligned feature frame, what is the row-level forecast for
   each ``(ticker, as_of)``?

Both helpers are split out so unit tests can drive them with the
``MockForecastModel`` instead of LightGBM/XGBoost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.external_features.ml.manifest import MLBundleManifest
from backtest.external_features.ml.models.base import ForecastModel
from backtest.external_features.ml.stage3_stacking import HORIZON_KEYS, ZScoreStats


@dataclass
class BundleEntry:
    """Resolved on-disk bundle with manifest + loaded artefacts."""

    manifest: MLBundleManifest
    bundle_dir: Path
    stage1: Dict[int, ForecastModel]
    stage2: Dict[int, Dict[str, ForecastModel]]
    stage3: ForecastModel
    zscore_stats: ZScoreStats
    imputer_state: dict


def discover_manifests(bundle_root: Path) -> List[Path]:
    bundle_root = Path(bundle_root)
    if not bundle_root.exists():
        return []
    return sorted(bundle_root.rglob("manifest.json"))


def select_bundle_for_as_of(
    bundle_root: Path,
    as_of: date,
    *,
    model_family: str = "lightgbm",
    override: Optional[Path] = None,
) -> Path:
    """Return the manifest path with the highest ``available_from`` that
    still respects ``available_from <= as_of`` (Codex D3/D17)."""

    if override:
        manifest_path = Path(override) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"override bundle missing manifest: {manifest_path}")
        return manifest_path

    candidates: List[tuple[str, Path]] = []
    for manifest_path in discover_manifests(bundle_root):
        try:
            manifest = MLBundleManifest.read(manifest_path)
        except Exception:
            continue
        if manifest.model_family != model_family:
            continue
        if manifest.available_from > as_of.isoformat():
            continue
        candidates.append((manifest.available_from, manifest_path))
    if not candidates:
        raise RuntimeError(
            f"No ML bundle with available_from <= {as_of.isoformat()} under {bundle_root}"
        )
    candidates.sort()
    return candidates[-1][1]


def load_bundle(manifest_path: Path) -> BundleEntry:
    manifest_path = Path(manifest_path)
    manifest = MLBundleManifest.read(manifest_path)
    bundle_dir = manifest_path.parent

    stage1_models: Dict[int, ForecastModel] = {}
    stage2_models: Dict[int, Dict[str, ForecastModel]] = {}
    for horizon in HORIZON_KEYS:
        s1_key = f"stage1_{horizon}d"
        s2_key = f"stage2_{horizon}d"
        if s1_key in manifest.stage_paths:
            stage1_models[horizon] = ForecastModel.load(bundle_dir / manifest.stage_paths[s1_key])
        if s2_key in manifest.stage_paths:
            # Stage 2 is a dict {ticker: ForecastModel}, not a single
            # model — load via raw pickle.
            import pickle as _pickle

            with (bundle_dir / manifest.stage_paths[s2_key]).open("rb") as _h:
                raw = _pickle.load(_h)
            if not isinstance(raw, dict):
                raise TypeError(
                    f"stage2 pickle at {manifest.stage_paths[s2_key]} expected "
                    f"a dict[ticker->ForecastModel], got {type(raw).__name__}"
                )
            stage2_models[horizon] = raw
    stage3 = ForecastModel.load(bundle_dir / manifest.stage_paths["stage3"])

    zscore_payload = json.loads(
        (bundle_dir / manifest.zscore_stats_path).read_text(encoding="utf-8")
    )
    imputer_state = json.loads(
        (bundle_dir / manifest.imputer_state_path).read_text(encoding="utf-8")
    )
    return BundleEntry(
        manifest=manifest,
        bundle_dir=bundle_dir,
        stage1=stage1_models,
        stage2=stage2_models,
        stage3=stage3,
        zscore_stats=ZScoreStats.from_dict(zscore_payload),
        imputer_state=imputer_state,
    )


def predict_bundle(
    bundle: BundleEntry,
    features_with_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Run Stage 1 -> Stage 2 -> Stage 3 -> tanh for a feature frame.

    Returns a frame with ``ticker``, ``as_of``, ``ml_forecast_score`` and
    one column ``ml_forecast_{h}d`` per stacked horizon.
    """

    feature_cols = [c for c in features_with_meta.columns if c not in {"as_of", "ticker"}]
    if not feature_cols:
        return pd.DataFrame(columns=["as_of", "ticker", "ml_forecast_score"])

    combined_cols: Dict[str, np.ndarray] = {}
    for horizon, model in bundle.stage1.items():
        stage1_pred = model.predict(features_with_meta[feature_cols])
        residual = pd.Series(0.0, index=features_with_meta.index, dtype=float)
        per_ticker = bundle.stage2.get(horizon, {})
        if per_ticker:
            for ticker, group in features_with_meta.groupby("ticker"):
                sub_model = per_ticker.get(str(ticker))
                if sub_model is None:
                    continue
                try:
                    res = sub_model.predict(group[feature_cols])
                except Exception:
                    continue
                residual.loc[group.index] = res
        combined_cols[f"combined_{horizon}d"] = stage1_pred + residual.to_numpy()
    combined = pd.DataFrame(combined_cols, index=features_with_meta.index)
    stage3_pred = bundle.stage3.predict(combined[[c for c in combined.columns]])
    out = pd.DataFrame(
        {
            "as_of": features_with_meta["as_of"].values,
            "ticker": features_with_meta["ticker"].values,
            "ml_forecast_score": np.tanh(stage3_pred),
        }
    )
    for horizon in HORIZON_KEYS:
        col = f"combined_{horizon}d"
        if col in combined.columns:
            out[f"ml_forecast_{horizon}d"] = np.tanh(combined[col].to_numpy())
    return out


__all__ = [
    "BundleEntry",
    "discover_manifests",
    "load_bundle",
    "predict_bundle",
    "select_bundle_for_as_of",
]
