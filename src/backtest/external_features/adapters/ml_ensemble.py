"""Phase D ML ensemble forecast adapter (T-0216).

Loads BOTH the latest LightGBM and XGBoost bundles whose
``available_from <= as_of`` and averages the final ``ml_forecast_*``
tanh-outputs. The sidecar carries one row per ``(ticker, horizon,
model_family)`` so attribution stays auditable per family.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from backtest.external_features.adapters.base import (
    ExternalFeatureFetchResult,
    SidecarBlob,
)
from backtest.external_features.adapters.lightgbm_forecast import (
    DEFAULT_BUNDLE_ROOT,
    LightGBMForecastAdapter,
    MLAdapterOptions,
    _BaseMLForecastAdapter,
    _build_csv_rows,
    _build_feature_frame,
    _build_sidecar_rows,
)
from backtest.external_features.adapters.xgboost_forecast import (
    XGBoostForecastAdapter,
)
from backtest.external_features.ml.features import FeatureMatrixState
from backtest.external_features.ml.inference import (
    BundleEntry,
    load_bundle,
    predict_bundle,
    select_bundle_for_as_of,
)
from backtest.external_features.ml.stage3_stacking import HORIZON_KEYS


class EnsembleForecastAdapter(_BaseMLForecastAdapter):
    """Stacked-mean ensemble of LightGBM + XGBoost bundles."""

    model_family = "ensemble"
    dataset_id_value = "ml_forecast_ensemble"
    source_name_value = "MLForecastEnsemble"
    engine_label = "ensemble"
    license_note = (
        "ensemble of internal LightGBM + XGBoost bundles; see per-family manifests"
    )

    def _select_bundle(self, as_of: date) -> BundleEntry:  # pragma: no cover
        raise RuntimeError(
            "EnsembleForecastAdapter does not select a single bundle. "
            "Use fetch_remote() which loads lightgbm + xgboost bundles."
        )

    def fetch_remote(
        self, tickers: List[str], as_of: date
    ) -> ExternalFeatureFetchResult:
        tickers_clean = [str(t).upper() for t in tickers if str(t).strip()]
        if not tickers_clean:
            return ExternalFeatureFetchResult(frame=pd.DataFrame(), sidecars=[])

        opts = self._options
        root = opts.bundle_root or DEFAULT_BUNDLE_ROOT
        bundles: List[BundleEntry] = []
        for family in ("lightgbm", "xgboost"):
            manifest_path = _try_select_bundle(
                root, as_of, model_family=family, override=opts.bundle_override
            )
            if manifest_path is None:
                continue
            bundles.append(load_bundle(manifest_path))
        if not bundles:
            raise RuntimeError(
                f"No lightgbm/xgboost ML bundle with available_from <= "
                f"{as_of.isoformat()} under {root}"
            )

        snapshot_ts = pd.Timestamp.utcnow().tz_localize(None)
        per_family_predictions: List[pd.DataFrame] = []
        per_family_csv_rows: List[List[dict]] = []
        sidecar_rows: List[dict] = []

        for bundle in bundles:
            state = FeatureMatrixState.from_dict(bundle.imputer_state or {})
            feature_frame = _build_feature_frame(
                options=opts,
                state=state,
                tickers=tickers_clean,
                as_of=as_of,
            )
            if feature_frame.empty:
                continue
            predictions = predict_bundle(bundle, feature_frame)
            if predictions.empty:
                continue
            engine_code = (
                f"{bundle.manifest.model_family}@"
                f"{bundle.manifest.lib_versions.get(bundle.manifest.model_family, '0.0')}"
                f"_{bundle.manifest.bundle_hash[:8]}"
            )
            per_family_predictions.append(predictions)
            per_family_csv_rows.append(
                _build_csv_rows(
                    as_of=as_of,
                    snapshot_ts=snapshot_ts,
                    dataset_id=bundle.manifest.model_family,  # placeholder
                    source_name=bundle.manifest.model_family,
                    bundle=bundle,
                    predictions=predictions,
                    engine_code=engine_code,
                )
            )
            sidecar_rows.extend(
                _build_sidecar_rows(
                    feature_frame=feature_frame,
                    predictions=predictions,
                    bundle=bundle,
                    engine_code=engine_code,
                )
            )
        if not per_family_predictions:
            return ExternalFeatureFetchResult(frame=pd.DataFrame(), sidecars=[])

        averaged = _average_predictions(per_family_predictions)
        # Use the *most recent* bundle as anchor for ``available_from``
        anchor = max(
            bundles, key=lambda b: b.manifest.available_from
        )
        engine_code = (
            f"ensemble@{anchor.manifest.bundle_hash[:8]}"
        )
        csv_rows = _build_csv_rows(
            as_of=as_of,
            snapshot_ts=snapshot_ts,
            dataset_id=self.dataset_id,
            source_name=self.source_name,
            bundle=anchor,
            predictions=averaged,
            engine_code=engine_code,
        )
        sidecar = SidecarBlob(
            relative_name=f"{as_of.isoformat()}.ml_attribution.ndjson",
            rows=sidecar_rows,
            kind="ml_attribution_ndjson",
        )
        return ExternalFeatureFetchResult(
            frame=pd.DataFrame(csv_rows), sidecars=[sidecar]
        )


def _try_select_bundle(
    root: Path, as_of: date, *, model_family: str, override: Optional[Path]
) -> Optional[Path]:
    try:
        return select_bundle_for_as_of(
            root, as_of, model_family=model_family, override=override
        )
    except (FileNotFoundError, RuntimeError):
        return None


def _average_predictions(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = list(frames)
    if not frames:
        return pd.DataFrame()
    base = frames[0][["as_of", "ticker"]].reset_index(drop=True).copy()
    score_cols = ["ml_forecast_score"] + [
        f"ml_forecast_{h}d" for h in HORIZON_KEYS
    ]
    sums = {col: np.zeros(len(base), dtype=float) for col in score_cols}
    counts = {col: np.zeros(len(base), dtype=float) for col in score_cols}
    for frame in frames:
        for col in score_cols:
            if col in frame.columns:
                values = pd.to_numeric(frame[col], errors="coerce").to_numpy()
                mask = ~np.isnan(values)
                sums[col] += np.where(mask, np.nan_to_num(values), 0.0)
                counts[col] += mask.astype(float)
    out = base.copy()
    for col in score_cols:
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(counts[col] > 0, sums[col] / counts[col], np.nan)
        out[col] = avg
    return out


__all__ = ["EnsembleForecastAdapter"]
