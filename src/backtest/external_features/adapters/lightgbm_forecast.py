"""Phase D LightGBM forecast adapter (T-0214).

Inference-only adapter that resolves the right bundle for ``as_of``
(Codex D3/D17), loads it, runs Stage 1 -> Stage 2 -> Stage 3 -> tanh
and emits the long-form ``ml_forecast_*`` rows plus a row-level
attributions sidecar via the registered ``ml_attribution_ndjson``
writer (Codex D7/D8).

The adapter uses :meth:`with_options` to receive the bundle directory
and an optional sentiment/analyst dataset config at pull-time. The
factory returns a new instance — the registry singleton stays
immutable (Codex D5/D20).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from backtest.external_features.adapters.base import (
    ExternalFeatureAdapter,
    ExternalFeatureFetchResult,
    SidecarBlob,
)
from backtest.external_features.ml.features import (
    FeatureMatrixBuilder,
    FeatureMatrixState,
)
from backtest.external_features.ml.inference import (
    BundleEntry,
    load_bundle,
    select_bundle_for_as_of,
)
from backtest.external_features.ml.manifest import MLBundleManifest
from backtest.external_features.ml.stage3_stacking import HORIZON_KEYS


DEFAULT_BUNDLE_ROOT = Path("data/external_features/ml/models")


def engine_code_uint32(engine_code: str) -> int:
    """Stable ``uint32`` for the ``ml_model_version`` row (Codex D25)."""

    digest = hashlib.sha256(engine_code.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True)
class MLAdapterOptions:
    """Runtime options for an ML adapter (set via :meth:`with_options`)."""

    bundle_root: Optional[Path] = None
    bundle_override: Optional[Path] = None
    prices: Optional[pd.DataFrame] = None
    external_features_provider: Optional[Any] = None
    fundamentals_loader: Optional[Any] = None
    analyst_dataset: Optional[str] = None
    news_dataset: Optional[str] = None
    sector_lookup: Mapping[str, str] = field(default_factory=dict)
    stacking_only: bool = False


def _build_feature_frame(
    *,
    options: MLAdapterOptions,
    state: FeatureMatrixState,
    tickers: List[str],
    as_of: date,
) -> pd.DataFrame:
    """Run :class:`FeatureMatrixBuilder.transform` for a single ``as_of``."""

    builder = FeatureMatrixBuilder(
        fundamentals_loader=options.fundamentals_loader,
        external_features_provider=options.external_features_provider,
        analyst_dataset=options.analyst_dataset,
        news_dataset=options.news_dataset,
        sector_lookup=options.sector_lookup,
    )
    prices = options.prices if options.prices is not None else pd.DataFrame()
    frame = builder.transform(
        prices=prices,
        as_of_index=[pd.Timestamp(as_of)],
        tickers=tickers,
        state=state,
    )
    return frame


def _build_csv_rows(
    *,
    as_of: date,
    snapshot_ts: pd.Timestamp,
    dataset_id: str,
    source_name: str,
    bundle: BundleEntry,
    predictions: pd.DataFrame,
    engine_code: str,
) -> List[dict]:
    manifest = bundle.manifest
    available_from = date.fromisoformat(manifest.available_from)
    feature_trained_through = date.fromisoformat(manifest.feature_trained_through)
    model_version = engine_code_uint32(engine_code)
    training_age = (as_of - available_from).days
    out: List[dict] = []
    for _, row in predictions.iterrows():
        base = {
            "ticker": str(row["ticker"]).upper(),
            "release_date": pd.Timestamp(as_of),
            "snapshot_ts": snapshot_ts,
            "source": source_name,
            "dataset": dataset_id,
        }
        score = float(row.get("ml_forecast_score", 0.0))
        out.append(
            {**base, "feature_name": "ml_forecast_score", "feature_value": score}
        )
        out.append(
            {
                **base,
                "feature_name": "ml_available_from_ordinal",
                "feature_value": float(available_from.toordinal()),
            }
        )
        out.append(
            {
                **base,
                "feature_name": "ml_feature_trained_through_ordinal",
                "feature_value": float(feature_trained_through.toordinal()),
            }
        )
        for h in HORIZON_KEYS:
            col = f"ml_forecast_{h}d"
            if col in predictions.columns:
                out.append(
                    {
                        **base,
                        "feature_name": col,
                        "feature_value": float(row.get(col, 0.0)),
                    }
                )
        out.append(
            {
                **base,
                "feature_name": "ml_model_version",
                "feature_value": float(model_version),
            }
        )
        out.append(
            {
                **base,
                "feature_name": "ml_training_age_days",
                "feature_value": float(training_age),
            }
        )
        out.append(
            {
                **base,
                "feature_name": "ml_stage2_fallback",
                "feature_value": 0.0,
            }
        )
    return out


def _build_sidecar_rows(
    *,
    feature_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    bundle: BundleEntry,
    engine_code: str,
) -> List[dict]:
    rows: List[dict] = []
    manifest = bundle.manifest
    feature_cols = [
        c for c in feature_frame.columns if c not in {"as_of", "ticker"}
    ]
    pred_lookup = predictions.set_index("ticker") if "ticker" in predictions.columns else None
    for _, frow in feature_frame.iterrows():
        ticker = str(frow["ticker"]).upper()
        # Single-row DataFrame with correct float dtypes; `frow.to_frame().T`
        # would produce object dtypes because the row has mixed types
        # (as_of/ticker as object). LightGBM would then reject it with
        # "pandas dtypes must be int, float or bool".
        X = pd.DataFrame(
            {col: [float(frow[col])] for col in feature_cols},
            index=[0],
        )
        for horizon in HORIZON_KEYS:
            stage1 = bundle.stage1.get(horizon)
            if stage1 is None:
                continue
            try:
                contributions_arr = stage1.predict_contributions(X)
                contributions = {
                    feature_cols[i]: float(contributions_arr[0, i])
                    for i in range(min(len(feature_cols), contributions_arr.shape[1]))
                }
                reason = None
            except Exception as exc:
                contributions = {}
                reason = type(exc).__name__
            raw_forecast = float(stage1.predict(X)[0])
            residual = 0.0
            sub_models = bundle.stage2.get(horizon, {})
            sub = sub_models.get(ticker)
            if sub is not None:
                try:
                    residual = float(sub.predict(X)[0])
                except Exception:
                    residual = 0.0
            blended = 0.0
            if pred_lookup is not None and ticker in pred_lookup.index:
                col = f"ml_forecast_{horizon}d"
                if col in pred_lookup.columns:
                    blended = float(pred_lookup.loc[ticker, col])
            entry: Dict[str, object] = {
                "ticker": ticker,
                "horizon": int(horizon),
                "contributions": contributions,
                "raw_forecast": raw_forecast,
                "residual": residual,
                "blended": blended,
                "model_family": manifest.model_family,
                "feature_trained_through": manifest.feature_trained_through,
                "available_from": manifest.available_from,
                "bundle_hash": manifest.bundle_hash,
                "engine_code": engine_code,
            }
            if reason:
                entry["contributions_unavailable_reason"] = reason
            rows.append(entry)
    return rows


class _BaseMLForecastAdapter(ExternalFeatureAdapter):
    """Shared logic for the LightGBM / XGBoost / Ensemble adapters."""

    model_family: str = "lightgbm"
    dataset_id_value: str = "lightgbm_forecast"
    source_name_value: str = "LightGBMForecast"
    engine_label: str = "lightgbm"
    license_note: str = (
        "internal model bundle; pickled LightGBM artefacts, see manifest"
    )

    def __init__(self, options: Optional[MLAdapterOptions] = None) -> None:
        self._options = options or MLAdapterOptions()

    @property
    def dataset_id(self) -> str:
        return self.dataset_id_value

    @property
    def source_name(self) -> str:
        return self.source_name_value

    @property
    def quality_tag(self) -> str:
        return "official"

    @property
    def license_tos_note(self) -> str:
        return self.license_note

    @property
    def plan_policy(self) -> Optional[str]:
        return "ml_model_bundle"

    @property
    def options(self) -> MLAdapterOptions:
        return self._options

    def with_options(self, **kwargs):
        """Codex D5/D20 — return a fresh instance, never mutate."""

        if not kwargs:
            return self
        current = self._options
        allowed = {
            "bundle_root",
            "bundle_override",
            "prices",
            "external_features_provider",
            "fundamentals_loader",
            "analyst_dataset",
            "news_dataset",
            "sector_lookup",
            "stacking_only",
        }
        bad = set(kwargs) - allowed
        if bad:
            raise TypeError(
                f"{type(self).__name__}.with_options does not accept: "
                + ", ".join(sorted(bad))
            )
        merged: Dict[str, Any] = {
            "bundle_root": current.bundle_root,
            "bundle_override": current.bundle_override,
            "prices": current.prices,
            "external_features_provider": current.external_features_provider,
            "fundamentals_loader": current.fundamentals_loader,
            "analyst_dataset": current.analyst_dataset,
            "news_dataset": current.news_dataset,
            "sector_lookup": current.sector_lookup,
            "stacking_only": current.stacking_only,
        }
        merged.update(kwargs)
        if merged["bundle_root"] is not None:
            merged["bundle_root"] = Path(merged["bundle_root"])
        if merged["bundle_override"] is not None:
            merged["bundle_override"] = Path(merged["bundle_override"])
        return type(self)(MLAdapterOptions(**merged))

    def _engine_code(self, bundle: BundleEntry) -> str:
        manifest = bundle.manifest
        lib_version = manifest.lib_versions.get(self.engine_label, "0.0")
        return f"{self.engine_label}@{lib_version}_{manifest.bundle_hash[:8]}"

    def _select_bundle(self, as_of: date) -> BundleEntry:
        opts = self._options
        if opts.bundle_override is not None:
            manifest_path = Path(opts.bundle_override) / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"override bundle missing manifest: {manifest_path}"
                )
        else:
            root = opts.bundle_root or DEFAULT_BUNDLE_ROOT
            manifest_path = select_bundle_for_as_of(
                root, as_of, model_family=self.model_family
            )
        return load_bundle(manifest_path)

    def _build_feature_frame_for_inference(
        self, bundle: BundleEntry, tickers: List[str], as_of: date
    ) -> pd.DataFrame:
        state = FeatureMatrixState.from_dict(bundle.imputer_state or {})
        return _build_feature_frame(
            options=self._options,
            state=state,
            tickers=tickers,
            as_of=as_of,
        )

    def fetch_remote(
        self, tickers: List[str], as_of: date
    ) -> ExternalFeatureFetchResult:
        from backtest.external_features.ml.inference import predict_bundle

        tickers_clean = [str(t).upper() for t in tickers if str(t).strip()]
        if not tickers_clean:
            return ExternalFeatureFetchResult(frame=pd.DataFrame(), sidecars=[])

        bundle = self._select_bundle(as_of)
        engine_code = self._engine_code(bundle)
        feature_frame = self._build_feature_frame_for_inference(
            bundle, tickers_clean, as_of
        )
        if feature_frame.empty:
            return ExternalFeatureFetchResult(frame=pd.DataFrame(), sidecars=[])
        predictions = predict_bundle(bundle, feature_frame)
        if predictions.empty:
            return ExternalFeatureFetchResult(frame=pd.DataFrame(), sidecars=[])
        snapshot_ts = pd.Timestamp.utcnow().tz_localize(None)
        csv_rows = _build_csv_rows(
            as_of=as_of,
            snapshot_ts=snapshot_ts,
            dataset_id=self.dataset_id,
            source_name=self.source_name,
            bundle=bundle,
            predictions=predictions,
            engine_code=engine_code,
        )
        sidecar_rows = _build_sidecar_rows(
            feature_frame=feature_frame,
            predictions=predictions,
            bundle=bundle,
            engine_code=engine_code,
        )
        sidecar = SidecarBlob(
            relative_name=f"{as_of.isoformat()}.ml_attribution.ndjson",
            rows=sidecar_rows,
            kind="ml_attribution_ndjson",
        )
        frame = pd.DataFrame(csv_rows)
        return ExternalFeatureFetchResult(frame=frame, sidecars=[sidecar])


class LightGBMForecastAdapter(_BaseMLForecastAdapter):
    """LightGBM inference adapter (Codex D5/D20)."""

    model_family = "lightgbm"
    dataset_id_value = "lightgbm_forecast"
    source_name_value = "LightGBMForecast"
    engine_label = "lightgbm"
    license_note = (
        "internal LightGBM bundle; pickled artefacts, manifest carries lib_versions"
    )


__all__ = [
    "DEFAULT_BUNDLE_ROOT",
    "LightGBMForecastAdapter",
    "MLAdapterOptions",
    "_BaseMLForecastAdapter",
    "engine_code_uint32",
]
