"""Phase D synthetic ML-forecast adapter (T-0217, Codex D15).

Generates deterministic ``ml_forecast_score`` values from a
``(ticker, as_of)`` seed so the ML score-mix path can be exercised
without training any real model. The adapter also writes a
deterministic ``ml_attribution_ndjson`` sidecar so the
``raw_payload_hash`` in the CSV stays anchored (DoD: sidecar hash
unchanged for synthetic too).

The synthetic engine does NOT consume a bundle directory. The
``engine_code`` is ``synthetic_ml@1.0`` so provenance stays auditable.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import List, Optional

import pandas as pd

from backtest.external_features.adapters.base import (
    ExternalFeatureAdapter,
    ExternalFeatureFetchResult,
    SidecarBlob,
)

DATASET_ID = "synthetic_ml_forecast"
SOURCE_NAME = "SyntheticMLForecast"
ENGINE_CODE = "synthetic_ml@1.0"
HORIZON_KEYS: tuple[int, ...] = (21, 63, 252)


def _engine_code_uint32(engine_code: str) -> int:
    """Stable ``uint32`` for the long-form ``ml_model_version`` row (Codex D25)."""

    digest = hashlib.sha256(engine_code.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _deterministic_horizon_score(ticker: str, as_of_iso: str, horizon: int) -> float:
    seed = f"{ticker.upper()}|{as_of_iso}|{horizon}".encode("utf-8")
    digest = hashlib.sha256(seed).digest()
    raw = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return round(raw * 2.0 - 1.0, 6)


class SyntheticMLForecastAdapter(ExternalFeatureAdapter):
    """Deterministic synthetic ML-forecast pull (T-0217).

    Has no bundle requirement and is the default smoke target. The
    ``available_from`` / ``feature_trained_through`` metadata fields are
    set to ``as_of`` itself so the ``assess_ml_evidence`` lookahead
    check (Codex D14/D18) never marks synthetic rows as invalid in
    legitimate test scenarios.
    """

    def __init__(self, *, engine_code: str = ENGINE_CODE) -> None:
        self._engine_code = str(engine_code)

    @property
    def dataset_id(self) -> str:
        return DATASET_ID

    @property
    def source_name(self) -> str:
        return SOURCE_NAME

    @property
    def quality_tag(self) -> str:
        return "proxy"

    @property
    def license_tos_note(self) -> str:
        return "synthetic ml forecast derived from deterministic (ticker, as_of) hash"

    @property
    def plan_policy(self) -> Optional[str]:
        return "synthetic_proxy"

    @property
    def source_url(self) -> Optional[str]:
        return None

    def with_options(self, **kwargs):
        """Codex D5/D20: non-mutating; accepts only ``engine_code``."""

        if not kwargs:
            return self
        engine_code = kwargs.pop("engine_code", self._engine_code)
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(
                f"SyntheticMLForecastAdapter.with_options does not accept: {unsupported}"
            )
        return SyntheticMLForecastAdapter(engine_code=engine_code)

    def fetch_remote(
        self, tickers: List[str], as_of: date
    ) -> ExternalFeatureFetchResult:
        tickers_clean = [str(t).upper() for t in tickers if str(t).strip()]
        if not tickers_clean:
            return ExternalFeatureFetchResult(frame=pd.DataFrame(), sidecars=[])
        snapshot_ts = pd.Timestamp.utcnow().tz_localize(None)
        as_of_iso = as_of.isoformat()
        as_of_ordinal = as_of.toordinal()
        model_version = _engine_code_uint32(self._engine_code)

        csv_rows: List[dict] = []
        sidecar_rows: List[dict] = []
        for ticker in tickers_clean:
            horizon_scores = {
                h: _deterministic_horizon_score(ticker, as_of_iso, h)
                for h in HORIZON_KEYS
            }
            # Stage-3 blended target uses tanh of mean. Stay determinstic
            # without re-introducing real model dependencies.
            blended = sum(horizon_scores.values()) / len(horizon_scores)
            score = round(_clip(blended), 6)
            base_row = {
                "ticker": ticker,
                "release_date": pd.Timestamp(as_of),
                "snapshot_ts": snapshot_ts,
                "source": self.source_name,
                "dataset": self.dataset_id,
            }
            csv_rows.append(
                {**base_row, "feature_name": "ml_forecast_score", "feature_value": score}
            )
            csv_rows.append(
                {
                    **base_row,
                    "feature_name": "ml_available_from_ordinal",
                    "feature_value": float(as_of_ordinal),
                }
            )
            csv_rows.append(
                {
                    **base_row,
                    "feature_name": "ml_feature_trained_through_ordinal",
                    "feature_value": float(as_of_ordinal),
                }
            )
            for horizon, h_score in horizon_scores.items():
                csv_rows.append(
                    {
                        **base_row,
                        "feature_name": f"ml_forecast_{horizon}d",
                        "feature_value": round(_clip(h_score), 6),
                    }
                )
            csv_rows.append(
                {
                    **base_row,
                    "feature_name": "ml_model_version",
                    "feature_value": float(model_version),
                }
            )
            csv_rows.append(
                {
                    **base_row,
                    "feature_name": "ml_training_age_days",
                    "feature_value": 0.0,
                }
            )
            csv_rows.append(
                {
                    **base_row,
                    "feature_name": "ml_stage2_fallback",
                    "feature_value": 1.0,
                }
            )
            for horizon, h_score in horizon_scores.items():
                sidecar_rows.append(
                    {
                        "ticker": ticker,
                        "horizon": int(horizon),
                        "contributions": {},
                        "raw_forecast": float(h_score),
                        "residual": 0.0,
                        "blended": float(score),
                        "model_family": "synthetic",
                        "feature_trained_through": as_of_iso,
                        "available_from": as_of_iso,
                        "bundle_hash": "",
                        "engine_code": self._engine_code,
                        "contributions_unavailable_reason": "synthetic_engine_no_contributions",
                    }
                )
        frame = pd.DataFrame(csv_rows)
        sidecar = SidecarBlob(
            relative_name=f"{as_of_iso}.ml_attribution.ndjson",
            rows=sidecar_rows,
            kind="ml_attribution_ndjson",
        )
        return ExternalFeatureFetchResult(frame=frame, sidecars=[sidecar])


def _clip(value: float) -> float:
    if value != value:  # NaN guard
        return 0.0
    return max(-1.0, min(1.0, float(value)))


__all__ = [
    "DATASET_ID",
    "ENGINE_CODE",
    "HORIZON_KEYS",
    "SyntheticMLForecastAdapter",
]
