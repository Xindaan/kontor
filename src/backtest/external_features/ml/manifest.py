"""Phase D bundle manifest (T-0211, Codex D9/D17/D29).

A manifest describes ONE training bundle (Stage 1 + Stage 2 + Stage 3
plus shared imputer + zscore stats). It lives next to the pickles and
is the canonical hash anchor for model provenance (Codex D9/D10).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def stable_model_entry_id(
    training_run_id: str,
    config_hash: str,
    bundle_hash: str,
) -> str:
    """Deterministic ``entry_id`` for bundle-level provenance (Codex D30)."""

    digest = hashlib.sha256(
        f"{training_run_id}|{config_hash}|{bundle_hash}".encode("utf-8")
    ).hexdigest()
    return f"ml-bundle-{digest[:16]}"


@dataclass
class MLBundleManifest:
    """All metadata required to reload one inference bundle."""

    horizons: List[int]
    feature_trained_through: str
    labels_known_through: str
    available_from: str
    lib_versions: Dict[str, str]
    seed: int
    training_run_id: str
    config_hash: str
    feature_schema_hash: str = ""
    stage_paths: Dict[str, str] = field(default_factory=dict)
    zscore_stats_path: str = "zscore_stats.json"
    imputer_state_path: str = "imputer_state.json"
    bundle_hash: str = ""
    model_family: str = "lightgbm"
    notes: Optional[str] = None
    # Phase E3 BC-additive fields (Codex R3.11):
    # `framework` marks DL bundles ("pytorch"); default "lightgbm"
    # remains for Phase D LightGBM/XGBoost/Ridge.
    framework: str = "lightgbm"
    # `sequence_length` is only set for DL bundles; None for
    # LightGBM/XGBoost/Ridge.
    sequence_length: Optional[int] = None

    def canonical_payload(self) -> str:
        payload: Dict[str, Any] = asdict(self)
        payload.pop("bundle_hash", None)
        return json.dumps(payload, sort_keys=True)

    def compute_bundle_hash(self) -> str:
        return hashlib.sha256(self.canonical_payload().encode("utf-8")).hexdigest()

    def with_bundle_hash(self) -> "MLBundleManifest":
        return MLBundleManifest(
            horizons=list(self.horizons),
            feature_trained_through=self.feature_trained_through,
            labels_known_through=self.labels_known_through,
            available_from=self.available_from,
            lib_versions=dict(self.lib_versions),
            seed=self.seed,
            training_run_id=self.training_run_id,
            config_hash=self.config_hash,
            feature_schema_hash=self.feature_schema_hash,
            stage_paths=dict(self.stage_paths),
            zscore_stats_path=self.zscore_stats_path,
            imputer_state_path=self.imputer_state_path,
            bundle_hash=self.compute_bundle_hash(),
            model_family=self.model_family,
            notes=self.notes,
            framework=self.framework,
            sequence_length=self.sequence_length,
        )

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        finalised = self.with_bundle_hash()
        payload = asdict(finalised)
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> "MLBundleManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        # BC protection (Phase E3 / Codex R3.11): Phase D bundles don't
        # have `framework`/`sequence_length`; filled in with defaults
        # here so `cls(**payload)` doesn't crash.
        payload.setdefault("framework", "lightgbm")
        payload.setdefault("sequence_length", None)
        return cls(**payload)


def feature_schema_hash(feature_columns: List[str]) -> str:
    canonical = "|".join(sorted(feature_columns))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "MLBundleManifest",
    "feature_schema_hash",
    "stable_model_entry_id",
]
