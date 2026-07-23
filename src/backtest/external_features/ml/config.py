"""Phase D ML training/inference configuration dataclasses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class MLTrainingConfig:
    """Inputs accepted by the training pipeline.

    Default values are intentionally conservative; real Phase-D runs
    override them via the ``backtest ml train`` CLI.
    """

    horizons: Tuple[int, ...] = (21, 63, 252)
    model_families: Tuple[str, ...] = ("lightgbm",)
    outer_train_years: float = 5.0
    outer_holdout_months: int = 1
    inner_train_years: float = 3.0
    inner_test_months: int = 6
    grid_size: int = 4
    min_ticker_history_days: int = 250
    feature_lookback_years: float = 6.0
    seed: int = 42
    tickers: Tuple[str, ...] = ()
    universe_source: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("horizons must not be empty")
        if not self.model_families:
            raise ValueError("model_families must not be empty")
        if self.outer_train_years <= 0:
            raise ValueError("outer_train_years must be > 0")
        if self.outer_holdout_months <= 0:
            raise ValueError("outer_holdout_months must be > 0")
        if not self.tickers and not self.universe_source:
            raise ValueError(
                "MLTrainingConfig requires either tickers or universe_source "
                "(Codex D4 — Survivorship-Schutz)"
            )

    @property
    def config_hash(self) -> str:
        payload = asdict(self)
        # Serialise tuples as lists for stable JSON.
        payload["horizons"] = list(self.horizons)
        payload["model_families"] = list(self.model_families)
        payload["tickers"] = list(self.tickers)
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MLInferenceConfig:
    """Bundle resolution settings used by ML adapters."""

    bundle_dir: str = "data/external_features/ml/models"
    model_family: str = "lightgbm"
    bundle_override: Optional[str] = None
    stacking_only: bool = False


__all__ = ["MLInferenceConfig", "MLTrainingConfig"]
