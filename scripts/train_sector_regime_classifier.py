"""Trains the sector regime classifier (ML Round 3 Phase 3A).

Walk-forward training of a LightGBM 3-class classifier (normal /
fragile / stressed) on 10 SPDR US sector ETFs (XLK, XLF, XLY, XLV,
XLE, XLI, XLP, XLB, XLU, XLRE) with ~24 years of history starting 2001.

Output: ~29 bundles under
``data/external_features/ml/models_regime/sectors/<holdout_start>/lightgbm/``
with ``manifest.json`` + ``imputer_state.json`` + ``classifier.pkl``.
Per-bundle mean AUC stress across walk-forward: ~0.625 (backtest-
validated value from the 2026-05-15 run).

Bundle PKLs are gitignored — JSON manifests are committed (see
``data/external_features/ml/models/semiconductor/`` as a template).
Reproducible by re-running this script (deterministic,
seed=42).

Prerequisites: ``poetry install --with ml`` or
``.venv/bin/pip install lightgbm scikit-learn``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.data import DataLoader
from backtest.external_features.ml.regime_classifier_training import (
    RegimeClassifierConfig,
    run_walk_forward_regime_training,
)

# US SPDR sector ETFs (10 classics, all since 1998)
SECTOR_TICKERS = (
    "XLK", "XLF", "XLY", "XLV", "XLE",
    "XLI", "XLP", "XLB", "XLU", "XLRE",
)

DEFAULT_OUTPUT_DIR = Path("data/external_features/ml/models_regime/sectors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", default="2001-01-01", help="Trainings-Start (default 2001-01-01)"
    )
    parser.add_argument(
        "--end", default="2024-12-31", help="Trainings-Ende (default 2024-12-31)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Bundle-Output-Verzeichnis (default data/external_features/ml/models_regime/sectors)",
    )
    parser.add_argument(
        "--outer-train-years", type=float, default=4.0,
        help="Outer-Training-Window in Jahren (default 4.0)",
    )
    parser.add_argument(
        "--outer-holdout-months", type=int, default=6,
        help="Outer-Holdout-Window in Monaten (default 6)",
    )
    parser.add_argument(
        "--label-horizon-days", type=int, default=21,
        help="Forward-Label-Horizont in Handelstagen (default 21)",
    )
    args = parser.parse_args()

    print(f"Loading {len(SECTOR_TICKERS)} sector ETFs {args.start}..{args.end} ...")
    t0 = time.time()
    pd_obj = DataLoader.yahoo(
        tickers=list(SECTOR_TICKERS),
        start=args.start,
        end=args.end,
        currency="EUR",
        align="ffill",
        skip_failed=True,
    )
    prices = pd_obj.prices
    print(f"  loaded in {time.time() - t0:.0f}s: shape={prices.shape}, "
          f"tickers={sorted(prices.columns)}")
    if len(prices.columns) < 5:
        raise RuntimeError(
            f"Zu wenig Sektoren geladen ({len(prices.columns)}/10). "
            "Yahoo-Daten unvollstaendig — Trainingslauf abgebrochen."
        )

    config = RegimeClassifierConfig(
        tickers=tuple(prices.columns),
        label_horizon_days=args.label_horizon_days,
        outer_train_years=args.outer_train_years,
        outer_holdout_months=args.outer_holdout_months,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting walk-forward training -> {out_dir} ...")
    t0 = time.time()
    result = run_walk_forward_regime_training(prices, config, output_dir=out_dir)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s, {len(result.manifest_paths)} bundles erzeugt.")
    if not result.manifest_paths:
        raise RuntimeError("Keine Bundles erzeugt - check Daten + Config.")
    print(f"  Erstes Bundle: {result.manifest_paths[0]}")
    print(f"  Letztes Bundle: {result.manifest_paths[-1]}")


if __name__ == "__main__":
    main()
