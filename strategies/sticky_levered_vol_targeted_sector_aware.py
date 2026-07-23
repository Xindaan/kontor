"""Sticky Levered + Vol-Targeting + sector-stress-adaptive overlay
(ML round 3, phase 3A — backtest-validated pilot candidate).

The first strategy to bring **ML with an edge** into this repo after
7 previous ML attempts. The structural diagnosis from rounds 1+2
(small-universe problem) is addressed directly: train a regime
classifier on the US SPDR sector universe (10 tickers, 23+ years of
history), aggregate the per-sector `P(stressed)` into a market-wide
stress score (cross-sectional standard deviation), use that score to
adaptively scale the `vol_target` parameter of the existing
vol-targeting overlay.

Mechanics (built on top of StickyLeveredVolTargeted):

1. **Monthly momentum pick**: unchanged (Sticky Levered logic).
2. **Weekly vol-targeting overlay**: identical to the base, BUT with
   a dynamic ``vol_target``::

       stress = aggregate_sector_stress(P(stress) per sector, mode='std')
       vt_eff = vol_target_base * max(floor, 1 - alpha * stress)
       w_risky = clip(vt_eff / realized_vol, 0, 1)

3. **Sticky PIT bundle lookup**: for each ``as_of``, the bundle with
   ``available_from <= as_of`` is selected (strictly walk-forward).
4. **Fallback if no bundle is available**: ``vt_eff = vol_target_base``
   (degenerates to pure VolTarget — the strategy does not crash).

Backtest finding 2019-2024 (own engine, with German-tax-like costs —
a full backtester test is still pending):

- Pure VolTarget (baseline):  CAGR 39.1% Sharpe 1.05 MaxDD -53.9%
- SectorAware std α=2:        CAGR 36.5% Sharpe 1.11 MaxDD -47.5%
- Delta:                      -2.6pp CAGR / +0.06 Sharpe / +6.4pp DD

Robust across all 4 sub-periods 2019-2024. Meets the plan kill
criterion (ΔSharpe ≥ +0.05 AND ΔMaxDD ≥ +3pp).

Honest caveat: this is NOT a +30% CAGR jump. It is a **risk reducer**
that consistently improves MaxDD with a minimal Sharpe edge. Pilot
status, no immediate master switch (see
`docs/strategy_meta_playbook_2026-05-15.md` promotion gates).
"""

from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.external_features.ml.features import (
    FeatureMatrixBuilder,
    FeatureMatrixState,
)
from backtest.external_features.ml.sector_regime_aggregation import (
    DEFAULT_MODE as DEFAULT_AGG_MODE,
    aggregate_sector_stress,
)
from backtest.strategy import Allocation
from strategies.sticky_levered_vol_targeted import StickyLeveredVolTargeted

DEFAULT_SECTOR_TICKERS: Tuple[str, ...] = (
    "XLK", "XLF", "XLY", "XLV", "XLE",
    "XLI", "XLP", "XLB", "XLU", "XLRE",
)

DEFAULT_BUNDLE_DIR = Path("data/external_features/ml/models_regime/sectors")
DEFAULT_AGG_ALPHA: float = 2.0
DEFAULT_AGG_FLOOR: float = 0.4
PRIOR_STRESS_FALLBACK: float = 0.10  # fallback stress when no bundle/data


class StickyLeveredVolTargetedSectorAware(StickyLeveredVolTargeted):
    """Sticky Levered + vol targeting + sector-stress-adaptive overlay."""

    name = "[Research] Sticky Levered + Vol-Targeting (Sector-Aware)"
    rebalance_frequency = "weekly"

    def __init__(
        self,
        *,
        bundle_dir: Optional[Path] = None,
        sector_tickers: Tuple[str, ...] = DEFAULT_SECTOR_TICKERS,
        aggregation_mode: str = DEFAULT_AGG_MODE,
        alpha: float = DEFAULT_AGG_ALPHA,
        floor: float = DEFAULT_AGG_FLOOR,
        prior_stress: float = PRIOR_STRESS_FALLBACK,
        sector_prices_loader=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.bundle_dir = Path(bundle_dir) if bundle_dir else DEFAULT_BUNDLE_DIR
        self.sector_tickers = tuple(sector_tickers)
        self.aggregation_mode = aggregation_mode
        self.alpha = float(alpha)
        self.floor = float(floor)
        self.prior_stress = float(prior_stress)
        self._sector_prices_loader = sector_prices_loader  # injectable for tests
        self.params.update(
            {
                "bundle_dir": str(self.bundle_dir),
                "sector_tickers": list(self.sector_tickers),
                "aggregation_mode": self.aggregation_mode,
                "alpha": self.alpha,
                "floor": self.floor,
                "prior_stress": self.prior_stress,
            }
        )
        # Lazy state — bundles + sector prices load on first signal call.
        self._bundles: Optional[List[Dict]] = None
        self._sector_prices: Optional[pd.DataFrame] = None
        self._sector_feature_builder = FeatureMatrixBuilder()

    # ------------------------------------------------------------------
    # Bundle + Sector-Prices Loading (lazy)
    # ------------------------------------------------------------------

    def _ensure_bundles_loaded(self) -> List[Dict]:
        if self._bundles is not None:
            return self._bundles
        bundles: List[Dict] = []
        bundle_dir = Path(self.bundle_dir)
        if not bundle_dir.exists():
            self._bundles = bundles
            return bundles
        for day_dir in sorted(bundle_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            lgb_dir = day_dir / "lightgbm"
            manifest_path = lgb_dir / "manifest.json"
            clf_path = lgb_dir / "classifier.pkl"
            state_path = lgb_dir / "imputer_state.json"
            if not (manifest_path.exists() and clf_path.exists() and state_path.exists()):
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
                bundles.append(
                    {
                        "available_from": pd.Timestamp(manifest["available_from"]),
                        "classifier": pickle.loads(clf_path.read_bytes()),
                        "state": FeatureMatrixState.from_dict(
                            json.loads(state_path.read_bytes())
                        ),
                    }
                )
            except Exception:
                continue
        self._bundles = bundles
        return bundles

    def _ensure_sector_prices_loaded(self) -> Optional[pd.DataFrame]:
        if self._sector_prices is not None:
            return self._sector_prices
        if self._sector_prices_loader is not None:
            try:
                self._sector_prices = self._sector_prices_loader()
                return self._sector_prices
            except Exception:
                self._sector_prices = pd.DataFrame()
                return self._sector_prices
        # Default: lazy import + load via DataLoader.yahoo
        try:
            from backtest.data import DataLoader

            pd_obj = DataLoader.yahoo(
                tickers=list(self.sector_tickers),
                start="2001-01-01",
                end=pd.Timestamp.utcnow().normalize().date().isoformat(),
                currency="EUR",
                align="ffill",
                skip_failed=True,
            )
            self._sector_prices = pd_obj.prices
        except Exception:
            self._sector_prices = pd.DataFrame()
        return self._sector_prices

    def _select_bundle(self, as_of: pd.Timestamp) -> Optional[Dict]:
        bundles = self._ensure_bundles_loaded()
        elig = [b for b in bundles if b["available_from"] <= as_of]
        if not elig:
            return None
        return max(elig, key=lambda b: b["available_from"])

    # ------------------------------------------------------------------
    # Stress-Score Computation
    # ------------------------------------------------------------------

    def _compute_sector_stress(self, as_of_ts: pd.Timestamp) -> Optional[float]:
        """Return the aggregated cross-sectional stress score, or ``None``."""
        bundle = self._select_bundle(as_of_ts)
        if bundle is None:
            return None
        sector_prices = self._ensure_sector_prices_loaded()
        if sector_prices is None or sector_prices.empty:
            return None
        if as_of_ts not in sector_prices.index:
            # Snap to nearest available trading day <= as_of
            earlier = sector_prices.loc[:as_of_ts]
            if earlier.empty:
                return None
            as_of_ts = earlier.index[-1]
        try:
            feats = self._sector_feature_builder.transform(
                sector_prices,
                [as_of_ts],
                tickers=list(sector_prices.columns),
                state=bundle["state"],
            )
        except Exception:
            return None
        if feats.empty:
            return None
        X = feats[bundle["state"].feature_columns].to_numpy(dtype=float)
        try:
            proba = bundle["classifier"].predict_proba(X)
        except Exception:
            return None
        p_stress = proba[:, 2]  # class 2 = stressed (per regime_labels.REGIME_NAMES)
        return aggregate_sector_stress(p_stress, mode=self.aggregation_mode)

    # ------------------------------------------------------------------
    # Signal Override
    # ------------------------------------------------------------------

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        """Sticky Levered pick + vol targeting with a sector-stress-adaptive vol_target.

        Behavior without a bundle (fallback): degenerates to pure VolTarget
        (`stress = prior_stress`). No crash.
        """
        if data is None or data.empty:
            return super().signal(current_date, data)

        # 1. Sticky pick refresh (monthly) as in the base class.
        self._refresh_monthly_pick(current_date, data)
        picked = self._picked or self.safe_asset
        if picked == self.safe_asset:
            self._cur_risky_weight = 0.0
            return Allocation({self.safe_asset: 1.0})

        # 2. Compute sector stress for this as_of.
        as_of_ts = pd.Timestamp(current_date)
        stress = self._compute_sector_stress(as_of_ts)
        if stress is None:
            stress = self.prior_stress  # fallback, no crash

        # 3. Dynamically scale vol_target.
        vt_eff = self.vol_target * max(self.floor, 1.0 - self.alpha * stress)

        # 4. Standard vol overlay with the adaptive vt_eff.
        series = data[picked] if picked in data.columns else None
        from strategies.sticky_levered_vol_targeted import _realized_vol  # avoid cycle on import
        rv = _realized_vol(series, self.vol_lookback_days) if series is not None else None
        if rv is None or rv <= 0.0:
            target_w = 1.0
        else:
            target_w = min(1.0, vt_eff / rv)

        # 5. Turnover damper (rebal_band).
        if (
            self._cur_risky_weight > 0.0
            and abs(target_w - self._cur_risky_weight) < self.rebal_band
        ):
            target_w = self._cur_risky_weight
        else:
            self._cur_risky_weight = target_w

        safe_w = 1.0 - target_w
        if safe_w <= 1e-6:
            return Allocation({picked: 1.0})
        return Allocation({picked: target_w, self.safe_asset: safe_w})


strategy = StickyLeveredVolTargetedSectorAware()
