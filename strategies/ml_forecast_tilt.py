"""ML-Forecast Tilt Demo Strategy (Phase D T-0229 + Phase E4 T-0328).

Reads ``ml_forecast_score`` per ticker from an external features
provider (default: ``synthetic_ml_forecast`` so the strategy stays
runnable without a trained model bundle), picks the top-N names whose
score exceeds ``score_cutoff`` and weights them via
``weighting in {"equal", "inverse_vol", "erc"}``. Falls back to an
equal-weight tilt over the full universe if no ticker meets the cutoff
— the demo never goes fully to cash.

Phase E4 (Codex R4.15) extends the strategy with the ``weighting``
parameter. With `weighting="equal"` the output is bit-identical to
Phase D.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.portfolio import erc_weights, inverse_vol_weights
from backtest.strategy import Allocation, Strategy


DEFAULT_UNIVERSE = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD")
DEFAULT_TOP_N = 5
DEFAULT_SCORE_CUTOFF = 0.0
DEFAULT_ML_DATASET = "synthetic_ml_forecast"
DEFAULT_WEIGHTING = "equal"
DEFAULT_TARGET_SUM = 1.0


class MLForecastTilt(Strategy):
    """Top-N tilt by ``ml_forecast_score``; equal/inverse_vol/erc weighting."""

    name = "ML Forecast Tilt"

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        top_n: int = DEFAULT_TOP_N,
        score_cutoff: float = DEFAULT_SCORE_CUTOFF,
        ml_dataset: Optional[str] = DEFAULT_ML_DATASET,
        weighting: str = DEFAULT_WEIGHTING,
        target_sum: float = DEFAULT_TARGET_SUM,
        max_weight: Optional[float] = None,
    ) -> None:
        if universe is None:
            universe = list(DEFAULT_UNIVERSE)
        self.assets = list(universe)
        weighting = str(weighting).strip().lower()
        if weighting not in {"equal", "inverse_vol", "erc"}:
            raise ValueError(
                "weighting must be 'equal', 'inverse_vol' or 'erc' (got "
                f"{weighting!r})"
            )
        self.params: Dict[str, Any] = {
            "universe": list(universe),
            "top_n": int(top_n),
            "score_cutoff": float(score_cutoff),
            "ml_dataset": ml_dataset,
            "weighting": weighting,
            "target_sum": float(target_sum),
            "max_weight": max_weight,
        }
        self._top_n = max(1, int(top_n))
        self._score_cutoff = float(score_cutoff)
        self._ml_dataset = ml_dataset
        self._weighting = weighting
        self._target_sum = float(target_sum)
        self._max_weight = (
            float(max_weight) if max_weight is not None else None
        )
        self.rebalance_frequency = "monthly"

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        usable_cols = [t for t in self.assets if t in data.columns]
        if not usable_cols:
            return Allocation({})

        scores = self._fetch_ml_scores(current_date, usable_cols)
        if not scores:
            # No provider configured / no rows yet — fall back to equal
            # weight over the full universe so the strategy keeps trading.
            return self._weight_kept(usable_cols, data)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        kept = [t for t, score in ranked if score >= self._score_cutoff][
            : self._top_n
        ]
        if not kept:
            # Fallback: all top-N rather than 100% cash.
            kept = [t for t, _ in ranked[: self._top_n]]
        return self._weight_kept(kept, data)

    def _weight_kept(
        self,
        kept: List[str],
        data: pd.DataFrame,
    ) -> Allocation:
        """Apply the chosen weighting scheme to the kept tickers."""

        if not kept:
            return Allocation({})

        if self._weighting == "equal":
            equal = self._target_sum / len(kept)
            return Allocation({t: equal for t in kept})

        # inverse_vol or erc need a returns stream for the picked tickers.
        returns = self._returns_for(kept, data)
        if returns is None or returns.empty:
            equal = self._target_sum / len(kept)
            return Allocation({t: equal for t in kept})

        if self._weighting == "inverse_vol":
            weights = inverse_vol_weights(returns, target_sum=self._target_sum)
        else:  # erc
            try:
                weights = erc_weights(
                    returns,
                    target_sum=self._target_sum,
                    max_weight=self._max_weight,
                )
            except ValueError:
                # Bounds infeasible -> fallback equal.
                equal = self._target_sum / len(kept)
                return Allocation({t: equal for t in kept})

        return Allocation(weights)

    @staticmethod
    def _returns_for(
        tickers: List[str],
        data: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        cols = [t for t in tickers if t in data.columns]
        if not cols:
            return None
        prices = data[cols].dropna(how="all")
        if len(prices) < 30:
            return None
        return prices.pct_change(fill_method=None).dropna(how="all")

    def _fetch_ml_scores(
        self,
        as_of: date,
        tickers: List[str],
    ) -> Dict[str, float]:
        provider = getattr(self, "_external_features_provider", None)
        if provider is None or not tickers or not self._ml_dataset:
            return {}
        try:
            if hasattr(provider, "snapshot_dataset"):
                snap = provider.snapshot_dataset(
                    self._ml_dataset, as_of=as_of, tickers=tickers
                )
            else:
                snap = provider.snapshot(as_of=as_of, tickers=tickers)
        except Exception:
            return {}
        if snap is None or getattr(snap, "data", None) is None:
            return {}
        df = snap.data
        if df.empty or "feature_name" not in df.columns:
            return {}
        rows = df.loc[df["feature_name"] == "ml_forecast_score"]
        scores: Dict[str, float] = {}
        for ticker, group in rows.groupby("ticker"):
            try:
                scores[str(ticker).upper()] = float(group["feature_value"].iloc[-1])
            except (TypeError, ValueError):
                continue
        return scores


strategy = MLForecastTilt()
