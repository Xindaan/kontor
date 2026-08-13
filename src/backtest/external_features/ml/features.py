"""Phase D feature-matrix builder (T-0203, Codex D13).

Combines price-derived technical features, fundamentals, analyst- and
news-snapshot scores into a single long DataFrame indexed by
``(as_of, ticker)``. The class enforces a Point-in-Time contract:

- For every row ``(as_of, ticker)`` every external feature is taken
  from ``snapshot(as_of)`` or ``history(... <= as_of)``.
- The median imputer is fit ONLY on the training window. The fitted
  medians are stored in :class:`FeatureMatrixState` and reused at
  inference time without recomputation (Leakage-Schutz).

The builder deliberately keeps zero hard dependency on lightgbm/xgboost
so unit tests can drive it with the synthetic adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


PRICE_FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_21d",
    "ret_63d",
    "ret_252d",
    "vol_63d",
    "maxdd_126d",
    "trend_spread",
)

EXTERNAL_FEATURE_SCORE_NAMES: tuple[str, ...] = (
    "analyst_score",
    "news_sentiment_score",
    "news_sentiment_dispersion",
)


@dataclass
class FeatureMatrixState:
    """Persisted state used by inference (Codex D13)."""

    feature_columns: List[str] = field(default_factory=list)
    imputer_medians: Dict[str, float] = field(default_factory=dict)
    tickers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_columns": list(self.feature_columns),
            "imputer_medians": dict(self.imputer_medians),
            "tickers": list(self.tickers),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FeatureMatrixState":
        return cls(
            feature_columns=list(payload.get("feature_columns") or []),
            imputer_medians={
                str(k): float(v)
                for k, v in (payload.get("imputer_medians") or {}).items()
            },
            tickers=list(payload.get("tickers") or []),
        )


class FeatureMatrixBuilder:
    """Build feature matrices for the ML training and inference paths.

    Concrete loaders are passed in by the caller, so tests can inject
    synthetic snapshots without touching disk.
    """

    def __init__(
        self,
        *,
        fundamentals_loader=None,
        external_features_provider=None,
        analyst_dataset: Optional[str] = None,
        news_dataset: Optional[str] = None,
        sector_lookup: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._fundamentals_loader = fundamentals_loader
        self._external_features_provider = external_features_provider
        self._analyst_dataset = analyst_dataset
        self._news_dataset = news_dataset
        self._sector_lookup = dict(sector_lookup or {})

    # ------------------------------------------------------------------
    # Fit (training) path
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        prices: pd.DataFrame,
        as_of_index: Iterable[pd.Timestamp],
        tickers: Sequence[str],
    ) -> tuple[pd.DataFrame, FeatureMatrixState]:
        """Build a long feature frame and the persisted imputer state."""

        raw = self._collect_rows(prices, as_of_index, tickers)
        if raw.empty:
            return raw, FeatureMatrixState(feature_columns=[], tickers=list(tickers))
        feature_cols = [
            col for col in raw.columns if col not in {"as_of", "ticker"}
        ]
        medians = {
            col: float(pd.to_numeric(raw[col], errors="coerce").median())
            if raw[col].notna().any()
            else 0.0
            for col in feature_cols
        }
        imputed = raw.copy()
        for col in feature_cols:
            value = medians[col]
            if not math.isfinite(value):
                value = 0.0
                medians[col] = 0.0
            imputed[col] = (
                pd.to_numeric(imputed[col], errors="coerce").fillna(value).astype(float)
            )
        state = FeatureMatrixState(
            feature_columns=feature_cols,
            imputer_medians=medians,
            tickers=list(dict.fromkeys(str(t).upper() for t in tickers)),
        )
        return imputed, state

    # ------------------------------------------------------------------
    # Transform (inference) path
    # ------------------------------------------------------------------

    def transform(
        self,
        prices: pd.DataFrame,
        as_of_index: Iterable[pd.Timestamp],
        tickers: Sequence[str],
        state: FeatureMatrixState,
    ) -> pd.DataFrame:
        """Apply a fitted state to a fresh feature window."""

        raw = self._collect_rows(prices, as_of_index, tickers)
        if raw.empty:
            return raw
        out = raw.copy()
        for col in state.feature_columns:
            if col not in out.columns:
                out[col] = state.imputer_medians.get(col, 0.0)
            value = state.imputer_medians.get(col, 0.0)
            out[col] = (
                pd.to_numeric(out[col], errors="coerce").fillna(value).astype(float)
            )
        # Drop accidental extra columns to keep feature order stable.
        keep = ["as_of", "ticker", *state.feature_columns]
        for required in keep:
            if required not in out.columns:
                out[required] = 0.0
        return out[keep]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _collect_rows(
        self,
        prices: pd.DataFrame,
        as_of_index: Iterable[pd.Timestamp],
        tickers: Sequence[str],
    ) -> pd.DataFrame:
        if prices is None or prices.empty or not list(tickers):
            return pd.DataFrame(columns=["as_of", "ticker"])

        tickers_upper = [str(t).upper() for t in tickers]
        available = [t for t in tickers_upper if t in prices.columns]
        if not available:
            return pd.DataFrame(columns=["as_of", "ticker"])

        as_of_list: List[pd.Timestamp] = []
        for raw in as_of_index:
            ts = pd.Timestamp(raw)
            if not pd.isna(ts):
                as_of_list.append(ts)
        if not as_of_list:
            return pd.DataFrame(columns=["as_of", "ticker"])

        price_index = pd.DatetimeIndex(prices.index)
        rows: List[dict] = []
        for as_of_ts in as_of_list:
            pos = price_index.searchsorted(as_of_ts, side="right") - 1
            if pos < 0:
                continue
            window = prices.iloc[: pos + 1]
            ext_rows = self._collect_external_features(as_of_ts, available)
            for ticker in available:
                series = pd.to_numeric(window[ticker], errors="coerce").dropna()
                row: Dict[str, object] = {
                    "as_of": as_of_ts,
                    "ticker": ticker,
                }
                row.update(self._compute_price_features(series))
                row.update(ext_rows.get(ticker, {}))
                if self._sector_lookup:
                    row["sector"] = self._sector_lookup.get(ticker, "OTHER")
                rows.append(row)
        if not rows:
            return pd.DataFrame(columns=["as_of", "ticker"])
        frame = pd.DataFrame(rows)
        # Stable column ordering (price features first).
        cols = ["as_of", "ticker"]
        for c in PRICE_FEATURE_COLUMNS:
            if c in frame.columns:
                cols.append(c)
        for c in EXTERNAL_FEATURE_SCORE_NAMES:
            if c in frame.columns:
                cols.append(c)
        for c in frame.columns:
            if c not in cols:
                cols.append(c)
        return frame[cols].sort_values(["as_of", "ticker"]).reset_index(drop=True)

    def _compute_price_features(self, series: pd.Series) -> Dict[str, float]:
        if series.empty:
            return {col: float("nan") for col in PRICE_FEATURE_COLUMNS}
        values = series.to_numpy()
        last = values[-1]
        out: Dict[str, float] = {}
        out["ret_21d"] = float(values[-1] / values[-22] - 1.0) if len(values) > 21 else float("nan")
        out["ret_63d"] = float(values[-1] / values[-64] - 1.0) if len(values) > 63 else float("nan")
        out["ret_252d"] = (
            float(values[-1] / values[-253] - 1.0) if len(values) > 252 else float("nan")
        )
        if len(values) > 63:
            returns = series.pct_change().tail(63).dropna()
            out["vol_63d"] = float(returns.std(ddof=0) * math.sqrt(252.0))
        else:
            out["vol_63d"] = float("nan")
        if len(values) > 126:
            window = series.tail(126)
            cummax = window.cummax()
            drawdown = (window / cummax - 1.0).min()
            out["maxdd_126d"] = float(drawdown)
        else:
            out["maxdd_126d"] = float("nan")
        ret21 = out["ret_21d"]
        ret63 = out["ret_63d"]
        out["trend_spread"] = (
            float(ret21 - ret63) if math.isfinite(ret21) and math.isfinite(ret63) else float("nan")
        )
        return out

    def _collect_external_features(
        self,
        as_of_ts: pd.Timestamp,
        tickers: Sequence[str],
    ) -> Dict[str, Dict[str, float]]:
        if self._external_features_provider is None:
            return {}
        per_ticker: Dict[str, Dict[str, float]] = {t: {} for t in tickers}
        analyst = self._fetch_snapshot(self._analyst_dataset, as_of_ts.date(), tickers)
        if analyst is not None and not analyst.empty:
            self._merge_score_rows(per_ticker, analyst, "analyst_score")
        news = self._fetch_snapshot(self._news_dataset, as_of_ts.date(), tickers)
        if news is not None and not news.empty:
            self._merge_score_rows(per_ticker, news, "news_sentiment_score")
            self._merge_score_rows(per_ticker, news, "news_sentiment_dispersion")
        return per_ticker

    def _fetch_snapshot(
        self,
        dataset: Optional[str],
        as_of: date,
        tickers: Sequence[str],
    ):
        if not dataset or self._external_features_provider is None:
            return None
        provider = self._external_features_provider
        try:
            if hasattr(provider, "snapshot_dataset"):
                snap = provider.snapshot_dataset(dataset, as_of=as_of, tickers=list(tickers))
            else:
                snap = provider.snapshot(as_of=as_of, tickers=list(tickers))
        except Exception:
            return None
        df = getattr(snap, "data", None) if snap is not None else None
        return df

    @staticmethod
    def _merge_score_rows(
        per_ticker: Dict[str, Dict[str, float]],
        frame: pd.DataFrame,
        feature_name: str,
    ) -> None:
        if frame is None or frame.empty or "feature_name" not in frame.columns:
            return
        rows = frame.loc[frame["feature_name"] == feature_name]
        if rows.empty:
            return
        for ticker, group in rows.groupby("ticker"):
            try:
                value = float(group["feature_value"].iloc[-1])
            except (TypeError, ValueError):
                continue
            target = per_ticker.setdefault(str(ticker).upper(), {})
            target[feature_name] = value


__all__ = [
    "EXTERNAL_FEATURE_SCORE_NAMES",
    "FeatureMatrixBuilder",
    "FeatureMatrixState",
    "PRICE_FEATURE_COLUMNS",
]
