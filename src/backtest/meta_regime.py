"""Generic regime/fragility classification from strategy equity curves."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd


RegimeProfile = Literal["defensiv", "ausgewogen", "aggressiv", "custom"]
RegimeBucket = Literal[
    "normal", "news_stressed", "fragile", "stressed", "insufficient_history"
]


REFERENCE_DAYS = 756
MIN_HISTORY_DAYS = 252

# Phase C news-percentile parameters (Codex C11).
NEWS_PERCENTILE_TRAILING_DAYS = 63
NEWS_PERCENTILE_MIN_HISTORY_DAYS = 21


PROFILE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "defensiv": {
        "weak_lower_pct": 0.40,
        "risk_upper_pct": 0.60,
        # Phase C: news-stress thresholds. Defaults are intentionally
        # extreme so that the trigger NEVER fires when no news provider
        # is configured.
        "news_sentiment_lower_pct": 0.05,
        "news_dispersion_upper_pct": 0.95,
    },
    "ausgewogen": {
        "weak_lower_pct": 0.30,
        "risk_upper_pct": 0.70,
        "news_sentiment_lower_pct": 0.10,
        "news_dispersion_upper_pct": 0.90,
    },
    "aggressiv": {
        "weak_lower_pct": 0.20,
        "risk_upper_pct": 0.80,
        "news_sentiment_lower_pct": 0.15,
        "news_dispersion_upper_pct": 0.85,
    },
}


# Phase C: news_stressed slots between normal and fragile;
# insufficient_history stays at 99 (Codex C12).
REGIME_BUCKET_ORDER: Dict[str, int] = {
    "normal": 0,
    "news_stressed": 1,
    "fragile": 2,
    "stressed": 3,
    "insufficient_history": 99,
}


@dataclass(frozen=True)
class RegimeMeasurements:
    """Profile-independent regime measurements and percentiles."""

    status: Literal["ok", "insufficient_history"]
    metrics: Dict[str, float] = field(default_factory=dict)
    percentiles: Dict[str, float] = field(default_factory=dict)
    reference_days: int = 0
    available_feature_days: int = 0
    history_reason: Optional[str] = None
    # Phase B: universum-wide analyst_score dispersion at as_of. None
    # when no provider/dataset is configured or no data is available.
    analyst_dispersion: Optional[float] = None
    # Phase C: news-sentiment regime features (Codex C11). All optional;
    # default None means the news regime trigger never fires.
    news_sentiment_mean: Optional[float] = None
    news_sentiment_dispersion: Optional[float] = None
    news_count_zscore: Optional[float] = None
    news_history_days: Optional[int] = None
    news_sentiment_mean_percentile: Optional[float] = None
    news_dispersion_percentile: Optional[float] = None
    # Phase D: cross-sectional dispersion of ml_forecast_score at as_of.
    # Informational only (Codex Architektur-Entscheidung 10) — no
    # regime-bucket trigger is wired to this field.
    ml_dispersion: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "status": self.status,
            "metrics": dict(self.metrics),
            "percentiles": dict(self.percentiles),
            "reference_days": int(self.reference_days),
            "available_feature_days": int(self.available_feature_days),
            "history_reason": self.history_reason,
        }
        if self.analyst_dispersion is not None:
            payload["analyst_dispersion"] = float(self.analyst_dispersion)
        for key in (
            "news_sentiment_mean",
            "news_sentiment_dispersion",
            "news_count_zscore",
            "news_history_days",
            "news_sentiment_mean_percentile",
            "news_dispersion_percentile",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = float(value) if key != "news_history_days" else int(value)
        if self.ml_dispersion is not None:
            payload["ml_dispersion"] = float(self.ml_dispersion)
        return payload


@dataclass(frozen=True)
class RegimeSnapshot:
    """Profile-specific regime bucket plus reasons."""

    profile: RegimeProfile
    bucket: RegimeBucket
    status: Literal["ok", "insufficient_history"]
    flags: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    percentiles: Dict[str, float] = field(default_factory=dict)
    reference_days: int = 0
    available_feature_days: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "profile": self.profile,
            "bucket": self.bucket,
            "status": self.status,
            "flags": dict(self.flags),
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "percentiles": dict(self.percentiles),
            "reference_days": int(self.reference_days),
            "available_feature_days": int(self.available_feature_days),
        }


def normalize_regime_profile(profile: RegimeProfile) -> Literal["defensiv", "ausgewogen", "aggressiv"]:
    """Normalize custom profile to the balanced defaults until custom thresholds exist."""
    if profile == "custom":
        return "ausgewogen"
    if profile not in PROFILE_THRESHOLDS:
        raise ValueError(f"Unsupported regime profile: {profile}")
    return profile


def bucket_rank(bucket: Optional[str]) -> int:
    """Lower rank means more robust / preferable."""
    if bucket is None:
        return REGIME_BUCKET_ORDER["insufficient_history"]
    return REGIME_BUCKET_ORDER.get(bucket, REGIME_BUCKET_ORDER["insufficient_history"])


def _rolling_return(series: pd.Series, window: int) -> pd.Series:
    return series / series.shift(window) - 1.0


def _rolling_annualized_vol(series: pd.Series, window: int) -> pd.Series:
    returns = series.pct_change(fill_method=None)
    return returns.rolling(window=window, min_periods=window).std(ddof=0) * np.sqrt(252.0)


def _window_max_drawdown(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    running_peak = np.maximum.accumulate(arr)
    drawdowns = arr / running_peak - 1.0
    return float(np.nanmin(drawdowns))


def _rolling_max_drawdown(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).apply(_window_max_drawdown, raw=True)


def _percentile_rank(series: pd.Series, value: float) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty or not np.isfinite(value):
        return 0.0
    arr = cleaned.to_numpy(dtype=float)
    value = float(value)
    # Degenerate reference (near-constant distribution): exact-equality tie
    # detection at float-noise scale (~1e-15) is FP/SIMD-path dependent and makes
    # the downstream regime buckets flaky across machines. A value inside a
    # near-constant distribution ranks at the median (0.5).
    scale = max(abs(value), float(np.abs(arr).max()))
    if float(np.ptp(arr)) <= 1e-9 * scale + 1e-12:
        return 0.5
    lower = float((arr < value).mean())
    ties = float((arr == value).mean())
    return lower + 0.5 * ties


def build_regime_measurements(
    equity_curve: pd.Series,
    reference_days: int = REFERENCE_DAYS,
    min_history_days: int = MIN_HISTORY_DAYS,
    *,
    external_provider: Any = None,
    as_of: Optional[date] = None,
    universe: Optional[Iterable[str]] = None,
    analyst_datasets: Sequence[str] = (),
    news_datasets: Sequence[str] = (),
    ml_datasets: Sequence[str] = (),
) -> RegimeMeasurements:
    """Build profile-independent measurements for the latest available date.

    Phase B extension: when ``external_provider``, ``as_of`` and a
    non-empty ``universe`` are supplied, the resulting measurements
    include ``analyst_dispersion`` — the universe-wide std of
    ``analyst_score`` values at ``as_of`` for the first matching
    dataset in ``analyst_datasets`` (or the provider's default).

    Phase C extension: news_datasets activates the news-sentiment
    regime features (mean, dispersion, count z-score plus rolling
    percentiles via :meth:`ExternalFeaturesLoader.history`).
    """
    analyst_dispersion = _compute_analyst_dispersion(
        provider=external_provider,
        as_of=as_of,
        universe=universe,
        datasets=analyst_datasets,
    )
    news_metrics = _compute_news_regime_metrics(
        provider=external_provider,
        as_of=as_of,
        universe=universe,
        datasets=news_datasets,
    )
    ml_dispersion = _compute_ml_dispersion(
        provider=external_provider,
        as_of=as_of,
        universe=universe,
        datasets=ml_datasets,
    )

    if equity_curve is None:
        return RegimeMeasurements(
            status="insufficient_history",
            available_feature_days=0,
            history_reason="No equity curve available",
            analyst_dispersion=analyst_dispersion,
            **news_metrics,
            ml_dispersion=ml_dispersion,
        )

    equity = pd.to_numeric(pd.Series(equity_curve).dropna(), errors="coerce").dropna()
    if equity.empty:
        return RegimeMeasurements(
            status="insufficient_history",
            available_feature_days=0,
            history_reason="No valid equity values available",
            analyst_dispersion=analyst_dispersion,
            **news_metrics,
            ml_dispersion=ml_dispersion,
        )

    ret_21 = _rolling_return(equity, 21)
    ret_63 = _rolling_return(equity, 63)
    vol_63 = _rolling_annualized_vol(equity, 63)
    maxdd_126 = _rolling_max_drawdown(equity, 126)
    trend_spread = ret_21 - ret_63

    features = pd.DataFrame(
        {
            "ret_21": ret_21,
            "ret_63": ret_63,
            "vol_63": vol_63,
            "maxdd_126": maxdd_126,
            "trend_spread": trend_spread,
        }
    ).dropna()

    available = len(features)
    if available < int(min_history_days):
        return RegimeMeasurements(
            status="insufficient_history",
            reference_days=min(available, int(reference_days)),
            available_feature_days=available,
            history_reason=(
                f"Insufficient regime history: {available} feature days < {int(min_history_days)} required"
            ),
            analyst_dispersion=analyst_dispersion,
            **news_metrics,
            ml_dispersion=ml_dispersion,
        )

    reference = features.tail(min(int(reference_days), available))
    current = reference.iloc[-1]
    metrics = {
        "ret_21": float(current["ret_21"]),
        "ret_63": float(current["ret_63"]),
        "vol_63": float(current["vol_63"]),
        "maxdd_126": float(current["maxdd_126"]),
        "trend_spread": float(current["trend_spread"]),
    }
    percentiles = {
        "ret_21": _percentile_rank(reference["ret_21"], metrics["ret_21"]),
        "ret_63": _percentile_rank(reference["ret_63"], metrics["ret_63"]),
        "vol_63": _percentile_rank(reference["vol_63"], metrics["vol_63"]),
        "maxdd_126": _percentile_rank(reference["maxdd_126"].abs(), abs(metrics["maxdd_126"])),
        "trend_spread": _percentile_rank(reference["trend_spread"], metrics["trend_spread"]),
    }
    return RegimeMeasurements(
        status="ok",
        metrics=metrics,
        percentiles=percentiles,
        reference_days=len(reference),
        available_feature_days=available,
        analyst_dispersion=analyst_dispersion,
        **news_metrics,
        ml_dispersion=ml_dispersion,
    )


def _compute_news_regime_metrics(
    *,
    provider: Any,
    as_of: Optional[date],
    universe: Optional[Iterable[str]],
    datasets: Sequence[str],
) -> Dict[str, Optional[float]]:
    """Compute the news regime features at ``as_of`` (Phase C T-0114).

    Returns a dict with keys matching :class:`RegimeMeasurements`
    fields so the caller can splat it with ``**``. All keys default to
    ``None`` so this function never accidentally activates the
    news-stress trigger when no provider is configured.
    """

    payload: Dict[str, Optional[float]] = {
        "news_sentiment_mean": None,
        "news_sentiment_dispersion": None,
        "news_count_zscore": None,
        "news_history_days": None,
        "news_sentiment_mean_percentile": None,
        "news_dispersion_percentile": None,
    }
    if provider is None or as_of is None or not datasets:
        return payload
    tickers = [str(t).upper() for t in (universe or []) if t]
    if not tickers:
        return payload
    dataset = list(datasets)[0]

    # Current-snapshot mean and dispersion (mirror analyst pattern).
    try:
        if hasattr(provider, "snapshot_dataset"):
            snap = provider.snapshot_dataset(dataset, as_of=as_of, tickers=tickers)
        else:
            snap = provider.snapshot(as_of=as_of, tickers=tickers)
    except Exception:
        snap = None
    score_values: List[float] = []
    if snap is not None and getattr(snap, "data", None) is not None and not snap.data.empty:
        df = snap.data
        if "feature_name" in df.columns:
            rows = df.loc[df["feature_name"] == "news_sentiment_score"]
            score_values = (
                pd.to_numeric(rows["feature_value"], errors="coerce").dropna().tolist()
            )
            if score_values:
                payload["news_sentiment_mean"] = float(np.mean(score_values))
                if len(score_values) >= 2:
                    payload["news_sentiment_dispersion"] = float(
                        np.std(score_values, ddof=0)
                    )

    # History-based percentile + count z-score via loader.history().
    history_provider = provider
    if hasattr(provider, "get") and dataset:
        try:
            history_provider = provider.get(dataset)
        except Exception:
            history_provider = provider
    history_fn = getattr(history_provider, "history", None)
    if history_fn is None:
        return payload
    start = pd.Timestamp(as_of) - pd.Timedelta(days=NEWS_PERCENTILE_TRAILING_DAYS)
    try:
        score_history = history_fn(
            feature_name="news_sentiment_score",
            start=start.date(),
            end=as_of,
            tickers=tickers,
        )
    except TypeError:
        # loader.history doesn't accept tickers kwarg yet
        score_history = history_fn(
            feature_name="news_sentiment_score",
            start=start.date(),
            end=as_of,
        )
    except Exception:
        return payload
    if score_history is None or score_history.empty:
        return payload
    days_seen = score_history["release_date"].nunique()
    payload["news_history_days"] = int(days_seen)
    if days_seen < NEWS_PERCENTILE_MIN_HISTORY_DAYS:
        return payload

    daily_mean = (
        score_history.groupby("release_date")["feature_value"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean())
        .dropna()
    )
    if payload["news_sentiment_mean"] is not None and not daily_mean.empty:
        payload["news_sentiment_mean_percentile"] = _percentile_rank(
            daily_mean, float(payload["news_sentiment_mean"])
        )

    daily_disp = (
        score_history.groupby("release_date")["feature_value"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().std(ddof=0))
        .dropna()
    )
    if payload["news_sentiment_dispersion"] is not None and not daily_disp.empty:
        payload["news_dispersion_percentile"] = _percentile_rank(
            daily_disp, float(payload["news_sentiment_dispersion"])
        )

    # News count z-score: how many score-rows today vs trailing-history.
    try:
        count_history = history_fn(
            feature_name="news_article_count",
            start=start.date(),
            end=as_of,
            tickers=tickers,
        )
    except TypeError:
        count_history = history_fn(
            feature_name="news_article_count",
            start=start.date(),
            end=as_of,
        )
    except Exception:
        count_history = None
    if count_history is not None and not count_history.empty:
        daily_counts = (
            count_history.groupby("release_date")["feature_value"]
            .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().sum())
            .dropna()
        )
        if len(daily_counts) >= 2:
            mean_count = float(daily_counts.mean())
            std_count = float(daily_counts.std(ddof=0))
            today_count = float(daily_counts.iloc[-1])
            if std_count > 0:
                payload["news_count_zscore"] = (today_count - mean_count) / std_count
    return payload


def _compute_analyst_dispersion(
    provider: Any,
    as_of: Optional[date],
    universe: Optional[Iterable[str]],
    datasets: Sequence[str],
) -> Optional[float]:
    """Return the std of analyst_score values across ``universe`` at ``as_of``.

    Returns ``None`` if the provider, as_of or universe are missing or
    yield no analyst rows.
    """
    if provider is None or as_of is None or universe is None:
        return None
    tickers = [str(t).upper() for t in universe if t]
    if not tickers:
        return None
    dataset = list(datasets)[0] if datasets else None
    try:
        if dataset and hasattr(provider, "snapshot_dataset"):
            snap = provider.snapshot_dataset(dataset, as_of=as_of, tickers=tickers)
        else:
            snap = provider.snapshot(as_of=as_of, tickers=tickers)
    except Exception:
        return None
    if snap is None or getattr(snap, "data", None) is None:
        return None
    df = snap.data
    if df.empty or "feature_name" not in df.columns:
        return None
    score_rows = df.loc[df["feature_name"] == "analyst_score"]
    if score_rows.empty:
        return None
    values = pd.to_numeric(score_rows["feature_value"], errors="coerce").dropna()
    if len(values) < 2:
        return None
    return float(values.std(ddof=0))


def _compute_ml_dispersion(
    *,
    provider: Any,
    as_of: Optional[date],
    universe: Optional[Iterable[str]],
    datasets: Sequence[str],
) -> Optional[float]:
    """Return the std of ``ml_forecast_score`` values across ``universe`` (Codex D, T-0219).

    Informational only — Phase D explicitly does NOT add a regime bucket
    trigger from ml_dispersion (Architektur-Entscheidung 10). Caller can
    log/audit but no profile reacts to it. Returns ``None`` when the
    provider, as_of, universe or dataset cannot yield ml_forecast_score
    rows.
    """

    if provider is None or as_of is None or universe is None or not datasets:
        return None
    tickers = [str(t).upper() for t in universe if t]
    if not tickers:
        return None
    dataset = list(datasets)[0]
    try:
        if hasattr(provider, "snapshot_dataset"):
            snap = provider.snapshot_dataset(dataset, as_of=as_of, tickers=tickers)
        else:
            snap = provider.snapshot(as_of=as_of, tickers=tickers)
    except Exception:
        return None
    if snap is None or getattr(snap, "data", None) is None:
        return None
    df = snap.data
    if df.empty or "feature_name" not in df.columns:
        return None
    rows = df.loc[df["feature_name"] == "ml_forecast_score"]
    if rows.empty:
        return None
    values = pd.to_numeric(rows["feature_value"], errors="coerce").dropna()
    if len(values) < 2:
        return None
    return float(values.std(ddof=0))


def classify_regime_measurements(
    measurements: RegimeMeasurements,
    profile: RegimeProfile = "ausgewogen",
) -> RegimeSnapshot:
    """Classify measurements into normal/fragile/stressed bucket."""
    normalized_profile = normalize_regime_profile(profile)
    if measurements.status != "ok":
        reason = measurements.history_reason or "Insufficient regime history"
        return RegimeSnapshot(
            profile=profile,
            bucket="insufficient_history",
            status="insufficient_history",
            flags={},
            reasons=[reason],
            metrics=dict(measurements.metrics),
            percentiles=dict(measurements.percentiles),
            reference_days=measurements.reference_days,
            available_feature_days=measurements.available_feature_days,
        )

    thresholds = PROFILE_THRESHOLDS[normalized_profile]
    lower = thresholds["weak_lower_pct"]
    upper = thresholds["risk_upper_pct"]
    news_lower = float(thresholds.get("news_sentiment_lower_pct", 0.0))
    news_upper = float(thresholds.get("news_dispersion_upper_pct", 1.0))
    percentiles = measurements.percentiles
    metrics = measurements.metrics

    flags = {
        "weak_short_momentum": percentiles.get("ret_21", 0.0) <= lower,
        "trend_deterioration": percentiles.get("trend_spread", 0.0) <= lower,
        "high_volatility": percentiles.get("vol_63", 0.0) >= upper,
        "deep_drawdown": percentiles.get("maxdd_126", 0.0) >= upper,
    }

    reasons: List[str] = []
    if flags["weak_short_momentum"]:
        reasons.append(
            f"Weak short momentum: ret_21={metrics.get('ret_21', 0.0):+.1%}, pct={percentiles.get('ret_21', 0.0):.0%}"
        )
    if flags["trend_deterioration"]:
        reasons.append(
            f"Trend deterioration: spread={metrics.get('trend_spread', 0.0):+.1%}, pct={percentiles.get('trend_spread', 0.0):.0%}"
        )
    if flags["high_volatility"]:
        reasons.append(
            f"High volatility: vol_63={metrics.get('vol_63', 0.0):.1%}, pct={percentiles.get('vol_63', 0.0):.0%}"
        )
    if flags["deep_drawdown"]:
        reasons.append(
            f"Deep drawdown: maxdd_126={metrics.get('maxdd_126', 0.0):.1%}, pct={percentiles.get('maxdd_126', 0.0):.0%}"
        )

    # Phase C news-stress flag (Codex C11): only fires when sufficient
    # trailing history is available and both a sentiment-percentile and
    # a dispersion-percentile reach their respective thresholds.
    news_history_days = measurements.news_history_days
    news_sentiment_pct = measurements.news_sentiment_mean_percentile
    news_dispersion_pct = measurements.news_dispersion_percentile
    news_stress = False
    if (
        news_history_days is not None
        and news_history_days >= NEWS_PERCENTILE_MIN_HISTORY_DAYS
        and news_sentiment_pct is not None
        and news_dispersion_pct is not None
    ):
        news_stress = (
            news_sentiment_pct <= news_lower and news_dispersion_pct >= news_upper
        )
    flags["news_sentiment_stress"] = news_stress
    if news_stress:
        reasons.append(
            "News sentiment stress: "
            f"mean_pct={news_sentiment_pct:.0%}<= {news_lower:.0%}, "
            f"dispersion_pct={news_dispersion_pct:.0%}>= {news_upper:.0%}"
        )

    # Bucket logic: only `news_sentiment_stress` -> `news_stressed`;
    # combined with classical fragility flags it folds into
    # `fragile`/`stressed` as before (Codex C15 / Plan).
    classical_flag_count = sum(
        1
        for name, value in flags.items()
        if value and name != "news_sentiment_stress"
    )
    if classical_flag_count == 0 and not news_stress:
        bucket: RegimeBucket = "normal"
        reasons = ["No fragility flags triggered"]
    elif classical_flag_count == 0 and news_stress:
        bucket = "news_stressed"
    elif classical_flag_count == 1:
        bucket = "fragile"
    else:
        bucket = "stressed"

    return RegimeSnapshot(
        profile=profile,
        bucket=bucket,
        status="ok",
        flags=flags,
        reasons=reasons,
        metrics=dict(metrics),
        percentiles=dict(percentiles),
        reference_days=measurements.reference_days,
        available_feature_days=measurements.available_feature_days,
    )


def assess_regime_from_equity_curve(
    equity_curve: pd.Series,
    profile: RegimeProfile = "ausgewogen",
    reference_days: int = REFERENCE_DAYS,
    min_history_days: int = MIN_HISTORY_DAYS,
) -> RegimeSnapshot:
    """Convenience wrapper: measure and classify latest regime from equity curve."""
    measurements = build_regime_measurements(
        equity_curve=equity_curve,
        reference_days=reference_days,
        min_history_days=min_history_days,
    )
    return classify_regime_measurements(measurements, profile=profile)
