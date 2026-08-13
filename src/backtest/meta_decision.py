"""Strategy-agnostic meta decision engine for live signals."""

from __future__ import annotations

import json
import math
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backtest.backtester import BacktestConfig, Backtester
from backtest.data import DataLoader
from backtest.meta_evidence import (
    DEFAULT_DECISION_LOG,
    EvidenceProfile,
    assess_analyst_evidence,
    assess_conditioned_analyst_evidence_status,
    assess_conditioned_ml_evidence_status,
    assess_conditioned_news_evidence_status,
    assess_evidence_status,
    assess_conditioned_evidence_status,
    assess_ml_evidence,
    assess_news_evidence,
    append_meta_decision_log,
    find_latest_evidence_artifact,
    load_recent_meta_decisions,
    resolve_conditioned_min_windows,
)
from backtest.meta_cross_product import (
    ComponentEvidence,
    CrossProductGateResult,
    evaluate_cross_product_gate,
)
from backtest.meta_regime import assess_regime_from_equity_curve, bucket_rank
from backtest.signals import Portfolio, SignalGenerator
from backtest.web.services.strategies import load_strategy_instance


ParamsSource = Literal["preset_first", "strategy_defaults", "manual_only"]
ScoringMode = Literal["hybrid", "gate_only", "performance_only"]
DecisionCadence = Literal["run_check_rebalance_switch", "immediate", "monthly_fixed"]
PlanMode = Literal["recommendation_only", "recommendation_with_portfolio_plan", "always_plan"]
GateFailAction = Literal["hold_current", "manual_override", "fallback_safe"]
RegimeMode = Literal["none", "strategy_fragility"]


DEFAULT_PERFORMANCE_WINDOWS_DAYS: List[int] = [63, 126, 252]
DEFAULT_PERFORMANCE_WEIGHTS: List[float] = [0.5, 0.3, 0.2]
REGIME_TOLERANCE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "defensiv": {
        "alpha_tie_band": 0.02,
        "stress_alpha_tolerance": 0.03,
        "conditioned_min_windows": 6,
        # Phase B: analyst_score weight default 0.0 -> back-compat.
        "analyst_score_weight": 0.0,
        # Phase C news_score_weight default 0.0 -> back-compat.
        "news_score_weight": 0.0,
        # Phase D ml_score_weight default 0.0 -> back-compat (T-0220).
        "ml_score_weight": 0.0,
        # Phase E1 cross_product_consensus_threshold (Codex R3.3 / T-0303).
        "cross_product_consensus_threshold": 0.7,
    },
    "ausgewogen": {
        "alpha_tie_band": 0.03,
        "stress_alpha_tolerance": 0.05,
        "conditioned_min_windows": 4,
        "analyst_score_weight": 0.0,
        "news_score_weight": 0.0,
        "ml_score_weight": 0.0,
        "cross_product_consensus_threshold": 0.5,
    },
    "aggressiv": {
        "alpha_tie_band": 0.05,
        "stress_alpha_tolerance": 0.08,
        "conditioned_min_windows": 3,
        "analyst_score_weight": 0.0,
        "news_score_weight": 0.0,
        "ml_score_weight": 0.0,
        "cross_product_consensus_threshold": 0.3,
    },
    "custom": {
        "alpha_tie_band": 0.03,
        "stress_alpha_tolerance": 0.05,
        "conditioned_min_windows": 4,
        "analyst_score_weight": 0.0,
        "news_score_weight": 0.0,
        "ml_score_weight": 0.0,
        "cross_product_consensus_threshold": 0.5,
    },
}


@dataclass
class CandidateScore:
    strategy: str
    params: Dict[str, Any]
    rebalance_frequency: str
    live_score: float
    performance_score: float
    gate_score: float
    trailing_returns: Dict[str, float]
    annualized_vol: float
    max_drawdown: float
    regime_bucket: str = "insufficient_history"
    regime_reasons: List[str] = field(default_factory=list)
    regime_metrics: Dict[str, float] = field(default_factory=dict)
    regime_percentiles: Dict[str, float] = field(default_factory=dict)
    regime_status: str = "insufficient_history"
    # Phase B analyst integration. Defaults keep Phase-A behaviour
    # bit-identical: weight 0 -> analyst component ignored entirely.
    analyst_score: float = 0.0
    analyst_score_effective_weight: float = 0.0
    analyst_evidence_status: str = "missing"
    # Phase B+ T-0099: conditioned analyst evidence per regime bucket.
    analyst_evidence_conditioned_status: str = "missing"
    # Phase C news integration. Defaults keep Phase B behaviour
    # numerically identical when no news provider/weight is configured.
    news_score: float = 0.0
    news_score_effective_weight: float = 0.0
    news_evidence_status: str = "missing"
    news_evidence_conditioned_status: str = "missing"
    # Phase D ML integration (T-0221). Defaults keep Phase-C behaviour
    # numerically identical when no ML provider/weight is configured.
    ml_score: float = 0.0
    ml_score_effective_weight: float = 0.0
    ml_evidence_status: str = "missing"
    ml_evidence_conditioned_status: str = "missing"
    # Phase E1 cross-product evidence (T-0304). All default to None,
    # so that Phase-D output stays bit-identical when
    # cross_product_require=False.
    cross_product_consensus_per_bucket: Optional[Dict[str, float]] = None
    cross_product_consensus_threshold: Optional[float] = None
    cross_product_status: Optional[str] = None
    cross_product_active_components: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "strategy": self.strategy,
            "params": self.params,
            "rebalance_frequency": self.rebalance_frequency,
            "live_score": self.live_score,
            "performance_score": self.performance_score,
            "gate_score": self.gate_score,
            "trailing_returns": self.trailing_returns,
            "annualized_vol": self.annualized_vol,
            "max_drawdown": self.max_drawdown,
            "regime_bucket": self.regime_bucket,
            "regime_reasons": list(self.regime_reasons),
            "regime_metrics": dict(self.regime_metrics),
            "regime_percentiles": dict(self.regime_percentiles),
            "regime_status": self.regime_status,
            "analyst_score": float(self.analyst_score),
            "analyst_score_effective_weight": float(self.analyst_score_effective_weight),
            "analyst_evidence_status": str(self.analyst_evidence_status),
            "analyst_evidence_conditioned_status": str(
                self.analyst_evidence_conditioned_status
            ),
            "news_score": float(self.news_score),
            "news_score_effective_weight": float(self.news_score_effective_weight),
            "news_evidence_status": str(self.news_evidence_status),
            "news_evidence_conditioned_status": str(
                self.news_evidence_conditioned_status
            ),
            "ml_score": float(self.ml_score),
            "ml_score_effective_weight": float(self.ml_score_effective_weight),
            "ml_evidence_status": str(self.ml_evidence_status),
            "ml_evidence_conditioned_status": str(
                self.ml_evidence_conditioned_status
            ),
        }
        # Phase E1: only emit cross-product fields when they are not
        # default-None (preserves Phase-D bit-identity when
        # cross_product_require=False).
        if self.cross_product_consensus_per_bucket is not None:
            payload["cross_product_consensus_per_bucket"] = dict(
                self.cross_product_consensus_per_bucket
            )
        if self.cross_product_consensus_threshold is not None:
            payload["cross_product_consensus_threshold"] = float(
                self.cross_product_consensus_threshold
            )
        if self.cross_product_status is not None:
            payload["cross_product_status"] = str(self.cross_product_status)
        if self.cross_product_active_components is not None:
            payload["cross_product_active_components"] = list(
                self.cross_product_active_components
            )
        return payload


def _coerce_date(value: Optional[date | str]) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed):
        return fallback
    return parsed


def _normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _weighted_score(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values or not weights:
        return 0.0
    n = min(len(values), len(weights))
    if n == 0:
        return 0.0
    v = np.asarray(values[:n], dtype=float)
    w = np.asarray(weights[:n], dtype=float)
    if np.sum(np.abs(w)) < 1e-12:
        return float(np.mean(v))
    w = w / np.sum(np.abs(w))
    return float(np.sum(v * w))


def _compute_equity_stats(equity_curve: pd.Series) -> Tuple[float, float]:
    if equity_curve is None or equity_curve.empty:
        return 0.0, 0.0
    returns = equity_curve.pct_change(fill_method=None).dropna()
    vol = float(returns.std(ddof=0) * np.sqrt(252)) if not returns.empty else 0.0
    running_peak = equity_curve.cummax()
    drawdowns = equity_curve / running_peak - 1.0
    max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0
    return vol, max_dd


def _compute_trailing_returns(
    equity_curve: pd.Series,
    windows_days: Sequence[int],
) -> Dict[str, float]:
    returns: Dict[str, float] = {}
    if equity_curve is None or equity_curve.empty:
        return returns
    for window in windows_days:
        key = f"{int(window)}d"
        if len(equity_curve) <= int(window):
            returns[key] = 0.0
            continue
        start = float(equity_curve.iloc[-int(window) - 1])
        end = float(equity_curve.iloc[-1])
        if start <= 0:
            returns[key] = 0.0
        else:
            returns[key] = end / start - 1.0
    return returns


def _resolve_regime_tolerances(
    regime_profile: str,
    alpha_tie_band: Optional[float],
    stress_alpha_tolerance: Optional[float],
    conditioned_min_windows: Optional[int],
) -> Tuple[float, float, int]:
    defaults = REGIME_TOLERANCE_DEFAULTS.get(regime_profile, REGIME_TOLERANCE_DEFAULTS["ausgewogen"])
    resolved_alpha_tie = float(
        defaults["alpha_tie_band"] if alpha_tie_band is None else max(0.0, float(alpha_tie_band))
    )
    resolved_stress_tol = float(
        defaults["stress_alpha_tolerance"]
        if stress_alpha_tolerance is None
        else max(0.0, float(stress_alpha_tolerance))
    )
    resolved_min_windows = resolve_conditioned_min_windows(
        evidence_profile=regime_profile if regime_profile in {"defensiv", "ausgewogen", "aggressiv", "custom"} else "ausgewogen",
        conditioned_min_windows=conditioned_min_windows
        if conditioned_min_windows is not None
        else int(defaults["conditioned_min_windows"]),
    )
    return resolved_alpha_tie, resolved_stress_tol, resolved_min_windows


def _bucket_not_worse(target_bucket: str, current_bucket: str) -> bool:
    if "insufficient_history" in {target_bucket, current_bucket}:
        return True
    return bucket_rank(target_bucket) <= bucket_rank(current_bucket)


def _bucket_is_better(target_bucket: str, current_bucket: str) -> bool:
    if "insufficient_history" in {target_bucket, current_bucket}:
        return False
    return bucket_rank(target_bucket) < bucket_rank(current_bucket)


def _resolve_rebalance_due(
    as_of: date,
    portfolio: Optional[Portfolio],
    frequency: str,
) -> bool:
    if portfolio is None or portfolio.last_rebalance is None:
        return True
    last = portfolio.last_rebalance
    if frequency == "daily":
        next_date = last + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        return as_of >= next_date
    if frequency == "weekly":
        return as_of >= (last + timedelta(days=7))
    if frequency == "monthly":
        year = last.year
        month = last.month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        _, day = monthrange(year, month)
        return as_of >= date(year, month, day)
    if frequency == "quarterly":
        return as_of >= (last + timedelta(days=90))
    if frequency == "yearly":
        return as_of >= date(last.year + 1, 12, 31)
    return True


def _resolve_confirmation_count(
    current_strategy: str,
    recommended_target: str,
    decision_log_path: str | Path,
    switch_margin: float,
) -> int:
    rows = load_recent_meta_decisions(log_path=decision_log_path, limit=300)
    count = 0
    for row in reversed(rows):
        if row.get("current_strategy") != current_strategy:
            continue
        if row.get("recommended_target") != recommended_target:
            break
        if _safe_float(row.get("score_margin", 0.0)) < switch_margin:
            break
        count += 1
    return count


def _extract_path_candidates(path: str) -> Iterable[str]:
    p = Path(path)
    resolved = str(p.expanduser().resolve())
    yield resolved
    yield path
    yield p.name
    yield p.stem


def _discover_latest_preset_file(explicit_file: Optional[str | Path] = None) -> Optional[Path]:
    if explicit_file:
        candidate = Path(explicit_file).expanduser().resolve()
        if candidate.exists():
            return candidate
        return None
    results_dir = Path("results").resolve()
    if not results_dir.exists():
        return None
    json_files = sorted(
        results_dir.glob("optimized_params*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return json_files[0] if json_files else None


def _load_preset_params_map(explicit_file: Optional[str | Path] = None) -> Dict[str, Dict[str, Any]]:
    path = _discover_latest_preset_file(explicit_file)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            normalized[str(key)] = dict(value)
    return normalized


def _resolve_candidate_params(
    strategy_path: str,
    explicit_params: Optional[Dict[str, Any]],
    params_source: ParamsSource,
    preset_map: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[str]]:
    params = dict(explicit_params or {})
    rebalance_override = None
    if params_source == "manual_only":
        return params, rebalance_override
    if params_source == "strategy_defaults":
        return params, rebalance_override
    if params_source != "preset_first":
        return params, rebalance_override

    if params:
        return params, rebalance_override

    match: Optional[Dict[str, Any]] = None
    for key in _extract_path_candidates(strategy_path):
        if key in preset_map:
            match = dict(preset_map[key])
            break
    if match is None:
        return params, rebalance_override
    if "rebalance_frequency" in match:
        rebalance_override = str(match.pop("rebalance_frequency"))
    return match, rebalance_override


def _score_candidate(
    strategy_path: str,
    params: Dict[str, Any],
    as_of: date,
    skip_failed: bool,
    scoring_mode: ScoringMode,
    performance_windows_days: Sequence[int],
    performance_weights: Sequence[float],
    regime_profile: str,
    rebalance_override: Optional[str] = None,
    *,
    analyst_provider: Any = None,
    analyst_dataset: Optional[str] = None,
    analyst_score_effective_weight: float = 0.0,
    analyst_evidence_status: str = "missing",
    analyst_evidence_conditioned_status: str = "missing",
    news_provider: Any = None,
    news_dataset: Optional[str] = None,
    news_score_effective_weight: float = 0.0,
    news_evidence_status: str = "missing",
    news_evidence_conditioned_status: str = "missing",
    ml_provider: Any = None,
    ml_dataset: Optional[str] = None,
    ml_score_effective_weight: float = 0.0,
    ml_evidence_status: str = "missing",
    ml_evidence_conditioned_status: str = "missing",
) -> CandidateScore:
    strategy = load_strategy_instance(Path(strategy_path), params=params or None)
    if rebalance_override:
        strategy.rebalance_frequency = rebalance_override

    end_date = as_of.isoformat()
    lookback_days = max(int(max(performance_windows_days, default=252)) * 4, 900)
    start_date = (as_of - timedelta(days=lookback_days)).isoformat()

    data = DataLoader.yahoo(
        tickers=list(strategy.assets),
        start=start_date,
        end=end_date,
        currency="EUR",
        align="ffill",
        skip_failed=skip_failed,
    )
    config = BacktestConfig(
        initial_capital=10_000.0,
        costs_pct=0.001,
        rebalance_frequency=getattr(strategy, "rebalance_frequency", "monthly"),
        benchmark="S&P 500",
        tax_enabled=False,
        metric_basis="gross",
        validate=False,
    )
    result = Backtester(strategy, data, config).run()
    equity_curve = result.equity_curve
    trailing = _compute_trailing_returns(equity_curve, performance_windows_days)
    trailing_values = [trailing.get(f"{int(day)}d", 0.0) for day in performance_windows_days]
    performance_score = _weighted_score(trailing_values, performance_weights)
    regime_snapshot = assess_regime_from_equity_curve(equity_curve, profile=regime_profile) if equity_curve is not None else None

    vol, max_dd = _compute_equity_stats(equity_curve)
    # Normalize into roughly [-1, +1] range.
    gate_score = max(-1.0, min(1.0, 1.0 - vol - abs(max_dd)))

    # Phase B: optional analyst component blended into live_score.
    analyst_score_value = 0.0
    if analyst_provider is not None and analyst_score_effective_weight > 0:
        analyst_score_value = _strategy_analyst_score(
            provider=analyst_provider,
            dataset=analyst_dataset,
            as_of=as_of,
            target_allocation=None,
            strategy=strategy,
            data=data,
        )
    # Phase C: optional news component blended into live_score.
    news_score_value = 0.0
    if news_provider is not None and news_score_effective_weight > 0:
        news_score_value = _strategy_news_score(
            provider=news_provider,
            dataset=news_dataset,
            as_of=as_of,
            target_allocation=None,
            strategy=strategy,
            data=data,
        )
    # Phase D: optional ML component blended into live_score.
    ml_score_value = 0.0
    if ml_provider is not None and ml_score_effective_weight > 0:
        ml_score_value = _strategy_ml_score(
            provider=ml_provider,
            dataset=ml_dataset,
            as_of=as_of,
            target_allocation=None,
            strategy=strategy,
            data=data,
        )

    w_analyst = float(analyst_score_effective_weight)
    w_news = float(news_score_effective_weight)
    w_ml = float(ml_score_effective_weight)
    if w_analyst + w_news + w_ml >= 1.0:
        raise ValueError(
            "analyst_score_effective_weight + news_score_effective_weight + "
            "ml_score_effective_weight must be < 1.0 "
            f"(got {w_analyst} + {w_news} + {w_ml} = {w_analyst + w_news + w_ml})"
        )

    if scoring_mode == "performance_only":
        live_score = performance_score
    elif scoring_mode == "gate_only":
        live_score = gate_score
    else:
        # Multi-Weight-Renormalisierung (Phase D / Codex Architektur 9):
        # `w_perf = 0.7*(1-w_a-w_n-w_ml), w_gate = 0.3*(1-w_a-w_n-w_ml)`.
        # When all component weights are 0 this collapses to the
        # Phase-A formula bit-identically; with only analyst/news set it
        # matches the Phase-C two-component formula.
        scale = 1.0 - w_analyst - w_news - w_ml
        w_perf = 0.7 * scale
        w_gate = 0.3 * scale
        live_score = (
            w_perf * performance_score
            + w_gate * gate_score
            + w_analyst * analyst_score_value
            + w_news * news_score_value
            + w_ml * ml_score_value
        )

    return CandidateScore(
        strategy=_normalize_path(strategy_path),
        params=dict(params),
        rebalance_frequency=getattr(strategy, "rebalance_frequency", "monthly"),
        live_score=float(live_score),
        performance_score=float(performance_score),
        gate_score=float(gate_score),
        trailing_returns=trailing,
        annualized_vol=float(vol),
        max_drawdown=float(max_dd),
        regime_bucket=regime_snapshot.bucket if regime_snapshot is not None else "insufficient_history",
        regime_reasons=list(regime_snapshot.reasons) if regime_snapshot is not None else [],
        regime_metrics=dict(regime_snapshot.metrics) if regime_snapshot is not None else {},
        regime_percentiles=dict(regime_snapshot.percentiles) if regime_snapshot is not None else {},
        regime_status=regime_snapshot.status if regime_snapshot is not None else "insufficient_history",
        analyst_score=float(analyst_score_value),
        analyst_score_effective_weight=float(analyst_score_effective_weight),
        analyst_evidence_status=str(analyst_evidence_status),
        analyst_evidence_conditioned_status=str(analyst_evidence_conditioned_status),
        news_score=float(news_score_value),
        news_score_effective_weight=float(news_score_effective_weight),
        news_evidence_status=str(news_evidence_status),
        news_evidence_conditioned_status=str(news_evidence_conditioned_status),
        ml_score=float(ml_score_value),
        ml_score_effective_weight=float(ml_score_effective_weight),
        ml_evidence_status=str(ml_evidence_status),
        ml_evidence_conditioned_status=str(ml_evidence_conditioned_status),
    )


def _strategy_ml_score(
    provider: Any,
    dataset: Optional[str],
    as_of: date,
    target_allocation: Optional[Dict[str, float]],
    strategy: Any,
    data: Any,
) -> float:
    """Weighted average of ``ml_forecast_score`` for the target allocation
    (Phase D, T-0222). Mirror of :func:`_strategy_news_score` so the
    contract is identical at the meta-decision call site."""

    weights = target_allocation
    if weights is None:
        assets = list(getattr(strategy, "assets", []) or [])
        if not assets:
            return 0.0
        weights = {a: 1.0 / len(assets) for a in assets}
    try:
        if dataset and hasattr(provider, "snapshot_dataset"):
            snap = provider.snapshot_dataset(dataset, as_of=as_of, tickers=list(weights))
        else:
            snap = provider.snapshot(as_of=as_of, tickers=list(weights))
    except Exception:
        return 0.0
    if snap is None or getattr(snap, "data", None) is None:
        return 0.0
    df = snap.data
    if df.empty or "feature_name" not in df.columns:
        return 0.0
    rows = df.loc[df["feature_name"] == "ml_forecast_score"]
    scores: Dict[str, float] = {}
    for ticker, group in rows.groupby("ticker"):
        try:
            scores[str(ticker).upper()] = float(group["feature_value"].iloc[-1])
        except (TypeError, ValueError):
            continue
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    weighted = 0.0
    for ticker, weight in weights.items():
        weighted += float(weight) * scores.get(str(ticker).upper(), 0.0)
    return weighted / total_weight


def _strategy_news_score(
    provider: Any,
    dataset: Optional[str],
    as_of: date,
    target_allocation: Optional[Dict[str, float]],
    strategy: Any,
    data: Any,
) -> float:
    """Weighted average of news_sentiment_score values for the target
    allocation; mirrors :func:`_strategy_analyst_score`. Missing scores
    count as 0.0 (neutral)."""

    weights = target_allocation
    if weights is None:
        # Default to equal-weight over the strategy's universe so the
        # function works even before the strategy is rebalanced.
        assets = list(getattr(strategy, "assets", []) or [])
        if not assets:
            return 0.0
        weights = {a: 1.0 / len(assets) for a in assets}
    try:
        if dataset and hasattr(provider, "snapshot_dataset"):
            snap = provider.snapshot_dataset(dataset, as_of=as_of, tickers=list(weights))
        else:
            snap = provider.snapshot(as_of=as_of, tickers=list(weights))
    except Exception:
        return 0.0
    if snap is None or getattr(snap, "data", None) is None:
        return 0.0
    df = snap.data
    if df.empty or "feature_name" not in df.columns:
        return 0.0
    rows = df.loc[df["feature_name"] == "news_sentiment_score"]
    scores: Dict[str, float] = {}
    for ticker, group in rows.groupby("ticker"):
        try:
            scores[str(ticker).upper()] = float(group["feature_value"].iloc[-1])
        except (TypeError, ValueError):
            continue
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    weighted = 0.0
    for ticker, weight in weights.items():
        weighted += float(weight) * scores.get(str(ticker).upper(), 0.0)
    return weighted / total_weight


def _strategy_analyst_score(
    provider: Any,
    dataset: Optional[str],
    as_of: date,
    target_allocation: Optional[Dict[str, float]],
    strategy: Any,
    data: Any,
) -> float:
    """Weighted average of analyst_score values for the target allocation.

    Missing scores count as 0.0 (neutral). Phase B contract.
    """

    weights: Dict[str, float] = {}
    if target_allocation:
        weights = {str(t).upper(): float(w) for t, w in target_allocation.items() if w > 0}
    if not weights:
        try:
            allocation = strategy.signal(as_of, data.prices)
            weights = {
                str(t).upper(): float(w)
                for t, w in getattr(allocation, "weights", {}).items()
                if w > 0
            }
        except Exception:
            weights = {}
    if not weights:
        return 0.0
    try:
        if dataset and hasattr(provider, "snapshot_dataset"):
            snap = provider.snapshot_dataset(dataset, as_of=as_of, tickers=list(weights))
        else:
            snap = provider.snapshot(as_of=as_of, tickers=list(weights))
    except Exception:
        return 0.0
    if snap is None or getattr(snap, "data", None) is None:
        return 0.0
    df = snap.data
    if df.empty or "feature_name" not in df.columns:
        return 0.0
    score_rows = df.loc[df["feature_name"] == "analyst_score"]
    scores: Dict[str, float] = {}
    for ticker, group in score_rows.groupby("ticker"):
        try:
            scores[str(ticker).upper()] = float(group["feature_value"].iloc[-1])
        except (TypeError, ValueError):
            continue
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    weighted = 0.0
    for ticker, weight in weights.items():
        weighted += weight * scores.get(ticker, 0.0)
    return weighted / total_weight


def _build_switch_plan(
    target_strategy_path: str,
    target_params: Dict[str, Any],
    as_of: date,
    portfolio: Optional[Portfolio],
    skip_failed: bool,
    drift_tolerance: float,
) -> Optional[Dict[str, Any]]:
    if portfolio is None:
        return None
    strategy = load_strategy_instance(Path(target_strategy_path), params=target_params or None)
    report = SignalGenerator(
        strategy=strategy,
        portfolio=portfolio,
        skip_failed=skip_failed,
        drift_tolerance=drift_tolerance,
    ).generate(as_of=as_of)
    payload = report.to_dict()
    return {
        "target_strategy": target_strategy_path,
        "target_strategy_name": report.strategy_name,
        "orders": payload.get("orders", []),
        "actionable_orders": payload.get("summary", {}).get("orders", {}).get("actionable", 0),
    }


def _evaluate_cross_product_gate(
    *,
    current_row: "CandidateScore",
    analyst_summary: Optional[Dict[str, Any]],
    news_summary: Optional[Dict[str, Any]],
    ml_summary: Optional[Dict[str, Any]],
    analyst_unconditional_status: str,
    news_unconditional_status: str,
    ml_unconditional_status: str,
    analyst_configured_weight: float,
    news_configured_weight: float,
    ml_configured_weight: float,
    profile: str,
    threshold_override: Optional[float],
    require: bool,
) -> CrossProductGateResult:
    """Phase E1 (T-0305): cross-product consensus gate for switches.

    Consensus inputs per component:
    - `configured_weight` (value set by the user, NOT effective).
    - `unconditional_status` from `assess_<comp>_evidence`.
    - `conditioned_status_by_bucket` from
      `summary["conditioned"][bucket].checks`, resp. the downstream
      `assess_conditioned_<comp>_evidence_status` results.

    We map the summary `conditioned[bucket]` dicts to PASS/FAIL
    status by replicating the same threshold behavior as
    `assess_conditioned_*_evidence_status` — per bucket: PASS iff
    all three sub-checks (min_windows, min_cagr_edge_pp,
    min_hit_rate) are satisfied.
    """

    def _bucket_pass_map(summary: Optional[Dict[str, Any]]) -> Dict[str, str]:
        # Map summary["conditioned"][bucket] -> "pass"/"fail" per bucket.
        if not summary:
            return {}
        thresholds = summary.get("thresholds") or {}
        if not thresholds:
            return {}
        out: Dict[str, str] = {}
        conditioned = summary.get("conditioned") or {}
        if not isinstance(conditioned, dict):
            return {}
        for bucket, bucket_summary in conditioned.items():
            if not isinstance(bucket_summary, dict):
                continue
            num_windows = int(bucket_summary.get("num_windows", 0))
            # Konsens nutzt relaxed min_windows wie
            # assess_conditioned_*_evidence_status (max(3, full // 2)).
            full_min_windows = int(thresholds.get("min_windows", 0) or 0)
            relaxed_min_windows = max(3, full_min_windows // 2) if full_min_windows else 3
            if num_windows < relaxed_min_windows:
                out[str(bucket)] = "fail"
                continue
            checks_ok = (
                float(bucket_summary.get("cagr_edge_pp", 0.0))
                >= float(thresholds.get("min_cagr_edge_pp", 0.0))
                and float(bucket_summary.get("hit_rate", 0.0))
                >= float(thresholds.get("min_hit_rate", 0.0))
            )
            out[str(bucket)] = "pass" if checks_ok else "fail"
        return out

    components = [
        ComponentEvidence(
            name="analyst",
            configured_weight=float(analyst_configured_weight or 0.0),
            unconditional_status=str(analyst_unconditional_status),
            conditioned_status_by_bucket=_bucket_pass_map(analyst_summary),
        ),
        ComponentEvidence(
            name="news",
            configured_weight=float(news_configured_weight or 0.0),
            unconditional_status=str(news_unconditional_status),
            conditioned_status_by_bucket=_bucket_pass_map(news_summary),
        ),
        ComponentEvidence(
            name="ml",
            configured_weight=float(ml_configured_weight or 0.0),
            unconditional_status=str(ml_unconditional_status),
            conditioned_status_by_bucket=_bucket_pass_map(ml_summary),
        ),
    ]

    return evaluate_cross_product_gate(
        components=components,
        current_regime_bucket=current_row.regime_bucket,
        profile=profile,
        threshold_override=threshold_override,
        require=require,
    )


def run_meta_decision(
    *,
    as_of: Optional[date | str],
    current_strategy: str,
    current_params: Optional[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    portfolio: Optional[Portfolio] = None,
    skip_failed: bool = True,
    drift_tolerance: float = 0.005,
    params_source: ParamsSource = "preset_first",
    preset_params_file: Optional[str] = None,
    scoring_mode: ScoringMode = "hybrid",
    performance_windows_days: Optional[Sequence[int]] = None,
    performance_weights: Optional[Sequence[float]] = None,
    confirm_points: int = 2,
    switch_margin: float = 0.10,
    decision_cadence: DecisionCadence = "run_check_rebalance_switch",
    plan_mode: PlanMode = "recommendation_with_portfolio_plan",
    evidence_required: bool = True,
    evidence_profile: str = "ausgewogen",
    evidence_compare_mode: str = "vs_current",
    evidence_max_age_days: int = 30,
    evidence_artifact_path: Optional[str] = None,
    gate_fail_action: GateFailAction = "hold_current",
    custom_thresholds: Optional[Dict[str, Any]] = None,
    regime_mode: RegimeMode = "strategy_fragility",
    regime_profile: str = "ausgewogen",
    alpha_tie_band: Optional[float] = None,
    stress_alpha_tolerance: Optional[float] = None,
    conditioned_min_windows: Optional[int] = None,
    decision_log_path: str | Path = DEFAULT_DECISION_LOG,
    external_features_provider: Any = None,
    analyst_score_weight: float = 0.0,
    analyst_datasets: Sequence[str] = (),
    allow_synthetic_analyst_evidence: bool = False,
    analyst_require_conditioned_evidence: bool = False,
    news_score_weight: float = 0.0,
    news_datasets: Sequence[str] = (),
    allow_synthetic_news_evidence: bool = False,
    news_require_conditioned_evidence: bool = False,
    ml_score_weight: float = 0.0,
    ml_datasets: Sequence[str] = (),
    allow_synthetic_ml_evidence: bool = False,
    ml_require_conditioned_evidence: bool = False,
    # Phase E1 Cross-Product-Evidence (T-0305).
    cross_product_require: bool = False,
    cross_product_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate live candidate ranking and evidence-gated switch decision.

    Phase B+ T-0099 adds a conditioned analyst-evidence gate. Phase C
    adds a parallel news-evidence gate driven by
    ``news_score_weight`` / ``news_datasets``. Defaults are all zero so
    Phase B behaviour stays bit-identical when no news provider is set.
    """
    as_of_date = _coerce_date(as_of)
    current_strategy = _normalize_path(current_strategy)
    current_params = dict(current_params or {})
    performance_windows_days = list(performance_windows_days or DEFAULT_PERFORMANCE_WINDOWS_DAYS)
    performance_weights = list(performance_weights or DEFAULT_PERFORMANCE_WEIGHTS)
    confirm_points = max(1, int(confirm_points))
    custom_thresholds = dict(custom_thresholds or {})
    resolved_alpha_tie_band, resolved_stress_alpha_tolerance, resolved_conditioned_min_windows = (
        _resolve_regime_tolerances(
            regime_profile=regime_profile,
            alpha_tie_band=alpha_tie_band,
            stress_alpha_tolerance=stress_alpha_tolerance,
            conditioned_min_windows=conditioned_min_windows,
        )
    )
    resolved_evidence_profile: EvidenceProfile = (
        evidence_profile
        if evidence_profile in {"defensiv", "ausgewogen", "aggressiv", "custom"}
        else "ausgewogen"
    )

    preset_map = _load_preset_params_map(preset_params_file)

    normalized_candidates: List[Tuple[str, Dict[str, Any], Optional[str]]] = []
    seen = set()
    for candidate in candidates:
        strategy_path = _normalize_path(candidate.get("strategy"))
        if strategy_path in seen:
            continue
        seen.add(strategy_path)
        params, rebalance_override = _resolve_candidate_params(
            strategy_path=strategy_path,
            explicit_params=candidate.get("params"),
            params_source=params_source,
            preset_map=preset_map,
        )
        normalized_candidates.append((strategy_path, params, rebalance_override))

    if current_strategy not in seen:
        normalized_candidates.append((current_strategy, current_params, None))

    # Phase B: analyst evidence check BEFORE ranking. The effective
    # analyst weight is 0.0 unless evidence says PASS — this guarantees
    # that switches stay OOS-backed.
    analyst_evidence_status: EvidenceStatus = "missing"
    analyst_evidence_summary: Optional[Dict[str, Any]] = None
    analyst_evidence_reasons: List[str] = []
    analyst_effective_weight = 0.0
    analyst_default_dataset: Optional[str] = None
    news_evidence_status: EvidenceStatus = "missing"
    news_evidence_summary: Optional[Dict[str, Any]] = None
    news_evidence_reasons: List[str] = []
    news_effective_weight = 0.0
    news_default_dataset: Optional[str] = None
    ml_evidence_status: EvidenceStatus = "missing"
    ml_evidence_summary: Optional[Dict[str, Any]] = None
    ml_evidence_reasons: List[str] = []
    ml_effective_weight = 0.0
    ml_default_dataset: Optional[str] = None

    candidate_assets: List[str] = []
    current_assets_list: List[str] = []
    needs_universe = (
        external_features_provider is not None
        and (
            analyst_score_weight > 0
            or news_score_weight > 0
            or ml_score_weight > 0
        )
    )
    if needs_universe:
        for path, params, _ in normalized_candidates:
            try:
                strategy = load_strategy_instance(Path(path), params=params or None)
                candidate_assets.extend(list(getattr(strategy, "assets", []) or []))
            except Exception:
                continue
        try:
            current_strategy_instance = load_strategy_instance(
                Path(current_strategy), params=current_params or None
            )
            current_assets_list = list(getattr(current_strategy_instance, "assets", []) or [])
        except Exception:
            current_assets_list = []

    if external_features_provider is not None and analyst_score_weight > 0:
        analyst_default_dataset = (
            list(analyst_datasets)[0] if analyst_datasets else None
        )
        (
            analyst_evidence_status,
            analyst_evidence_reasons,
            analyst_evidence_summary,
        ) = assess_analyst_evidence(
            external_provider=external_features_provider,
            current_assets=current_assets_list,
            candidate_assets=candidate_assets,
            as_of=as_of_date,
            analyst_dataset=analyst_default_dataset,
            profile=resolved_evidence_profile,
            allow_synthetic_analyst_evidence=allow_synthetic_analyst_evidence,
        )
        if analyst_evidence_status == "pass":
            analyst_effective_weight = float(analyst_score_weight)

    if external_features_provider is not None and news_score_weight > 0:
        news_default_dataset = (
            list(news_datasets)[0] if news_datasets else None
        )
        (
            news_evidence_status,
            news_evidence_reasons,
            news_evidence_summary,
        ) = assess_news_evidence(
            external_provider=external_features_provider,
            current_assets=current_assets_list,
            candidate_assets=candidate_assets,
            as_of=as_of_date,
            news_dataset=news_default_dataset,
            profile=resolved_evidence_profile,
            allow_synthetic_news_evidence=allow_synthetic_news_evidence,
        )
        if news_evidence_status == "pass":
            news_effective_weight = float(news_score_weight)

    # Phase D: ML evidence gate (Codex D14/D18). Lookahead filter applied
    # in :func:`assess_ml_evidence` based on the snapshot's
    # ``ml_available_from_ordinal`` rows so the hot-path stays
    # provenance-free.
    if external_features_provider is not None and ml_score_weight > 0:
        ml_default_dataset = list(ml_datasets)[0] if ml_datasets else None
        (
            ml_evidence_status,
            ml_evidence_reasons,
            ml_evidence_summary,
        ) = assess_ml_evidence(
            external_provider=external_features_provider,
            current_assets=current_assets_list,
            candidate_assets=candidate_assets,
            as_of=as_of_date,
            ml_dataset=ml_default_dataset,
            profile=resolved_evidence_profile,
            allow_synthetic_ml_evidence=allow_synthetic_ml_evidence,
        )
        if ml_evidence_status == "pass":
            ml_effective_weight = float(ml_score_weight)

    scores: List[CandidateScore] = []
    for strategy_path, params, rebalance_override in normalized_candidates:
        score = _score_candidate(
            strategy_path=strategy_path,
            params=params,
            as_of=as_of_date,
            skip_failed=skip_failed,
            scoring_mode=scoring_mode,
            performance_windows_days=performance_windows_days,
            performance_weights=performance_weights,
            regime_profile=regime_profile,
            rebalance_override=rebalance_override,
            analyst_provider=external_features_provider,
            analyst_dataset=analyst_default_dataset,
            analyst_score_effective_weight=analyst_effective_weight,
            analyst_evidence_status=analyst_evidence_status,
            news_provider=external_features_provider,
            news_dataset=news_default_dataset,
            news_score_effective_weight=news_effective_weight,
            news_evidence_status=news_evidence_status,
            ml_provider=external_features_provider,
            ml_dataset=ml_default_dataset,
            ml_score_effective_weight=ml_effective_weight,
            ml_evidence_status=ml_evidence_status,
        )
        scores.append(score)

    ranked = sorted(scores, key=lambda row: row.live_score, reverse=True)
    current_row = next((row for row in ranked if row.strategy == current_strategy), None)
    if current_row is None:
        raise ValueError("Current strategy missing from candidate score table.")

    # T-0099 (analyst) + Phase C (news): conditioned evidence gates.
    # Phase C adds a parallel news gate; both share the same downgrade
    # logic — if either gate is required and not PASS, the corresponding
    # effective weight is zeroed and the table is re-scored.
    analyst_evidence_conditioned_status: EvidenceStatus = "missing"
    analyst_evidence_conditioned_reasons: List[str] = []
    analyst_evidence_conditioned_bucket: Optional[Dict[str, Any]] = None
    if analyst_evidence_summary is not None:
        (
            analyst_evidence_conditioned_status,
            analyst_evidence_conditioned_reasons,
            _conditioned_windows,
            analyst_evidence_conditioned_bucket,
        ) = assess_conditioned_analyst_evidence_status(
            analyst_evidence_summary,
            current_bucket=current_row.regime_bucket,
            profile=resolved_evidence_profile,
            conditioned_min_windows=conditioned_min_windows,
        )
    news_evidence_conditioned_status: EvidenceStatus = "missing"
    news_evidence_conditioned_reasons: List[str] = []
    news_evidence_conditioned_bucket: Optional[Dict[str, Any]] = None
    if news_evidence_summary is not None:
        (
            news_evidence_conditioned_status,
            news_evidence_conditioned_reasons,
            _conditioned_news_windows,
            news_evidence_conditioned_bucket,
        ) = assess_conditioned_news_evidence_status(
            news_evidence_summary,
            current_bucket=current_row.regime_bucket,
            profile=resolved_evidence_profile,
            conditioned_min_windows=conditioned_min_windows,
        )
    ml_evidence_conditioned_status: EvidenceStatus = "missing"
    ml_evidence_conditioned_reasons: List[str] = []
    ml_evidence_conditioned_bucket: Optional[Dict[str, Any]] = None
    if ml_evidence_summary is not None:
        (
            ml_evidence_conditioned_status,
            ml_evidence_conditioned_reasons,
            _conditioned_ml_windows,
            ml_evidence_conditioned_bucket,
        ) = assess_conditioned_ml_evidence_status(
            ml_evidence_summary,
            current_bucket=current_row.regime_bucket,
            profile=resolved_evidence_profile,
            conditioned_min_windows=conditioned_min_windows,
        )

    analyst_downgrade = (
        analyst_require_conditioned_evidence
        and analyst_effective_weight > 0
        and analyst_evidence_conditioned_status != "pass"
    )
    news_downgrade = (
        news_require_conditioned_evidence
        and news_effective_weight > 0
        and news_evidence_conditioned_status != "pass"
    )
    ml_downgrade = (
        ml_require_conditioned_evidence
        and ml_effective_weight > 0
        and ml_evidence_conditioned_status != "pass"
    )
    if analyst_downgrade or news_downgrade or ml_downgrade:
        # Downgrade any/all weights and re-score the table so that ranks
        # reflect the gated effective weights.
        if analyst_downgrade:
            analyst_effective_weight = 0.0
        if news_downgrade:
            news_effective_weight = 0.0
        if ml_downgrade:
            ml_effective_weight = 0.0
        scores = []
        for strategy_path, params, rebalance_override in normalized_candidates:
            score = _score_candidate(
                strategy_path=strategy_path,
                params=params,
                as_of=as_of_date,
                skip_failed=skip_failed,
                scoring_mode=scoring_mode,
                performance_windows_days=performance_windows_days,
                performance_weights=performance_weights,
                regime_profile=regime_profile,
                rebalance_override=rebalance_override,
                analyst_provider=external_features_provider,
                analyst_dataset=analyst_default_dataset,
                analyst_score_effective_weight=analyst_effective_weight,
                analyst_evidence_status=analyst_evidence_status,
                analyst_evidence_conditioned_status=analyst_evidence_conditioned_status,
                news_provider=external_features_provider,
                news_dataset=news_default_dataset,
                news_score_effective_weight=news_effective_weight,
                news_evidence_status=news_evidence_status,
                news_evidence_conditioned_status=news_evidence_conditioned_status,
                ml_provider=external_features_provider,
                ml_dataset=ml_default_dataset,
                ml_score_effective_weight=ml_effective_weight,
                ml_evidence_status=ml_evidence_status,
                ml_evidence_conditioned_status=ml_evidence_conditioned_status,
            )
            scores.append(score)
        ranked = sorted(scores, key=lambda row: row.live_score, reverse=True)
        current_row = next((row for row in ranked if row.strategy == current_strategy), None)
        if current_row is None:
            raise ValueError("Current strategy missing from candidate score table.")
    else:
        # No re-scoring; just decorate each row with the conditioned
        # status fields so the JSON payload carries them.
        for row in scores:
            row.analyst_evidence_conditioned_status = str(analyst_evidence_conditioned_status)
            row.news_evidence_conditioned_status = str(news_evidence_conditioned_status)
            row.ml_evidence_conditioned_status = str(ml_evidence_conditioned_status)
    challenger_rows = [row for row in ranked if row.strategy != current_strategy]
    top_challenger = challenger_rows[0] if challenger_rows else None

    live_reasons: List[str] = []
    decision_rule = "hold"
    selected_row = current_row
    switch_checks: List[Dict[str, Any]] = []

    fragility_candidate: Optional[CandidateScore] = None
    if regime_mode == "strategy_fragility" and current_row.regime_bucket in {"fragile", "stressed"}:
        tolerance = (
            resolved_alpha_tie_band
            if current_row.regime_bucket == "fragile"
            else resolved_stress_alpha_tolerance
        )
        fragility_candidates = [
            row
            for row in challenger_rows
            if _bucket_is_better(row.regime_bucket, current_row.regime_bucket)
            and row.performance_score >= (current_row.performance_score - tolerance)
        ]
        fragility_candidates = sorted(
            fragility_candidates,
            key=lambda row: (
                bucket_rank(row.regime_bucket),
                -row.performance_score,
                row.regime_metrics.get("vol_63", float("inf")),
            ),
        )
        if fragility_candidates:
            fragility_candidate = fragility_candidates[0]
        else:
            live_reasons.append(
                "Current strategy is fragile/stressed, but no better-bucket challenger stayed within the allowed performance tolerance"
            )

    alpha_candidate = next(
        (
            row
            for row in challenger_rows
            if (row.live_score - current_row.live_score) >= float(switch_margin)
            and (
                regime_mode != "strategy_fragility"
                or _bucket_not_worse(row.regime_bucket, current_row.regime_bucket)
            )
        ),
        None,
    )

    if fragility_candidate is not None:
        decision_rule = "fragility_driven"
        selected_row = fragility_candidate
    elif alpha_candidate is not None:
        decision_rule = "alpha_driven"
        selected_row = alpha_candidate
    else:
        if top_challenger is None:
            live_reasons.append("No challenger strategies configured")
        elif top_challenger.live_score - current_row.live_score < float(switch_margin):
            live_reasons.append(
                f"Score margin {top_challenger.live_score - current_row.live_score:.4f} below switch_margin {switch_margin:.4f}"
            )
        elif (
            regime_mode == "strategy_fragility"
            and not _bucket_not_worse(top_challenger.regime_bucket, current_row.regime_bucket)
        ):
            live_reasons.append(
                "Top alpha challenger blocked because its regime bucket is worse than the current strategy"
            )
        if ranked and ranked[0].strategy == current_strategy:
            live_reasons.append("Current strategy remains top-ranked")

    recommended_target = selected_row.strategy
    score_margin = selected_row.live_score - current_row.live_score
    performance_gap = selected_row.performance_score - current_row.performance_score

    live_switch_ready = recommended_target != current_strategy
    confirmation_history_hits: Optional[int] = None
    confirmation_required_history = max(0, int(confirm_points) - 1)
    cadence_gate_passed: Optional[bool] = None
    cadence_gate_detail: Optional[str] = None
    if live_switch_ready and decision_rule == "alpha_driven" and confirm_points > 1:
        history_hits = _resolve_confirmation_count(
            current_strategy=current_strategy,
            recommended_target=recommended_target,
            decision_log_path=decision_log_path,
            switch_margin=float(switch_margin),
        )
        confirmation_history_hits = history_hits
        required_history = confirm_points - 1
        if history_hits < required_history:
            live_switch_ready = False
            live_reasons.append(
                f"Confirmation history insufficient: {history_hits}/{required_history}"
            )
    elif live_switch_ready and decision_rule == "alpha_driven":
        confirmation_history_hits = confirmation_required_history
    elif decision_rule == "fragility_driven" and live_switch_ready:
        confirmation_history_hits = None

    if live_switch_ready and decision_cadence == "run_check_rebalance_switch":
        due = _resolve_rebalance_due(
            as_of=as_of_date,
            portfolio=portfolio,
            frequency=current_row.rebalance_frequency,
        )
        if not due:
            live_switch_ready = False
            cadence_gate_passed = False
            cadence_gate_detail = "Rebalance cadence gate not due yet"
            live_reasons.append("Rebalance cadence gate not due yet")
        else:
            cadence_gate_passed = True
            cadence_gate_detail = "Current cadence is due"
    elif live_switch_ready and decision_cadence == "monthly_fixed" and as_of_date.day < 20:
        live_switch_ready = False
        cadence_gate_passed = False
        cadence_gate_detail = "Monthly fixed cadence gate not reached"
        live_reasons.append("Monthly fixed cadence gate not reached")
    elif live_switch_ready:
        cadence_gate_passed = True
        cadence_gate_detail = "Cadence allows switching"

    artifact = None
    evidence_status = None
    evidence_reasons: List[str] = []
    evidence_age_days = None
    conditioned_evidence_status = None
    conditioned_evidence_reasons: List[str] = []
    conditioned_windows = None
    conditioned_summary = None
    switch_allowed = bool(live_switch_ready)

    if recommended_target != current_strategy:
        if evidence_artifact_path:
            artifact = json.loads(Path(evidence_artifact_path).read_text())
        else:
            artifact = find_latest_evidence_artifact(
                current_strategy=current_strategy,
                target_strategy=recommended_target,
            )
        evidence_status, evidence_reasons, evidence_age_days = assess_evidence_status(
            artifact=artifact,
            evidence_max_age_days=evidence_max_age_days,
            as_of=as_of_date,
            evidence_profile=resolved_evidence_profile,
            custom_thresholds=custom_thresholds,
        )
        if evidence_required and evidence_status != "pass":
            switch_allowed = False
            if evidence_status == "missing":
                live_reasons.append("No valid evidence artifact available")
            elif evidence_status == "stale":
                live_reasons.append("Evidence artifact is stale")
            else:
                live_reasons.append("Evidence gates failed")

        if decision_rule == "fragility_driven":
            conditioned_evidence_status, conditioned_evidence_reasons, conditioned_windows, conditioned_summary = (
                assess_conditioned_evidence_status(
                    artifact=artifact,
                    current_bucket=current_row.regime_bucket,
                    evidence_profile=resolved_evidence_profile,
                    conditioned_min_windows=resolved_conditioned_min_windows,
                    custom_thresholds=custom_thresholds,
                )
            )
            if evidence_required and conditioned_evidence_status != "pass":
                switch_allowed = False
                live_reasons.append("Conditioned evidence gate blocked fragility-driven switch")
    else:
        switch_allowed = False

    # Phase E1: cross-product consensus gate (T-0305, Codex R3.7).
    # Acts as an additional pre-switch gate on
    # `decision_bucket = current_regime_bucket`. live_scores remain
    # untouched. When `cross_product_require=False`, the gate is
    # marked `not_applicable` and blocks nothing.
    cross_product_gate_result = _evaluate_cross_product_gate(
        current_row=current_row,
        analyst_summary=analyst_evidence_summary,
        news_summary=news_evidence_summary,
        ml_summary=ml_evidence_summary,
        analyst_unconditional_status=analyst_evidence_status,
        news_unconditional_status=news_evidence_status,
        ml_unconditional_status=ml_evidence_status,
        analyst_configured_weight=float(analyst_score_weight),
        news_configured_weight=float(news_score_weight),
        ml_configured_weight=float(ml_score_weight),
        profile=resolved_evidence_profile,
        threshold_override=cross_product_threshold,
        require=cross_product_require,
    )
    # Decorate all score rows with the consensus output, so that the
    # JSON payload carries it along.
    if cross_product_require:
        for row in scores:
            row.cross_product_consensus_per_bucket = dict(
                cross_product_gate_result.consensus_per_bucket
            )
            row.cross_product_consensus_threshold = float(
                cross_product_gate_result.threshold
            )
            row.cross_product_status = str(cross_product_gate_result.status)
            row.cross_product_active_components = list(
                cross_product_gate_result.active_components
            )
    if cross_product_require and recommended_target != current_strategy:
        if cross_product_gate_result.status != "pass":
            switch_allowed = False
            live_reasons.extend(
                f"Cross-product gate blocked switch: {r}"
                for r in cross_product_gate_result.reasons
            )

    if conditioned_evidence_reasons:
        evidence_reasons = list(evidence_reasons) + [
            f"[conditioned:{current_row.regime_bucket}] {reason}"
            for reason in conditioned_evidence_reasons
        ]

    candidate_selected = recommended_target != current_strategy
    if candidate_selected:
        switch_checks.append(
            {
                "key": "candidate",
                "label": "Challenger Selected",
                "status": "pass",
                "detail": f"Recommended target differs from current strategy ({Path(recommended_target).name})",
                "next_step": None,
            }
        )
    else:
        switch_checks.append(
            {
                "key": "candidate",
                "label": "Challenger Selected",
                "status": "fail",
                "detail": "Current strategy remains top-ranked; no different target selected",
                "next_step": "No switch path available until a different challenger becomes top-ranked",
            }
        )

    if decision_rule == "alpha_driven" and candidate_selected:
        score_gate_passed = score_margin >= float(switch_margin)
        switch_checks.append(
            {
                "key": "score_margin",
                "label": "Score Margin",
                "status": "pass" if score_gate_passed else "fail",
                "detail": (
                    f"score_margin={score_margin:.4f} vs required switch_margin={float(switch_margin):.4f}"
                ),
                "next_step": None if score_gate_passed else f"Need score_margin >= {float(switch_margin):.4f}",
            }
        )
    elif decision_rule == "fragility_driven" and candidate_selected:
        switch_checks.append(
            {
                "key": "score_margin",
                "label": "Score Margin",
                "status": "not_applicable",
                "detail": "Fragility-driven switch uses bucket improvement plus alpha tolerance instead of switch_margin",
                "next_step": None,
            }
        )
    else:
        switch_checks.append(
            {
                "key": "score_margin",
                "label": "Score Margin",
                "status": "fail",
                "detail": f"No alpha challenger exceeded switch_margin={float(switch_margin):.4f}",
                "next_step": f"Need a challenger with score_margin >= {float(switch_margin):.4f}",
            }
        )

    if decision_rule == "alpha_driven" and candidate_selected:
        if confirm_points > 1:
            hits = int(confirmation_history_hits or 0)
            switch_checks.append(
                {
                    "key": "confirmation",
                    "label": "Confirmation History",
                    "status": "pass" if hits >= confirmation_required_history else "fail",
                    "detail": f"{hits}/{confirmation_required_history} prior confirmations with score_margin >= {float(switch_margin):.4f}",
                    "next_step": None if hits >= confirmation_required_history else f"Need {confirmation_required_history - hits} additional confirmation run(s)",
                }
            )
        else:
            switch_checks.append(
                {
                    "key": "confirmation",
                    "label": "Confirmation History",
                    "status": "pass",
                    "detail": "confirm_points=1, so no prior confirmation is required",
                    "next_step": None,
                }
            )
    elif decision_rule == "fragility_driven" and candidate_selected:
        switch_checks.append(
            {
                "key": "confirmation",
                "label": "Confirmation History",
                "status": "not_applicable",
                "detail": "Fragility-driven switches do not require alpha confirmation history",
                "next_step": None,
            }
        )
    else:
        switch_checks.append(
            {
                "key": "confirmation",
                "label": "Confirmation History",
                "status": "fail",
                "detail": "No active switch candidate to confirm",
                "next_step": "A valid challenger must be selected before confirmation can accumulate",
            }
        )

    if candidate_selected:
        switch_checks.append(
            {
                "key": "cadence",
                "label": "Decision Cadence",
                "status": "pass" if cadence_gate_passed is not False else "fail",
                "detail": cadence_gate_detail or "Cadence check not required",
                "next_step": None if cadence_gate_passed is not False else "Wait until the next eligible cadence window",
            }
        )
    else:
        switch_checks.append(
            {
                "key": "cadence",
                "label": "Decision Cadence",
                "status": "fail",
                "detail": "No switch candidate selected, so cadence cannot trigger a switch",
                "next_step": "A valid challenger must be selected before cadence matters",
            }
        )

    if candidate_selected:
        if evidence_required:
            evidence_next_step = None
            if evidence_status == "missing":
                evidence_next_step = "Run or load a recent evidence artifact and get PASS"
            elif evidence_status == "stale":
                evidence_next_step = f"Refresh the evidence artifact so it is <= {int(evidence_max_age_days)} days old and PASS"
            elif evidence_status != "pass":
                evidence_next_step = "Current evidence artifact must pass all unconditional gates"
            switch_checks.append(
                {
                    "key": "evidence",
                    "label": "Evidence Gate",
                    "status": "pass" if evidence_status == "pass" else "fail",
                    "detail": f"Evidence status={evidence_status or 'missing'}",
                    "next_step": evidence_next_step,
                }
            )
        else:
            switch_checks.append(
                {
                    "key": "evidence",
                    "label": "Evidence Gate",
                    "status": "not_applicable",
                    "detail": f"Evidence bypassed; latest status={evidence_status or 'missing'}",
                    "next_step": None,
                }
            )
    else:
        switch_checks.append(
            {
                "key": "evidence",
                "label": "Evidence Gate",
                "status": "fail",
                "detail": "No switch candidate selected",
                "next_step": "A valid challenger must be selected before evidence can be evaluated",
            }
        )

    if candidate_selected and analyst_score_weight > 0:
        analyst_pass = analyst_evidence_status == "pass"
        switch_checks.append(
            {
                "key": "analyst_evidence",
                "label": "Analyst Evidence",
                "status": "pass" if analyst_pass else "fail",
                "detail": (
                    f"Analyst evidence status={analyst_evidence_status}; "
                    f"effective_weight={analyst_effective_weight:.3f}"
                ),
                "next_step": (
                    None
                    if analyst_pass
                    else "Refresh analyst evidence (PASS required to apply analyst_score_weight > 0)"
                ),
            }
        )
    elif candidate_selected:
        switch_checks.append(
            {
                "key": "analyst_evidence",
                "label": "Analyst Evidence",
                "status": "not_applicable",
                "detail": "analyst_score_weight = 0; analyst component is inactive",
                "next_step": None,
            }
        )
    else:
        switch_checks.append(
            {
                "key": "analyst_evidence",
                "label": "Analyst Evidence",
                "status": "fail",
                "detail": "No switch candidate selected",
                "next_step": "A valid challenger must be selected before analyst evidence can be evaluated",
            }
        )

    # Phase C: news evidence switch_check (mirrors analyst pattern).
    if candidate_selected and news_score_weight > 0:
        news_pass = news_evidence_status == "pass"
        switch_checks.append(
            {
                "key": "news_evidence",
                "label": "News Evidence",
                "status": "pass" if news_pass else "fail",
                "detail": (
                    f"News evidence status={news_evidence_status}; "
                    f"effective_weight={news_effective_weight:.3f}"
                ),
                "next_step": (
                    None
                    if news_pass
                    else "Refresh news evidence (PASS required to apply news_score_weight > 0)"
                ),
            }
        )
    elif candidate_selected:
        switch_checks.append(
            {
                "key": "news_evidence",
                "label": "News Evidence",
                "status": "not_applicable",
                "detail": "news_score_weight = 0; news component is inactive",
                "next_step": None,
            }
        )
    else:
        switch_checks.append(
            {
                "key": "news_evidence",
                "label": "News Evidence",
                "status": "fail",
                "detail": "No switch candidate selected",
                "next_step": "A valid challenger must be selected before news evidence can be evaluated",
            }
        )

    # Phase C: conditioned news evidence (mirror T-0099 pattern).
    if candidate_selected and news_score_weight > 0:
        cond_pass = news_evidence_conditioned_status == "pass"
        if news_require_conditioned_evidence:
            cond_token = "pass" if cond_pass else "fail"
        else:
            cond_token = "pass" if cond_pass else "not_applicable"
        detail_bits = [
            f"Conditioned news status={news_evidence_conditioned_status}"
            f" for bucket {current_row.regime_bucket}"
        ]
        if news_evidence_conditioned_bucket:
            detail_bits.append(
                f"windows={int(news_evidence_conditioned_bucket.get('num_windows', 0))}"
            )
        next_step = None
        if news_require_conditioned_evidence and not cond_pass:
            next_step = (
                f"Need conditioned news evidence PASS for bucket "
                f"{current_row.regime_bucket} (set "
                f"news_require_conditioned_evidence=False to bypass)"
            )
        switch_checks.append(
            {
                "key": "news_conditioned_evidence",
                "label": "News Evidence (Conditioned)",
                "status": cond_token,
                "detail": "; ".join(detail_bits),
                "next_step": next_step,
            }
        )

    # Phase D: ML evidence switch_check (mirrors analyst/news pattern).
    if candidate_selected and ml_score_weight > 0:
        ml_pass = ml_evidence_status == "pass"
        switch_checks.append(
            {
                "key": "ml_evidence",
                "label": "ML Evidence",
                "status": "pass" if ml_pass else "fail",
                "detail": (
                    f"ML evidence status={ml_evidence_status}; "
                    f"effective_weight={ml_effective_weight:.3f}"
                ),
                "next_step": (
                    None
                    if ml_pass
                    else "Refresh ML evidence (PASS required to apply ml_score_weight > 0)"
                ),
            }
        )
    elif candidate_selected:
        switch_checks.append(
            {
                "key": "ml_evidence",
                "label": "ML Evidence",
                "status": "not_applicable",
                "detail": "ml_score_weight = 0; ml component is inactive",
                "next_step": None,
            }
        )
    else:
        switch_checks.append(
            {
                "key": "ml_evidence",
                "label": "ML Evidence",
                "status": "fail",
                "detail": "No switch candidate selected",
                "next_step": "A valid challenger must be selected before ml evidence can be evaluated",
            }
        )

    # Phase D: conditioned ml evidence (mirror T-0099 / Phase C pattern).
    if candidate_selected and ml_score_weight > 0:
        cond_pass = ml_evidence_conditioned_status == "pass"
        if ml_require_conditioned_evidence:
            cond_token = "pass" if cond_pass else "fail"
        else:
            cond_token = "pass" if cond_pass else "not_applicable"
        detail_bits = [
            f"Conditioned ml status={ml_evidence_conditioned_status}"
            f" for bucket {current_row.regime_bucket}"
        ]
        if ml_evidence_conditioned_bucket:
            detail_bits.append(
                f"windows={int(ml_evidence_conditioned_bucket.get('num_windows', 0))}"
            )
        next_step = None
        if ml_require_conditioned_evidence and not cond_pass:
            next_step = (
                f"Need conditioned ml evidence PASS for bucket "
                f"{current_row.regime_bucket} (set "
                f"ml_require_conditioned_evidence=False to bypass)"
            )
        switch_checks.append(
            {
                "key": "ml_conditioned_evidence",
                "label": "ML Evidence (Conditioned)",
                "status": cond_token,
                "detail": "; ".join(detail_bits),
                "next_step": next_step,
            }
        )

    # Phase E1: cross-product consensus (T-0309 / Codex R3.7). Entry always
    # made on `candidate_selected`. Gate active ONLY when `cross_product_require`.
    if candidate_selected:
        cp_status = cross_product_gate_result.status
        if cross_product_require:
            cp_token = "pass" if cp_status == "pass" else (
                "fail" if cp_status == "fail" else "missing"
            )
        else:
            cp_token = "not_applicable"
        cp_detail_bits = [
            f"Cross-product status={cp_status}",
            f"decision_bucket={cross_product_gate_result.decision_bucket}",
            f"threshold={cross_product_gate_result.threshold:.3f}",
            f"active=[{','.join(cross_product_gate_result.active_components)}]",
        ]
        bucket_consensus = cross_product_gate_result.consensus_per_bucket.get(
            current_row.regime_bucket
        )
        if bucket_consensus is not None:
            cp_detail_bits.append(
                f"consensus[{current_row.regime_bucket}]={bucket_consensus:.3f}"
            )
        cp_next_step = None
        if cross_product_require and cp_status != "pass":
            cp_next_step = (
                "Cross-product consensus must reach threshold for current "
                "regime bucket (set cross_product_require=False to bypass)"
            )
        switch_checks.append(
            {
                "key": "cross_product_consensus",
                "label": "Cross-Product Consensus",
                "status": cp_token,
                "detail": "; ".join(cp_detail_bits),
                "next_step": cp_next_step,
            }
        )

    # T-0099: conditioned analyst evidence per regime bucket. Reported
    # in switch_checks whenever analyst weight > 0; gate-active only
    # when caller set analyst_require_conditioned_evidence=True.
    if candidate_selected and analyst_score_weight > 0:
        cond_pass = analyst_evidence_conditioned_status == "pass"
        if analyst_require_conditioned_evidence:
            cond_status_token = "pass" if cond_pass else "fail"
        else:
            cond_status_token = "pass" if cond_pass else "not_applicable"
        detail_bits = [
            f"Conditioned analyst status={analyst_evidence_conditioned_status}"
            f" for bucket {current_row.regime_bucket}"
        ]
        if analyst_evidence_conditioned_bucket:
            detail_bits.append(
                f"windows={int(analyst_evidence_conditioned_bucket.get('num_windows', 0))}"
            )
        next_step = None
        if analyst_require_conditioned_evidence and not cond_pass:
            next_step = (
                f"Need conditioned analyst evidence PASS for bucket "
                f"{current_row.regime_bucket} (set "
                f"analyst_require_conditioned_evidence=False to bypass)"
            )
        switch_checks.append(
            {
                "key": "analyst_conditioned_evidence",
                "label": "Analyst Evidence (Conditioned)",
                "status": cond_status_token,
                "detail": "; ".join(detail_bits),
                "next_step": next_step,
            }
        )

    if decision_rule == "fragility_driven" and candidate_selected:
        if evidence_required:
            conditioned_next_step = None
            if conditioned_evidence_status != "pass":
                conditioned_next_step = (
                    f"Need conditioned evidence PASS with at least {resolved_conditioned_min_windows} windows for bucket {current_row.regime_bucket}"
                )
            switch_checks.append(
                {
                    "key": "conditioned_evidence",
                    "label": "Conditioned Evidence",
                    "status": "pass" if conditioned_evidence_status == "pass" else "fail",
                    "detail": (
                        f"Conditioned evidence status={conditioned_evidence_status or 'missing'}; windows={conditioned_windows if conditioned_windows is not None else '-'}"
                    ),
                    "next_step": conditioned_next_step,
                }
            )
        else:
            switch_checks.append(
                {
                    "key": "conditioned_evidence",
                    "label": "Conditioned Evidence",
                    "status": "not_applicable",
                    "detail": "Evidence bypassed for fragility-driven switch",
                    "next_step": None,
                }
            )
    elif candidate_selected:
        switch_checks.append(
            {
                "key": "conditioned_evidence",
                "label": "Conditioned Evidence",
                "status": "not_applicable",
                "detail": "Only required for fragility-driven switches",
                "next_step": None,
            }
        )
    else:
        switch_checks.append(
            {
                "key": "conditioned_evidence",
                "label": "Conditioned Evidence",
                "status": "fail",
                "detail": "No switch candidate selected",
                "next_step": "A valid challenger must be selected before conditioned evidence can be evaluated",
            }
        )

    blocked_checks = [row["label"] for row in switch_checks if row["status"] == "fail"]

    if switch_allowed:
        executed_action = "switch_to_target"
    elif recommended_target == current_strategy:
        executed_action = "hold_current"
    elif gate_fail_action == "fallback_safe":
        executed_action = "fallback_safe"
    elif gate_fail_action == "manual_override":
        executed_action = "manual_override_required"
    else:
        executed_action = "hold_current"

    switch_plan = None
    if plan_mode in {"recommendation_with_portfolio_plan", "always_plan"} and recommended_target != current_strategy:
        candidate_for_plan = next((row for row in ranked if row.strategy == recommended_target), selected_row)
        switch_plan = _build_switch_plan(
            target_strategy_path=recommended_target,
            target_params=candidate_for_plan.params,
            as_of=as_of_date,
            portfolio=portfolio,
            skip_failed=skip_failed,
            drift_tolerance=drift_tolerance,
        )
        if switch_plan is not None:
            switch_plan["switch_allowed"] = switch_allowed
            switch_plan["blocked_reason"] = None if switch_allowed else "Evidence/live gate blocked switch"

    result = {
        "enabled": True,
        "as_of": as_of_date.isoformat(),
        "current_strategy": current_strategy,
        "current_params": current_params,
        "recommended_target": recommended_target,
        "score_margin": score_margin,
        "performance_gap": performance_gap,
        "switch_allowed": switch_allowed,
        "executed_action": executed_action,
        "decision_rule": decision_rule,
        "scoring_mode": scoring_mode,
        "switch_margin": float(switch_margin),
        "confirm_points": int(confirm_points),
        "decision_cadence": decision_cadence,
        "gate_fail_action": gate_fail_action,
        "regime_mode": regime_mode,
        "regime_profile": regime_profile,
        "alpha_tie_band": resolved_alpha_tie_band,
        "stress_alpha_tolerance": resolved_stress_alpha_tolerance,
        "conditioned_min_windows": resolved_conditioned_min_windows,
        "live_reasons": live_reasons,
        "candidates": [row.to_dict() for row in ranked],
        "evidence_required": bool(evidence_required),
        "evidence_profile": evidence_profile,
        "evidence_compare_mode": evidence_compare_mode,
        "evidence_status": evidence_status,
        "evidence_summary": (artifact.get("unconditional_summary") or artifact.get("summary")) if artifact else None,
        "evidence_reasons": evidence_reasons,
        "evidence_artifact_id": artifact.get("artifact_id") if artifact else None,
        "evidence_artifact_path": artifact.get("artifact_path") if artifact else None,
        "evidence_age_days": evidence_age_days,
        "conditioned_evidence_status": conditioned_evidence_status,
        "conditioned_windows": conditioned_windows,
        "conditioned_evidence_summary": conditioned_summary,
        "current_regime_bucket": current_row.regime_bucket,
        "current_regime_reasons": list(current_row.regime_reasons),
        "target_regime_bucket": selected_row.regime_bucket,
        "target_regime_reasons": list(selected_row.regime_reasons),
        "switch_checks": switch_checks,
        "blocked_checks": blocked_checks,
    }
    if switch_plan is not None:
        result["switch_plan"] = switch_plan

    append_meta_decision_log(
        {
            "timestamp": date.today().isoformat(),
            "as_of": as_of_date.isoformat(),
            "current_strategy": current_strategy,
            "recommended_target": recommended_target,
            "score_margin": score_margin,
            "performance_gap": performance_gap,
            "decision_rule": decision_rule,
            "current_regime_bucket": current_row.regime_bucket,
            "target_regime_bucket": selected_row.regime_bucket,
            "switch_allowed": switch_allowed,
            "executed_action": executed_action,
            "evidence_status": evidence_status,
            "conditioned_evidence_status": conditioned_evidence_status,
            "evidence_artifact_id": result.get("evidence_artifact_id"),
            "reasons": live_reasons + list(evidence_reasons or []),
        },
        log_path=decision_log_path,
    )

    return result
