"""Meta-switch evidence generation and persistence utilities."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backtest.backtester import BacktestConfig, Backtester
from backtest.data import DataLoader, PriceData
from backtest.metadata import collect_environment_metadata, collect_git_metadata
from backtest.meta_regime import (
    RegimeProfile,
    build_regime_measurements,
    classify_regime_measurements,
)
from backtest.strategy import Strategy


EvidenceProfile = Literal["defensiv", "ausgewogen", "aggressiv", "custom"]
EvidenceStatus = Literal["pass", "fail", "stale", "missing"]


DEFAULT_EVIDENCE_DIR = Path("results/meta_evidence")
DEFAULT_DECISION_LOG = Path("results/meta_decisions/decision_log.jsonl")


EVIDENCE_PROFILE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "defensiv": {
        "min_windows": 8,
        "min_cagr_edge_pp": 2.0,
        "min_hit_rate": 0.60,
        "max_degradation_pct": 25.0,
        "max_dd_worsening_pp": 5.0,
    },
    "ausgewogen": {
        "min_windows": 6,
        "min_cagr_edge_pp": 1.0,
        "min_hit_rate": 0.55,
        "max_degradation_pct": 35.0,
        "max_dd_worsening_pp": 8.0,
    },
    "aggressiv": {
        "min_windows": 4,
        "min_cagr_edge_pp": 0.5,
        "min_hit_rate": 0.50,
        "max_degradation_pct": 45.0,
        "max_dd_worsening_pp": 12.0,
    },
}

CONDITIONED_MIN_WINDOWS_DEFAULTS: Dict[str, int] = {
    "defensiv": 6,
    "ausgewogen": 4,
    "aggressiv": 3,
}


@dataclass
class EvidenceGateResult:
    """Gate result bundle for an evidence artifact."""

    passed: bool
    checks: Dict[str, bool]
    reasons: List[str]
    thresholds: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass": self.passed,
            "checks": self.checks,
            "reasons": self.reasons,
            "thresholds": self.thresholds,
        }


def _coerce_iso_date(value: Optional[str | date]) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed):
        return fallback
    return parsed


def _degradation_pct(train_metric: float, test_metric: float) -> float:
    if abs(train_metric) < 1e-12:
        return 0.0
    return (train_metric - test_metric) / abs(train_metric) * 100.0


def _load_strategy_instance(strategy_path: str | Path, params: Optional[Dict[str, Any]] = None) -> Strategy:
    path = Path(strategy_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    module_name = f"meta_evidence_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load strategy module: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    strategy_instance: Optional[Strategy] = None
    strategy_class: Optional[type[Strategy]] = None

    if hasattr(module, "strategy"):
        candidate = getattr(module, "strategy")
        if isinstance(candidate, Strategy):
            strategy_instance = candidate
            strategy_class = type(candidate)

    if strategy_class is None:
        for name in dir(module):
            candidate = getattr(module, name)
            if isinstance(candidate, type) and issubclass(candidate, Strategy) and candidate is not Strategy:
                strategy_class = candidate
                break

    if strategy_class is None and strategy_instance is None:
        raise ValueError(f"No Strategy subclass found in {path}")

    if params:
        if strategy_class is None:
            raise ValueError(f"Cannot apply params for strategy without class: {path}")
        return strategy_class(**params)

    if strategy_instance is not None:
        return strategy_instance
    return strategy_class()


def _build_walk_forward_windows(
    dates: pd.DatetimeIndex,
    train_days: int,
    test_days: int,
    step_days: int,
    anchored: bool = False,
) -> List[Dict[str, pd.Timestamp]]:
    windows: List[Dict[str, pd.Timestamp]] = []
    n = len(dates)
    if n < (train_days + test_days):
        return windows

    cursor = 0
    while True:
        if anchored:
            train_start_idx = 0
            train_end_idx = train_days - 1 + cursor
        else:
            train_start_idx = cursor
            train_end_idx = train_start_idx + train_days - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = test_start_idx + test_days - 1

        if test_end_idx >= n:
            break

        windows.append(
            {
                "train_start": pd.Timestamp(dates[train_start_idx]),
                "train_end": pd.Timestamp(dates[train_end_idx]),
                "test_start": pd.Timestamp(dates[test_start_idx]),
                "test_end": pd.Timestamp(dates[test_end_idx]),
            }
        )
        cursor += step_days
        if cursor >= n:
            break

    return windows


def _run_backtest_metrics(
    strategy: Strategy,
    data: PriceData,
    rebalance_frequency: Optional[str] = None,
    initial_capital: float = 10_000.0,
    costs_pct: float = 0.001,
    metric_basis: str = "gross",
) -> Dict[str, float]:
    return _run_backtest_bundle(
        strategy=strategy,
        data=data,
        rebalance_frequency=rebalance_frequency,
        initial_capital=initial_capital,
        costs_pct=costs_pct,
        metric_basis=metric_basis,
    )["metrics"]


def _resolve_profile_thresholds(
    evidence_profile: EvidenceProfile,
    custom_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    if evidence_profile == "custom":
        base = EVIDENCE_PROFILE_THRESHOLDS["ausgewogen"].copy()
        if custom_thresholds:
            for key, value in custom_thresholds.items():
                base[key] = _safe_float(value, fallback=base.get(key, 0.0))
        return base
    if evidence_profile not in EVIDENCE_PROFILE_THRESHOLDS:
        raise ValueError(f"Unsupported evidence profile: {evidence_profile}")
    return EVIDENCE_PROFILE_THRESHOLDS[evidence_profile].copy()


def resolve_conditioned_min_windows(
    evidence_profile: EvidenceProfile,
    conditioned_min_windows: Optional[int] = None,
) -> int:
    if conditioned_min_windows is not None:
        return max(1, int(conditioned_min_windows))
    if evidence_profile == "custom":
        return CONDITIONED_MIN_WINDOWS_DEFAULTS["ausgewogen"]
    return CONDITIONED_MIN_WINDOWS_DEFAULTS[evidence_profile]


def evaluate_evidence_gates(
    summary: Dict[str, Any],
    evidence_profile: EvidenceProfile = "ausgewogen",
    custom_thresholds: Optional[Dict[str, Any]] = None,
    min_windows_override: Optional[int] = None,
) -> EvidenceGateResult:
    thresholds = _resolve_profile_thresholds(evidence_profile, custom_thresholds)
    if min_windows_override is not None:
        thresholds["min_windows"] = max(1.0, float(min_windows_override))
    num_windows = int(summary.get("num_windows", 0))
    cagr_edge_pp = _safe_float(summary.get("oos_cagr_edge_pp", 0.0))
    hit_rate = _safe_float(summary.get("oos_hit_rate", 0.0))
    degradation_pct = _safe_float(summary.get("oos_degradation_pct", 0.0))
    dd_delta_pp = _safe_float(summary.get("oos_dd_delta_pp", 0.0))

    checks = {
        "min_windows": num_windows >= int(thresholds["min_windows"]),
        "cagr_edge": cagr_edge_pp >= thresholds["min_cagr_edge_pp"],
        "hit_rate": hit_rate >= thresholds["min_hit_rate"],
        "degradation": degradation_pct <= thresholds["max_degradation_pct"],
        "dd_worsening": dd_delta_pp <= thresholds["max_dd_worsening_pp"],
    }

    reasons: List[str] = []
    if not checks["min_windows"]:
        reasons.append(
            f"Too few OOS windows: {num_windows} < {int(thresholds['min_windows'])}"
        )
    if not checks["cagr_edge"]:
        reasons.append(
            f"OOS CAGR edge too low: {cagr_edge_pp:.2f}pp < {thresholds['min_cagr_edge_pp']:.2f}pp"
        )
    if not checks["hit_rate"]:
        reasons.append(
            f"OOS hit-rate too low: {hit_rate:.1%} < {thresholds['min_hit_rate']:.1%}"
        )
    if not checks["degradation"]:
        reasons.append(
            f"OOS degradation too high: {degradation_pct:.1f}% > {thresholds['max_degradation_pct']:.1f}%"
        )
    if not checks["dd_worsening"]:
        reasons.append(
            f"OOS drawdown worsening too high: {dd_delta_pp:.2f}pp > {thresholds['max_dd_worsening_pp']:.2f}pp"
        )

    return EvidenceGateResult(
        passed=all(checks.values()),
        checks=checks,
        reasons=reasons,
        thresholds=thresholds,
    )


def _run_backtest_bundle(
    strategy: Strategy,
    data: PriceData,
    rebalance_frequency: Optional[str] = None,
    initial_capital: float = 10_000.0,
    costs_pct: float = 0.001,
    metric_basis: str = "gross",
) -> Dict[str, Any]:
    freq = rebalance_frequency or getattr(strategy, "rebalance_frequency", "monthly")
    config = BacktestConfig(
        initial_capital=initial_capital,
        costs_pct=costs_pct,
        rebalance_frequency=freq,
        benchmark="S&P 500",
        tax_enabled=False,
        metric_basis=metric_basis,
        validate=False,
    )
    result = Backtester(strategy, data, config).run()
    metrics = result.metrics
    if metrics is None:
        metric_payload = {"cagr": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}
    else:
        metric_payload = {
            "cagr": _safe_float(getattr(metrics, "cagr", 0.0)),
            "max_drawdown": _safe_float(getattr(metrics, "max_drawdown", 0.0)),
            "sharpe_ratio": _safe_float(getattr(metrics, "sharpe_ratio", 0.0)),
        }
    return {
        "metrics": metric_payload,
        "equity_curve": result.equity_curve.copy(),
    }


def _summarize_window_rows(window_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not window_rows:
        return {
            "num_windows": 0,
            "oos_cagr_edge_pp": 0.0,
            "oos_hit_rate": 0.0,
            "oos_degradation_pct": 0.0,
            "oos_dd_delta_pp": 0.0,
        }

    edge_pp_values = [_safe_float(row.get("cagr_edge_pp", 0.0)) for row in window_rows]
    hit_values = [1.0 if _safe_float(row.get("cagr_edge_pp", 0.0)) > 0.0 else 0.0 for row in window_rows]
    degradation_values = [_safe_float(row.get("degradation_pct", 0.0)) for row in window_rows]
    dd_delta_values = [_safe_float(row.get("dd_delta_pp", 0.0)) for row in window_rows]
    return {
        "num_windows": len(window_rows),
        "oos_cagr_edge_pp": _safe_mean(edge_pp_values),
        "oos_hit_rate": _safe_mean(hit_values),
        "oos_degradation_pct": _safe_mean(degradation_values),
        "oos_dd_delta_pp": _safe_mean(dd_delta_values),
    }


def _classify_current_regime(equity_curve: pd.Series) -> Dict[str, Any]:
    measurements = build_regime_measurements(equity_curve)
    snapshots = {
        profile: classify_regime_measurements(measurements, profile=profile)
        for profile in ("defensiv", "ausgewogen", "aggressiv")
    }
    return {
        "status": measurements.status,
        "reference_days": measurements.reference_days,
        "available_feature_days": measurements.available_feature_days,
        "history_reason": measurements.history_reason,
        "metrics": dict(measurements.metrics),
        "percentiles": dict(measurements.percentiles),
        "buckets": {profile: snapshot.bucket for profile, snapshot in snapshots.items()},
        "reasons": {profile: list(snapshot.reasons) for profile, snapshot in snapshots.items()},
        "flags": {profile: dict(snapshot.flags) for profile, snapshot in snapshots.items()},
    }


def _build_conditioned_summary(window_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    conditioned: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for profile in ("defensiv", "ausgewogen", "aggressiv"):
        profile_summary: Dict[str, Dict[str, Any]] = {}
        for bucket in ("normal", "fragile", "stressed", "insufficient_history"):
            bucket_rows = [
                row for row in window_rows
                if ((row.get("current_regime") or {}).get("buckets") or {}).get(profile) == bucket
            ]
            profile_summary[bucket] = _summarize_window_rows(bucket_rows)
        conditioned[profile] = profile_summary
    return conditioned


def _strategy_label(strategy_path: str | Path) -> str:
    return Path(strategy_path).stem.lower()


def _pair_id(current_strategy: str | Path, target_strategy: str | Path) -> str:
    return f"{_strategy_label(current_strategy)}__to__{_strategy_label(target_strategy)}"


def _artifact_id(payload: Dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{digest}"


def _select_artifact_output_path(
    pair_id: str,
    artifact_id: str,
    evidence_artifact_path: Optional[str | Path],
) -> Path:
    if evidence_artifact_path:
        return Path(evidence_artifact_path).expanduser().resolve()
    return (DEFAULT_EVIDENCE_DIR / pair_id / f"{artifact_id}.json").resolve()


def run_meta_evidence_analysis(
    current_strategy: str | Path,
    target_strategy: str | Path,
    current_params: Optional[Dict[str, Any]] = None,
    target_params: Optional[Dict[str, Any]] = None,
    as_of: Optional[str | date] = None,
    evidence_profile: EvidenceProfile = "ausgewogen",
    evidence_compare_mode: Literal["vs_current"] = "vs_current",
    evidence_max_age_days: int = 30,
    evidence_artifact_path: Optional[str | Path] = None,
    custom_thresholds: Optional[Dict[str, Any]] = None,
    train_years: float = 5.0,
    test_years: float = 1.0,
    step_months: int = 12,
    anchored: bool = False,
    start_date: str = "2010-01-01",
    initial_capital: float = 10_000.0,
    costs_pct: float = 0.001,
    skip_failed: bool = True,
    metric_basis: str = "gross",
    save_artifact: bool = True,
) -> Dict[str, Any]:
    """Run OOS evidence analysis and persist a JSON artifact."""
    if evidence_compare_mode != "vs_current":
        raise ValueError("Only evidence_compare_mode='vs_current' is supported.")

    as_of_date = _coerce_iso_date(as_of)
    as_of_ts = pd.Timestamp(as_of_date)

    current_path = str(Path(current_strategy).expanduser().resolve())
    target_path = str(Path(target_strategy).expanduser().resolve())
    current_params = dict(current_params or {})
    target_params = dict(target_params or {})

    current_probe = _load_strategy_instance(current_path, current_params)
    target_probe = _load_strategy_instance(target_path, target_params)

    tickers = sorted(set(current_probe.assets) | set(target_probe.assets))
    if not tickers:
        raise ValueError("No assets found for evidence analysis.")

    data = DataLoader.yahoo(
        tickers=tickers,
        start=start_date,
        end=as_of_date.isoformat(),
        currency="EUR",
        align="ffill",
        skip_failed=skip_failed,
    )
    if data.prices.empty:
        raise ValueError("No price data available for evidence analysis.")

    train_days = max(21, int(train_years * 252))
    test_days = max(21, int(test_years * 252))
    step_days = max(21, int(step_months * 21))
    windows = _build_walk_forward_windows(
        dates=data.prices.index,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        anchored=anchored,
    )
    if not windows:
        raise ValueError(
            "Could not generate OOS windows. Increase data range or reduce train/test years."
        )

    window_rows: List[Dict[str, Any]] = []

    for window in windows:
        if window["test_end"] > as_of_ts:
            continue

        train_data = data.filter_dates(window["train_start"], window["train_end"])
        test_data = data.filter_dates(window["test_start"], window["test_end"])
        if train_data.prices.empty or test_data.prices.empty:
            continue

        target_train_strategy = _load_strategy_instance(target_path, target_params)
        target_test_strategy = _load_strategy_instance(target_path, target_params)
        current_test_strategy = _load_strategy_instance(current_path, current_params)

        target_train_bundle = _run_backtest_bundle(
            target_train_strategy,
            train_data,
            rebalance_frequency=getattr(target_train_strategy, "rebalance_frequency", "monthly"),
            initial_capital=initial_capital,
            costs_pct=costs_pct,
            metric_basis=metric_basis,
        )
        target_test_bundle = _run_backtest_bundle(
            target_test_strategy,
            test_data,
            rebalance_frequency=getattr(target_test_strategy, "rebalance_frequency", "monthly"),
            initial_capital=initial_capital,
            costs_pct=costs_pct,
            metric_basis=metric_basis,
        )
        current_test_bundle = _run_backtest_bundle(
            current_test_strategy,
            test_data,
            rebalance_frequency=getattr(current_test_strategy, "rebalance_frequency", "monthly"),
            initial_capital=initial_capital,
            costs_pct=costs_pct,
            metric_basis=metric_basis,
        )
        current_history_strategy = _load_strategy_instance(current_path, current_params)
        current_history_data = data.filter_dates(start_date, window["test_start"])
        current_history_bundle = _run_backtest_bundle(
            current_history_strategy,
            current_history_data,
            rebalance_frequency=getattr(current_history_strategy, "rebalance_frequency", "monthly"),
            initial_capital=initial_capital,
            costs_pct=costs_pct,
            metric_basis=metric_basis,
        )
        current_regime = _classify_current_regime(current_history_bundle["equity_curve"])

        target_train = target_train_bundle["metrics"]
        target_test = target_test_bundle["metrics"]
        current_test = current_test_bundle["metrics"]

        cagr_edge_pp = (target_test["cagr"] - current_test["cagr"]) * 100.0
        degradation_pct = _degradation_pct(target_train["cagr"], target_test["cagr"])
        dd_delta_pp = (abs(target_test["max_drawdown"]) - abs(current_test["max_drawdown"])) * 100.0

        window_rows.append(
            {
                "train_start": window["train_start"].strftime("%Y-%m-%d"),
                "train_end": window["train_end"].strftime("%Y-%m-%d"),
                "test_start": window["test_start"].strftime("%Y-%m-%d"),
                "test_end": window["test_end"].strftime("%Y-%m-%d"),
                "target_train": target_train,
                "target_test": target_test,
                "current_test": current_test,
                "cagr_edge_pp": cagr_edge_pp,
                "degradation_pct": degradation_pct,
                "dd_delta_pp": dd_delta_pp,
                "target_outperformed": cagr_edge_pp > 0.0,
                "current_regime": current_regime,
            }
        )

    if not window_rows:
        raise ValueError("No valid OOS windows found up to as_of date.")

    unconditional_summary = _summarize_window_rows(window_rows)
    unconditional_summary.update(
        {
            "train_years": train_years,
            "test_years": test_years,
            "step_months": step_months,
            "anchored": bool(anchored),
        }
    )
    conditioned_summary = _build_conditioned_summary(window_rows)
    summary = {
        **unconditional_summary,
        "train_years": train_years,
        "test_years": test_years,
        "step_months": step_months,
        "anchored": bool(anchored),
    }
    gates = evaluate_evidence_gates(
        summary=summary,
        evidence_profile=evidence_profile,
        custom_thresholds=custom_thresholds,
    )

    pair_id = _pair_id(current_path, target_path)
    identity_payload = {
        "pair_id": pair_id,
        "as_of": as_of_date.isoformat(),
        "evidence_profile": evidence_profile,
        "current_strategy": current_path,
        "target_strategy": target_path,
        "current_params": current_params,
        "target_params": target_params,
        "summary": summary,
    }
    artifact_id = _artifact_id(identity_payload)
    output_path = _select_artifact_output_path(
        pair_id=pair_id,
        artifact_id=artifact_id,
        evidence_artifact_path=evidence_artifact_path,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "artifact_id": artifact_id,
        "pair_id": pair_id,
        "as_of": as_of_date.isoformat(),
        "created_at": created_at,
        "evidence_profile": evidence_profile,
        "evidence_compare_mode": evidence_compare_mode,
        "evidence_max_age_days": int(evidence_max_age_days),
        "current_strategy": {
            "path": current_path,
            "params": current_params,
        },
        "target_strategy": {
            "path": target_path,
            "params": target_params,
        },
        "summary": summary,
        "unconditional_summary": unconditional_summary,
        "conditioned_summary": conditioned_summary,
        "gates": gates.to_dict(),
        "windows": window_rows,
        "run_metadata": {
            "git": collect_git_metadata().to_dict(),
            "environment": collect_environment_metadata().to_dict(),
            "metric_basis": metric_basis,
            "initial_capital": initial_capital,
            "costs_pct": costs_pct,
            "start_date": start_date,
        },
        "artifact_path": str(output_path),
    }

    if save_artifact:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2))

    return artifact


def load_evidence_artifact(path: str | Path) -> Dict[str, Any]:
    """Load a persisted evidence artifact JSON file."""
    artifact_path = Path(path).expanduser().resolve()
    return json.loads(artifact_path.read_text())


def find_latest_evidence_artifact(
    current_strategy: str | Path,
    target_strategy: str | Path,
    base_dir: str | Path = DEFAULT_EVIDENCE_DIR,
) -> Optional[Dict[str, Any]]:
    """Return latest artifact payload for a strategy pair, if present."""
    pair_id = _pair_id(current_strategy, target_strategy)
    pair_dir = Path(base_dir).expanduser().resolve() / pair_id
    if not pair_dir.exists():
        return None
    files = sorted(pair_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    return load_evidence_artifact(files[0])


def find_evidence_artifact_by_id(
    artifact_id: str,
    base_dir: str | Path = DEFAULT_EVIDENCE_DIR,
) -> Optional[Dict[str, Any]]:
    """Find an evidence artifact by id under the base evidence directory."""
    base = Path(base_dir).expanduser().resolve()
    if not base.exists():
        return None
    candidate = next(base.rglob(f"{artifact_id}.json"), None)
    if candidate is None:
        return None
    return load_evidence_artifact(candidate)


def assess_evidence_status(
    artifact: Optional[Dict[str, Any]],
    evidence_max_age_days: int = 30,
    as_of: Optional[str | date] = None,
    evidence_profile: Optional[EvidenceProfile] = None,
    custom_thresholds: Optional[Dict[str, Any]] = None,
) -> Tuple[EvidenceStatus, List[str], Optional[float]]:
    """Assess artifact status as pass/fail/stale/missing."""
    if artifact is None:
        return "missing", ["No evidence artifact found"], None

    reasons = list(artifact.get("gates", {}).get("reasons", []) or [])
    created_at_raw = artifact.get("created_at")
    age_days: Optional[float] = None
    if created_at_raw:
        try:
            created_dt = datetime.fromisoformat(str(created_at_raw).replace("Z", "+00:00"))
            ref_date = _coerce_iso_date(as_of)
            age_days = max(0.0, (ref_date - created_dt.date()).days)
        except Exception:
            reasons.append("Could not parse evidence created_at timestamp")

    if age_days is not None and age_days > float(evidence_max_age_days):
        return "stale", reasons + [f"Evidence stale: {age_days:.0f}d > {evidence_max_age_days}d"], age_days

    if evidence_profile is not None:
        summary = artifact.get("unconditional_summary") or artifact.get("summary") or {}
        if summary:
            gate_result = evaluate_evidence_gates(
                summary=summary,
                evidence_profile=evidence_profile,
                custom_thresholds=custom_thresholds,
            )
            passed = gate_result.passed
            reasons = gate_result.reasons
        else:
            passed = bool(artifact.get("gates", {}).get("pass", False))
    else:
        passed = bool(artifact.get("gates", {}).get("pass", False))
    if passed:
        return "pass", reasons, age_days
    return "fail", reasons or ["Evidence gates failed"], age_days


def assess_conditioned_evidence_status(
    artifact: Optional[Dict[str, Any]],
    *,
    current_bucket: Optional[str],
    evidence_profile: EvidenceProfile = "ausgewogen",
    conditioned_min_windows: Optional[int] = None,
    custom_thresholds: Optional[Dict[str, Any]] = None,
) -> Tuple[EvidenceStatus, List[str], int, Optional[Dict[str, Any]]]:
    """Assess the conditioned evidence bucket for fragility-driven switches."""
    if artifact is None:
        return "missing", ["No evidence artifact found"], 0, None
    if current_bucket is None:
        return "missing", ["No current regime bucket available"], 0, None

    profile_key = evidence_profile if evidence_profile in {"defensiv", "ausgewogen", "aggressiv"} else "ausgewogen"
    conditioned_summary = (
        ((artifact.get("conditioned_summary") or {}).get(profile_key) or {}).get(current_bucket)
    )
    if not conditioned_summary:
        return "missing", [f"No conditioned evidence found for bucket '{current_bucket}'"], 0, None

    num_windows = int(conditioned_summary.get("num_windows", 0))
    if num_windows <= 0:
        return "missing", [f"No conditioned evidence windows for bucket '{current_bucket}'"], 0, conditioned_summary

    gate_result = evaluate_evidence_gates(
        summary=conditioned_summary,
        evidence_profile=evidence_profile,
        custom_thresholds=custom_thresholds,
        min_windows_override=resolve_conditioned_min_windows(
            evidence_profile=evidence_profile,
            conditioned_min_windows=conditioned_min_windows,
        ),
    )
    if gate_result.passed:
        return "pass", gate_result.reasons, num_windows, conditioned_summary
    return "fail", gate_result.reasons, num_windows, conditioned_summary


def append_meta_decision_log(
    row: Dict[str, Any],
    log_path: str | Path = DEFAULT_DECISION_LOG,
) -> None:
    """Append one line to decision audit log."""
    path = Path(log_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str))
        handle.write("\n")


def load_recent_meta_decisions(
    log_path: str | Path = DEFAULT_DECISION_LOG,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Load most recent decision audit rows."""
    path = Path(log_path).expanduser().resolve()
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit <= 0:
        return rows
    return rows[-limit:]


# ---------------------------------------------------------------------------
# Phase B analyst-evidence OOS engine (T-0066).
# ---------------------------------------------------------------------------

ANALYST_EVIDENCE_PROFILE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "defensiv": {
        "min_windows": 12,
        "min_cagr_edge_pp": 1.5,
        "min_hit_rate": 0.58,
    },
    "ausgewogen": {
        "min_windows": 8,
        "min_cagr_edge_pp": 0.75,
        "min_hit_rate": 0.54,
    },
    "aggressiv": {
        "min_windows": 6,
        "min_cagr_edge_pp": 0.25,
        "min_hit_rate": 0.50,
    },
}


def _resolve_analyst_evidence_thresholds(
    profile: EvidenceProfile,
    custom_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    if profile == "custom":
        base = ANALYST_EVIDENCE_PROFILE_THRESHOLDS["ausgewogen"].copy()
        if custom_thresholds:
            for key, value in custom_thresholds.items():
                if key in base:
                    base[key] = float(value)
        return base
    return ANALYST_EVIDENCE_PROFILE_THRESHOLDS.get(
        profile, ANALYST_EVIDENCE_PROFILE_THRESHOLDS["ausgewogen"]
    ).copy()


def _analyst_tilt_weights(
    snapshot_df: pd.DataFrame,
    universe: Sequence[str],
) -> Dict[str, float]:
    """Build the monthly Analyst-Tilt allocation per plan T-0066.

    Top-third by score, only positive scores, equal-weight; if no
    positive scores exist, fall back to equal-weight on the whole
    universe.
    """

    universe_upper = [str(u).upper() for u in universe]
    if snapshot_df is None or snapshot_df.empty or "feature_name" not in snapshot_df.columns:
        return _equal_weights(universe_upper)
    score_rows = snapshot_df.loc[snapshot_df["feature_name"] == "analyst_score"]
    if score_rows.empty:
        return _equal_weights(universe_upper)
    scores: Dict[str, float] = {}
    for ticker, group in score_rows.groupby("ticker"):
        ticker_upper = str(ticker).upper()
        if ticker_upper not in universe_upper:
            continue
        try:
            scores[ticker_upper] = float(group["feature_value"].iloc[-1])
        except (TypeError, ValueError):
            continue
    positive = {t: s for t, s in scores.items() if s >= 0.0}
    if not positive:
        return _equal_weights(universe_upper)
    sorted_picks = sorted(positive.items(), key=lambda kv: kv[1], reverse=True)
    cutoff = max(1, len(sorted_picks) // 3)
    top = [t for t, _ in sorted_picks[:cutoff]]
    if not top:
        return _equal_weights(universe_upper)
    weight = 1.0 / len(top)
    return {t: weight for t in top}


def _equal_weights(universe: Sequence[str]) -> Dict[str, float]:
    if not universe:
        return {}
    weight = 1.0 / len(universe)
    return {t: weight for t in universe}


def _portfolio_return(weights: Dict[str, float], asset_returns: Dict[str, float]) -> float:
    total = 0.0
    used = 0.0
    for ticker, weight in weights.items():
        ret = asset_returns.get(ticker)
        if ret is None:
            continue
        total += float(weight) * float(ret)
        used += float(weight)
    if used <= 0:
        return 0.0
    return total / used


def _provider_snapshot(
    provider: Any,
    dataset: Optional[str],
    as_of: date,
    tickers: Sequence[str],
):
    if provider is None:
        return None
    try:
        if dataset and hasattr(provider, "snapshot_dataset"):
            return provider.snapshot_dataset(dataset, as_of=as_of, tickers=list(tickers))
        return provider.snapshot(as_of=as_of, tickers=list(tickers))
    except Exception:
        return None


def _snapshot_source_is_synthetic(snapshot_df: pd.DataFrame) -> bool:
    if snapshot_df is None or snapshot_df.empty or "source" not in snapshot_df.columns:
        return False
    sources = {str(s).strip().lower() for s in snapshot_df["source"].unique()}
    if not sources:
        return False
    # If every row originates from SyntheticPIT, this is synthetic data.
    return sources == {"syntheticpit"}


def assess_analyst_evidence(
    *,
    external_provider: Any,
    current_assets: Iterable[str],
    candidate_assets: Iterable[str],
    as_of: Optional[date | str] = None,
    analyst_dataset: Optional[str] = None,
    lookback_months: int = 24,
    price_loader: Optional[Any] = None,
    profile: EvidenceProfile = "ausgewogen",
    allow_synthetic_analyst_evidence: bool = False,
    custom_thresholds: Optional[Dict[str, Any]] = None,
    regime_classifier: Optional[Any] = None,
) -> Tuple[EvidenceStatus, List[str], Optional[Dict[str, Any]]]:
    """Walk-Forward OOS check for the analyst tilt vs equal-weight B&H.

    Returns ``(status, reasons, summary)``. ``summary`` is the artifact-
    style dict with per-window metrics; ``None`` when status is
    ``missing``.
    """

    reasons: List[str] = []
    universe = sorted(
        {str(t).upper() for t in list(current_assets) + list(candidate_assets) if t}
    )
    if not universe:
        return "missing", ["Universe is empty"], None
    if external_provider is None:
        return "missing", ["No external features provider available"], None

    as_of_date = _coerce_iso_date(as_of)
    thresholds = _resolve_analyst_evidence_thresholds(profile, custom_thresholds)

    month_ends = pd.date_range(
        end=pd.Timestamp(as_of_date),
        periods=lookback_months + 1,
        freq="ME",
    )
    month_ends = [d.date() for d in month_ends]
    if len(month_ends) < 2:
        return "missing", ["Insufficient monthly lookback"], None

    # Load price history covering the lookback + one extra month for the
    # final forward return.
    history_start = (as_of_date - pd.Timedelta(days=int(lookback_months * 32))).isoformat()
    price_data = _load_universe_prices(price_loader, universe, history_start, as_of_date.isoformat())
    if price_data is None or price_data.empty:
        return "missing", ["No price data available for evidence universe"], None
    available = [c for c in universe if c in price_data.columns]
    if not available:
        return "missing", ["None of the universe tickers have price data"], None
    price_data = price_data[available]

    window_rows: List[Dict[str, Any]] = []
    synthetic_detected = False
    classifier = regime_classifier or _default_regime_classifier_factory(profile)

    for idx in range(len(month_ends) - 1):
        t0 = month_ends[idx]
        t1 = month_ends[idx + 1]
        snap = _provider_snapshot(external_provider, analyst_dataset, t0, available)
        snapshot_df = getattr(snap, "data", None) if snap is not None else None
        if snapshot_df is not None and not snapshot_df.empty:
            if _snapshot_source_is_synthetic(snapshot_df):
                synthetic_detected = True
        weights_tilt = _analyst_tilt_weights(snapshot_df, available)
        weights_bah = _equal_weights(available)

        forward = _forward_returns(price_data, t0, t1, available)
        if not forward:
            continue
        tilt_ret = _portfolio_return(weights_tilt, forward)
        bah_ret = _portfolio_return(weights_bah, forward)
        # PIT-safe regime classification: use only price data up to and
        # including t0 — never the forward window we are about to score.
        bucket = "insufficient_history"
        if classifier is not None:
            try:
                bucket = classifier(price_data, t0) or "insufficient_history"
            except Exception:
                bucket = "insufficient_history"
        window_rows.append(
            {
                "month_end": t0.isoformat(),
                "tilt_return": float(tilt_ret),
                "bah_return": float(bah_ret),
                "edge": float(tilt_ret - bah_ret),
                "regime_bucket": str(bucket),
            }
        )

    if synthetic_detected and not allow_synthetic_analyst_evidence:
        return (
            "missing",
            [
                "Analyst snapshots originate from SyntheticPIT; "
                "set allow_synthetic_analyst_evidence=True to evaluate"
            ],
            None,
        )

    if not window_rows:
        return "missing", ["No usable monthly windows"], None

    edges = [row["edge"] for row in window_rows]
    hit_rate = float(sum(1 for e in edges if e > 0) / len(edges))
    mean_edge = float(np.mean(edges))
    months = len(window_rows)
    cagr_edge_pp = mean_edge * 12.0 * 100.0  # annualised average monthly edge in pp

    checks = {
        "min_windows": months >= int(thresholds["min_windows"]),
        "min_cagr_edge_pp": cagr_edge_pp >= float(thresholds["min_cagr_edge_pp"]),
        "min_hit_rate": hit_rate >= float(thresholds["min_hit_rate"]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        reasons.append("Failed analyst gates: " + ", ".join(failed))
        status: EvidenceStatus = "fail"
    else:
        status = "pass"

    conditioned = _aggregate_conditioned_buckets(window_rows)

    summary = {
        "profile": profile,
        "thresholds": thresholds,
        "checks": checks,
        "windows": months,
        "hit_rate": hit_rate,
        "cagr_edge_pp": cagr_edge_pp,
        "as_of": as_of_date.isoformat(),
        "universe": available,
        "row_summary": window_rows[-min(6, len(window_rows)):],
        "synthetic_source": synthetic_detected,
        "conditioned": conditioned,
    }
    return status, reasons, summary


def _aggregate_conditioned_buckets(
    window_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Group window-level edges by regime bucket (T-0099).

    Each window carries a ``regime_bucket`` produced at t0 (PIT-safe).
    The result maps bucket -> aggregated metrics in the same shape as
    the unconditional headline summary so callers can reuse the gate
    evaluation logic.
    """

    if not window_rows:
        return {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in window_rows:
        bucket = str(row.get("regime_bucket") or "insufficient_history")
        grouped.setdefault(bucket, []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for bucket, rows in grouped.items():
        edges = [float(r["edge"]) for r in rows]
        if not edges:
            continue
        hit_rate = float(sum(1 for e in edges if e > 0) / len(edges))
        mean_edge = float(np.mean(edges))
        cagr_edge_pp = mean_edge * 12.0 * 100.0
        out[bucket] = {
            "num_windows": len(rows),
            "hit_rate": hit_rate,
            "mean_edge": mean_edge,
            "cagr_edge_pp": cagr_edge_pp,
            "rows": rows[-min(6, len(rows)):],
        }
    return out


def _default_regime_classifier_factory(profile: EvidenceProfile) -> Any:
    """Return a PIT-safe regime classifier that uses an equal-weight
    universe equity curve up to t0 as a proxy.

    The returned callable signature is ``classifier(price_data, t0) -> bucket``.
    Imports happen lazily so that meta_evidence stays import-light at
    module load time.
    """

    profile_key = profile if profile in {"defensiv", "ausgewogen", "aggressiv"} else "ausgewogen"

    def _classify(price_data: pd.DataFrame, t0: date) -> str:
        from backtest.meta_regime import assess_regime_from_equity_curve

        try:
            slice_df = price_data.loc[: pd.Timestamp(t0)]
            if slice_df is None or slice_df.empty:
                return "insufficient_history"
            # Equal-weight equity curve normalised to start at 1.0; we
            # only need the SHAPE of returns, not absolute levels.
            normed = slice_df.divide(slice_df.iloc[0])
            curve = normed.mean(axis=1).dropna()
            if curve.empty:
                return "insufficient_history"
            snapshot = assess_regime_from_equity_curve(curve, profile=profile_key)
            return getattr(snapshot, "bucket", "insufficient_history") or "insufficient_history"
        except Exception:
            return "insufficient_history"

    return _classify


def assess_conditioned_analyst_evidence_status(
    summary: Optional[Dict[str, Any]],
    *,
    current_bucket: Optional[str],
    profile: EvidenceProfile = "ausgewogen",
    conditioned_min_windows: Optional[int] = None,
    custom_thresholds: Optional[Dict[str, Any]] = None,
) -> Tuple[EvidenceStatus, List[str], int, Optional[Dict[str, Any]]]:
    """Bucket-conditioned PASS/FAIL gate over an analyst evidence summary.

    Mirrors :func:`assess_conditioned_evidence_status` but works against
    the analyst-specific conditioned aggregation produced by
    :func:`assess_analyst_evidence`. Returns ``(status, reasons,
    num_windows, bucket_summary)``.

    Thresholds reuse the unconditional analyst gates but with a relaxed
    ``min_windows`` value, because conditioned buckets receive a subset
    of the lookback windows.
    """

    if summary is None:
        return "missing", ["No analyst evidence summary available"], 0, None
    if current_bucket is None:
        return "missing", ["No current regime bucket available"], 0, None

    conditioned = summary.get("conditioned") or {}
    bucket_summary = conditioned.get(current_bucket) if isinstance(conditioned, dict) else None
    if not bucket_summary:
        return (
            "missing",
            [f"No conditioned analyst evidence for bucket '{current_bucket}'"],
            0,
            None,
        )

    num_windows = int(bucket_summary.get("num_windows", 0))
    if num_windows <= 0:
        return (
            "missing",
            [f"Conditioned analyst evidence empty for bucket '{current_bucket}'"],
            0,
            bucket_summary,
        )

    base_thresholds = _resolve_analyst_evidence_thresholds(profile, custom_thresholds)
    relaxed_min = conditioned_min_windows
    if relaxed_min is None:
        # default: half of unconditional, but at least three windows.
        relaxed_min = max(3, int(base_thresholds["min_windows"]) // 2)

    checks = {
        "min_windows": num_windows >= int(relaxed_min),
        "min_cagr_edge_pp": float(bucket_summary.get("cagr_edge_pp", 0.0))
        >= float(base_thresholds["min_cagr_edge_pp"]),
        "min_hit_rate": float(bucket_summary.get("hit_rate", 0.0))
        >= float(base_thresholds["min_hit_rate"]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    reasons = [f"Bucket '{current_bucket}' windows={num_windows}, "
               f"cagr_edge_pp={float(bucket_summary.get('cagr_edge_pp', 0.0)):.2f}, "
               f"hit_rate={float(bucket_summary.get('hit_rate', 0.0)):.2f}"]
    if failed:
        reasons.append("Failed conditioned analyst gates: " + ", ".join(failed))
        status: EvidenceStatus = "fail"
    else:
        status = "pass"
    enriched = dict(bucket_summary)
    enriched["checks"] = checks
    enriched["min_windows_required"] = int(relaxed_min)
    enriched["thresholds"] = base_thresholds
    return status, reasons, num_windows, enriched


def _forward_returns(
    price_data: pd.DataFrame,
    t0: date,
    t1: date,
    tickers: Sequence[str],
) -> Dict[str, float]:
    if price_data is None or price_data.empty:
        return {}
    end_ts = pd.Timestamp(t1)
    start_ts = pd.Timestamp(t0)
    try:
        start_row = price_data.loc[:start_ts].iloc[-1]
        end_row = price_data.loc[:end_ts].iloc[-1]
    except (KeyError, IndexError):
        return {}
    forward: Dict[str, float] = {}
    for ticker in tickers:
        if ticker not in price_data.columns:
            continue
        try:
            p0 = float(start_row[ticker])
            p1 = float(end_row[ticker])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(p0) or not math.isfinite(p1) or p0 <= 0:
            continue
        forward[ticker] = (p1 / p0) - 1.0
    return forward


def _load_universe_prices(
    price_loader: Optional[Any],
    universe: Sequence[str],
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """Resolve a price DataFrame for the universe.

    Accepts either a callable ``price_loader(tickers, start, end) -> DataFrame``
    (the common test injection point) or ``None`` to fall back to
    DataLoader.yahoo.
    """

    if price_loader is not None:
        try:
            df = price_loader(list(universe), start, end)
            if isinstance(df, pd.DataFrame):
                return df
            return getattr(df, "prices", None)
        except Exception:
            return None
    try:
        from backtest.data import DataLoader

        data = DataLoader.yahoo(
            tickers=list(universe),
            start=start,
            end=end,
            currency="EUR",
            align="ffill",
            skip_failed=True,
        )
        return data.prices
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase C news-evidence OOS engine (T-0119/T-0120).
# ---------------------------------------------------------------------------

NEWS_EVIDENCE_PROFILE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "defensiv": {
        "min_windows": 12,
        "min_cagr_edge_pp": 1.5,
        "min_hit_rate": 0.58,
    },
    "ausgewogen": {
        "min_windows": 8,
        "min_cagr_edge_pp": 0.75,
        "min_hit_rate": 0.54,
    },
    "aggressiv": {
        "min_windows": 6,
        "min_cagr_edge_pp": 0.25,
        "min_hit_rate": 0.50,
    },
}


def _resolve_news_evidence_thresholds(
    profile: EvidenceProfile,
    custom_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    if profile == "custom":
        base = NEWS_EVIDENCE_PROFILE_THRESHOLDS["ausgewogen"].copy()
        if custom_thresholds:
            for key, value in custom_thresholds.items():
                if key in base:
                    base[key] = float(value)
        return base
    return NEWS_EVIDENCE_PROFILE_THRESHOLDS.get(
        profile, NEWS_EVIDENCE_PROFILE_THRESHOLDS["ausgewogen"]
    ).copy()


def _news_tilt_weights(
    snapshot_df: pd.DataFrame,
    universe: Sequence[str],
) -> Dict[str, float]:
    """Build the monthly news-tilt allocation.

    Top-quartile by ``news_sentiment_score``, only positive scores,
    equal-weight; fallback to equal-weight on the full universe if no
    positive scores exist.
    """

    universe_upper = [str(u).upper() for u in universe]
    if snapshot_df is None or snapshot_df.empty or "feature_name" not in snapshot_df.columns:
        return _equal_weights(universe_upper)
    score_rows = snapshot_df.loc[snapshot_df["feature_name"] == "news_sentiment_score"]
    if score_rows.empty:
        return _equal_weights(universe_upper)
    scores: Dict[str, float] = {}
    for ticker, group in score_rows.groupby("ticker"):
        ticker_upper = str(ticker).upper()
        if ticker_upper not in universe_upper:
            continue
        try:
            scores[ticker_upper] = float(group["feature_value"].iloc[-1])
        except (TypeError, ValueError):
            continue
    positive = {t: s for t, s in scores.items() if s > 0.0}
    if not positive:
        return _equal_weights(universe_upper)
    sorted_picks = sorted(positive.items(), key=lambda kv: kv[1], reverse=True)
    cutoff = max(1, len(sorted_picks) // 4)
    top = [t for t, _ in sorted_picks[:cutoff]]
    if not top:
        return _equal_weights(universe_upper)
    weight = 1.0 / len(top)
    return {t: weight for t in top}


def _news_snapshot_is_synthetic(snapshot_df: pd.DataFrame) -> bool:
    if snapshot_df is None or snapshot_df.empty or "source" not in snapshot_df.columns:
        return False
    sources = {str(s).strip().lower() for s in snapshot_df["source"].unique()}
    if not sources:
        return False
    return sources == {"syntheticnewspit"}


def assess_news_evidence(
    *,
    external_provider: Any,
    current_assets: Iterable[str],
    candidate_assets: Iterable[str],
    as_of: Optional[date | str] = None,
    news_dataset: Optional[str] = None,
    lookback_months: int = 24,
    price_loader: Optional[Any] = None,
    profile: EvidenceProfile = "ausgewogen",
    allow_synthetic_news_evidence: bool = False,
    custom_thresholds: Optional[Dict[str, Any]] = None,
    regime_classifier: Optional[Any] = None,
) -> Tuple[EvidenceStatus, List[str], Optional[Dict[str, Any]]]:
    """Walk-Forward OOS check for the news-sentiment tilt vs equal-weight B&H.

    Mirrors :func:`assess_analyst_evidence` and re-uses the same
    PIT-safe helpers; tilt logic differs (top-quartile of positive
    sentiment scores, not top-third of analyst score).
    """

    reasons: List[str] = []
    universe = sorted(
        {str(t).upper() for t in list(current_assets) + list(candidate_assets) if t}
    )
    if not universe:
        return "missing", ["Universe is empty"], None
    if external_provider is None:
        return "missing", ["No external features provider available"], None

    as_of_date = _coerce_iso_date(as_of)
    thresholds = _resolve_news_evidence_thresholds(profile, custom_thresholds)

    month_ends = pd.date_range(
        end=pd.Timestamp(as_of_date),
        periods=lookback_months + 1,
        freq="ME",
    )
    month_ends = [d.date() for d in month_ends]
    if len(month_ends) < 2:
        return "missing", ["Insufficient monthly lookback"], None

    history_start = (as_of_date - pd.Timedelta(days=int(lookback_months * 32))).isoformat()
    price_data = _load_universe_prices(
        price_loader, universe, history_start, as_of_date.isoformat()
    )
    if price_data is None or price_data.empty:
        return "missing", ["No price data available for news evidence universe"], None
    available = [c for c in universe if c in price_data.columns]
    if not available:
        return "missing", ["None of the universe tickers have price data"], None
    price_data = price_data[available]

    window_rows: List[Dict[str, Any]] = []
    synthetic_detected = False
    classifier = regime_classifier or _default_regime_classifier_factory(profile)

    for idx in range(len(month_ends) - 1):
        t0 = month_ends[idx]
        t1 = month_ends[idx + 1]
        snap = _provider_snapshot(external_provider, news_dataset, t0, available)
        snapshot_df = getattr(snap, "data", None) if snap is not None else None
        if snapshot_df is not None and not snapshot_df.empty:
            if _news_snapshot_is_synthetic(snapshot_df):
                synthetic_detected = True
        weights_tilt = _news_tilt_weights(snapshot_df, available)
        weights_bah = _equal_weights(available)

        forward = _forward_returns(price_data, t0, t1, available)
        if not forward:
            continue
        tilt_ret = _portfolio_return(weights_tilt, forward)
        bah_ret = _portfolio_return(weights_bah, forward)
        bucket = "insufficient_history"
        if classifier is not None:
            try:
                bucket = classifier(price_data, t0) or "insufficient_history"
            except Exception:
                bucket = "insufficient_history"
        window_rows.append(
            {
                "month_end": t0.isoformat(),
                "tilt_return": float(tilt_ret),
                "bah_return": float(bah_ret),
                "edge": float(tilt_ret - bah_ret),
                "regime_bucket": str(bucket),
            }
        )

    if synthetic_detected and not allow_synthetic_news_evidence:
        return (
            "missing",
            [
                "News snapshots originate from SyntheticNewsPIT; "
                "set allow_synthetic_news_evidence=True to evaluate"
            ],
            None,
        )

    if not window_rows:
        return "missing", ["No usable monthly windows"], None

    edges = [row["edge"] for row in window_rows]
    hit_rate = float(sum(1 for e in edges if e > 0) / len(edges))
    mean_edge = float(np.mean(edges))
    months = len(window_rows)
    cagr_edge_pp = mean_edge * 12.0 * 100.0

    checks = {
        "min_windows": months >= int(thresholds["min_windows"]),
        "min_cagr_edge_pp": cagr_edge_pp >= float(thresholds["min_cagr_edge_pp"]),
        "min_hit_rate": hit_rate >= float(thresholds["min_hit_rate"]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        reasons.append("Failed news gates: " + ", ".join(failed))
        status: EvidenceStatus = "fail"
    else:
        status = "pass"

    conditioned = _aggregate_conditioned_buckets(window_rows)

    summary = {
        "profile": profile,
        "thresholds": thresholds,
        "checks": checks,
        "windows": months,
        "hit_rate": hit_rate,
        "cagr_edge_pp": cagr_edge_pp,
        "as_of": as_of_date.isoformat(),
        "universe": available,
        "row_summary": window_rows[-min(6, len(window_rows)):],
        "synthetic_source": synthetic_detected,
        "conditioned": conditioned,
    }
    return status, reasons, summary


def assess_conditioned_news_evidence_status(
    summary: Optional[Dict[str, Any]],
    *,
    current_bucket: Optional[str],
    profile: EvidenceProfile = "ausgewogen",
    conditioned_min_windows: Optional[int] = None,
    custom_thresholds: Optional[Dict[str, Any]] = None,
) -> Tuple[EvidenceStatus, List[str], int, Optional[Dict[str, Any]]]:
    """Bucket-conditioned PASS/FAIL gate over a news evidence summary."""

    if summary is None:
        return "missing", ["No news evidence summary available"], 0, None
    if current_bucket is None:
        return "missing", ["No current regime bucket available"], 0, None

    conditioned = summary.get("conditioned") or {}
    bucket_summary = conditioned.get(current_bucket) if isinstance(conditioned, dict) else None
    if not bucket_summary:
        return (
            "missing",
            [f"No conditioned news evidence for bucket '{current_bucket}'"],
            0,
            None,
        )

    num_windows = int(bucket_summary.get("num_windows", 0))
    if num_windows <= 0:
        return (
            "missing",
            [f"Conditioned news evidence empty for bucket '{current_bucket}'"],
            0,
            bucket_summary,
        )

    base_thresholds = _resolve_news_evidence_thresholds(profile, custom_thresholds)
    relaxed_min = conditioned_min_windows
    if relaxed_min is None:
        relaxed_min = max(3, int(base_thresholds["min_windows"]) // 2)

    checks = {
        "min_windows": num_windows >= int(relaxed_min),
        "min_cagr_edge_pp": float(bucket_summary.get("cagr_edge_pp", 0.0))
        >= float(base_thresholds["min_cagr_edge_pp"]),
        "min_hit_rate": float(bucket_summary.get("hit_rate", 0.0))
        >= float(base_thresholds["min_hit_rate"]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    reasons = [
        f"News bucket '{current_bucket}' windows={num_windows}, "
        f"cagr_edge_pp={float(bucket_summary.get('cagr_edge_pp', 0.0)):.2f}, "
        f"hit_rate={float(bucket_summary.get('hit_rate', 0.0)):.2f}"
    ]
    if failed:
        reasons.append("Failed conditioned news gates: " + ", ".join(failed))
        status: EvidenceStatus = "fail"
    else:
        status = "pass"
    enriched = dict(bucket_summary)
    enriched["checks"] = checks
    enriched["min_windows_required"] = int(relaxed_min)
    enriched["thresholds"] = base_thresholds
    return status, reasons, num_windows, enriched


# ---------------------------------------------------------------------------
# Phase D ml-forecast evidence OOS engine (T-0223/T-0224).
# ---------------------------------------------------------------------------

ML_EVIDENCE_PROFILE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "defensiv": {
        "min_windows": 12,
        "min_cagr_edge_pp": 1.5,
        "min_hit_rate": 0.58,
    },
    "ausgewogen": {
        "min_windows": 8,
        "min_cagr_edge_pp": 0.75,
        "min_hit_rate": 0.54,
    },
    "aggressiv": {
        "min_windows": 6,
        "min_cagr_edge_pp": 0.25,
        "min_hit_rate": 0.50,
    },
}


def _resolve_ml_evidence_thresholds(
    profile: EvidenceProfile,
    custom_thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    if profile == "custom":
        base = ML_EVIDENCE_PROFILE_THRESHOLDS["ausgewogen"].copy()
        if custom_thresholds:
            for key, value in custom_thresholds.items():
                if key in base:
                    base[key] = float(value)
        return base
    return ML_EVIDENCE_PROFILE_THRESHOLDS.get(
        profile, ML_EVIDENCE_PROFILE_THRESHOLDS["ausgewogen"]
    ).copy()


def _ml_tilt_weights(
    snapshot_df: pd.DataFrame,
    universe: Sequence[str],
) -> Dict[str, float]:
    """Build the monthly ML-forecast tilt allocation (top-quartile, equal-weight).

    Mirrors :func:`_news_tilt_weights`. Falls back to equal-weight if no
    ticker has a positive ``ml_forecast_score`` row in the snapshot.
    """

    universe_upper = [str(u).upper() for u in universe]
    if (
        snapshot_df is None
        or snapshot_df.empty
        or "feature_name" not in snapshot_df.columns
    ):
        return _equal_weights(universe_upper)
    score_rows = snapshot_df.loc[snapshot_df["feature_name"] == "ml_forecast_score"]
    if score_rows.empty:
        return _equal_weights(universe_upper)
    scores: Dict[str, float] = {}
    for ticker, group in score_rows.groupby("ticker"):
        ticker_upper = str(ticker).upper()
        if ticker_upper not in universe_upper:
            continue
        try:
            scores[ticker_upper] = float(group["feature_value"].iloc[-1])
        except (TypeError, ValueError):
            continue
    positive = {t: s for t, s in scores.items() if s > 0.0}
    if not positive:
        return _equal_weights(universe_upper)
    sorted_picks = sorted(positive.items(), key=lambda kv: kv[1], reverse=True)
    cutoff = max(1, len(sorted_picks) // 4)
    top = [t for t, _ in sorted_picks[:cutoff]]
    if not top:
        return _equal_weights(universe_upper)
    weight = 1.0 / len(top)
    return {t: weight for t in top}


def _ml_snapshot_is_synthetic(snapshot_df: pd.DataFrame) -> bool:
    if snapshot_df is None or snapshot_df.empty or "source" not in snapshot_df.columns:
        return False
    sources = {str(s).strip().lower() for s in snapshot_df["source"].unique()}
    if not sources:
        return False
    return sources == {"syntheticmlforecast"}


def _ml_snapshot_filter_lookahead(
    snapshot_df: pd.DataFrame,
    release_date: date,
) -> pd.DataFrame:
    """Codex D14/D18: drop snapshot rows where the underlying bundle was
    trained on data more recent than ``release_date``.

    The check reads ``ml_available_from_ordinal`` straight from the
    DataFrame — no manifest/provenance look-up — so the hot-path stays
    cheap. Tickers whose bundle was effectively a "future model" at
    ``release_date`` get filtered out entirely.
    """

    if snapshot_df is None or snapshot_df.empty:
        return snapshot_df
    if "feature_name" not in snapshot_df.columns:
        return snapshot_df
    cutoff = int(release_date.toordinal())
    af = snapshot_df.loc[
        snapshot_df["feature_name"] == "ml_available_from_ordinal"
    ]
    if af.empty:
        return snapshot_df
    af_by_ticker: Dict[str, int] = {}
    for ticker, group in af.groupby("ticker"):
        try:
            af_by_ticker[str(ticker).upper()] = int(
                float(group["feature_value"].iloc[-1])
            )
        except (TypeError, ValueError):
            continue
    valid_tickers = {
        t for t, ordinal in af_by_ticker.items() if ordinal <= cutoff
    }
    if not valid_tickers:
        return snapshot_df.iloc[0:0]
    return snapshot_df.loc[
        snapshot_df["ticker"].astype(str).str.upper().isin(valid_tickers)
    ]


def assess_ml_evidence(
    *,
    external_provider: Any,
    current_assets: Iterable[str],
    candidate_assets: Iterable[str],
    as_of: Optional[date | str] = None,
    ml_dataset: Optional[str] = None,
    lookback_months: int = 24,
    price_loader: Optional[Any] = None,
    profile: EvidenceProfile = "ausgewogen",
    allow_synthetic_ml_evidence: bool = False,
    custom_thresholds: Optional[Dict[str, Any]] = None,
    regime_classifier: Optional[Any] = None,
) -> Tuple[EvidenceStatus, List[str], Optional[Dict[str, Any]]]:
    """Walk-Forward OOS check for the ML-forecast tilt vs equal-weight B&H.

    Mirrors :func:`assess_news_evidence`. The tilt is top-quartile by
    ``ml_forecast_score`` (positive scores only). Snapshots whose
    bundle was effectively a "future model" relative to the window's
    release date are excluded via
    :func:`_ml_snapshot_filter_lookahead` (Codex D14/D18).
    """

    reasons: List[str] = []
    universe = sorted(
        {str(t).upper() for t in list(current_assets) + list(candidate_assets) if t}
    )
    if not universe:
        return "missing", ["Universe is empty"], None
    if external_provider is None:
        return "missing", ["No external features provider available"], None

    as_of_date = _coerce_iso_date(as_of)
    thresholds = _resolve_ml_evidence_thresholds(profile, custom_thresholds)

    month_ends = pd.date_range(
        end=pd.Timestamp(as_of_date),
        periods=lookback_months + 1,
        freq="ME",
    )
    month_ends = [d.date() for d in month_ends]
    if len(month_ends) < 2:
        return "missing", ["Insufficient monthly lookback"], None

    history_start = (
        as_of_date - pd.Timedelta(days=int(lookback_months * 32))
    ).isoformat()
    price_data = _load_universe_prices(
        price_loader, universe, history_start, as_of_date.isoformat()
    )
    if price_data is None or price_data.empty:
        return "missing", ["No price data available for ml evidence universe"], None
    available = [c for c in universe if c in price_data.columns]
    if not available:
        return "missing", ["None of the universe tickers have price data"], None
    price_data = price_data[available]

    window_rows: List[Dict[str, Any]] = []
    synthetic_detected = False
    classifier = regime_classifier or _default_regime_classifier_factory(profile)

    for idx in range(len(month_ends) - 1):
        t0 = month_ends[idx]
        t1 = month_ends[idx + 1]
        snap = _provider_snapshot(external_provider, ml_dataset, t0, available)
        snapshot_df = getattr(snap, "data", None) if snap is not None else None
        if snapshot_df is not None and not snapshot_df.empty:
            if _ml_snapshot_is_synthetic(snapshot_df):
                synthetic_detected = True
            snapshot_df = _ml_snapshot_filter_lookahead(snapshot_df, t0)
        weights_tilt = _ml_tilt_weights(snapshot_df, available)
        weights_bah = _equal_weights(available)

        forward = _forward_returns(price_data, t0, t1, available)
        if not forward:
            continue
        tilt_ret = _portfolio_return(weights_tilt, forward)
        bah_ret = _portfolio_return(weights_bah, forward)
        bucket = "insufficient_history"
        if classifier is not None:
            try:
                bucket = classifier(price_data, t0) or "insufficient_history"
            except Exception:
                bucket = "insufficient_history"
        window_rows.append(
            {
                "month_end": t0.isoformat(),
                "tilt_return": float(tilt_ret),
                "bah_return": float(bah_ret),
                "edge": float(tilt_ret - bah_ret),
                "regime_bucket": str(bucket),
            }
        )

    if synthetic_detected and not allow_synthetic_ml_evidence:
        return (
            "missing",
            [
                "ML snapshots originate from SyntheticMLForecast; "
                "set allow_synthetic_ml_evidence=True to evaluate"
            ],
            None,
        )

    if not window_rows:
        return "missing", ["No usable monthly windows"], None

    edges = [row["edge"] for row in window_rows]
    hit_rate = float(sum(1 for e in edges if e > 0) / len(edges))
    mean_edge = float(np.mean(edges))
    months = len(window_rows)
    cagr_edge_pp = mean_edge * 12.0 * 100.0

    checks = {
        "min_windows": months >= int(thresholds["min_windows"]),
        "min_cagr_edge_pp": cagr_edge_pp >= float(thresholds["min_cagr_edge_pp"]),
        "min_hit_rate": hit_rate >= float(thresholds["min_hit_rate"]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        reasons.append("Failed ml gates: " + ", ".join(failed))
        status: EvidenceStatus = "fail"
    else:
        status = "pass"

    conditioned = _aggregate_conditioned_buckets(window_rows)

    summary = {
        "profile": profile,
        "thresholds": thresholds,
        "checks": checks,
        "windows": months,
        "hit_rate": hit_rate,
        "cagr_edge_pp": cagr_edge_pp,
        "as_of": as_of_date.isoformat(),
        "universe": available,
        "row_summary": window_rows[-min(6, len(window_rows)):],
        "synthetic_source": synthetic_detected,
        "conditioned": conditioned,
    }
    return status, reasons, summary


def assess_conditioned_ml_evidence_status(
    summary: Optional[Dict[str, Any]],
    *,
    current_bucket: Optional[str],
    profile: EvidenceProfile = "ausgewogen",
    conditioned_min_windows: Optional[int] = None,
    custom_thresholds: Optional[Dict[str, Any]] = None,
) -> Tuple[EvidenceStatus, List[str], int, Optional[Dict[str, Any]]]:
    """Bucket-conditioned PASS/FAIL gate over an ML evidence summary."""

    if summary is None:
        return "missing", ["No ml evidence summary available"], 0, None
    if current_bucket is None:
        return "missing", ["No current regime bucket available"], 0, None

    conditioned = summary.get("conditioned") or {}
    bucket_summary = (
        conditioned.get(current_bucket) if isinstance(conditioned, dict) else None
    )
    if not bucket_summary:
        return (
            "missing",
            [f"No conditioned ml evidence for bucket '{current_bucket}'"],
            0,
            None,
        )

    num_windows = int(bucket_summary.get("num_windows", 0))
    if num_windows <= 0:
        return (
            "missing",
            [f"Conditioned ml evidence empty for bucket '{current_bucket}'"],
            0,
            bucket_summary,
        )

    base_thresholds = _resolve_ml_evidence_thresholds(profile, custom_thresholds)
    relaxed_min = conditioned_min_windows
    if relaxed_min is None:
        relaxed_min = max(3, int(base_thresholds["min_windows"]) // 2)

    checks = {
        "min_windows": num_windows >= int(relaxed_min),
        "min_cagr_edge_pp": float(bucket_summary.get("cagr_edge_pp", 0.0))
        >= float(base_thresholds["min_cagr_edge_pp"]),
        "min_hit_rate": float(bucket_summary.get("hit_rate", 0.0))
        >= float(base_thresholds["min_hit_rate"]),
    }
    failed = [name for name, ok in checks.items() if not ok]
    reasons = [
        f"ML bucket '{current_bucket}' windows={num_windows}, "
        f"cagr_edge_pp={float(bucket_summary.get('cagr_edge_pp', 0.0)):.2f}, "
        f"hit_rate={float(bucket_summary.get('hit_rate', 0.0)):.2f}"
    ]
    if failed:
        reasons.append("Failed conditioned ml gates: " + ", ".join(failed))
        status: EvidenceStatus = "fail"
    else:
        status = "pass"
    enriched = dict(bucket_summary)
    enriched["checks"] = checks
    enriched["min_windows_required"] = int(relaxed_min)
    enriched["thresholds"] = base_thresholds
    return status, reasons, num_windows, enriched
