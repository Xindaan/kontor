"""Evidence-backed bootstrap start decision between two strategies."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from backtest.backtester import BacktestConfig, Backtester
from backtest.data import DataLoader
from backtest.meta_evidence import run_meta_evidence_analysis
from backtest.metadata import collect_environment_metadata, collect_git_metadata
from backtest.web.services.strategies import load_strategy_instance


FallbackTieBreaker = Literal["maxdd", "sharpe"]


DEFAULT_BOOTSTRAP_DIR = Path("results/meta_bootstrap")
DEFAULT_BOOTSTRAP_LOG = Path("results/meta_decisions/bootstrap_log.jsonl")


def _coerce_date(value: Optional[str | date]) -> date:
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
    if not (parsed == parsed) or parsed in {float("inf"), float("-inf")}:
        return fallback
    return parsed


def _normalize_strategy_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _pair_id(strategy_a: str | Path, strategy_b: str | Path) -> str:
    a = Path(strategy_a).stem.lower()
    b = Path(strategy_b).stem.lower()
    ordered = sorted([a, b])
    return f"{ordered[0]}__vs__{ordered[1]}"


def _artifact_id(payload: Dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{digest}"


def _resolve_output_path(
    pair_id: str,
    artifact_id: str,
    output_path: Optional[str | Path] = None,
) -> Path:
    if output_path:
        return Path(output_path).expanduser().resolve()
    return (DEFAULT_BOOTSTRAP_DIR / pair_id / f"{artifact_id}.json").resolve()


def _append_bootstrap_log(row: Dict[str, Any], log_path: str | Path = DEFAULT_BOOTSTRAP_LOG) -> None:
    path = Path(log_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str))
        handle.write("\n")


def _run_full_period_metrics(
    strategy_path: str | Path,
    params: Optional[Dict[str, Any]],
    *,
    as_of: date,
    start_date: str,
    initial_capital: float,
    costs_pct: float,
    skip_failed: bool,
    metric_basis: str,
) -> Dict[str, float]:
    strategy = load_strategy_instance(Path(strategy_path), params=params or None)
    data = DataLoader.yahoo(
        tickers=list(strategy.assets),
        start=start_date,
        end=as_of.isoformat(),
        currency="EUR",
        align="ffill",
        skip_failed=skip_failed,
    )
    config = BacktestConfig(
        initial_capital=initial_capital,
        costs_pct=costs_pct,
        rebalance_frequency=getattr(strategy, "rebalance_frequency", "monthly"),
        benchmark="S&P 500",
        tax_enabled=False,
        metric_basis=metric_basis,
        validate=False,
    )
    result = Backtester(strategy, data, config).run()
    metrics = result.metrics
    if metrics is None:
        return {"cagr": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0}
    return {
        "cagr": _safe_float(getattr(metrics, "cagr", 0.0)),
        "max_drawdown": _safe_float(getattr(metrics, "max_drawdown", 0.0)),
        "sharpe_ratio": _safe_float(getattr(metrics, "sharpe_ratio", 0.0)),
    }


def _direction_summary(artifact: Dict[str, Any]) -> Dict[str, Any]:
    summary = artifact.get("summary", {}) or {}
    gates = artifact.get("gates", {}) or {}
    return {
        "artifact_id": artifact.get("artifact_id"),
        "artifact_path": artifact.get("artifact_path"),
        "pass": bool(gates.get("pass", False)),
        "reasons": list(gates.get("reasons", []) or []),
        "num_windows": int(summary.get("num_windows", 0) or 0),
        "oos_cagr_edge_pp": _safe_float(summary.get("oos_cagr_edge_pp", 0.0)),
        "oos_hit_rate": _safe_float(summary.get("oos_hit_rate", 0.0)),
        "oos_degradation_pct": _safe_float(summary.get("oos_degradation_pct", 0.0)),
        "oos_dd_delta_pp": _safe_float(summary.get("oos_dd_delta_pp", 0.0)),
    }


def run_meta_bootstrap_decision(
    *,
    strategy_a: str | Path,
    strategy_b: str | Path,
    strategy_a_params: Optional[Dict[str, Any]] = None,
    strategy_b_params: Optional[Dict[str, Any]] = None,
    as_of: Optional[str | date] = None,
    evidence_profile: str = "ausgewogen",
    evidence_compare_mode: str = "vs_current",
    evidence_max_age_days: int = 30,
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
    fallback_cagr_tie_band_pp: float = 1.0,
    fallback_tie_breaker: FallbackTieBreaker = "maxdd",
    artifact_path: Optional[str | Path] = None,
    save_artifact: bool = True,
    log_path: str | Path = DEFAULT_BOOTSTRAP_LOG,
) -> Dict[str, Any]:
    """Select a start strategy using bilateral evidence plus deterministic fallback."""
    as_of_date = _coerce_date(as_of)
    strategy_a = _normalize_strategy_path(strategy_a)
    strategy_b = _normalize_strategy_path(strategy_b)
    strategy_a_params = dict(strategy_a_params or {})
    strategy_b_params = dict(strategy_b_params or {})
    custom_thresholds = dict(custom_thresholds or {})
    fallback_cagr_tie_band_pp = max(0.0, float(fallback_cagr_tie_band_pp))

    a_to_b = run_meta_evidence_analysis(
        current_strategy=strategy_a,
        target_strategy=strategy_b,
        current_params=strategy_a_params,
        target_params=strategy_b_params,
        as_of=as_of_date,
        evidence_profile=evidence_profile,  # type: ignore[arg-type]
        evidence_compare_mode=evidence_compare_mode,  # type: ignore[arg-type]
        evidence_max_age_days=evidence_max_age_days,
        custom_thresholds=custom_thresholds or None,
        train_years=train_years,
        test_years=test_years,
        step_months=step_months,
        anchored=anchored,
        start_date=start_date,
        initial_capital=initial_capital,
        costs_pct=costs_pct,
        skip_failed=skip_failed,
        metric_basis=metric_basis,
        save_artifact=True,
    )
    b_to_a = run_meta_evidence_analysis(
        current_strategy=strategy_b,
        target_strategy=strategy_a,
        current_params=strategy_b_params,
        target_params=strategy_a_params,
        as_of=as_of_date,
        evidence_profile=evidence_profile,  # type: ignore[arg-type]
        evidence_compare_mode=evidence_compare_mode,  # type: ignore[arg-type]
        evidence_max_age_days=evidence_max_age_days,
        custom_thresholds=custom_thresholds or None,
        train_years=train_years,
        test_years=test_years,
        step_months=step_months,
        anchored=anchored,
        start_date=start_date,
        initial_capital=initial_capital,
        costs_pct=costs_pct,
        skip_failed=skip_failed,
        metric_basis=metric_basis,
        save_artifact=True,
    )

    summary_a_to_b = _direction_summary(a_to_b)
    summary_b_to_a = _direction_summary(b_to_a)

    pass_a_to_b = bool(summary_a_to_b["pass"])
    pass_b_to_a = bool(summary_b_to_a["pass"])

    decision_rule = ""
    reasons: List[str] = []
    fallback_metrics: Optional[Dict[str, Any]] = None

    if pass_a_to_b and not pass_b_to_a:
        recommended = strategy_b
        decision_rule = "evidence_unilateral"
        reasons.append("A->B evidence PASS while B->A evidence FAIL")
    elif pass_b_to_a and not pass_a_to_b:
        recommended = strategy_a
        decision_rule = "evidence_unilateral"
        reasons.append("B->A evidence PASS while A->B evidence FAIL")
    else:
        metrics_a = _run_full_period_metrics(
            strategy_path=strategy_a,
            params=strategy_a_params,
            as_of=as_of_date,
            start_date=start_date,
            initial_capital=initial_capital,
            costs_pct=costs_pct,
            skip_failed=skip_failed,
            metric_basis=metric_basis,
        )
        metrics_b = _run_full_period_metrics(
            strategy_path=strategy_b,
            params=strategy_b_params,
            as_of=as_of_date,
            start_date=start_date,
            initial_capital=initial_capital,
            costs_pct=costs_pct,
            skip_failed=skip_failed,
            metric_basis=metric_basis,
        )
        cagr_edge_b_minus_a_pp = (metrics_b["cagr"] - metrics_a["cagr"]) * 100.0
        fallback_metrics = {
            "strategy_a": metrics_a,
            "strategy_b": metrics_b,
            "cagr_edge_b_minus_a_pp": cagr_edge_b_minus_a_pp,
            "tie_band_pp": fallback_cagr_tie_band_pp,
            "tie_breaker": fallback_tie_breaker,
        }

        if abs(cagr_edge_b_minus_a_pp) >= fallback_cagr_tie_band_pp:
            recommended = strategy_b if cagr_edge_b_minus_a_pp > 0 else strategy_a
            decision_rule = "fallback_cagr"
            reasons.append(
                f"No unilateral evidence pass; selected higher full-period CAGR "
                f"(edge {cagr_edge_b_minus_a_pp:+.2f}pp)"
            )
        elif fallback_tie_breaker == "sharpe":
            if metrics_b["sharpe_ratio"] > metrics_a["sharpe_ratio"]:
                recommended = strategy_b
            elif metrics_b["sharpe_ratio"] < metrics_a["sharpe_ratio"]:
                recommended = strategy_a
            else:
                recommended = strategy_b if metrics_b["cagr"] >= metrics_a["cagr"] else strategy_a
            decision_rule = "fallback_tie_breaker"
            reasons.append(
                "No unilateral evidence pass and CAGR within tie-band; tie-breaker=sharpe"
            )
        else:
            # Less negative max_drawdown is better.
            if metrics_b["max_drawdown"] > metrics_a["max_drawdown"]:
                recommended = strategy_b
            elif metrics_b["max_drawdown"] < metrics_a["max_drawdown"]:
                recommended = strategy_a
            else:
                recommended = strategy_b if metrics_b["cagr"] >= metrics_a["cagr"] else strategy_a
            decision_rule = "fallback_tie_breaker"
            reasons.append(
                "No unilateral evidence pass and CAGR within tie-band; tie-breaker=maxdd"
            )

    pair_id = _pair_id(strategy_a, strategy_b)
    identity_payload = {
        "pair_id": pair_id,
        "as_of": as_of_date.isoformat(),
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "strategy_a_params": strategy_a_params,
        "strategy_b_params": strategy_b_params,
        "evidence_profile": evidence_profile,
        "decision_rule": decision_rule,
        "recommended_start_strategy": recommended,
        "summary_a_to_b": summary_a_to_b,
        "summary_b_to_a": summary_b_to_a,
        "fallback_metrics": fallback_metrics,
    }
    artifact_id = _artifact_id(identity_payload)
    output_path = _resolve_output_path(pair_id, artifact_id, artifact_path)

    created_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "artifact_id": artifact_id,
        "pair_id": pair_id,
        "as_of": as_of_date.isoformat(),
        "created_at": created_at,
        "strategy_a": {"path": strategy_a, "params": strategy_a_params},
        "strategy_b": {"path": strategy_b, "params": strategy_b_params},
        "evidence_profile": evidence_profile,
        "evidence_compare_mode": evidence_compare_mode,
        "evidence_max_age_days": int(evidence_max_age_days),
        "evidence": {
            "a_to_b": summary_a_to_b,
            "b_to_a": summary_b_to_a,
        },
        "fallback": fallback_metrics,
        "decision": {
            "recommended_start_strategy": recommended,
            "decision_rule": decision_rule,
            "reasons": reasons,
        },
        "run_metadata": {
            "git": collect_git_metadata().to_dict(),
            "environment": collect_environment_metadata().to_dict(),
            "metric_basis": metric_basis,
            "initial_capital": initial_capital,
            "costs_pct": costs_pct,
            "start_date": start_date,
            "train_years": train_years,
            "test_years": test_years,
            "step_months": step_months,
            "anchored": bool(anchored),
            "fallback_cagr_tie_band_pp": fallback_cagr_tie_band_pp,
            "fallback_tie_breaker": fallback_tie_breaker,
        },
        "artifact_path": str(output_path),
    }

    if save_artifact:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2))

    _append_bootstrap_log(
        {
            "timestamp": date.today().isoformat(),
            "as_of": as_of_date.isoformat(),
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "recommended_start_strategy": recommended,
            "decision_rule": decision_rule,
            "reasons": reasons,
            "artifact_id": artifact_id,
        },
        log_path=log_path,
    )

    return artifact

