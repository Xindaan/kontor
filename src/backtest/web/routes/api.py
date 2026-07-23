"""JSON API routes for the web frontend."""

import math
import json
import time
from uuid import uuid4
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse


def safe_float(value: Any) -> Optional[float]:
    """Convert value to float, returning None for NaN/Inf/None."""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def coerce_numeric_inputs(value: Any) -> Any:
    """Coerce numeric-like inputs to floats, preserving non-numeric values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        coerced = safe_float(stripped)
        if coerced is None:
            return value
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(coerced)
        if coerced.is_integer() and "." not in stripped:
            return int(coerced)
        return coerced
    if isinstance(value, dict):
        return {key: coerce_numeric_inputs(val) for key, val in value.items()}
    if isinstance(value, list):
        return [coerce_numeric_inputs(item) for item in value]
    return value


def _needs_volume_data(max_volume_participation: Optional[float], min_daily_dollar_volume: float) -> bool:
    """True if request needs volume data for liquidity guards."""
    return (
        (max_volume_participation is not None and max_volume_participation > 0)
        or float(min_daily_dollar_volume) > 0.0
    )


def _build_risk_overlay_payload(request: Any) -> Optional[Dict[str, Any]]:
    """Build risk overlay dictionary from request fields."""
    sector_caps_raw = getattr(request, "sector_caps", {}) or {}
    ticker_sectors_raw = getattr(request, "ticker_sectors", {}) or {}

    sector_caps: Dict[str, float] = {}
    for key, value in sector_caps_raw.items():
        cap = safe_float(value)
        if cap is None:
            continue
        sector_caps[str(key).strip().lower()] = cap

    ticker_sectors: Dict[str, str] = {}
    for ticker, sector in ticker_sectors_raw.items():
        sector_name = str(sector).strip().lower()
        if not sector_name:
            continue
        ticker_sectors[str(ticker)] = sector_name

    max_position = safe_float(getattr(request, "max_position", None))
    turnover_budget = safe_float(getattr(request, "turnover_budget", None))
    drawdown_threshold = safe_float(getattr(request, "drawdown_brake_threshold", None))
    drawdown_cash_target = safe_float(getattr(request, "drawdown_brake_cash_target", None))
    drawdown_release = safe_float(getattr(request, "drawdown_brake_release", None))

    drawdown_brake = None
    if drawdown_threshold is not None:
        drawdown_brake = {
            "threshold": drawdown_threshold,
            "cash_target": drawdown_cash_target if drawdown_cash_target is not None else 1.0,
        }
        if drawdown_release is not None:
            drawdown_brake["release_drawdown"] = drawdown_release

    if not any(
        [
            max_position is not None,
            turnover_budget is not None,
            bool(sector_caps),
            bool(ticker_sectors),
            drawdown_brake is not None,
        ]
    ):
        return None

    payload: Dict[str, Any] = {}
    if max_position is not None:
        payload["max_position"] = max_position
    if turnover_budget is not None:
        payload["turnover_budget"] = turnover_budget
    if sector_caps:
        payload["sector_caps"] = sector_caps
    if ticker_sectors:
        payload["ticker_sectors"] = ticker_sectors
    if drawdown_brake is not None:
        payload["drawdown_brake"] = drawdown_brake
    return payload


def _build_exposure_policy_payload(request: Any) -> Optional[Dict[str, Any]]:
    """Build optional exposure policy dictionary from request fields."""
    raw = getattr(request, "exposure_policy", None)
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("exposure_policy must be an object/dictionary")
    return coerce_numeric_inputs(raw)


def _extend_assets_for_exposure_policy(assets: List[str], exposure_policy: Optional[Dict[str, Any]]) -> List[str]:
    """Add policy fallback/proxy assets to the market-data request."""
    if not exposure_policy:
        return list(assets)
    from backtest.risk.exposure_policy import required_assets_from_raw

    ordered = list(assets)
    seen = set(ordered)
    for ticker in required_assets_from_raw(exposure_policy):
        if ticker not in seen:
            ordered.append(ticker)
            seen.add(ticker)
    return ordered

from backtest.web.deps import templates
from backtest.web.models.schemas import (
    RunRequest,
    StrategyInfo,
    StrategySchema,
    BacktestResultResponse,
    MetricsResponse,
    TaxSummaryResponse,
    TradeResponse,
    TradingCostsResponse,
    MonthlyReturnResponse,
    ErrorResponse,
    CompareRequest,
    CompareResultResponse,
    CompareRowResponse,
    SweepRequest,
    SweepResultResponse,
    SweepSummaryRow,
    SweepWindowResult,
    SweepProgressResponse,
    RunProgressResponse,
    CompareProgressResponse,
    BatchOptimizeProgressResponse,
    OptimizeProgressResponse,
    OptimizeRequest,
    OptimizeResultResponse,
    OptimizeResultRow,
    WalkForwardResponse,
    WalkForwardWindowResponse,
    BatchOptimizeRequest,
    BatchOptimizeResponse,
    BatchOptimizeResultRow,
    WalkForwardSummaryResponse,
    ManualProvenanceCreateRequest,
    ManualProvenanceEntryResponse,
    ManualProvenanceListResponse,
    ManualProvenanceVerifyResponse,
    ManualProvenanceIssueResponse,
    SignalRequest,
    SignalResponse,
    MetaEvidenceRunRequest,
    MetaEvidenceResponse,
    MetaPromotionRunRequest,
    MetaPromotionResponse,
    MetaPromotionListResponse,
    MetaBootstrapRunRequest,
    MetaBootstrapResponse,
)
from backtest.sweep import SweepCancelled
from backtest.web.services.strategies import (
    list_strategies,
    get_param_schema,
    load_strategy_instance,
    coerce_params_to_signature,
)
from backtest.web.services.charts import generate_charts
from backtest.batch_optimize import BatchOptimizeCancelled

router = APIRouter(tags=["api"])

RUN_PROGRESS: Dict[str, Dict[str, Any]] = {}
COMPARE_PROGRESS: Dict[str, Dict[str, Any]] = {}
SWEEP_PROGRESS: Dict[str, Dict[str, Any]] = {}
BATCH_PROGRESS: Dict[str, Dict[str, Any]] = {}
OPTIMIZE_PROGRESS: Dict[str, Dict[str, Any]] = {}


def _set_progress(store: Dict[str, Dict[str, Any]], run_id: str, **updates: Any) -> None:
    entry = store.get(run_id, {})
    entry.update(updates)
    entry["updated_at"] = time.time()
    store[run_id] = entry


def _set_sweep_progress(run_id: str, **updates: Any) -> None:
    _set_progress(SWEEP_PROGRESS, run_id, **updates)


@router.get("/run/progress/{run_id}", response_model=RunProgressResponse)
async def get_run_progress(run_id: str):
    """Get progress updates for a running backtest."""
    progress = RUN_PROGRESS.get(run_id)
    if not progress:
        return RunProgressResponse(
            run_id=run_id,
            status="pending",
            message="Waiting for run to start...",
            elapsed_seconds=0.0,
        )
    return RunProgressResponse(
        run_id=run_id,
        status=progress.get("status", "running"),
        message=progress.get("message"),
        elapsed_seconds=progress.get("elapsed_seconds", 0.0),
    )


@router.post("/run/cancel/{run_id}")
async def cancel_run(run_id: str):
    """Request cancellation for a running backtest."""
    progress = RUN_PROGRESS.get(run_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Run progress not found")
    progress["cancel_requested"] = True
    progress["status"] = "cancelling"
    progress["updated_at"] = time.time()
    RUN_PROGRESS[run_id] = progress
    return {"status": "cancelling"}


@router.get("/compare/progress/{run_id}", response_model=CompareProgressResponse)
async def get_compare_progress(run_id: str):
    """Get progress updates for a running comparison."""
    progress = COMPARE_PROGRESS.get(run_id)
    if not progress:
        return CompareProgressResponse(
            run_id=run_id,
            status="pending",
            strategy_index=0,
            strategy_total=0,
            strategy_name=None,
            elapsed_seconds=0.0,
        )
    return CompareProgressResponse(
        run_id=run_id,
        status=progress.get("status", "running"),
        strategy_index=progress.get("strategy_index", 0),
        strategy_total=progress.get("strategy_total", 0),
        strategy_name=progress.get("strategy_name"),
        elapsed_seconds=progress.get("elapsed_seconds", 0.0),
    )


@router.post("/compare/cancel/{run_id}")
async def cancel_compare(run_id: str):
    """Request cancellation for a running comparison."""
    progress = COMPARE_PROGRESS.get(run_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Compare progress not found")
    progress["cancel_requested"] = True
    progress["status"] = "cancelling"
    progress["updated_at"] = time.time()
    COMPARE_PROGRESS[run_id] = progress
    return {"status": "cancelling"}


@router.get("/batch-optimize/progress/{run_id}", response_model=BatchOptimizeProgressResponse)
async def get_batch_optimize_progress(run_id: str):
    """Get progress updates for a running batch optimization."""
    progress = BATCH_PROGRESS.get(run_id)
    if not progress:
        return BatchOptimizeProgressResponse(
            run_id=run_id,
            status="pending",
            strategy_index=0,
            strategy_total=0,
            strategy_name=None,
            run_count=0,
            total_runs=0,
            elapsed_seconds=0.0,
        )
    return BatchOptimizeProgressResponse(
        run_id=run_id,
        status=progress.get("status", "running"),
        strategy_index=progress.get("strategy_index", 0),
        strategy_total=progress.get("strategy_total", 0),
        strategy_name=progress.get("strategy_name"),
        run_count=progress.get("run_count", 0),
        total_runs=progress.get("total_runs", 0),
        elapsed_seconds=progress.get("elapsed_seconds", 0.0),
    )


@router.get("/optimize/progress/{run_id}", response_model=OptimizeProgressResponse)
async def get_optimize_progress(run_id: str):
    """Get progress updates for a running optimization."""
    progress = OPTIMIZE_PROGRESS.get(run_id)
    if not progress:
        return OptimizeProgressResponse(
            run_id=run_id,
            status="pending",
            run_count=0,
            total_runs=0,
            elapsed_seconds=0.0,
            message="Waiting for optimization to start...",
        )
    return OptimizeProgressResponse(
        run_id=run_id,
        status=progress.get("status", "running"),
        run_count=progress.get("run_count", 0),
        total_runs=progress.get("total_runs", 0),
        elapsed_seconds=progress.get("elapsed_seconds", 0.0),
        message=progress.get("message"),
    )


@router.post("/batch-optimize/cancel/{run_id}")
async def cancel_batch_optimize(run_id: str):
    """Request cancellation for a running batch optimization."""
    progress = BATCH_PROGRESS.get(run_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Batch optimization progress not found")
    progress["cancel_requested"] = True
    progress["status"] = "cancelling"
    progress["updated_at"] = time.time()
    BATCH_PROGRESS[run_id] = progress
    return {"status": "cancelling"}


@router.post("/sweep/cancel/{run_id}")
async def cancel_sweep(run_id: str):
    """Request cancellation for a running sweep."""
    progress = SWEEP_PROGRESS.get(run_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Sweep progress not found")
    progress["cancel_requested"] = True
    progress["status"] = "cancelling"
    progress["updated_at"] = time.time()
    SWEEP_PROGRESS[run_id] = progress
    return {"status": "cancelling"}


@router.get("/sweep/progress/{run_id}", response_model=SweepProgressResponse)
async def get_sweep_progress(run_id: str):
    """Get progress updates for a running sweep."""
    progress = SWEEP_PROGRESS.get(run_id)
    if not progress:
        return SweepProgressResponse(
            run_id=run_id,
            status="pending",
            run_count=0,
            total_runs=0,
            strategy_index=0,
            strategy_total=0,
            strategy_name=None,
            iteration_index=0,
            iteration_total=0,
            elapsed_seconds=0.0,
        )

    return SweepProgressResponse(
        run_id=run_id,
        status=progress.get("status", "running"),
        run_count=progress.get("run_count", 0),
        total_runs=progress.get("total_runs", 0),
        strategy_index=progress.get("strategy_index", 0),
        strategy_total=progress.get("strategy_total", 0),
        strategy_name=progress.get("strategy_name"),
        iteration_index=progress.get("iteration_index", 0),
        iteration_total=progress.get("iteration_total", 0),
        elapsed_seconds=progress.get("elapsed_seconds", 0.0),
    )


@router.get("/strategies", response_model=List[StrategyInfo])
async def get_strategies():
    """List all available strategies."""
    strategies = list_strategies()
    return [StrategyInfo(**s) for s in strategies]


@router.get("/strategies/{strategy_name}/schema", response_model=StrategySchema)
async def get_strategy_schema(strategy_name: str):
    """Get the parameter schema for a strategy.

    strategy_name can be either the file name (e.g., 'dual_momentum.py')
    or the full path.
    """
    # Find the strategy file
    strategies = list_strategies()
    strategy_path = None

    for s in strategies:
        if s["file_name"] == strategy_name or s["file_path"] == strategy_name:
            strategy_path = s["file_path"]
            break

    if not strategy_path:
        raise HTTPException(status_code=404, detail=f"Strategy not found: {strategy_name}")

    try:
        schema = get_param_schema(Path(strategy_path))
        return StrategySchema(**schema)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/{strategy_name}/params", response_class=HTMLResponse)
async def get_strategy_params_html(request: Request, strategy_name: str):
    """Get HTML partial for strategy parameters (for HTMX).

    This endpoint returns an HTML fragment that can be inserted
    into the form via HTMX.
    """
    # Find the strategy file
    strategies = list_strategies()
    strategy_path = None

    for s in strategies:
        if s["file_name"] == strategy_name or s["file_path"] == strategy_name:
            strategy_path = s["file_path"]
            break

    if not strategy_path:
        return HTMLResponse(
            content='<p class="text-red-500">Strategy not found</p>',
            status_code=404
        )

    try:
        schema = get_param_schema(Path(strategy_path))
        return templates.TemplateResponse(
            request,
            "partials/_param_form.html",
            {
                "schema": schema,
            },
        )
    except Exception as e:
        return HTMLResponse(
            content=f'<p class="text-red-500">Error loading strategy: {e}</p>',
            status_code=500
        )


@router.get("/data/manual/provenance", response_model=ManualProvenanceListResponse)
async def get_manual_data_provenance(
    dataset: Optional[str] = None,
    source: Optional[str] = None,
):
    """List manual data provenance entries."""
    from backtest.provenance import ManualDataProvenanceRegistry

    try:
        registry = ManualDataProvenanceRegistry()
        entries = registry.list_entries(dataset=dataset, source=source)
        rows = [ManualProvenanceEntryResponse(**entry.to_dict()) for entry in entries]
        return ManualProvenanceListResponse(total=len(rows), entries=rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/manual/provenance", response_model=ManualProvenanceEntryResponse)
async def create_manual_data_provenance(request: ManualProvenanceCreateRequest):
    """Create manual data provenance entry."""
    from backtest.provenance import ManualDataProvenanceRegistry

    try:
        registry = ManualDataProvenanceRegistry()
        entry = registry.register_entry(
            file_path=request.file_path,
            dataset=request.dataset,
            source=request.source,
            quality_tag=request.quality_tag,
            as_of_date=request.as_of_date,
            import_method=request.import_method,
            license_tos_note=request.license_tos_note,
            source_url=request.source_url,
            notes=request.notes,
            entry_id=request.entry_id,
        )
        return ManualProvenanceEntryResponse(**entry.to_dict())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/manual/provenance/verify", response_model=ManualProvenanceVerifyResponse)
async def verify_manual_data_provenance(skip_hash: bool = False):
    """Verify manual data provenance registry entries."""
    from backtest.provenance import ManualDataProvenanceRegistry

    try:
        registry = ManualDataProvenanceRegistry()
        result = registry.verify_entries(check_hash=not skip_hash)
        return ManualProvenanceVerifyResponse(
            registry_path=result["registry_path"],
            total_entries=result["total_entries"],
            ok_entries=result["ok_entries"],
            issue_count=result["issue_count"],
            issues=[ManualProvenanceIssueResponse(**issue) for issue in result["issues"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/manual/provenance/{entry_id}", response_model=ManualProvenanceEntryResponse)
async def get_manual_data_provenance_entry(entry_id: str):
    """Get one manual data provenance entry by id."""
    from backtest.provenance import ManualDataProvenanceRegistry

    try:
        registry = ManualDataProvenanceRegistry()
        entry = registry.get_entry(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Entry not found: {entry_id}")
        return ManualProvenanceEntryResponse(**entry.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signals", response_model=SignalResponse)
async def generate_live_signals(request: SignalRequest):
    """Generate live trading signals including order/drift diagnostics."""
    from datetime import date
    from backtest.signals import SignalGenerator, Portfolio
    from backtest.meta_decision import run_meta_decision

    try:
        strategy_path = Path(request.strategy)
        params = coerce_numeric_inputs(request.params or {})
        strategy = load_strategy_instance(strategy_path, params=params)

        if request.rebalance_frequency:
            strategy.rebalance_frequency = request.rebalance_frequency

        as_of = None
        if request.signal_date:
            as_of = date.fromisoformat(request.signal_date)

        portfolio = None
        if request.portfolio is not None:
            last_rebalance = None
            if request.portfolio.last_rebalance:
                last_rebalance = date.fromisoformat(request.portfolio.last_rebalance)
            portfolio = Portfolio(
                positions={str(k): float(v) for k, v in request.portfolio.positions.items()},
                cash=float(request.portfolio.cash),
                last_rebalance=last_rebalance,
            )

        generator = SignalGenerator(
            strategy=strategy,
            portfolio=portfolio,
            skip_failed=request.skip_failed,
            drift_tolerance=float(request.drift_tolerance),
            exposure_policy=_build_exposure_policy_payload(request),
        )
        report = generator.generate(as_of=as_of)

        if request.meta_decision is not None and request.meta_decision.enabled:
            meta_candidates = [
                {
                    "strategy": candidate.strategy,
                    "params": coerce_numeric_inputs(candidate.params or {}),
                }
                for candidate in request.meta_decision.candidates
            ]
            meta_result = run_meta_decision(
                as_of=as_of or report.as_of,
                current_strategy=str(strategy_path),
                current_params=params,
                candidates=meta_candidates,
                portfolio=portfolio,
                skip_failed=request.skip_failed,
                drift_tolerance=float(request.drift_tolerance),
                params_source=request.meta_decision.params_source,
                preset_params_file=request.meta_decision.preset_params_file,
                scoring_mode=request.meta_decision.scoring_mode,
                performance_windows_days=request.meta_decision.performance_windows_days,
                performance_weights=request.meta_decision.performance_weights,
                confirm_points=request.meta_decision.confirm_points,
                switch_margin=request.meta_decision.switch_margin,
                decision_cadence=request.meta_decision.decision_cadence,
                plan_mode=request.meta_decision.plan_mode,
                evidence_required=request.meta_decision.evidence_required,
                evidence_profile=request.meta_decision.evidence_profile,
                evidence_compare_mode=request.meta_decision.evidence_compare_mode,
                evidence_max_age_days=request.meta_decision.evidence_max_age_days,
                evidence_artifact_path=request.meta_decision.evidence_artifact_path,
                gate_fail_action=request.meta_decision.gate_fail_action,
                custom_thresholds=request.meta_decision.custom_thresholds,
                regime_mode=request.meta_decision.regime_mode,
                regime_profile=request.meta_decision.regime_profile,
                alpha_tie_band=request.meta_decision.alpha_tie_band,
                stress_alpha_tolerance=request.meta_decision.stress_alpha_tolerance,
                conditioned_min_windows=request.meta_decision.conditioned_min_windows,
            )
            report.meta_decision = meta_result
        return SignalResponse(report=report.to_dict())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/meta-evidence/run", response_model=MetaEvidenceResponse)
async def run_meta_evidence(request: MetaEvidenceRunRequest):
    """Run evidence analysis for a strategy switch pair and persist artifact."""
    from backtest.meta_evidence import run_meta_evidence_analysis
    import itertools

    try:
        artifact = run_meta_evidence_analysis(
            current_strategy=request.current_strategy,
            target_strategy=request.target_strategy,
            current_params=coerce_numeric_inputs(request.current_params or {}),
            target_params=coerce_numeric_inputs(request.target_params or {}),
            as_of=request.as_of,
            evidence_profile=request.evidence_profile,
            evidence_compare_mode=request.evidence_compare_mode,
            evidence_max_age_days=request.evidence_max_age_days,
            evidence_artifact_path=request.evidence_artifact_path,
            custom_thresholds=coerce_numeric_inputs(request.custom_thresholds or {}),
            train_years=request.train_years,
            test_years=request.test_years,
            step_months=request.step_months,
            anchored=request.anchored,
            start_date=request.start_date,
            initial_capital=request.initial_capital,
            costs_pct=request.costs_pct,
            skip_failed=request.skip_failed,
            metric_basis=request.metric_basis,
            save_artifact=True,
        )

        if request.tuning_enabled:
            confirm_points_grid = sorted(set(int(v) for v in request.grid_confirm_points if int(v) >= 1))
            switch_margin_grid = sorted(set(float(v) for v in request.grid_switch_margin if float(v) >= 0.0))
            combos = list(itertools.product(confirm_points_grid, switch_margin_grid))
            total_combinations = len(combos)
            capped_combinations = combos[: int(request.max_combinations)]

            edge_pp = safe_float(artifact.get("summary", {}).get("oos_cagr_edge_pp")) or 0.0
            degradation_pct = safe_float(artifact.get("summary", {}).get("oos_degradation_pct")) or 0.0
            dd_delta_pp = safe_float(artifact.get("summary", {}).get("oos_dd_delta_pp")) or 0.0

            stage1_rows = []
            for confirm_points, switch_margin in capped_combinations:
                # Stage 1: fast heuristic from evidence summary.
                stage1_score = (
                    edge_pp
                    - 0.35 * float(confirm_points - 1)
                    - 2.0 * float(switch_margin)
                )
                stage1_rows.append(
                    {
                        "confirm_points": int(confirm_points),
                        "switch_margin": float(switch_margin),
                        "stage1_score": float(stage1_score),
                    }
                )

            stage1_rows.sort(key=lambda row: row["stage1_score"], reverse=True)
            top_k = max(1, int(request.top_k))
            stage1_top = stage1_rows[:top_k]

            stage2_rows = []
            for row in stage1_top:
                # Stage 2: robustness penalty pass.
                stage2_score = (
                    row["stage1_score"]
                    - 0.02 * max(0.0, degradation_pct - 25.0)
                    - 0.05 * max(0.0, dd_delta_pp - 5.0)
                )
                stage2_rows.append(
                    {
                        **row,
                        "stage2_score": float(stage2_score),
                    }
                )
            stage2_rows.sort(key=lambda row: row["stage2_score"], reverse=True)
            best_setup = stage2_rows[0] if stage2_rows else None

            artifact["tuning"] = {
                "enabled": True,
                "mode": "2-stage-smart",
                "total_combinations": total_combinations,
                "capped_combinations": len(capped_combinations),
                "max_combinations": int(request.max_combinations),
                "top_k": top_k,
                "stage1_top": stage1_top,
                "stage2_results": stage2_rows,
                "best_setup": best_setup,
            }

            if best_setup is not None:
                defaults_path = Path("results/meta_allocator_defaults.json").resolve()
                defaults_path.parent.mkdir(parents=True, exist_ok=True)
                defaults_payload = {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "current_strategy": request.current_strategy,
                    "target_strategy": request.target_strategy,
                    "evidence_profile": request.evidence_profile,
                    "evidence_max_age_days": request.evidence_max_age_days,
                    "confirm_points": best_setup["confirm_points"],
                    "switch_margin": best_setup["switch_margin"],
                    "tuning_source_artifact_id": artifact.get("artifact_id"),
                }
                defaults_path.write_text(json.dumps(defaults_payload, indent=2))
                artifact["defaults_path"] = str(defaults_path)

        return MetaEvidenceResponse(artifact=artifact)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/meta-evidence/latest", response_model=MetaEvidenceResponse)
async def get_latest_meta_evidence(current_strategy: str, target_strategy: str):
    """Load latest evidence artifact for a strategy pair."""
    from backtest.meta_evidence import find_latest_evidence_artifact

    artifact = find_latest_evidence_artifact(
        current_strategy=current_strategy,
        target_strategy=target_strategy,
    )
    if artifact is None:
        raise HTTPException(status_code=404, detail="No evidence artifact found for strategy pair")
    return MetaEvidenceResponse(artifact=artifact)


@router.get("/meta-evidence/{artifact_id}", response_model=MetaEvidenceResponse)
async def get_meta_evidence_by_id(artifact_id: str):
    """Load one evidence artifact by id."""
    from backtest.meta_evidence import find_evidence_artifact_by_id

    artifact = find_evidence_artifact_by_id(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")
    return MetaEvidenceResponse(artifact=artifact)


@router.post("/meta-promotion/run", response_model=MetaPromotionResponse)
async def run_meta_promotion(request: MetaPromotionRunRequest):
    """Run the strategy promotion governance report and persist artifact."""
    from backtest.meta_promotion import (
        DEFAULT_STRATEGIES,
        DEFAULT_BROKERS,
        SOXL_PROXY_BASELINE,
        SOXL_PROXY_STRATEGIES,
        run_meta_promotion_report,
    )

    try:
        if request.research_proxy_mode == "soxl_proxy":
            default_strategies = SOXL_PROXY_STRATEGIES
            default_baseline = SOXL_PROXY_BASELINE
        else:
            default_strategies = DEFAULT_STRATEGIES
            default_baseline = DEFAULT_STRATEGIES[0]
        strategies = request.strategies or list(default_strategies)
        baseline = request.baseline or default_baseline
        brokers = request.brokers or list(DEFAULT_BROKERS)

        artifact = run_meta_promotion_report(
            strategy_paths=strategies,
            baseline_path=baseline,
            start=request.start,
            end=request.end,
            initial_capital=request.initial_capital,
            costs_pct=request.costs_pct,
            metric_basis=request.metric_basis,
            tax_enabled=request.tax_enabled,
            brokers=brokers,
            align=request.align,
            skip_failed=request.skip_failed,
            validate=request.validate_data,
            research_proxy_mode=request.research_proxy_mode,
            tail_risk_gate_basis=request.tail_risk_gate_basis,
        )
        return MetaPromotionResponse(artifact=artifact)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/meta-promotion/list", response_model=MetaPromotionListResponse)
async def list_meta_promotion(limit: int = 50):
    """List recent promotion artifacts (metadata only) for the browser."""
    from backtest.meta_promotion import list_promotion_artifacts

    artifacts = list_promotion_artifacts(limit=limit)
    return MetaPromotionListResponse(artifacts=artifacts)


@router.get("/meta-promotion/latest", response_model=MetaPromotionResponse)
async def get_latest_meta_promotion():
    """Load the latest promotion artifact, if any."""
    from backtest.meta_promotion import find_latest_promotion_artifact

    artifact = find_latest_promotion_artifact()
    if artifact is None:
        raise HTTPException(status_code=404, detail="No promotion artifact found")
    return MetaPromotionResponse(artifact=artifact)


@router.get("/meta-promotion/{artifact_id}", response_model=MetaPromotionResponse)
async def get_meta_promotion_by_id(artifact_id: str):
    """Load one promotion artifact by id."""
    from backtest.meta_promotion import find_promotion_artifact_by_id

    artifact = find_promotion_artifact_by_id(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")
    return MetaPromotionResponse(artifact=artifact)


@router.get("/meta-promotion/{artifact_id}/markdown")
async def get_meta_promotion_markdown(artifact_id: str):
    """Return the promotion_report.md content as plain text."""
    from fastapi.responses import PlainTextResponse
    from backtest.meta_promotion import read_promotion_markdown_by_id

    text = read_promotion_markdown_by_id(artifact_id)
    if text is None:
        raise HTTPException(status_code=404, detail=f"Markdown not found: {artifact_id}")
    return PlainTextResponse(text)


@router.post("/meta-bootstrap/run", response_model=MetaBootstrapResponse)
async def run_meta_bootstrap(request: MetaBootstrapRunRequest):
    """Run bilateral evidence bootstrap to select a neutral start strategy."""
    from backtest.meta_bootstrap import run_meta_bootstrap_decision

    try:
        artifact = run_meta_bootstrap_decision(
            strategy_a=request.strategy_a,
            strategy_b=request.strategy_b,
            strategy_a_params=coerce_numeric_inputs(request.strategy_a_params or {}),
            strategy_b_params=coerce_numeric_inputs(request.strategy_b_params or {}),
            as_of=request.as_of,
            evidence_profile=request.evidence_profile,
            evidence_compare_mode=request.evidence_compare_mode,
            evidence_max_age_days=request.evidence_max_age_days,
            custom_thresholds=coerce_numeric_inputs(request.custom_thresholds or {}),
            train_years=request.train_years,
            test_years=request.test_years,
            step_months=request.step_months,
            anchored=request.anchored,
            start_date=request.start_date,
            initial_capital=request.initial_capital,
            costs_pct=request.costs_pct,
            skip_failed=request.skip_failed,
            metric_basis=request.metric_basis,
            fallback_cagr_tie_band_pp=request.fallback_cagr_tie_band_pp,
            fallback_tie_breaker=request.fallback_tie_breaker,
            artifact_path=request.artifact_path,
            save_artifact=True,
        )
        return MetaBootstrapResponse(artifact=artifact)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run", response_model=BacktestResultResponse)
async def run_backtest(request: RunRequest):
    """Run a backtest and return results."""
    from backtest.backtester import Backtester, BacktestConfig
    from backtest.data import DataLoader
    from backtest.utils import MONTH_END_FREQ

    try:
        run_id = request.run_id or str(uuid4())
        start_time = time.time()
        _set_progress(
            RUN_PROGRESS,
            run_id,
            status="running",
            message="Preparing backtest...",
            elapsed_seconds=0.0,
            cancel_requested=False,
        )
        request_params = coerce_numeric_inputs(request.params or {})
        initial_capital = safe_float(request.initial_capital)
        costs_pct = safe_float(request.costs_pct)
        tax_exemption = safe_float(request.tax_exemption)
        execution_lag_days = int(request.execution_lag_days or 0)
        max_volume_participation = safe_float(request.max_volume_participation)
        min_daily_dollar_volume = safe_float(request.min_daily_dollar_volume) or 0.0
        load_volumes = _needs_volume_data(max_volume_participation, min_daily_dollar_volume)
        exposure_policy = _build_exposure_policy_payload(request)
        risk_overlay = _build_risk_overlay_payload(request)

        def cancel_check():
            return RUN_PROGRESS.get(run_id, {}).get("cancel_requested", False)

        def run_sync():
            if cancel_check():
                raise RuntimeError("Run cancelled by user.")

            _set_progress(
                RUN_PROGRESS,
                run_id,
                status="loading_data",
                message="Loading strategy...",
                elapsed_seconds=time.time() - start_time,
            )

            strategy = load_strategy_instance(
                Path(request.strategy),
                params=request_params if request_params else None
            )

            if cancel_check():
                raise RuntimeError("Run cancelled by user.")

            # Prepare assets list
            assets = _extend_assets_for_exposure_policy(strategy.assets.copy(), exposure_policy)

            # Add benchmark ticker if needed
            if request.benchmark:
                from backtest.assets import get_benchmark_ticker
                bench_ticker = get_benchmark_ticker(request.benchmark)
                if bench_ticker not in assets:
                    assets.append(bench_ticker)

            _set_progress(
                RUN_PROGRESS,
                run_id,
                status="loading_data",
                message="Loading price data...",
                elapsed_seconds=time.time() - start_time,
            )

            # Load data
            # Use align="ffill" to handle PIT strategies with many historical tickers
            # that may not all have overlapping data
            data = DataLoader.yahoo(
                tickers=assets,
                start=request.start_date,
                end=request.end_date,
                currency="EUR",
                align="ffill",
                skip_failed=request.skip_failed,
                load_dividends=request.drip_enabled or request.tax_enabled,
                load_volumes=load_volumes,
                validate=request.enable_validation,
            )

            _set_progress(
                RUN_PROGRESS,
                run_id,
                status="running",
                message="Running backtest...",
                elapsed_seconds=time.time() - start_time,
            )

            # Configure backtest
            config = BacktestConfig(
                initial_capital=initial_capital if initial_capital is not None else request.initial_capital,
                costs_pct=costs_pct if costs_pct is not None else request.costs_pct,
                benchmark=request.benchmark,
                rebalance_frequency=request.rebalance_frequency,
                execution_lag_days=execution_lag_days,
                max_volume_participation=max_volume_participation,
                min_daily_dollar_volume=min_daily_dollar_volume,
                liquidity_on_missing_volume=request.liquidity_on_missing_volume,
                exposure_policy=exposure_policy,
                risk_overlay=risk_overlay,
                tax_enabled=request.tax_enabled,
                tax_exemption_amount=tax_exemption if tax_exemption is not None else request.tax_exemption,
                metric_basis=request.metric_basis,
                validate=request.enable_validation,
                drip_enabled=request.drip_enabled,
            )

            # Run backtest
            backtester = Backtester(strategy, data, config)
            return backtester.run()

        result = await anyio.to_thread.run_sync(run_sync)

        # Check if equity curve is empty
        if result.equity_curve is None or len(result.equity_curve) == 0:
            raise ValueError(
                f"Strategy '{result.strategy.name}' produced no results. "
                "This may happen if:\n"
                "- The date range doesn't have enough data\n"
                "- For PIT strategies: The universe CSV is missing or has no data for this period\n"
                "- All tickers failed to download (try enabling 'Skip Failed Tickers')"
            )

        # Generate charts
        charts = generate_charts(result)

        # Calculate years
        years = (result.equity_curve.index[-1] - result.equity_curve.index[0]).days / 365.25

        # Build response
        metrics = _metrics_to_response(result.metrics)
        metrics_gross = _metrics_to_response(result.metrics_gross) if result.metrics_gross else None
        metrics_net = _metrics_to_response(result.metrics_net) if result.metrics_net else None

        # Tax summary
        tax_summary = None
        t = result.tax_summary
        if t:
            tax_summary = TaxSummaryResponse(
                total_tax_paid=t.total_tax_paid,
                total_realized_gains=t.total_realized_gains,
                total_realized_losses=t.total_realized_losses,
                net_realized_gain=t.net_realized_gain,
                exemption_used=t.exemption_used,
                effective_tax_rate=t.effective_tax_rate,
                tax_drag_cagr_pp=t.tax_drag_cagr_pp,
                tax_drag_final_pct=t.tax_drag_final_pct,
                final_value_gross=t.final_value_gross,
                final_value_net_realized=t.final_value_net_realized,
                final_value_net_liquidation=t.final_value_net_liquidation,
                tax_paid_liquidation=t.tax_paid_liquidation,
                unrealized_gain_at_end=t.unrealized_gain_at_end,
                cagr_gross=t.cagr_gross,
                cagr_net_realized=t.cagr_net_realized,
                cagr_net_liquidation=t.cagr_net_liquidation,
            )

        # Trading costs
        m = result.metrics
        trading_costs = TradingCostsResponse(
            total_costs=m.total_costs,
            costs_per_year=m.total_costs / years if years > 0 else 0,
            costs_pct_of_final=m.total_costs / result.equity_curve.iloc[-1] * 100 if result.equity_curve.iloc[-1] > 0 else 0,
            num_trades=m.num_trades,
            turnover_annual=m.turnover_annual or 0,
        )

        # Trades (last 50)
        trades = []
        for trade in result.trades[-50:]:
            # For SELL trades, always show tax_paid (even if 0), for BUY show None
            tax_paid_value = None
            if trade.action == "SELL":
                tax_paid_value = getattr(trade, 'tax_paid_trade', 0.0) or 0.0
            trades.append(TradeResponse(
                date=str(trade.date.date()) if hasattr(trade.date, 'date') else str(trade.date),
                ticker=trade.ticker,
                action=trade.action,
                shares=trade.shares,
                price=trade.price_exec,
                value=trade.value_exec,
                costs=trade.costs,
                tax_paid=tax_paid_value,
            ))

        # Monthly returns
        monthly_returns = []
        try:
            equity_ffill = result.equity_curve.ffill()
            monthly = equity_ffill.resample(MONTH_END_FREQ).last().pct_change().dropna()
            for idx, ret in monthly.items():
                monthly_returns.append(MonthlyReturnResponse(
                    year=idx.year,
                    month=idx.month,
                    return_pct=float(ret * 100),
                ))
        except Exception:
            pass  # Skip if monthly returns fail

        # Prepare equity curve data
        equity_data = {
            "dates": [d.isoformat() for d in result.equity_curve.index],
            "values": result.equity_curve.values.tolist(),
        }

        benchmark_data = None
        if result.benchmark_curve is not None:
            benchmark_data = {
                "dates": [d.isoformat() for d in result.benchmark_curve.index],
                "values": result.benchmark_curve.values.tolist(),
            }

        # Final values
        final_value_gross = None
        final_value_net = None
        if result.equity_curve_gross is not None:
            final_value_gross = float(result.equity_curve_gross.iloc[-1])
        if result.equity_curve_net is not None:
            final_value_net = float(result.equity_curve_net.iloc[-1])

        _set_progress(
            RUN_PROGRESS,
            run_id,
            status="complete",
            message="Complete",
            elapsed_seconds=time.time() - start_time,
        )

        return BacktestResultResponse(
            strategy_name=result.strategy.name,
            start_date=str(result.equity_curve.index[0].date()),
            end_date=str(result.equity_curve.index[-1].date()),
            initial_capital=request.initial_capital,
            final_value=float(result.equity_curve.iloc[-1]),
            final_value_gross=final_value_gross,
            final_value_net=final_value_net,
            years=years,
            benchmark_name=request.benchmark,
            metrics=metrics,
            metrics_gross=metrics_gross,
            metrics_net=metrics_net,
            tax_enabled=request.tax_enabled,
            tax_summary=tax_summary,
            trading_costs=trading_costs,
            trades=trades,
            monthly_returns=monthly_returns,
            equity_chart=charts.get("equity"),
            drawdown_chart=charts.get("drawdown"),
            monthly_returns_chart=charts.get("monthly_returns"),
            equity_curve=equity_data,
            benchmark_curve=benchmark_data,
            constraint_impact=getattr(result, "constraint_impact", None),
        )

    except RuntimeError as e:
        _set_progress(
            RUN_PROGRESS,
            run_id,
            status="cancelled",
            message=str(e),
            elapsed_seconds=time.time() - start_time,
        )
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        _set_progress(
            RUN_PROGRESS,
            run_id,
            status="error",
            message=str(e),
            elapsed_seconds=time.time() - start_time,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/html", response_class=HTMLResponse)
async def run_backtest_html(request: Request, run_request: RunRequest):
    """Run a backtest and return HTML result partial (for HTMX)."""
    try:
        result = await run_backtest(run_request)
        return templates.TemplateResponse(
            request,
            "partials/_result.html",
            {
                "result": result.model_dump(),
            },
        )
    except HTTPException as e:
        return templates.TemplateResponse(
            request,
            "partials/_error.html",
            {
                "error": e.detail,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "partials/_error.html",
            {
                "error": str(e),
            },
        )


def _metrics_to_response(metrics) -> MetricsResponse:
    """Convert a Metrics dataclass to MetricsResponse."""
    return MetricsResponse(
        total_return=metrics.total_return,
        cagr=metrics.cagr,
        volatility=metrics.volatility,
        sharpe_ratio=metrics.sharpe_ratio,
        sortino_ratio=getattr(metrics, "sortino_ratio", None),
        max_drawdown=metrics.max_drawdown,
        max_drawdown_duration=metrics.max_drawdown_duration,
        calmar_ratio=getattr(metrics, "calmar_ratio", None),
        win_rate_monthly=getattr(metrics, "win_rate_monthly", None),
        best_month=getattr(metrics, "best_month", None),
        worst_month=getattr(metrics, "worst_month", None),
        num_trades=getattr(metrics, "num_trades", 0),
        turnover_annual=getattr(metrics, "turnover_annual", None),
        total_costs=getattr(metrics, "total_costs", 0.0),
    )


@router.post("/compare", response_model=CompareResultResponse)
async def compare_strategies(request: CompareRequest):
    """Compare multiple strategies side by side."""
    from backtest.backtester import Backtester, BacktestConfig
    from backtest.data import DataLoader

    try:
        run_id = request.run_id or str(uuid4())
        start_time = time.time()
        _set_progress(
            COMPARE_PROGRESS,
            run_id,
            status="running",
            strategy_index=0,
            strategy_total=0,
            strategy_name=None,
            elapsed_seconds=0.0,
            cancel_requested=False,
        )

        def cancel_check():
            return COMPARE_PROGRESS.get(run_id, {}).get("cancel_requested", False)

        def run_sync():
            if cancel_check():
                raise RuntimeError("Compare cancelled by user.")

            initial_capital = safe_float(request.initial_capital)
            costs_pct = safe_float(request.costs_pct)
            tax_exemption = safe_float(request.tax_exemption)
            execution_lag_days = int(request.execution_lag_days or 0)
            max_volume_participation = safe_float(request.max_volume_participation)
            min_daily_dollar_volume = safe_float(request.min_daily_dollar_volume) or 0.0
            load_volumes = _needs_volume_data(max_volume_participation, min_daily_dollar_volume)
            exposure_policy = _build_exposure_policy_payload(request)
            risk_overlay = _build_risk_overlay_payload(request)

            # Load strategies first
            strategy_instances = []
            for strat_config in request.strategies:
                if cancel_check():
                    raise RuntimeError("Compare cancelled by user.")
                params = coerce_numeric_inputs(strat_config.params or {})
                strategy = load_strategy_instance(
                    Path(strat_config.strategy),
                    params=params if params else None
                )
                strategy_instances.append((strat_config, strategy))

            total_strategies = len(strategy_instances)
            _set_progress(
                COMPARE_PROGRESS,
                run_id,
                status="running",
                strategy_index=0,
                strategy_total=total_strategies,
                strategy_name=None,
                elapsed_seconds=time.time() - start_time,
            )

            # Add benchmark ticker
            bench_ticker = None
            if request.benchmark:
                from backtest.assets import get_benchmark_ticker
                bench_ticker = get_benchmark_ticker(request.benchmark)

            # Run backtests and collect results
            # Load data PER STRATEGY to support PIT strategies with dynamic universes
            rows = []
            benchmark_cagr = None
            benchmark_curve = None
            start_date = None
            end_date = None
            years = 0.0

            for idx, (strat_config, strategy) in enumerate(strategy_instances, 1):
                if cancel_check():
                    raise RuntimeError("Compare cancelled by user.")
                _set_progress(
                    COMPARE_PROGRESS,
                    run_id,
                    status="running",
                    strategy_index=idx,
                    strategy_total=total_strategies,
                    strategy_name=strategy.name,
                    elapsed_seconds=time.time() - start_time,
                )

                # Load data for this strategy (includes benchmark)
                assets_for_strategy = _extend_assets_for_exposure_policy(list(strategy.assets), exposure_policy)
                if bench_ticker and bench_ticker not in assets_for_strategy:
                    assets_for_strategy.append(bench_ticker)

                data = DataLoader.yahoo(
                    tickers=assets_for_strategy,
                    start=request.start_date,
                    end=request.end_date,
                    currency="EUR",
                    align="ffill",  # Required for PIT strategies with dynamic universes
                    skip_failed=request.skip_failed,
                    load_dividends=request.drip_enabled or request.tax_enabled,
                    load_volumes=load_volumes,
                    validate=request.enable_validation,
                )
                config = BacktestConfig(
                    initial_capital=initial_capital if initial_capital is not None else request.initial_capital,
                    costs_pct=costs_pct if costs_pct is not None else request.costs_pct,
                    benchmark=request.benchmark,
                    rebalance_frequency=strategy.rebalance_frequency,
                    execution_lag_days=execution_lag_days,
                    max_volume_participation=max_volume_participation,
                    min_daily_dollar_volume=min_daily_dollar_volume,
                    liquidity_on_missing_volume=request.liquidity_on_missing_volume,
                    exposure_policy=exposure_policy,
                    risk_overlay=risk_overlay,
                    tax_enabled=request.tax_enabled,
                    tax_exemption_amount=tax_exemption if tax_exemption is not None else request.tax_exemption,
                    metric_basis=request.metric_basis,
                    validate=request.enable_validation,
                    drip_enabled=request.drip_enabled,
                )

                backtester = Backtester(strategy, data, config)
                result = backtester.run()

                # Check if equity curve is empty
                if result.equity_curve is None or len(result.equity_curve) == 0:
                    raise ValueError(f"Strategy '{strategy.name}' produced no results. Check if data is available for the selected date range.")

                # Get dates from first result
                if start_date is None:
                    start_date = str(result.equity_curve.index[0].date())
                    end_date = str(result.equity_curve.index[-1].date())
                    years = (result.equity_curve.index[-1] - result.equity_curve.index[0]).days / 365.25

                    # Get benchmark curve
                    if result.benchmark_curve is not None and len(result.benchmark_curve) > 0:
                        benchmark_curve = {
                            "dates": [d.isoformat() for d in result.benchmark_curve.index],
                            "values": result.benchmark_curve.values.tolist(),
                        }
                        # Calculate benchmark CAGR from the benchmark curve
                        bm_start = result.benchmark_curve.iloc[0]
                        bm_end = result.benchmark_curve.iloc[-1]
                        if bm_start > 0 and years > 0:
                            benchmark_cagr = (bm_end / bm_start) ** (1 / years) - 1

                # Determine which metrics to use based on metric_basis
                t = result.tax_summary
                m = result.metrics
                mn = result.metrics_net

                if t and t.tax_enabled:
                    if request.metric_basis == "net_liquidation":
                        cagr = t.cagr_net_liquidation
                        final_value = t.final_value_net_liquidation
                        total_tax = t.total_tax_paid + t.tax_paid_liquidation
                    elif request.metric_basis == "net_realized":
                        cagr = t.cagr_net_realized
                        final_value = t.final_value_net_realized
                        total_tax = t.total_tax_paid
                    else:
                        cagr = t.cagr_gross
                        final_value = t.final_value_gross
                        total_tax = 0.0

                    cagr_gross = t.cagr_gross
                    tax_drag_pp = (cagr_gross - cagr) * 100 if request.metric_basis != "gross" else None

                    # Use net metrics for risk measures
                    metrics_for_risk = mn if mn else m
                else:
                    cagr = m.cagr
                    cagr_gross = m.cagr
                    final_value = float(result.equity_curve.iloc[-1])
                    total_tax = None
                    tax_drag_pp = None
                    metrics_for_risk = m

                # Calculate excess CAGR
                excess_cagr = (cagr - benchmark_cagr) if benchmark_cagr is not None else None

                # Equity curve for chart
                equity_data = {
                    "dates": [d.isoformat() for d in result.equity_curve.index],
                    "values": result.equity_curve.values.tolist(),
                }

                rows.append(CompareRowResponse(
                    strategy_name=strategy.name,
                    is_benchmark=False,
                    final_value=final_value,
                    cagr=cagr,
                    cagr_gross=cagr_gross,
                    excess_cagr=excess_cagr,
                    tax_drag_pp=tax_drag_pp,
                    volatility=metrics_for_risk.volatility,
                    sharpe_ratio=metrics_for_risk.sharpe_ratio,
                    sortino_ratio=getattr(metrics_for_risk, "sortino_ratio", 0.0) or 0.0,
                    max_drawdown=metrics_for_risk.max_drawdown,
                    calmar_ratio=getattr(metrics_for_risk, "calmar_ratio", 0.0) or 0.0,
                    win_rate_monthly=getattr(metrics_for_risk, "win_rate_monthly", None),
                    total_tax=total_tax,
                    total_costs=m.total_costs,
                    num_trades=m.num_trades,
                    equity_curve=equity_data,
                ))

            # Add benchmark row if available
            if benchmark_cagr is not None and benchmark_curve is not None:
                rows.append(CompareRowResponse(
                    strategy_name=f"Benchmark ({request.benchmark})",
                    is_benchmark=True,
                    final_value=float(benchmark_curve["values"][-1]),
                    cagr=benchmark_cagr,
                    cagr_gross=benchmark_cagr,
                    excess_cagr=None,
                    tax_drag_pp=None,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    sortino_ratio=0.0,
                    max_drawdown=0.0,
                    calmar_ratio=0.0,
                    total_tax=None,
                    total_costs=0.0,
                    num_trades=0,
                    equity_curve=benchmark_curve,
                ))

            return CompareResultResponse(
                start_date=start_date or request.start_date,
                end_date=end_date or "",
                initial_capital=request.initial_capital,
                years=years,
                benchmark_name=request.benchmark,
                metric_basis=request.metric_basis,
                tax_enabled=request.tax_enabled,
                rows=rows,
                benchmark_curve=benchmark_curve,
            )

        result = await anyio.to_thread.run_sync(run_sync)
        _set_progress(
            COMPARE_PROGRESS,
            run_id,
            status="complete",
            strategy_index=result.rows.__len__(),
            strategy_total=len(request.strategies),
            elapsed_seconds=time.time() - start_time,
        )
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        _set_progress(
            COMPARE_PROGRESS,
            run_id,
            status="cancelled",
            message=str(e),
            elapsed_seconds=time.time() - start_time,
        )
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        _set_progress(
            COMPARE_PROGRESS,
            run_id,
            status="error",
            elapsed_seconds=time.time() - start_time,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sweep", response_model=SweepResultResponse)
async def run_sweep(request: SweepRequest):
    """Run sweep (robustness) analysis on multiple strategies."""
    import copy
    from backtest.sweep import (
        SweepConfig,
        run_sweep as run_sweep_analysis,
        run_sweep_with_strategies,
        resolve_strategy_paths,
    )
    from backtest.batch_optimize import load_strategy_from_file

    try:
        run_id = request.run_id or str(uuid4())
        start_time = time.time()
        _set_sweep_progress(
            run_id,
            status="running",
            cancel_requested=False,
            run_count=0,
            total_runs=0,
            strategy_index=0,
            strategy_total=0,
            strategy_name=None,
            iteration_index=0,
            iteration_total=0,
            elapsed_seconds=0.0,
        )
        initial_capital = safe_float(request.initial_capital)
        costs_pct = safe_float(request.costs_pct)
        tax_exemption = safe_float(request.tax_exemption)
        execution_lag_days = int(request.execution_lag_days or 0)
        max_volume_participation = safe_float(request.max_volume_participation)
        min_daily_dollar_volume = safe_float(request.min_daily_dollar_volume) or 0.0
        risk_overlay = _build_risk_overlay_payload(request)
        strategy_params = coerce_numeric_inputs(request.strategy_params or {})

        # Resolve strategy paths
        strategy_paths = resolve_strategy_paths(request.strategies)

        # Create sweep config
        config = SweepConfig(
            mode=request.mode,
            window_length=request.window_length,
            from_date=request.from_date,
            to_date=request.to_date,
            end_date=request.end_date,
            start_grid=request.start_grid,
            step=request.step,
            initial_capital=initial_capital if initial_capital is not None else request.initial_capital,
            rebalance_frequency=request.rebalance_frequency,
            costs_pct=costs_pct if costs_pct is not None else request.costs_pct,
            benchmark_ticker=request.benchmark,
            execution_lag_days=execution_lag_days,
            max_volume_participation=max_volume_participation,
            min_daily_dollar_volume=min_daily_dollar_volume,
            liquidity_on_missing_volume=request.liquidity_on_missing_volume,
            risk_overlay=risk_overlay,
            tax_enabled=request.tax_enabled,
            tax_exemption=tax_exemption if tax_exemption is not None else request.tax_exemption,
            metric_basis=request.metric_basis,
            drip_enabled=request.drip_enabled,
            skip_failed=request.skip_failed,
            validate=False,  # Suppress warnings in web UI
        )

        # Run sweep analysis
        if strategy_params:
            strategies = []
            strategy_files = {}

            for path in strategy_paths:
                instance, cls, _ = load_strategy_from_file(str(path))
                params_to_apply = {}
                rebal_freq = instance.rebalance_frequency if instance else "monthly"

                if cls.__name__ in strategy_params:
                    opt_params = copy.deepcopy(strategy_params[cls.__name__])
                    rebal_freq = opt_params.pop("rebalance_frequency", rebal_freq)
                    params_to_apply = {k: v for k, v in opt_params.items() if not k.startswith("_")}

                    if params_to_apply:
                        params_to_apply = coerce_params_to_signature(cls, params_to_apply)
                        try:
                            instance = cls(**params_to_apply)
                        except TypeError:
                            instance = cls()
                            for k, v in params_to_apply.items():
                                setattr(instance, k, v)

                if instance is None:
                    instance = cls()

                instance.rebalance_frequency = rebal_freq
                strategies.append(instance)
                strategy_files[instance.name] = str(path)

            def progress_callback(**kwargs):
                _set_sweep_progress(
                    run_id,
                    elapsed_seconds=time.time() - start_time,
                    **kwargs,
                )

            def cancel_check():
                return SWEEP_PROGRESS.get(run_id, {}).get("cancel_requested", False)

            result = await anyio.to_thread.run_sync(
                lambda: run_sweep_with_strategies(
                    strategies=strategies,
                    strategy_files=strategy_files,
                    config=config,
                    progress=False,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            )
        else:
            def progress_callback(**kwargs):
                _set_sweep_progress(
                    run_id,
                    elapsed_seconds=time.time() - start_time,
                    **kwargs,
                )

            def cancel_check():
                return SWEEP_PROGRESS.get(run_id, {}).get("cancel_requested", False)

            result = await anyio.to_thread.run_sync(
                lambda: run_sweep_analysis(
                    strategy_paths,
                    config,
                    progress=False,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            )

        # Convert summaries to response format (with NaN handling)
        summaries = []
        for s in result.summaries:
            summaries.append(SweepSummaryRow(
                strategy_name=s.strategy_name,
                strategy_file=s.strategy_file,
                rank=s.rank,
                pareto_front=s.pareto_front,
                num_windows=s.num_windows,
                num_ok=s.num_ok,
                num_skipped=s.num_skipped,
                median_cagr=safe_float(s.median_cagr),
                p10_cagr=safe_float(s.p10_cagr),
                p90_cagr=safe_float(s.p90_cagr),
                worst_cagr=safe_float(s.worst_cagr),
                best_cagr=safe_float(s.best_cagr),
                prob_negative_cagr=safe_float(s.prob_negative_cagr),
                median_sharpe=safe_float(s.median_sharpe),
                p10_sharpe=safe_float(s.p10_sharpe),
                median_sortino=safe_float(s.median_sortino),
                median_maxdd=safe_float(s.median_maxdd),
                worst_maxdd=safe_float(s.worst_maxdd),
                median_vol=safe_float(s.median_vol),
                median_calmar=safe_float(s.median_calmar),
                median_cagr_gross=safe_float(s.median_cagr_gross),
                median_cagr_net_realized=safe_float(s.median_cagr_net_realized),
                median_cagr_net_liquidation=safe_float(s.median_cagr_net_liquidation),
                p10_cagr_gross=safe_float(s.p10_cagr_gross),
                p10_cagr_net_realized=safe_float(s.p10_cagr_net_realized),
                p10_cagr_net_liquidation=safe_float(s.p10_cagr_net_liquidation),
                hit_rate_vs_benchmark=safe_float(s.hit_rate_vs_benchmark),
                median_excess_cagr=safe_float(s.median_excess_cagr),
                p10_excess_cagr=safe_float(s.p10_excess_cagr),
                prob_underperform_benchmark=safe_float(s.prob_underperform_benchmark),
            ))

        # Convert window results for charts (with NaN handling)
        window_results = []
        for wr in result.window_results:
            window_results.append(SweepWindowResult(
                strategy_name=wr.strategy_name,
                window_start=wr.window_start.strftime("%Y-%m-%d"),
                window_end=wr.window_end.strftime("%Y-%m-%d"),
                window_years=wr.window_years,
                status=wr.status,
                cagr=safe_float(wr.cagr) or 0.0,
                cagr_gross=safe_float(wr.cagr_gross) or 0.0,
                cagr_net_realized=safe_float(wr.cagr_net_realized) or 0.0,
                cagr_net_liquidation=safe_float(wr.cagr_net_liquidation) or 0.0,
                sharpe_ratio=safe_float(wr.sharpe_ratio) or 0.0,
                sortino_ratio=safe_float(wr.sortino_ratio) or 0.0,
                max_drawdown=safe_float(wr.max_drawdown) or 0.0,
                volatility=safe_float(wr.volatility) or 0.0,
                benchmark_cagr=safe_float(wr.benchmark_cagr),
                excess_cagr=safe_float(wr.excess_cagr),
            ))

        _set_sweep_progress(
            run_id,
            status="complete",
            elapsed_seconds=time.time() - start_time,
            run_count=len(result.window_results),
            total_runs=len(result.window_results),
        )

        return SweepResultResponse(
            mode=config.mode,
            window_length=config.window_length,
            start_grid=config.start_grid,
            metric_basis=config.metric_basis,
            tax_enabled=config.tax_enabled,
            benchmark_ticker=config.benchmark_ticker,
            num_windows=len(result.windows),
            first_window_start=result.windows[0].start.strftime("%Y-%m-%d") if result.windows else None,
            last_window_end=result.windows[-1].end.strftime("%Y-%m-%d") if result.windows else None,
            summaries=summaries,
            window_results=window_results,
        )

    except SweepCancelled as e:
        _set_sweep_progress(
            run_id,
            status="cancelled",
            elapsed_seconds=time.time() - start_time,
        )
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        _set_sweep_progress(run_id, status="error", elapsed_seconds=time.time() - start_time)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize", response_model=OptimizeResultResponse)
async def run_optimization(request: OptimizeRequest):
    """Run parameter optimization for a strategy."""
    import itertools
    import inspect
    import copy

    from backtest.backtester import Backtester, BacktestConfig
    from backtest.data import DataLoader
    from backtest.strategy import Strategy

    run_id = request.run_id or str(uuid4())
    start_time = time.time()
    _set_progress(
        OPTIMIZE_PROGRESS,
        run_id,
        status="running",
        run_count=0,
        total_runs=0,
        elapsed_seconds=0.0,
        message="Preparing optimization...",
    )

    def run_sync() -> tuple[OptimizeResultResponse, int, int]:
        initial_capital = safe_float(request.initial_capital)
        costs_pct = safe_float(request.costs_pct)
        tax_exemption = safe_float(request.tax_exemption)
        wf_train_years = safe_float(request.wf_train_years)
        wf_test_years = safe_float(request.wf_test_years)
        wf_step_months = max(1, int(request.wf_step_months or 12))
        wf_nested = bool(request.walk_forward_nested)
        wf_inner_train_years = safe_float(request.wf_inner_train_years)
        wf_inner_test_years = safe_float(request.wf_inner_test_years)
        wf_inner_step_months = max(1, int(request.wf_inner_step_months or 6))
        wf_inner_anchored = bool(request.wf_inner_anchored)
        execution_lag_days = int(request.execution_lag_days or 0)
        max_volume_participation = safe_float(request.max_volume_participation)
        min_daily_dollar_volume = safe_float(request.min_daily_dollar_volume) or 0.0
        load_volumes = _needs_volume_data(max_volume_participation, min_daily_dollar_volume)
        risk_overlay = _build_risk_overlay_payload(request)

        _set_progress(
            OPTIMIZE_PROGRESS,
            run_id,
            status="running",
            elapsed_seconds=time.time() - start_time,
            message="Loading strategy...",
        )

        strategy_path = Path(request.strategy)
        if not strategy_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {strategy_path}")

        # Import strategy module
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["strategy_module"] = module
        spec.loader.exec_module(module)

        # Find strategy instance and class
        strategy_class = None
        strategy_instance = None

        if hasattr(module, 'strategy'):
            obj = getattr(module, 'strategy')
            if isinstance(obj, Strategy):
                strategy_instance = obj
                strategy_class = type(obj)

        if strategy_class is None:
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                    strategy_class = obj
                    break
                elif isinstance(obj, Strategy) and strategy_instance is None:
                    strategy_instance = obj
                    strategy_class = type(obj)

        if strategy_class is None:
            raise ValueError("No Strategy class found in file")

        # Get base assets from strategy instance
        # If no instance exists, try to create one with default parameters
        if strategy_instance:
            base_assets = list(strategy_instance.assets)
        else:
            # Try to instantiate with default parameters
            try:
                temp_strategy = strategy_class()
                base_assets = list(temp_strategy.assets)
                strategy_instance = temp_strategy  # Use this as base for later
            except Exception:
                # Fallback: try __new__ (for strategies that set assets as class attribute)
                try:
                    temp_strategy = strategy_class.__new__(strategy_class)
                    if hasattr(temp_strategy, 'assets'):
                        base_assets = list(temp_strategy.assets)
                    else:
                        base_assets = []
                except Exception:
                    base_assets = []

        if not base_assets:
            raise ValueError(
                f"Could not determine strategy assets. "
                f"Ensure the strategy file has a 'strategy = StrategyClass(...)' instance "
                f"or the class can be instantiated with default parameters."
            )

        _set_progress(
            OPTIMIZE_PROGRESS,
            run_id,
            status="loading_data",
            elapsed_seconds=time.time() - start_time,
            message="Loading price data...",
        )

        # Load data
        data = DataLoader.yahoo(
            tickers=base_assets,
            start=request.start_date,
            end=request.end_date,
            currency="EUR",
            align="ffill",
            skip_failed=request.skip_failed,
            load_dividends=request.drip_enabled or request.tax_enabled,
            load_volumes=load_volumes,
            validate=request.enable_validation,
        )

        # Build parameter grid
        param_grid = {}
        for item in request.param_grid:
            param_grid[item.name] = coerce_numeric_inputs(item.values)

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        if param_values:
            param_combinations = list(itertools.product(*param_values))
        else:
            param_combinations = [()]

        # End date from data (needed for both paths)
        end_date_str = request.end_date
        if not end_date_str and len(data.prices) > 0:
            end_date_str = str(data.prices.index[-1].date())

        # Walk-Forward Optimization Path
        if request.walk_forward:
            from backtest.research.walk_forward import WalkForwardAnalysis
            from backtest.config.run_config import RunConfig, CostConfig, TaxConfig
            from backtest.cli import _build_walk_forward_windows, _instantiate_strategy_for_params

            _set_progress(
                OPTIMIZE_PROGRESS,
                run_id,
                status="running",
                run_count=0,
                total_runs=0,
                elapsed_seconds=time.time() - start_time,
                message=(
                    "Running nested walk-forward optimization..."
                    if wf_nested
                    else "Running walk-forward optimization..."
                ),
            )

            # Convert years to months
            train_months = int(((wf_train_years if wf_train_years is not None else 5) * 12))
            test_months = int(((wf_test_years if wf_test_years is not None else 1) * 12))

            def _select_best_index(values: List[Optional[float]], minimize: bool) -> Optional[int]:
                valid_indices = [idx for idx, value in enumerate(values) if value is not None]
                if not valid_indices:
                    return None
                if minimize:
                    return min(
                        valid_indices,
                        key=lambda idx: values[idx] if values[idx] is not None else float("inf"),
                    )
                return max(
                    valid_indices,
                    key=lambda idx: values[idx] if values[idx] is not None else float("-inf"),
                )

            # Nested walk-forward path for CLI/Web parity.
            if wf_nested:
                outer_train_days = int((wf_train_years if wf_train_years is not None else 5) * 252)
                outer_test_days = int((wf_test_years if wf_test_years is not None else 1) * 252)
                outer_step_days = int(wf_step_months * 21)

                if outer_train_days <= 0 or outer_test_days <= 0 or outer_step_days <= 0:
                    raise ValueError(
                        "Walk-forward requires positive train/test/step settings."
                    )

                inner_train_years = wf_inner_train_years if wf_inner_train_years is not None else 3.0
                inner_test_years = wf_inner_test_years if wf_inner_test_years is not None else 1.0
                inner_train_days = int(inner_train_years * 252)
                inner_test_days = int(inner_test_years * 252)
                inner_step_days = int(wf_inner_step_months * 21)

                if inner_train_days <= 0 or inner_test_days <= 0 or inner_step_days <= 0:
                    raise ValueError(
                        "Nested walk-forward requires positive inner train/test/step settings."
                    )
                if outer_train_days < inner_train_days + inner_test_days:
                    raise ValueError(
                        "Outer training window is too short for nested walk-forward "
                        f"(need at least {inner_train_years}y + {inner_test_years}y)."
                    )

                outer_windows = _build_walk_forward_windows(
                    dates=data.prices.index,
                    train_days=outer_train_days,
                    test_days=outer_test_days,
                    step_days=outer_step_days,
                    anchored=request.wf_anchored or False,
                )
                if not outer_windows:
                    raise ValueError("No valid outer walk-forward windows could be generated.")

                frequencies = request.rebalance_frequencies or ["monthly"]
                if not frequencies:
                    frequencies = ["monthly"]

                def run_metric_backtest(eval_data, rebalance_frequency, params):
                    strategy = _instantiate_strategy_for_params(strategy_class, strategy_instance, params)
                    config = BacktestConfig(
                        initial_capital=initial_capital if initial_capital is not None else request.initial_capital,
                        costs_pct=costs_pct if costs_pct is not None else request.costs_pct,
                        execution_lag_days=execution_lag_days,
                        max_volume_participation=max_volume_participation,
                        min_daily_dollar_volume=min_daily_dollar_volume,
                        liquidity_on_missing_volume=request.liquidity_on_missing_volume,
                        risk_overlay=risk_overlay,
                        rebalance_frequency=rebalance_frequency,
                        tax_enabled=request.tax_enabled,
                        tax_exemption_amount=tax_exemption if tax_exemption is not None else request.tax_exemption,
                        metric_basis=request.metric_basis,
                        validate=request.enable_validation,
                        drip_enabled=request.drip_enabled,
                    )
                    backtester = Backtester(strategy, eval_data, config)
                    result = backtester.run()
                    metrics = result.metrics
                    metric_value = safe_float(getattr(metrics, request.metric, None))
                    if metric_value is None:
                        raise ValueError(f"Metric '{request.metric}' is not available.")
                    return float(metric_value), metrics

                windows_response: List[WalkForwardWindowResponse] = []
                train_sharpes: List[float] = []
                test_sharpes: List[float] = []
                degradations: List[float] = []
                oos_metric_values: List[Optional[float]] = []

                for idx, outer_window in enumerate(outer_windows, start=1):
                    _set_progress(
                        OPTIMIZE_PROGRESS,
                        run_id,
                        status="running",
                        run_count=idx - 1,
                        total_runs=len(outer_windows),
                        elapsed_seconds=time.time() - start_time,
                        message=f"Running nested outer window {idx}/{len(outer_windows)}...",
                    )

                    outer_train_data = data.filter_dates(
                        outer_window["train_start"],
                        outer_window["train_end"],
                    )
                    outer_test_data = data.filter_dates(
                        outer_window["test_start"],
                        outer_window["test_end"],
                    )

                    inner_windows = _build_walk_forward_windows(
                        dates=outer_train_data.prices.index,
                        train_days=inner_train_days,
                        test_days=inner_test_days,
                        step_days=inner_step_days,
                        anchored=wf_inner_anchored,
                    )
                    if not inner_windows:
                        continue

                    candidate_scores = []
                    for rebalance_frequency in frequencies:
                        for param_combo in param_combinations:
                            params = dict(zip(param_names, param_combo))
                            inner_scores: List[float] = []
                            for inner_window in inner_windows:
                                inner_test_data = outer_train_data.filter_dates(
                                    inner_window["test_start"],
                                    inner_window["test_end"],
                                )
                                try:
                                    metric_value, _ = run_metric_backtest(
                                        eval_data=inner_test_data,
                                        rebalance_frequency=rebalance_frequency,
                                        params=params,
                                    )
                                    inner_scores.append(metric_value)
                                except Exception:
                                    continue
                            if inner_scores:
                                mean_score = float(sum(inner_scores) / len(inner_scores))
                                variance = float(
                                    sum((x - mean_score) ** 2 for x in inner_scores) / len(inner_scores)
                                )
                                score_std = math.sqrt(variance)
                                candidate_scores.append(
                                    (
                                        rebalance_frequency,
                                        params,
                                        mean_score,
                                        score_std,
                                        len(inner_scores),
                                    )
                                )

                    if not candidate_scores:
                        continue

                    if request.minimize:
                        candidate_scores.sort(key=lambda row: (row[2], row[3]))
                    else:
                        candidate_scores.sort(key=lambda row: (-row[2], row[3]))

                    best_rebalance, best_params, inner_score_mean, inner_score_std, inner_windows_used = candidate_scores[0]

                    try:
                        train_metric, train_metrics = run_metric_backtest(
                            eval_data=outer_train_data,
                            rebalance_frequency=best_rebalance,
                            params=best_params,
                        )
                        test_metric, test_metrics = run_metric_backtest(
                            eval_data=outer_test_data,
                            rebalance_frequency=best_rebalance,
                            params=best_params,
                        )
                    except Exception:
                        continue

                    train_sharpe = safe_float(getattr(train_metrics, "sharpe_ratio", None)) or 0.0
                    test_sharpe = safe_float(getattr(test_metrics, "sharpe_ratio", None)) or 0.0
                    test_metric_value = safe_float(test_metric)
                    if train_metric == 0:
                        degradation = 0.0
                    elif request.minimize:
                        degradation = (test_metric - train_metric) / abs(train_metric)
                    else:
                        degradation = (train_metric - test_metric) / abs(train_metric)

                    train_sharpes.append(train_sharpe)
                    test_sharpes.append(test_sharpe)
                    degradations.append(degradation)
                    oos_metric_values.append(test_metric_value)

                    windows_response.append(
                        WalkForwardWindowResponse(
                            window_num=idx,
                            train_start=str(outer_window["train_start"].date()),
                            train_end=str(outer_window["train_end"].date()),
                            test_start=str(outer_window["test_start"].date()),
                            test_end=str(outer_window["test_end"].date()),
                            best_params=best_params,
                            train_sharpe=train_sharpe,
                            test_sharpe=test_sharpe,
                            test_cagr=safe_float(getattr(test_metrics, "cagr", None)),
                            degradation=safe_float(degradation),
                            best_rebalance_frequency=best_rebalance,
                            inner_score_mean=safe_float(inner_score_mean),
                            inner_score_std=safe_float(inner_score_std),
                            inner_windows=inner_windows_used,
                        )
                    )

                if not windows_response:
                    raise ValueError("No valid nested walk-forward windows produced results.")

                avg_is_sharpe = safe_float(sum(train_sharpes) / len(train_sharpes)) if train_sharpes else 0.0
                avg_oos_sharpe = safe_float(sum(test_sharpes) / len(test_sharpes)) if test_sharpes else 0.0
                degradation_ratio = safe_float(sum(degradations) / len(degradations)) if degradations else 0.0
                overfitting_score = min(1.0, max(0.0, degradation_ratio or 0.0))
                best_sharpe_window = max(
                    windows_response,
                    key=lambda w: w.test_sharpe if w.test_sharpe is not None else float("-inf"),
                )
                best_metric_idx = _select_best_index(oos_metric_values, request.minimize)
                best_metric_window = (
                    windows_response[best_metric_idx]
                    if best_metric_idx is not None
                    else best_sharpe_window
                )
                best_metric_value = (
                    oos_metric_values[best_metric_idx]
                    if best_metric_idx is not None
                    else (best_sharpe_window.test_sharpe if best_sharpe_window else None)
                )

                wf_response = WalkForwardResponse(
                    strategy_name=strategy_instance.name if strategy_instance else strategy_class.__name__,
                    strategy_file=str(strategy_path),
                    metric=request.metric,
                    start_date=request.start_date,
                    end_date=end_date_str or "",
                    train_months=train_months,
                    test_months=test_months,
                    step_months=wf_step_months,
                    anchored=request.wf_anchored or False,
                    mode="nested",
                    nested=True,
                    inner_train_months=int(inner_train_years * 12),
                    inner_test_months=int(inner_test_years * 12),
                    inner_step_months=wf_inner_step_months,
                    inner_anchored=wf_inner_anchored,
                    num_windows=len(windows_response),
                    avg_is_sharpe=avg_is_sharpe,
                    avg_oos_sharpe=avg_oos_sharpe,
                    degradation_ratio=degradation_ratio,
                    overfitting_score=safe_float(overfitting_score),
                    windows=windows_response,
                    best_params=best_metric_window.best_params if best_metric_window else None,
                    best_oos_sharpe=best_sharpe_window.test_sharpe if best_sharpe_window else None,
                )

                response = OptimizeResultResponse(
                    strategy_name=wf_response.strategy_name,
                    strategy_file=wf_response.strategy_file,
                    metric=request.metric,
                    minimize=request.minimize,
                    start_date=request.start_date,
                    end_date=end_date_str or "",
                    metric_basis=request.metric_basis,
                    tax_enabled=request.tax_enabled,
                    total_combinations=len(param_combinations),
                    successful_runs=len(windows_response),
                    failed_runs=max(0, len(outer_windows) - len(windows_response)),
                    results=[],  # No grid results for walk-forward
                    best_params=wf_response.best_params,
                    best_rebalance_frequency=best_metric_window.best_rebalance_frequency if best_metric_window else None,
                    best_metric_value=best_metric_value,
                    walk_forward_result=wf_response,
                )
                return response, 0, 0

            # Standard walk-forward path.
            run_config = RunConfig(
                start_date=request.start_date,
                end_date=request.end_date,
                initial_capital=initial_capital if initial_capital is not None else request.initial_capital,
                rebalance_frequency=request.rebalance_frequencies[0] if request.rebalance_frequencies else "monthly",
                costs=CostConfig(commission_pct=costs_pct if costs_pct is not None else request.costs_pct),
                execution_lag_days=execution_lag_days,
                max_volume_participation=max_volume_participation,
                min_daily_dollar_volume=min_daily_dollar_volume,
                liquidity_on_missing_volume=request.liquidity_on_missing_volume,
                risk_overlay=risk_overlay,
                tax=TaxConfig(
                    enabled=request.tax_enabled,
                    exemption_amount=tax_exemption if tax_exemption is not None else request.tax_exemption,
                ),
            )

            wfa = WalkForwardAnalysis(
                strategy_class=strategy_class,
                param_grid=param_grid,
                train_months=train_months,
                test_months=test_months,
                step_months=wf_step_months,
                anchored=request.wf_anchored or False,
                metric=request.metric,
            )

            wf_result = wfa.run(data, run_config, progress=False)

            windows_response = []
            oos_metric_values: List[Optional[float]] = []
            for i, w in enumerate(wf_result.windows):
                test_metric_value = safe_float(w.test_metrics.get(request.metric))
                oos_metric_values.append(test_metric_value)
                windows_response.append(
                    WalkForwardWindowResponse(
                        window_num=i + 1,
                        train_start=str(w.train_start.date()),
                        train_end=str(w.train_end.date()),
                        test_start=str(w.test_start.date()),
                        test_end=str(w.test_end.date()),
                        best_params=w.best_params,
                        train_sharpe=safe_float(w.train_metrics.get("sharpe_ratio")),
                        test_sharpe=safe_float(w.test_metrics.get("sharpe_ratio")),
                        test_cagr=safe_float(w.test_metrics.get("cagr")),
                        degradation=safe_float(w.degradation),
                        best_rebalance_frequency=request.rebalance_frequencies[0] if request.rebalance_frequencies else "monthly",
                    )
                )

            best_sharpe_window = (
                max(wf_result.windows, key=lambda w: w.test_metrics.get("sharpe_ratio", float("-inf")))
                if wf_result.windows
                else None
            )
            best_metric_idx = _select_best_index(oos_metric_values, request.minimize)
            best_metric_window = (
                wf_result.windows[best_metric_idx]
                if best_metric_idx is not None
                else best_sharpe_window
            )
            best_metric_value = (
                oos_metric_values[best_metric_idx]
                if best_metric_idx is not None
                else (
                    safe_float(best_sharpe_window.test_metrics.get("sharpe_ratio"))
                    if best_sharpe_window
                    else None
                )
            )

            wf_response = WalkForwardResponse(
                strategy_name=strategy_instance.name if strategy_instance else strategy_class.__name__,
                strategy_file=str(strategy_path),
                metric=request.metric,
                start_date=request.start_date,
                end_date=end_date_str or "",
                train_months=train_months,
                test_months=test_months,
                step_months=wf_step_months,
                anchored=request.wf_anchored or False,
                mode="standard",
                nested=False,
                num_windows=len(wf_result.windows),
                avg_is_sharpe=safe_float(wf_result.overall_is_sharpe),
                avg_oos_sharpe=safe_float(wf_result.overall_oos_sharpe),
                degradation_ratio=safe_float(wf_result.degradation_ratio),
                overfitting_score=safe_float(wf_result.overfitting_score),
                windows=windows_response,
                best_params=best_metric_window.best_params if best_metric_window else None,
                best_oos_sharpe=safe_float(best_sharpe_window.test_metrics.get("sharpe_ratio")) if best_sharpe_window else None,
            )

            response = OptimizeResultResponse(
                strategy_name=wf_response.strategy_name,
                strategy_file=wf_response.strategy_file,
                metric=request.metric,
                minimize=request.minimize,
                start_date=request.start_date,
                end_date=end_date_str or "",
                metric_basis=request.metric_basis,
                tax_enabled=request.tax_enabled,
                total_combinations=len(param_combinations),
                successful_runs=len(wf_result.windows),
                failed_runs=0,
                results=[],  # No grid results for walk-forward
                best_params=wf_response.best_params,
                best_rebalance_frequency=request.rebalance_frequencies[0] if request.rebalance_frequencies else None,
                best_metric_value=best_metric_value,
                walk_forward_result=wf_response,
            )
            return response, 0, 0

        # Normal Grid Optimization (existing code)
        all_results = []
        total_runs = len(param_combinations) * len(request.rebalance_frequencies)
        run_count = 0

        _set_progress(
            OPTIMIZE_PROGRESS,
            run_id,
            status="running",
            run_count=0,
            total_runs=total_runs,
            elapsed_seconds=time.time() - start_time,
            message="Running optimization...",
        )

        for rebal_freq in request.rebalance_frequencies:
            for param_combo in param_combinations:
                run_count += 1
                params = dict(zip(param_names, param_combo))

                # Create strategy with these parameters
                try:
                    if strategy_instance:
                        init_sig = inspect.signature(strategy_class.__init__)
                        init_params = [p for p in init_sig.parameters.keys() if p != 'self']

                        kwargs = {}
                        for p in init_params:
                            if p in params:
                                kwargs[p] = params[p]
                            elif hasattr(strategy_instance, p):
                                kwargs[p] = getattr(strategy_instance, p)

                        strategy = strategy_class(**kwargs)
                    else:
                        strategy = strategy_class(**params)
                except Exception:
                    if strategy_instance:
                        strategy = copy.deepcopy(strategy_instance)
                        for k, v in params.items():
                            setattr(strategy, k, v)
                        if hasattr(strategy, '_rebuild_assets'):
                            strategy._rebuild_assets()
                    else:
                        all_results.append({
                            'rebalance_frequency': rebal_freq,
                            'params': params,
                            'error': "Could not create strategy",
                            '_metric_value': None,
                        })
                        _set_progress(
                            OPTIMIZE_PROGRESS,
                            run_id,
                            run_count=run_count,
                            total_runs=total_runs,
                            elapsed_seconds=time.time() - start_time,
                        )
                        continue

                # Configure backtest
                config = BacktestConfig(
                    initial_capital=initial_capital if initial_capital is not None else request.initial_capital,
                    costs_pct=costs_pct if costs_pct is not None else request.costs_pct,
                    execution_lag_days=execution_lag_days,
                    max_volume_participation=max_volume_participation,
                    min_daily_dollar_volume=min_daily_dollar_volume,
                    liquidity_on_missing_volume=request.liquidity_on_missing_volume,
                    risk_overlay=risk_overlay,
                    rebalance_frequency=rebal_freq,
                    tax_enabled=request.tax_enabled,
                    tax_exemption_amount=tax_exemption if tax_exemption is not None else request.tax_exemption,
                    metric_basis=request.metric_basis,
                    validate=request.enable_validation,
                    drip_enabled=request.drip_enabled,
                )

                # Run backtest
                try:
                    backtester = Backtester(strategy, data, config)
                    result = backtester.run()

                    # Use net metrics if tax enabled and metric_basis is net
                    if request.tax_enabled and request.metric_basis != "gross" and result.metrics_net:
                        m = result.metrics_net
                    else:
                        m = result.metrics

                    metric_value = safe_float(getattr(m, request.metric, None))

                    all_results.append({
                        'rebalance_frequency': rebal_freq,
                        'params': params,
                        'sharpe_ratio': safe_float(m.sharpe_ratio),
                        'sortino_ratio': safe_float(getattr(m, 'sortino_ratio', None)),
                        'cagr': safe_float(m.cagr),
                        'volatility': safe_float(m.volatility),
                        'max_drawdown': safe_float(m.max_drawdown),
                        'calmar_ratio': safe_float(getattr(m, 'calmar_ratio', None)),
                        'total_return': safe_float(m.total_return),
                        '_metric_value': metric_value,
                    })
                except Exception as e:
                    all_results.append({
                        'rebalance_frequency': rebal_freq,
                        'params': params,
                        'error': str(e),
                        '_metric_value': None,
                    })
                finally:
                    _set_progress(
                        OPTIMIZE_PROGRESS,
                        run_id,
                        run_count=run_count,
                        total_runs=total_runs,
                        elapsed_seconds=time.time() - start_time,
                    )

        # Sort results
        reverse = not request.minimize
        valid_results = [r for r in all_results if r.get('_metric_value') is not None]
        failed_results = [r for r in all_results if r.get('_metric_value') is None]

        sorted_results = sorted(
            valid_results,
            key=lambda x: x['_metric_value'],
            reverse=reverse
        )

        # Build response rows
        result_rows = []
        for i, r in enumerate(sorted_results[:request.top_n], 1):
            result_rows.append(OptimizeResultRow(
                rank=i,
                rebalance_frequency=r['rebalance_frequency'],
                params=r['params'],
                cagr=r.get('cagr'),
                sharpe_ratio=r.get('sharpe_ratio'),
                sortino_ratio=r.get('sortino_ratio'),
                max_drawdown=r.get('max_drawdown'),
                volatility=r.get('volatility'),
                calmar_ratio=r.get('calmar_ratio'),
                total_return=r.get('total_return'),
                metric_value=r.get('_metric_value'),
            ))

        # Best parameters
        best_params = None
        best_rebalance = None
        best_metric = None
        if sorted_results:
            best = sorted_results[0]
            best_params = best['params']
            best_rebalance = best['rebalance_frequency']
            best_metric = best['_metric_value']

        response = OptimizeResultResponse(
            strategy_name=strategy_instance.name if strategy_instance else strategy_class.__name__,
            strategy_file=str(strategy_path),
            metric=request.metric,
            minimize=request.minimize,
            start_date=request.start_date,
            end_date=end_date_str or "",
            metric_basis=request.metric_basis,
            tax_enabled=request.tax_enabled,
            total_combinations=total_runs,
            successful_runs=len(valid_results),
            failed_runs=len(failed_results),
            results=result_rows,
            best_params=best_params,
            best_rebalance_frequency=best_rebalance,
            best_metric_value=best_metric,
        )
        return response, run_count, total_runs

    try:
        response, run_count, total_runs = await anyio.to_thread.run_sync(run_sync)
        _set_progress(
            OPTIMIZE_PROGRESS,
            run_id,
            status="complete",
            run_count=run_count,
            total_runs=total_runs,
            elapsed_seconds=time.time() - start_time,
            message="Optimization completed.",
        )
        return response
    except FileNotFoundError as e:
        _set_progress(
            OPTIMIZE_PROGRESS,
            run_id,
            status="error",
            elapsed_seconds=time.time() - start_time,
        )
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        _set_progress(
            OPTIMIZE_PROGRESS,
            run_id,
            status="error",
            elapsed_seconds=time.time() - start_time,
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        _set_progress(
            OPTIMIZE_PROGRESS,
            run_id,
            status="error",
            elapsed_seconds=time.time() - start_time,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-optimize", response_model=BatchOptimizeResponse)
async def run_batch_optimization_api(request: BatchOptimizeRequest):
    """Run parameter optimization for multiple strategies.

    This endpoint optimizes parameters for all provided strategies using
    predefined parameter grids, then returns the best parameters for each
    strategy ranked by the selected metric.
    """
    from backtest.batch_optimize import run_batch_optimization

    try:
        run_id = request.run_id or str(uuid4())
        start_time = time.time()
        _set_progress(
            BATCH_PROGRESS,
            run_id,
            status="running",
            strategy_index=0,
            strategy_total=0,
            strategy_name=None,
            run_count=0,
            total_runs=0,
            elapsed_seconds=0.0,
            cancel_requested=False,
        )
        initial_capital = safe_float(request.initial_capital)
        costs_pct = safe_float(request.costs_pct)
        tax_exemption = safe_float(request.tax_exemption)
        wf_train_years = safe_float(request.wf_train_years)
        wf_test_years = safe_float(request.wf_test_years)
        wf_step_months = max(1, int(request.wf_step_months or 12))
        wf_anchored = bool(request.wf_anchored)
        wf_nested = bool(request.walk_forward_nested)
        wf_inner_train_years = safe_float(request.wf_inner_train_years)
        wf_inner_test_years = safe_float(request.wf_inner_test_years)
        wf_inner_step_months = max(1, int(request.wf_inner_step_months or 6))
        wf_inner_anchored = bool(request.wf_inner_anchored)
        execution_lag_days = int(request.execution_lag_days or 0)
        max_volume_participation = safe_float(request.max_volume_participation)
        min_daily_dollar_volume = safe_float(request.min_daily_dollar_volume) or 0.0
        risk_overlay = _build_risk_overlay_payload(request)

        # Resolve strategy file names to full paths
        all_strategies = list_strategies()
        strategy_paths = []
        for strat_name in request.strategies:
            # Find matching strategy
            found = False
            for s in all_strategies:
                if s["file_name"] == strat_name or s["file_path"] == strat_name:
                    strategy_paths.append(s["file_path"])
                    found = True
                    break
            if not found:
                raise ValueError(f"Strategy not found: {strat_name}")

        if not strategy_paths:
            raise ValueError("No valid strategies selected")

        def progress_callback(**kwargs):
            _set_progress(
                BATCH_PROGRESS,
                run_id,
                elapsed_seconds=time.time() - start_time,
                **kwargs,
            )

        def cancel_check():
            return BATCH_PROGRESS.get(run_id, {}).get("cancel_requested", False)

        # Run batch optimization
        batch_result = await anyio.to_thread.run_sync(
            lambda: run_batch_optimization(
                strategy_files=strategy_paths,
                start=request.start_date,
                end=request.end_date,
                metric=request.metric,
                minimize=request.minimize,
                rebalance_frequencies=request.rebalance_frequencies,
                initial_capital=initial_capital if initial_capital is not None else request.initial_capital,
                costs_pct=costs_pct if costs_pct is not None else request.costs_pct,
                execution_lag_days=execution_lag_days,
                max_volume_participation=max_volume_participation,
                min_daily_dollar_volume=min_daily_dollar_volume,
                liquidity_on_missing_volume=request.liquidity_on_missing_volume,
                risk_overlay=risk_overlay,
                tax_enabled=request.tax_enabled,
                metric_basis=request.metric_basis,
                skip_failed=request.skip_failed,
                validate=request.enable_validation,
                drip_enabled=request.drip_enabled,
                progress=False,  # Disable progress output in web API
                # Walk-Forward options
                walk_forward=request.walk_forward,
                walk_forward_nested=wf_nested,
                train_years=wf_train_years if wf_train_years is not None else request.wf_train_years,
                test_years=wf_test_years if wf_test_years is not None else request.wf_test_years,
                step_months=wf_step_months,
                anchored=wf_anchored,
                inner_train_years=wf_inner_train_years if wf_inner_train_years is not None else request.wf_inner_train_years,
                inner_test_years=wf_inner_test_years if wf_inner_test_years is not None else request.wf_inner_test_years,
                inner_step_months=wf_inner_step_months,
                inner_anchored=wf_inner_anchored,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
        )

        # Convert to response format
        rows = []
        export_params = {}

        for opt_result in batch_result.results:
            # Get metrics from best result in all_results
            best_metrics = {}
            if opt_result.all_results:
                # Find the best result entry (first one with matching metric value)
                for entry in opt_result.all_results:
                    if entry.get('_metric_value') == opt_result.best_metric_value:
                        best_metrics = {
                            'cagr': safe_float(entry.get('cagr')),
                            'sharpe_ratio': safe_float(entry.get('sharpe_ratio')),
                            'sortino_ratio': safe_float(entry.get('sortino_ratio')),
                            'max_drawdown': safe_float(entry.get('max_drawdown')),
                            'volatility': safe_float(entry.get('volatility')),
                            'calmar_ratio': safe_float(entry.get('calmar_ratio')),
                        }
                        break

            # Build walk-forward summary if available
            wf_summary = None
            if opt_result.walk_forward:
                wf = opt_result.walk_forward
                wf_summary = WalkForwardSummaryResponse(
                    num_windows=wf.num_windows,
                    avg_is_sharpe=safe_float(wf.avg_is_sharpe) or 0.0,
                    avg_oos_sharpe=safe_float(wf.avg_oos_sharpe) or 0.0,
                    degradation_ratio=safe_float(wf.degradation_ratio) or 0.0,
                    overfitting_score=safe_float(wf.overfitting_score) or 0.0,
                    best_oos_sharpe=safe_float(wf.best_oos_sharpe) or 0.0,
                    mode=getattr(wf, "mode", "standard"),
                    nested=bool(getattr(wf, "nested", False)),
                    inner_train_years=safe_float(getattr(wf, "inner_train_years", None)),
                    inner_test_years=safe_float(getattr(wf, "inner_test_years", None)),
                    inner_step_months=getattr(wf, "inner_step_months", None),
                    inner_anchored=getattr(wf, "inner_anchored", None),
                )
                # For walk-forward, use OOS sharpe as metric value if optimizing sharpe
                if request.metric == "sharpe_ratio":
                    best_metrics['sharpe_ratio'] = safe_float(wf.avg_oos_sharpe)

            rows.append(BatchOptimizeResultRow(
                rank=0,  # Will be set after sorting
                strategy_name=opt_result.strategy_name,
                strategy_file=opt_result.strategy_file,
                strategy_class=opt_result.strategy_class,
                best_params=opt_result.best_params,
                best_rebalance_frequency=opt_result.best_rebalance_frequency,
                metric_value=safe_float(opt_result.best_metric_value),
                total_combinations=len(opt_result.all_results) if opt_result.all_results else 0,
                walk_forward=wf_summary,
                **best_metrics,
            ))

            # Build export params
            export_params[opt_result.strategy_class] = {
                "rebalance_frequency": opt_result.best_rebalance_frequency,
                **opt_result.best_params,
            }

        # Sort by metric (descending for most metrics, ascending for max_drawdown)
        reverse = not request.minimize
        rows.sort(
            key=lambda r: r.metric_value if r.metric_value is not None else float('-inf'),
            reverse=reverse
        )

        # Update ranks after sorting
        for i, row in enumerate(rows, 1):
            row.rank = i

        _set_progress(
            BATCH_PROGRESS,
            run_id,
            status="complete",
            strategy_index=len(rows),
            strategy_total=len(request.strategies),
            elapsed_seconds=time.time() - start_time,
        )

        return BatchOptimizeResponse(
            metric=request.metric,
            minimize=request.minimize,
            start_date=request.start_date,
            end_date=batch_result.end_date or "",
            metric_basis=request.metric_basis,
            tax_enabled=request.tax_enabled,
            walk_forward_enabled=request.walk_forward,
            walk_forward_nested=request.walk_forward_nested if request.walk_forward else None,
            wf_train_years=request.wf_train_years if request.walk_forward else None,
            wf_test_years=request.wf_test_years if request.walk_forward else None,
            wf_step_months=request.wf_step_months if request.walk_forward else None,
            wf_anchored=request.wf_anchored if request.walk_forward else None,
            wf_inner_train_years=request.wf_inner_train_years if request.walk_forward and request.walk_forward_nested else None,
            wf_inner_test_years=request.wf_inner_test_years if request.walk_forward and request.walk_forward_nested else None,
            wf_inner_step_months=request.wf_inner_step_months if request.walk_forward and request.walk_forward_nested else None,
            wf_inner_anchored=request.wf_inner_anchored if request.walk_forward and request.walk_forward_nested else None,
            total_strategies=len(request.strategies),
            successful_strategies=len(rows),
            failed_strategies=len(request.strategies) - len(rows),
            results=rows,
            export_params=export_params,
        )

    except BatchOptimizeCancelled as e:
        _set_progress(
            BATCH_PROGRESS,
            run_id,
            status="cancelled",
            elapsed_seconds=time.time() - start_time,
        )
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
