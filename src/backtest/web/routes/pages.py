"""HTML page routes for the web frontend."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Form, Query
from fastapi.responses import HTMLResponse

from backtest.web.deps import templates
from backtest.web.services.strategies import list_strategies, get_param_schema
from backtest.web.config import DEFAULT_STRATEGIES_DIR

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Dashboard / landing page."""
    strategies = list_strategies()
    return templates.TemplateResponse(
        request,
        "pages/index.html",
        {
            "strategies": strategies,
            "page_title": "Dashboard",
        },
    )


@router.get("/run", response_class=HTMLResponse)
async def run_form(
    request: Request,
    strategy: Optional[str] = Query(None, description="Pre-selected strategy path")
):
    """Backtest form page."""
    strategies = list_strategies()

    # Get strategy schema if pre-selected
    strategy_schema = None
    if strategy:
        try:
            strategy_schema = get_param_schema(Path(strategy))
        except Exception:
            pass

    return templates.TemplateResponse(
        request,
        "pages/run.html",
        {
            "strategies": strategies,
            "selected_strategy": strategy,
            "strategy_schema": strategy_schema,
            "page_title": "Run Backtest",
        },
    )


@router.get("/run/result", response_class=HTMLResponse)
async def run_result(request: Request):
    """Backtest result page (after form submission)."""
    # This will be populated by HTMX after the API call
    return templates.TemplateResponse(
        request,
        "pages/result.html",
        {
            "page_title": "Backtest Result",
        },
    )


@router.get("/signals", response_class=HTMLResponse)
async def signals_form(request: Request):
    """Live signals page."""
    strategies = list_strategies()
    return templates.TemplateResponse(
        request,
        "pages/signals.html",
        {
            "strategies": strategies,
            "page_title": "Signals",
        },
    )


# Placeholder routes for future features

@router.get("/compare", response_class=HTMLResponse)
async def compare_form(request: Request):
    """Strategy comparison page."""
    strategies = list_strategies()
    return templates.TemplateResponse(
        request,
        "pages/compare.html",
        {
            "strategies": strategies,
            "page_title": "Compare Strategies",
        },
    )


@router.get("/sweep", response_class=HTMLResponse)
async def sweep_form(request: Request):
    """Sweep analysis page."""
    strategies = list_strategies()
    return templates.TemplateResponse(
        request,
        "pages/sweep.html",
        {
            "strategies": strategies,
            "page_title": "Sweep Analysis",
        },
    )


@router.get("/optimize", response_class=HTMLResponse)
async def optimize_form(request: Request):
    """Parameter optimization page."""
    strategies = list_strategies()
    return templates.TemplateResponse(
        request,
        "pages/optimize.html",
        {
            "strategies": strategies,
            "page_title": "Optimize",
        },
    )


@router.get("/batch-optimize", response_class=HTMLResponse)
async def batch_optimize_form(request: Request):
    """Batch optimization page - optimize multiple strategies at once."""
    strategies = list_strategies()
    return templates.TemplateResponse(
        request,
        "pages/batch_optimize.html",
        {
            "strategies": strategies,
            "page_title": "Batch Optimize",
        },
    )


@router.get("/playbook", response_class=HTMLResponse)
async def playbook_form(request: Request):
    """Meta-Playbook (strategy promotion governance) page."""
    from backtest.meta_promotion import (
        DEFAULT_STRATEGIES,
        DEFAULT_BROKERS,
        SOXL_PROXY_BASELINE,
        SOXL_PROXY_STRATEGIES,
    )

    strategies = list_strategies()
    return templates.TemplateResponse(
        request,
        "pages/playbook.html",
        {
            "strategies": strategies,
            "default_strategies": list(DEFAULT_STRATEGIES),
            "default_baseline": DEFAULT_STRATEGIES[0],
            "soxl_proxy_strategies": list(SOXL_PROXY_STRATEGIES),
            "soxl_proxy_baseline": SOXL_PROXY_BASELINE,
            "default_brokers": list(DEFAULT_BROKERS),
            "page_title": "Meta-Playbook",
        },
    )


@router.get("/manual-data", response_class=HTMLResponse)
async def manual_data_page(request: Request):
    """Manual data provenance management page."""
    return templates.TemplateResponse(
        request,
        "pages/manual_data.html",
        {
            "page_title": "Manual Data",
        },
    )


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_list(request: Request):
    """Job history page (coming soon)."""
    return templates.TemplateResponse(
        request,
        "pages/coming_soon.html",
        {
            "page_title": "Jobs",
            "feature": "Job History",
        },
    )
