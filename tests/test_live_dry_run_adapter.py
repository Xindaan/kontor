"""Phase E2 — Dry-Run-Adapter Tests (T-0366, T-0383)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backtest.live.dry_run_adapter import DryRunAdapter
from backtest.live.order_plan_log import OrderPlanLog
from backtest.live.orders import (
    Order,
    compute_run_id,
    stable_order_plan_id,
)


def _make_order(ticker: str, run_id: str = "RUN") -> Order:
    spi = stable_order_plan_id(
        run_id=run_id, strategy_hash="S",
        portfolio_snapshot_hash="P", broker_label="dry_run",
        ticker=ticker, action="BUY", target_shares=10.0,
    )
    return Order(
        ticker=ticker, action="BUY", target_shares=10.0, target_value=1700.0,
        broker_label="dry_run", stable_order_plan_id=spi, run_id=run_id,
        strategy_hash="S", portfolio_snapshot_hash="P",
        signals_as_of_iso="2024-05-13",
    )


def test_dry_run_emits_planned_status(tmp_path: Path):
    adapter = DryRunAdapter(output_dir=tmp_path / "dry_run")
    log = OrderPlanLog(tmp_path / "log.jsonl")
    receipts = adapter.emit_order_plan(
        [_make_order("AAPL"), _make_order("MSFT")],
        log=log,
    )
    assert len(receipts) == 2
    assert all(r.status == "planned" for r in receipts)
    assert all(r.plan_only is True for r in receipts)


def test_dry_run_appends_idempotent_second_call(tmp_path: Path):
    adapter = DryRunAdapter(output_dir=tmp_path / "dry_run")
    log = OrderPlanLog(tmp_path / "log.jsonl")
    orders = [_make_order("AAPL")]
    adapter.emit_order_plan(orders, log=log)
    adapter.emit_order_plan(orders, log=log)
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_dry_run_writes_json_dump(tmp_path: Path):
    output_dir = tmp_path / "dry_run"
    adapter = DryRunAdapter(output_dir=output_dir)
    log = OrderPlanLog(tmp_path / "log.jsonl")
    orders = [_make_order("AAPL", run_id="MYRUN")]
    adapter.emit_order_plan(orders, log=log)
    dump = output_dir / "MYRUN.json"
    assert dump.exists()
    payload = json.loads(dump.read_text(encoding="utf-8"))
    assert payload[0]["ticker"] == "AAPL"
    assert payload[0]["plan_only"] is True
