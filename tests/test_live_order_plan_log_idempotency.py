"""Phase E2 — OrderPlanLog append-only idempotency (T-0370, Codex R3.5+R3.6+R2.15)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backtest.live.order_plan_log import (
    ALLOWED_RECEIPT_KEYS,
    OrderPlanLog,
    now_iso,
)
from backtest.live.orders import OrderPlanReceipt


def _make_receipt(spi: str = "ABC123") -> OrderPlanReceipt:
    return OrderPlanReceipt(
        stable_order_plan_id=spi,
        status="planned",
        emitted_at_iso=now_iso(),
        broker_label="dry_run",
        run_id="R",
        signals_as_of_iso="2024-05-13",
        plan_only=True,
        ticker="AAPL",
        action="BUY",
        target_shares=10.0,
        target_value=1700.0,
    )


def test_append_if_new_writes_first_time(tmp_path: Path):
    log = OrderPlanLog(tmp_path / "log.jsonl")
    receipt = _make_receipt("FIRST")
    stored, appended = log.append_if_new(receipt)
    assert appended is True
    assert log.path.exists()
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_append_if_new_does_not_duplicate(tmp_path: Path):
    """Codex R3.6: identical `stable_order_plan_id` -> no second
    entry."""
    log = OrderPlanLog(tmp_path / "log.jsonl")
    receipt = _make_receipt("SAME")
    log.append_if_new(receipt)
    log.append_if_new(receipt)
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_find_by_stable_order_plan_id(tmp_path: Path):
    log = OrderPlanLog(tmp_path / "log.jsonl")
    log.append_if_new(_make_receipt("A"))
    log.append_if_new(_make_receipt("B"))
    found = log.find_by_stable_order_plan_id("B")
    assert found is not None
    assert found["stable_order_plan_id"] == "B"


def test_log_allowlist_drops_unknown_keys(tmp_path: Path):
    """Codex R2.15: API key injection does not end up in the log.

    We can only test this indirectly, since OrderPlanReceipt is a
    dataclass with fixed fields and cannot have mock API key fields
    added to it. Instead: the allowlist contains NO
    `api_key`/`account_id` entries.
    """
    assert "api_key" not in ALLOWED_RECEIPT_KEYS
    assert "account_id" not in ALLOWED_RECEIPT_KEYS
    assert "secret" not in ALLOWED_RECEIPT_KEYS


def test_log_lines_are_canonical_json(tmp_path: Path):
    log = OrderPlanLog(tmp_path / "log.jsonl")
    log.append_if_new(_make_receipt("CANON"))
    line = log.path.read_text(encoding="utf-8").splitlines()[0]
    parsed = json.loads(line)
    assert parsed["stable_order_plan_id"] == "CANON"
    assert parsed["plan_only"] is True
