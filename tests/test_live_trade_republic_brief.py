"""Phase E2 — TR/Maxblue order brief tests (T-0369, T-0384, Codex R2.14+response)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from backtest.live.maxblue_brief import MaxblueBriefAdapter
from backtest.live.order_plan_log import OrderPlanLog
from backtest.live.orders import Order, stable_order_plan_id
from backtest.live.trade_republic_brief import (
    GERMAN_BRIEF_COLUMNS,
    TradeRepublicBriefAdapter,
)


def _make_order(ticker: str, broker: str = "trade_republic_brief") -> Order:
    spi = stable_order_plan_id(
        run_id="RUN", strategy_hash="S", portfolio_snapshot_hash="P",
        broker_label=broker, ticker=ticker, action="BUY",
        target_shares=10.0,
    )
    return Order(
        ticker=ticker, action="BUY", target_shares=10.0, target_value=1700.0,
        broker_label=broker, stable_order_plan_id=spi, run_id="RUN",
        strategy_hash="S", portfolio_snapshot_hash="P",
        signals_as_of_iso="2024-05-13",
    )


def test_tr_brief_writes_german_columns(tmp_path: Path):
    output_dir = tmp_path / "tr"
    adapter = TradeRepublicBriefAdapter(
        output_dir=output_dir,
        mapping_root="data/live/instrument_mapping",
    )
    log = OrderPlanLog(tmp_path / "log.jsonl")
    receipts = adapter.emit_order_plan([_make_order("AAPL")], log=log)
    assert receipts[0].status == "planned"
    csv_path = output_dir / "RUN.csv"
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    assert tuple(header) == GERMAN_BRIEF_COLUMNS


def test_tr_brief_skips_when_mapping_missing(tmp_path: Path):
    """Codex R2.14: without mapping -> skipped."""
    output_dir = tmp_path / "tr"
    # Empty mapping source.
    empty_root = tmp_path / "empty_mapping"
    empty_root.mkdir()
    adapter = TradeRepublicBriefAdapter(
        output_dir=output_dir, mapping_root=empty_root
    )
    log = OrderPlanLog(tmp_path / "log.jsonl")
    receipts = adapter.emit_order_plan([_make_order("UNKNOWN")], log=log)
    assert receipts[0].status == "skipped"
    assert receipts[0].reason == "missing_instrument_mapping"


def test_maxblue_uses_separate_mapping(tmp_path: Path):
    output_dir = tmp_path / "mb"
    adapter = MaxblueBriefAdapter(
        output_dir=output_dir,
        mapping_root="data/live/instrument_mapping",
    )
    log = OrderPlanLog(tmp_path / "log.jsonl")
    receipts = adapter.emit_order_plan(
        [_make_order("AAPL", broker="maxblue_brief")], log=log
    )
    assert receipts[0].status == "planned"


def test_brief_is_deterministic(tmp_path: Path):
    """Two identical calls -> same CSV."""
    output_dir = tmp_path / "tr"
    adapter = TradeRepublicBriefAdapter(
        output_dir=output_dir,
        mapping_root="data/live/instrument_mapping",
    )
    log_a = OrderPlanLog(tmp_path / "log_a.jsonl")
    log_b = OrderPlanLog(tmp_path / "log_b.jsonl")
    adapter.emit_order_plan([_make_order("AAPL")], log=log_a)
    csv_a = (output_dir / "RUN.csv").read_text(encoding="utf-8")
    adapter.emit_order_plan([_make_order("AAPL")], log=log_b)
    csv_b = (output_dir / "RUN.csv").read_text(encoding="utf-8")
    assert csv_a == csv_b
