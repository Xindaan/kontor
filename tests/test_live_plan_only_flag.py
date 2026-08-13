"""Phase E2 — all adapters set plan_only=True (T-0379, Codex R3.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from backtest.live.alpaca_paper_adapter import AlpacaPaperAdapter
from backtest.live.dry_run_adapter import DryRunAdapter
from backtest.live.ibkr_basket_csv_adapter import IBKRBasketCsvAdapter
from backtest.live.maxblue_brief import MaxblueBriefAdapter
from backtest.live.order_plan_log import OrderPlanLog
from backtest.live.orders import Order, stable_order_plan_id
from backtest.live.trade_republic_brief import TradeRepublicBriefAdapter


def _make_order(ticker: str, broker: str) -> Order:
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


@pytest.mark.parametrize(
    "adapter_cls,broker_label",
    [
        (DryRunAdapter, "dry_run"),
        (IBKRBasketCsvAdapter, "ibkr_basket_csv"),
        (AlpacaPaperAdapter, "alpaca_paper_preview"),
    ],
)
def test_simple_adapter_sets_plan_only_true(tmp_path: Path, adapter_cls, broker_label):
    adapter = adapter_cls(output_dir=tmp_path / broker_label)
    log = OrderPlanLog(tmp_path / f"{broker_label}.jsonl")
    receipts = adapter.emit_order_plan(
        [_make_order("AAPL", broker_label)], log=log
    )
    assert receipts
    for r in receipts:
        assert r.plan_only is True, f"{broker_label} adapter must set plan_only=True"


@pytest.mark.parametrize(
    "adapter_cls,broker_label",
    [
        (TradeRepublicBriefAdapter, "trade_republic_brief"),
        (MaxblueBriefAdapter, "maxblue_brief"),
    ],
)
def test_brief_adapter_sets_plan_only_true(tmp_path: Path, adapter_cls, broker_label):
    adapter = adapter_cls(
        output_dir=tmp_path / broker_label,
        mapping_root="data/live/instrument_mapping",
    )
    log = OrderPlanLog(tmp_path / f"{broker_label}.jsonl")
    receipts = adapter.emit_order_plan(
        [_make_order("AAPL", broker_label)], log=log
    )
    assert receipts
    for r in receipts:
        assert r.plan_only is True
