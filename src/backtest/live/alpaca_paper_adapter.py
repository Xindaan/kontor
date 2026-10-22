"""Phase E2 — Alpaca paper-preview adapter (T-0368, Codex R2.2).

Paper mode only. Phase E code **NEVER** calls `submit_order`.
Optionally, `alpaca-py` can be wired in for reconciliation
(`get_account()`/`list_positions()`) — there too, ONLY read calls.

The adapter produces OrderPlanReceipts; an additional JSON dump
lands in `results/live_orders/alpaca_paper/<run_id>.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

from backtest.live.execution_plan_base import ExecutionPlanAdapter
from backtest.live.order_plan_log import OrderPlanLog, now_iso
from backtest.live.orders import Order, OrderPlanReceipt


ALPACA_PAPER_ENDPOINT = "https://paper-api.alpaca.markets"


class AlpacaPaperAdapter(ExecutionPlanAdapter):
    """Produces plans for the Alpaca paper path. No submit."""

    def __init__(self, output_dir: str | Path = "results/live_orders/alpaca_paper") -> None:
        self._output_dir = Path(output_dir).expanduser()
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def broker_label(self) -> str:
        return "alpaca_paper_preview"

    def _route_order_plan(self, order: Order) -> OrderPlanReceipt:
        return OrderPlanReceipt(
            stable_order_plan_id=order.stable_order_plan_id,
            status="planned",
            emitted_at_iso=now_iso(),
            broker_label=self.broker_label,
            run_id=order.run_id,
            signals_as_of_iso=order.signals_as_of_iso,
            plan_only=True,
            ticker=order.ticker,
            action=order.action,
            target_shares=order.target_shares,
            target_value=order.target_value,
        )

    def emit_order_plan(
        self,
        orders: Sequence[Order],
        *,
        log: OrderPlanLog,
    ) -> List[OrderPlanReceipt]:
        receipts = super().emit_order_plan(orders, log=log)
        if not receipts:
            return receipts
        run_id = receipts[0].run_id
        dump_path = self._output_dir / f"{run_id}.json"
        payload = {
            "endpoint": ALPACA_PAPER_ENDPOINT,
            "plan_only": True,
            "receipts": [r.to_dict() for r in receipts],
        }
        dump_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return receipts


__all__ = ["ALPACA_PAPER_ENDPOINT", "AlpacaPaperAdapter"]
