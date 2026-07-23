"""Phase E2 — IBKR BasketTrader CSV adapter (T-0367, Codex R2.2+R3.10).

**NO ``ib_insync`` import**. Adapter only writes a TWS-importable
CSV. Real submit comes in Phase F.

CSV format (TWS BasketTrader):
``Action,Quantity,Symbol,SecType,Exchange,Currency,TimeInForce,
OrderType,LmtPrice,BasketTag``.

Plus a canonical `OrderPlanReceipt` dump as JSON.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Sequence

from backtest.live.execution_plan_base import ExecutionPlanAdapter
from backtest.live.order_plan_log import OrderPlanLog, now_iso
from backtest.live.orders import Order, OrderPlanReceipt


BASKET_TRADER_COLUMNS: tuple[str, ...] = (
    "Action",
    "Quantity",
    "Symbol",
    "SecType",
    "Exchange",
    "Currency",
    "TimeInForce",
    "OrderType",
    "LmtPrice",
    "BasketTag",
)


class IBKRBasketCsvAdapter(ExecutionPlanAdapter):
    """Writes TWS BasketTrader CSV + JSON receipt dump."""

    def __init__(
        self,
        output_dir: str | Path = "results/live_orders/ibkr_basket",
        *,
        exchange: str = "SMART",
        currency: str = "USD",
        time_in_force: str = "DAY",
        order_type: str = "MKT",
    ) -> None:
        self._output_dir = Path(output_dir).expanduser()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._exchange = exchange
        self._currency = currency
        self._time_in_force = time_in_force
        self._order_type = order_type

    @property
    def broker_label(self) -> str:
        return "ibkr_basket_csv"

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
        csv_path = self._output_dir / f"{run_id}.csv"
        json_path = self._output_dir / f"{run_id}.json"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(BASKET_TRADER_COLUMNS))
            for order, receipt in zip(orders, receipts):
                if receipt.status != "planned":
                    continue
                writer.writerow(
                    [
                        order.action.upper(),
                        # TWS BasketTrader Quantity is always positive;
                        # Action gives the direction. Negative Quantity
                        # is rejected by the importer.
                        abs(int(round(order.target_shares))),
                        order.ticker.upper(),
                        "STK",
                        self._exchange,
                        self._currency,
                        self._time_in_force,
                        self._order_type,
                        "",  # LmtPrice empty for MKT
                        run_id,
                    ]
                )
        json_path.write_text(
            json.dumps([r.to_dict() for r in receipts], sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return receipts


__all__ = ["BASKET_TRADER_COLUMNS", "IBKRBasketCsvAdapter"]
