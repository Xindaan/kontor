"""Phase E2 — execution plan layer.

This layer produces broker-ready order plans, but NO real
broker orders. Real-money submit is Phase F.

Subpackage layout:
- `execution_plan_base`: `ExecutionPlanAdapter` ABC.
- `orders`: `Order`, `OrderPlanReceipt`, `stable_order_plan_id()`,
  `compute_run_id()`.
- `signal_report_io`: `SignalReport.from_json`,
  `canonical_signal_report_hash()`.
- `instrument_mapping`: `InstrumentMapping`, `load_mapping()`.
- `order_plan_log`: `OrderPlanLog` (append-only).
- Adapters: `dry_run_adapter`, `ibkr_basket_csv_adapter`,
  `alpaca_paper_adapter`, `trade_republic_brief`, `maxblue_brief`.
"""

from backtest.live.execution_plan_base import ExecutionPlanAdapter
from backtest.live.orders import (
    Order,
    OrderPlanReceipt,
    compute_run_id,
    stable_order_plan_id,
)

__all__ = [
    "ExecutionPlanAdapter",
    "Order",
    "OrderPlanReceipt",
    "compute_run_id",
    "stable_order_plan_id",
]
