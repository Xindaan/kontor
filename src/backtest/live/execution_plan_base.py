"""Phase E2 — ExecutionPlanAdapter ABC (T-0364, Codex R2.1+R3.2).

Template method ``emit_order_plan(orders, *, log, ...)`` is the
only public interface. Concrete adapters override
``_route_order_plan()`` (NO `submit` vocabulary).

Codex R2.1: method names containing `execute`/`submit`/`placeOrder`
are forbidden throughout the ``src/backtest/live/`` tree and are
greped for via ``test_live_no_submit_vocab.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence

from backtest.live.order_plan_log import OrderPlanLog, now_iso
from backtest.live.orders import Order, OrderPlanReceipt


class ExecutionPlanAdapter(ABC):
    """Phase E template-method base for order plan emission.

    Concrete adapters override only :meth:`_route_order_plan` and
    optionally :meth:`_reconcile_positions`. The base handles
    idempotency (`OrderPlanLog.append_if_new`), timestamps and
    setting the `plan_only` flag.
    """

    @property
    @abstractmethod
    def broker_label(self) -> str:
        """Stable broker identifier, e.g. ``"dry_run"`` or
        ``"trade_republic_brief"``."""

    @property
    def plan_only(self) -> bool:
        """Codex R3.2: Phase E adapters all set `plan_only=True`."""

        return True

    @abstractmethod
    def _route_order_plan(self, order: Order) -> OrderPlanReceipt:
        """Produces a receipt for a single order.

        Concrete adapters set `status` to one of
        ``{"planned", "skipped", "rejected"}`` (Codex R2.1) and
        `reason` if needed. They call NO
        ``submit_order``/``placeOrder`` APIs.
        """

    def emit_order_plan(
        self,
        orders: Sequence[Order],
        *,
        log: OrderPlanLog,
    ) -> List[OrderPlanReceipt]:
        """Produces receipts for all orders + appends (idempotent)."""

        out: List[OrderPlanReceipt] = []
        for order in orders:
            receipt = self._route_order_plan(order)
            # Phase E safety: `plan_only=True` is non-negotiable.
            # Adapters that set it differently are corrected here.
            if not receipt.plan_only:
                receipt = OrderPlanReceipt(
                    **{**receipt.to_dict(), "plan_only": True}
                )
            stored, _appended = log.append_if_new(receipt)
            out.append(stored)
        return out


__all__ = ["ExecutionPlanAdapter"]
