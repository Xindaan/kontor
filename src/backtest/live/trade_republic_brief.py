"""Phase E2 — Trade Republic order brief (T-0369, Codex R2.14+R3.10+Antwort).

Writes an order brief CSV with German columns:
``ISIN,WKN,Stueckzahl,Aktion,Limit,Begruendung``.

Mapping is mandatory (Codex R2.14): without an entry in
`data/live/instrument_mapping/trade_republic.csv` -> skip with
`reason="missing_instrument_mapping"`.

No API submit. User types it in manually.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from backtest.live.execution_plan_base import ExecutionPlanAdapter
from backtest.live.instrument_mapping import InstrumentMapping, load_mapping
from backtest.live.order_plan_log import OrderPlanLog, now_iso
from backtest.live.orders import Order, OrderPlanReceipt


GERMAN_BRIEF_COLUMNS: tuple[str, ...] = (
    "ISIN",
    "WKN",
    "Stueckzahl",
    "Aktion",
    "Limit",
    "Begruendung",
)


class _BriefAdapter(ExecutionPlanAdapter):
    """Shared logic for DE brokers (TR / Maxblue)."""

    broker_label_value: str = "trade_republic_brief"
    mapping_broker_key: str = "trade_republic"
    output_dir: str = "results/live_orders/trade_republic"

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        mapping_root: Path | str = "data/live/instrument_mapping",
    ) -> None:
        self._output_dir = Path(output_dir or self.output_dir).expanduser()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._mapping = load_mapping(self.mapping_broker_key, root=mapping_root)
        # Carry reason suffixes from the orders — injected by the CLI
        # via a reason map.
        self._reasons_by_ticker: Dict[str, str] = {}

    def with_reasons(self, reasons: Dict[str, str]) -> "_BriefAdapter":
        """Sets a ticker->reason map before `emit_order_plan`."""
        self._reasons_by_ticker = {
            str(k).upper(): str(v) for k, v in (reasons or {}).items()
        }
        return self

    @property
    def broker_label(self) -> str:
        return self.broker_label_value

    def _lookup_mapping(self, ticker: str) -> Optional[InstrumentMapping]:
        return self._mapping.get(str(ticker).upper())

    def _route_order_plan(self, order: Order) -> OrderPlanReceipt:
        mapping = self._lookup_mapping(order.ticker)
        if mapping is None or not mapping.isin:
            return OrderPlanReceipt(
                stable_order_plan_id=order.stable_order_plan_id,
                status="skipped",
                emitted_at_iso=now_iso(),
                broker_label=self.broker_label,
                run_id=order.run_id,
                signals_as_of_iso=order.signals_as_of_iso,
                plan_only=True,
                reason="missing_instrument_mapping",
                ticker=order.ticker,
                action=order.action,
                target_shares=order.target_shares,
                target_value=order.target_value,
            )
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
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(GERMAN_BRIEF_COLUMNS))
            for order, receipt in zip(orders, receipts):
                if receipt.status != "planned":
                    continue
                mapping = self._lookup_mapping(order.ticker)
                if mapping is None or not mapping.isin:
                    continue
                writer.writerow(
                    [
                        mapping.isin or "",
                        mapping.wkn or "",
                        # Stueckzahl (quantity) is always positive; Aktion
                        # (BUY/SELL) carries the direction. Trade Republic /
                        # Maxblue don't accept negative quantities.
                        abs(int(round(order.target_shares))),
                        order.action.upper(),
                        "",  # Limit leer
                        self._reasons_by_ticker.get(
                            order.ticker.upper(),
                            f"Empfehlung Strategie {order.strategy_hash[:8]}",
                        ),
                    ]
                )
        return receipts


class TradeRepublicBriefAdapter(_BriefAdapter):
    broker_label_value = "trade_republic_brief"
    mapping_broker_key = "trade_republic"
    output_dir = "results/live_orders/trade_republic"


__all__ = [
    "GERMAN_BRIEF_COLUMNS",
    "TradeRepublicBriefAdapter",
    "_BriefAdapter",
]
