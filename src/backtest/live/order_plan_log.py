"""Phase E2 — OrderPlanLog (T-0370, Codex R3.5+R3.6+R2.15).

Append-only NDJSON audit log with:
- Append only on a new ``stable_order_plan_id`` (R3.6).
- Secret allowlist when writing (R2.15) — all keys outside the
  allowlist are filtered out.
- ``find_by_stable_order_plan_id()`` as a lookup.

Deliberately NO reuse of `ManualDataProvenanceRegistry` (R3.5).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backtest.live.orders import OrderPlanReceipt


# Codex R2.15: allowlist for NDJSON writes. No API keys, no
# account IDs.
ALLOWED_RECEIPT_KEYS: tuple[str, ...] = (
    "stable_order_plan_id",
    "status",
    "emitted_at_iso",
    "broker_label",
    "run_id",
    "signals_as_of_iso",
    "plan_only",
    "reason",
    "broker_specific_ticket",
    "ticker",
    "action",
    "target_shares",
    "target_value",
)


def _filter_to_allowlist(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Removes all keys outside the allowlist (Codex R2.15)."""

    return {k: v for k, v in payload.items() if k in ALLOWED_RECEIPT_KEYS}


def _serialise_receipt(receipt: OrderPlanReceipt) -> str:
    payload = _filter_to_allowlist(receipt.to_dict())
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class OrderPlanLog:
    """Append-only NDJSON audit log for OrderPlanReceipts.

    Lives under ``results/live_orders/order_plan_log.jsonl`` by
    default. ``append_if_new()`` is idempotent.
    """

    def __init__(self, path: str | Path = "results/live_orders/order_plan_log.jsonl") -> None:
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def _existing_ids(self) -> set[str]:
        ids: set[str] = set()
        if not self._path.exists():
            return ids
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            spi = payload.get("stable_order_plan_id")
            if spi:
                ids.add(str(spi))
        return ids

    def find_by_stable_order_plan_id(self, spi: str) -> Optional[Dict[str, Any]]:
        if not self._path.exists():
            return None
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("stable_order_plan_id")) == str(spi):
                return payload
        return None

    def append_if_new(
        self,
        receipt: OrderPlanReceipt,
    ) -> Tuple[OrderPlanReceipt, bool]:
        """Append only on a new `stable_order_plan_id` (Codex R3.6).

        Returns ``(receipt, appended)``. On a duplicate ID the
        existing entry is NOT overwritten; ``appended=False``.
        """

        existing = self.find_by_stable_order_plan_id(receipt.stable_order_plan_id)
        if existing is not None:
            return receipt, False
        line = _serialise_receipt(receipt) + "\n"
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        return receipt, True

    def read_all(self) -> List[Dict[str, Any]]:
        if not self._path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows


def now_iso() -> str:
    """UTC ISO timestamp without microseconds — for ``emitted_at_iso``."""

    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "ALLOWED_RECEIPT_KEYS",
    "OrderPlanLog",
    "now_iso",
]
