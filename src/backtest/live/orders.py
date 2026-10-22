"""Phase E2 — Order/OrderPlanReceipt dataclasses (T-0362, T-0363).

Order idempotency via a deterministic ``run_id`` (Codex R2.3, R3.4)
and ``stable_order_plan_id`` without ``created_at`` in the hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def stable_order_plan_id(
    *,
    run_id: str,
    strategy_hash: str,
    portfolio_snapshot_hash: str,
    broker_label: str,
    ticker: str,
    action: str,
    target_shares: float,
) -> str:
    """Deterministic order plan identifier (Codex R2.3 + R3.4).

    **Contains NO created_at** — idempotency depends solely on the
    ``run_id``, which is itself deterministically derived from
    SignalReport + broker + portfolio snapshot.
    """

    payload = "|".join(
        [
            str(run_id),
            str(strategy_hash),
            str(portfolio_snapshot_hash),
            str(broker_label),
            str(ticker).upper(),
            str(action).upper(),
            f"{float(target_shares):.6f}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_run_id(
    *,
    signal_report_hash: str,
    broker_label: str,
    portfolio_snapshot_hash: Optional[str],
    new_run_token: str = "",
) -> str:
    """Deterministic ``run_id`` from SignalReport + broker + portfolio.

    Codex R3.4: ``portfolio_snapshot_hash`` is mandatory. If it's
    missing (`None` or empty string), raises `RuntimeError`. Optional
    `new_run_token` mixes in a user-intentional new token.
    """

    if not portfolio_snapshot_hash:
        raise RuntimeError(
            "portfolio_snapshot_hash is required for live plan idempotency. "
            "Provide --portfolio PATH or store portfolio_snapshot_hash in "
            "the SignalReport JSON."
        )
    payload = "|".join(
        [
            str(signal_report_hash),
            str(broker_label),
            str(portfolio_snapshot_hash),
            str(new_run_token or ""),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Order:
    """Single order plan entry (Codex R2.1 — no submit vocabulary)."""

    ticker: str
    action: str  # "BUY" | "SELL"
    target_shares: float
    target_value: float
    broker_label: str
    stable_order_plan_id: str
    run_id: str
    strategy_hash: str
    portfolio_snapshot_hash: str
    signals_as_of_iso: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": str(self.ticker).upper(),
            "action": str(self.action).upper(),
            "target_shares": float(self.target_shares),
            "target_value": float(self.target_value),
            "broker_label": str(self.broker_label),
            "stable_order_plan_id": str(self.stable_order_plan_id),
            "run_id": str(self.run_id),
            "strategy_hash": str(self.strategy_hash),
            "portfolio_snapshot_hash": str(self.portfolio_snapshot_hash),
            "signals_as_of_iso": str(self.signals_as_of_iso),
        }


@dataclass(frozen=True)
class OrderPlanReceipt:
    """Audit receipt for an `Order`. Codex R3.2: `plan_only` field."""

    stable_order_plan_id: str
    status: str  # "planned" | "skipped" | "rejected"
    emitted_at_iso: str
    broker_label: str
    run_id: str
    signals_as_of_iso: str
    plan_only: bool = True
    reason: Optional[str] = None
    broker_specific_ticket: Optional[str] = None
    # Audit fields for the log allowlist (Codex R2.15).
    ticker: Optional[str] = None
    action: Optional[str] = None
    target_shares: Optional[float] = None
    target_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "stable_order_plan_id": str(self.stable_order_plan_id),
            "status": str(self.status),
            "emitted_at_iso": str(self.emitted_at_iso),
            "broker_label": str(self.broker_label),
            "run_id": str(self.run_id),
            "signals_as_of_iso": str(self.signals_as_of_iso),
            "plan_only": bool(self.plan_only),
        }
        if self.reason is not None:
            payload["reason"] = str(self.reason)
        if self.broker_specific_ticket is not None:
            payload["broker_specific_ticket"] = str(self.broker_specific_ticket)
        if self.ticker is not None:
            payload["ticker"] = str(self.ticker).upper()
        if self.action is not None:
            payload["action"] = str(self.action).upper()
        if self.target_shares is not None:
            payload["target_shares"] = float(self.target_shares)
        if self.target_value is not None:
            payload["target_value"] = float(self.target_value)
        return payload


__all__ = [
    "Order",
    "OrderPlanReceipt",
    "compute_run_id",
    "stable_order_plan_id",
]
