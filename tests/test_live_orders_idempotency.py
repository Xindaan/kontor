"""Phase E2 — order plan idempotency (T-0362, Codex R2.3+R3.4)."""

from __future__ import annotations

import pytest

from backtest.live.orders import compute_run_id, stable_order_plan_id


def test_compute_run_id_deterministic():
    a = compute_run_id(
        signal_report_hash="abc", broker_label="dry_run",
        portfolio_snapshot_hash="port",
    )
    b = compute_run_id(
        signal_report_hash="abc", broker_label="dry_run",
        portfolio_snapshot_hash="port",
    )
    assert a == b


def test_compute_run_id_changes_with_new_run_token():
    a = compute_run_id(
        signal_report_hash="abc", broker_label="dry_run",
        portfolio_snapshot_hash="port",
    )
    b = compute_run_id(
        signal_report_hash="abc", broker_label="dry_run",
        portfolio_snapshot_hash="port", new_run_token="X",
    )
    assert a != b


def test_compute_run_id_missing_portfolio_hash_raises():
    """Codex R3.4: no silent empty hash."""
    with pytest.raises(RuntimeError, match="portfolio_snapshot_hash"):
        compute_run_id(
            signal_report_hash="abc", broker_label="dry_run",
            portfolio_snapshot_hash=None,
        )
    with pytest.raises(RuntimeError):
        compute_run_id(
            signal_report_hash="abc", broker_label="dry_run",
            portfolio_snapshot_hash="",
        )


def test_stable_order_plan_id_does_not_depend_on_created_at():
    """Codex R2.3: no `created_at` in the hash."""
    a = stable_order_plan_id(
        run_id="R", strategy_hash="S", portfolio_snapshot_hash="P",
        broker_label="dry_run", ticker="AAPL", action="BUY",
        target_shares=10.123456,
    )
    b = stable_order_plan_id(
        run_id="R", strategy_hash="S", portfolio_snapshot_hash="P",
        broker_label="dry_run", ticker="AAPL", action="BUY",
        target_shares=10.123456,
    )
    assert a == b


def test_stable_order_plan_id_normalizes_ticker_action():
    """Lowercase ticker and action should produce the same hash as
    uppercase."""
    a = stable_order_plan_id(
        run_id="R", strategy_hash="S", portfolio_snapshot_hash="P",
        broker_label="dry_run", ticker="aapl", action="buy",
        target_shares=10.0,
    )
    b = stable_order_plan_id(
        run_id="R", strategy_hash="S", portfolio_snapshot_hash="P",
        broker_label="dry_run", ticker="AAPL", action="BUY",
        target_shares=10.0,
    )
    assert a == b


def test_stable_order_plan_id_round_target_shares():
    """target_shares is rounded to 6 decimal places (6th decimal digit stable)."""
    # Both values round to 10.123457 (6th digit 6.789 -> 7).
    a = stable_order_plan_id(
        run_id="R", strategy_hash="S", portfolio_snapshot_hash="P",
        broker_label="dry_run", ticker="AAPL", action="BUY",
        target_shares=10.1234566789,
    )
    b = stable_order_plan_id(
        run_id="R", strategy_hash="S", portfolio_snapshot_hash="P",
        broker_label="dry_run", ticker="AAPL", action="BUY",
        target_shares=10.1234566001,
    )
    assert a == b
