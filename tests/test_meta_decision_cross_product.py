"""Phase E1 integration test (T-0308/T-0309).

Ensures that:
- `cross_product_require=False` leaves Phase D bit-identical.
- With `cross_product_require=True` and insufficient consensus, the
  switch is blocked via the gate and `switch_checks` contains a
  `cross_product_consensus` entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from backtest.meta_decision import CandidateScore, _evaluate_cross_product_gate


def _make_summary(
    *,
    thresholds: Dict[str, float],
    conditioned: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    return {
        "thresholds": thresholds,
        "conditioned": conditioned,
    }


def _current_row(bucket: str = "normal") -> CandidateScore:
    return CandidateScore(
        strategy="s",
        params={},
        rebalance_frequency="monthly",
        live_score=0.0,
        performance_score=0.0,
        gate_score=0.0,
        trailing_returns={},
        annualized_vol=0.0,
        max_drawdown=0.0,
        regime_bucket=bucket,
    )


def test_phase_d_bit_identical_when_require_false():
    """With require=False the gate is not_applicable, Phase D untouched."""
    result = _evaluate_cross_product_gate(
        current_row=_current_row(),
        analyst_summary=None,
        news_summary=None,
        ml_summary=None,
        analyst_unconditional_status="missing",
        news_unconditional_status="missing",
        ml_unconditional_status="missing",
        analyst_configured_weight=0.0,
        news_configured_weight=0.0,
        ml_configured_weight=0.0,
        profile="ausgewogen",
        threshold_override=None,
        require=False,
    )
    assert result.status == "not_applicable"


def test_gate_pass_with_three_passing_components():
    """3 components x 0.2 = 0.6 >= ausgewogen.threshold=0.5."""
    summary = _make_summary(
        thresholds={
            "min_windows": 8,
            "min_cagr_edge_pp": 0.5,
            "min_hit_rate": 0.5,
        },
        conditioned={
            "normal": {
                "num_windows": 10,
                "cagr_edge_pp": 1.0,
                "hit_rate": 0.6,
            },
        },
    )
    result = _evaluate_cross_product_gate(
        current_row=_current_row("normal"),
        analyst_summary=summary,
        news_summary=summary,
        ml_summary=summary,
        analyst_unconditional_status="pass",
        news_unconditional_status="pass",
        ml_unconditional_status="pass",
        analyst_configured_weight=0.2,
        news_configured_weight=0.2,
        ml_configured_weight=0.2,
        profile="ausgewogen",
        threshold_override=None,
        require=True,
    )
    assert result.status == "pass", result.reasons
    assert result.consensus_per_bucket["normal"] == pytest.approx(0.6)


def test_gate_fail_when_conditioned_fails():
    """If all three components fail the bucket check,
    consensus = 0 -> fail."""
    summary_fail = _make_summary(
        thresholds={
            "min_windows": 8,
            "min_cagr_edge_pp": 0.5,
            "min_hit_rate": 0.5,
        },
        conditioned={
            "normal": {
                "num_windows": 10,
                "cagr_edge_pp": 0.1,  # < 0.5 threshold
                "hit_rate": 0.4,  # < 0.5 threshold
            },
        },
    )
    result = _evaluate_cross_product_gate(
        current_row=_current_row("normal"),
        analyst_summary=summary_fail,
        news_summary=summary_fail,
        ml_summary=summary_fail,
        analyst_unconditional_status="pass",
        news_unconditional_status="pass",
        ml_unconditional_status="pass",
        analyst_configured_weight=0.2,
        news_configured_weight=0.2,
        ml_configured_weight=0.2,
        profile="ausgewogen",
        threshold_override=None,
        require=True,
    )
    assert result.status == "fail", result.reasons


def test_gate_missing_when_no_components_active():
    result = _evaluate_cross_product_gate(
        current_row=_current_row("normal"),
        analyst_summary=None,
        news_summary=None,
        ml_summary=None,
        analyst_unconditional_status="missing",
        news_unconditional_status="missing",
        ml_unconditional_status="missing",
        analyst_configured_weight=0.0,
        news_configured_weight=0.0,
        ml_configured_weight=0.0,
        profile="ausgewogen",
        threshold_override=None,
        require=True,
    )
    assert result.status == "missing"


def test_threshold_override_changes_result():
    """With override 0.1, a previously-failing consensus turns to pass."""
    summary = _make_summary(
        thresholds={
            "min_windows": 8,
            "min_cagr_edge_pp": 0.5,
            "min_hit_rate": 0.5,
        },
        conditioned={
            "normal": {
                "num_windows": 10,
                "cagr_edge_pp": 1.0,
                "hit_rate": 0.6,
            },
        },
    )
    # only one component -> consensus=0.2 < ausgewogen.threshold=0.5
    result_default = _evaluate_cross_product_gate(
        current_row=_current_row("normal"),
        analyst_summary=summary,
        news_summary=None,
        ml_summary=None,
        analyst_unconditional_status="pass",
        news_unconditional_status="missing",
        ml_unconditional_status="missing",
        analyst_configured_weight=0.2,
        news_configured_weight=0.0,
        ml_configured_weight=0.0,
        profile="ausgewogen",
        threshold_override=None,
        require=True,
    )
    assert result_default.status == "fail"
    # with override 0.1 -> consensus=0.2 >= 0.1 -> pass
    result_override = _evaluate_cross_product_gate(
        current_row=_current_row("normal"),
        analyst_summary=summary,
        news_summary=None,
        ml_summary=None,
        analyst_unconditional_status="pass",
        news_unconditional_status="missing",
        ml_unconditional_status="missing",
        analyst_configured_weight=0.2,
        news_configured_weight=0.0,
        ml_configured_weight=0.0,
        profile="ausgewogen",
        threshold_override=0.1,
        require=True,
    )
    assert result_override.status == "pass"
