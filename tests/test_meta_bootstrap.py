from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from backtest.meta_bootstrap import run_meta_bootstrap_decision


def test_bootstrap_uses_unilateral_evidence_pass(monkeypatch, tmp_path: Path):
    strategy_a = str((tmp_path / "a.py").resolve())
    strategy_b = str((tmp_path / "b.py").resolve())

    def _fake_meta_evidence(**kwargs):
        if kwargs["current_strategy"] == strategy_a and kwargs["target_strategy"] == strategy_b:
            return {
                "artifact_id": "ab",
                "artifact_path": "/tmp/ab.json",
                "summary": {
                    "num_windows": 8,
                    "oos_cagr_edge_pp": 2.0,
                    "oos_hit_rate": 0.60,
                    "oos_degradation_pct": 20.0,
                    "oos_dd_delta_pp": 3.0,
                },
                "gates": {"pass": True, "reasons": []},
            }
        return {
            "artifact_id": "ba",
            "artifact_path": "/tmp/ba.json",
            "summary": {
                "num_windows": 8,
                "oos_cagr_edge_pp": -2.0,
                "oos_hit_rate": 0.40,
                "oos_degradation_pct": 20.0,
                "oos_dd_delta_pp": -3.0,
            },
            "gates": {"pass": False, "reasons": ["hit-rate low"]},
        }

    monkeypatch.setattr("backtest.meta_bootstrap.run_meta_evidence_analysis", _fake_meta_evidence)
    monkeypatch.setattr(
        "backtest.meta_bootstrap._run_full_period_metrics",
        lambda *args, **kwargs: {"cagr": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0},
    )

    artifact = run_meta_bootstrap_decision(
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        as_of=date(2026, 2, 17),
        save_artifact=False,
        log_path=tmp_path / "bootstrap.jsonl",
    )

    assert artifact["decision"]["recommended_start_strategy"] == strategy_b
    assert artifact["decision"]["decision_rule"] == "evidence_unilateral"
    assert artifact["fallback"] is None


def test_bootstrap_fallback_uses_cagr_when_no_pass(monkeypatch, tmp_path: Path):
    strategy_a = str((tmp_path / "a.py").resolve())
    strategy_b = str((tmp_path / "b.py").resolve())

    monkeypatch.setattr(
        "backtest.meta_bootstrap.run_meta_evidence_analysis",
        lambda **kwargs: {
            "artifact_id": "x",
            "artifact_path": "/tmp/x.json",
            "summary": {
                "num_windows": 8,
                "oos_cagr_edge_pp": 0.0,
                "oos_hit_rate": 0.5,
                "oos_degradation_pct": 20.0,
                "oos_dd_delta_pp": 0.0,
            },
            "gates": {"pass": False, "reasons": ["edge low"]},
        },
    )

    def _fake_metrics(strategy_path, params, **kwargs):
        _ = (params, kwargs)
        if str(Path(strategy_path).resolve()) == strategy_b:
            return {"cagr": 0.20, "max_drawdown": -0.55, "sharpe_ratio": 0.9}
        return {"cagr": 0.15, "max_drawdown": -0.45, "sharpe_ratio": 1.0}

    monkeypatch.setattr("backtest.meta_bootstrap._run_full_period_metrics", _fake_metrics)

    artifact = run_meta_bootstrap_decision(
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        as_of=date(2026, 2, 17),
        fallback_cagr_tie_band_pp=1.0,
        save_artifact=False,
        log_path=tmp_path / "bootstrap.jsonl",
    )

    assert artifact["decision"]["recommended_start_strategy"] == strategy_b
    assert artifact["decision"]["decision_rule"] == "fallback_cagr"
    assert artifact["fallback"]["cagr_edge_b_minus_a_pp"] == pytest.approx(5.0)


def test_bootstrap_fallback_tie_breaker_maxdd(monkeypatch, tmp_path: Path):
    strategy_a = str((tmp_path / "a.py").resolve())
    strategy_b = str((tmp_path / "b.py").resolve())

    monkeypatch.setattr(
        "backtest.meta_bootstrap.run_meta_evidence_analysis",
        lambda **kwargs: {
            "artifact_id": "x",
            "artifact_path": "/tmp/x.json",
            "summary": {
                "num_windows": 8,
                "oos_cagr_edge_pp": 0.0,
                "oos_hit_rate": 0.5,
                "oos_degradation_pct": 20.0,
                "oos_dd_delta_pp": 0.0,
            },
            "gates": {"pass": False, "reasons": ["edge low"]},
        },
    )

    def _fake_metrics(strategy_path, params, **kwargs):
        _ = (params, kwargs)
        if str(Path(strategy_path).resolve()) == strategy_b:
            return {"cagr": 0.151, "max_drawdown": -0.60, "sharpe_ratio": 1.1}
        return {"cagr": 0.150, "max_drawdown": -0.40, "sharpe_ratio": 0.9}

    monkeypatch.setattr("backtest.meta_bootstrap._run_full_period_metrics", _fake_metrics)

    artifact = run_meta_bootstrap_decision(
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        as_of=date(2026, 2, 17),
        fallback_cagr_tie_band_pp=1.0,
        fallback_tie_breaker="maxdd",
        save_artifact=False,
        log_path=tmp_path / "bootstrap.jsonl",
    )

    assert artifact["decision"]["recommended_start_strategy"] == strategy_a
    assert artifact["decision"]["decision_rule"] == "fallback_tie_breaker"
