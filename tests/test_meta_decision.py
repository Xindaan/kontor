from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from backtest.meta_decision import CandidateScore, run_meta_decision


def _candidate(
    strategy_path: str,
    *,
    live_score: float,
    performance_score: float | None = None,
    regime_bucket: str = "normal",
    regime_reasons: list[str] | None = None,
    vol_63: float = 0.20,
):
    return CandidateScore(
        strategy=str(Path(strategy_path).resolve()),
        params={},
        rebalance_frequency="monthly",
        live_score=live_score,
        performance_score=live_score if performance_score is None else performance_score,
        gate_score=0.1,
        trailing_returns={"63d": 0.1, "126d": 0.2, "252d": 0.3},
        annualized_vol=0.2,
        max_drawdown=-0.2,
        regime_bucket=regime_bucket,
        regime_reasons=regime_reasons or [regime_bucket],
        regime_metrics={"vol_63": vol_63},
        regime_percentiles={"vol_63": 0.5},
        regime_status="ok",
    )


def _fake_score_factory(score_map: dict[str, CandidateScore]):
    normalized = {str(Path(path).resolve()): value for path, value in score_map.items()}

    def _fake_score(
        strategy_path,
        params,
        as_of,
        skip_failed,
        scoring_mode,
        performance_windows_days,
        performance_weights,
        regime_profile,
        rebalance_override=None,
        **_kwargs,  # tolerate new analyst_* kwargs from Phase B
    ):
        _ = (
            params,
            as_of,
            skip_failed,
            scoring_mode,
            performance_windows_days,
            performance_weights,
            regime_profile,
            rebalance_override,
            _kwargs,
        )
        strategy_path = str(Path(strategy_path).resolve())
        return normalized[strategy_path]

    return _fake_score


def test_meta_decision_blocks_switch_when_evidence_missing(monkeypatch, tmp_path: Path):
    current = str((tmp_path / "current.py").resolve())
    target = str((tmp_path / "target.py").resolve())

    monkeypatch.setattr(
        "backtest.meta_decision._score_candidate",
        _fake_score_factory(
            {
                current: _candidate(current, live_score=0.40, performance_score=0.40, regime_bucket="normal"),
                target: _candidate(target, live_score=0.80, performance_score=0.80, regime_bucket="normal"),
            }
        ),
    )
    monkeypatch.setattr(
        "backtest.meta_decision.find_latest_evidence_artifact",
        lambda current_strategy, target_strategy: None,
    )

    result = run_meta_decision(
        as_of=date(2026, 2, 17),
        current_strategy=current,
        current_params={},
        candidates=[{"strategy": target, "params": {}}],
        decision_cadence="immediate",
        confirm_points=1,
        switch_margin=0.05,
        evidence_required=True,
        gate_fail_action="hold_current",
        plan_mode="recommendation_only",
        decision_log_path=tmp_path / "meta_decision.jsonl",
    )

    assert result["recommended_target"] == target
    assert result["evidence_status"] == "missing"
    assert result["switch_allowed"] is False
    assert result["executed_action"] == "hold_current"
    assert result["blocked_checks"] == ["Evidence Gate"]
    check_map = {item["key"]: item for item in result["switch_checks"]}
    assert check_map["candidate"]["status"] == "pass"
    assert check_map["score_margin"]["status"] == "pass"
    assert check_map["evidence"]["status"] == "fail"
    assert "Run or load a recent evidence artifact" in check_map["evidence"]["next_step"]


def test_meta_decision_allows_switch_with_passing_evidence(monkeypatch, tmp_path: Path):
    current = str((tmp_path / "current.py").resolve())
    target = str((tmp_path / "target.py").resolve())

    monkeypatch.setattr(
        "backtest.meta_decision._score_candidate",
        _fake_score_factory(
            {
                current: _candidate(current, live_score=0.40, performance_score=0.40, regime_bucket="normal"),
                target: _candidate(target, live_score=0.80, performance_score=0.80, regime_bucket="normal"),
            }
        ),
    )
    monkeypatch.setattr(
        "backtest.meta_decision.find_latest_evidence_artifact",
        lambda current_strategy, target_strategy: {
            "artifact_id": "artifact-1",
            "artifact_path": "/tmp/artifact-1.json",
            "created_at": "2026-02-10T00:00:00+00:00",
            "summary": {
                "oos_cagr_edge_pp": 2.0,
                "oos_hit_rate": 0.6,
                "oos_degradation_pct": 20.0,
                "oos_dd_delta_pp": 2.0,
                "num_windows": 8,
            },
            "gates": {"pass": True, "reasons": []},
        },
    )

    result = run_meta_decision(
        as_of=date(2026, 2, 17),
        current_strategy=current,
        current_params={},
        candidates=[{"strategy": target, "params": {}}],
        decision_cadence="immediate",
        confirm_points=1,
        switch_margin=0.05,
        evidence_required=True,
        gate_fail_action="hold_current",
        plan_mode="recommendation_only",
        decision_log_path=tmp_path / "meta_decision.jsonl",
    )

    assert result["recommended_target"] == target
    assert result["evidence_status"] == "pass"
    assert result["switch_allowed"] is True
    assert result["executed_action"] == "switch_to_target"
    assert result["evidence_artifact_id"] == "artifact-1"
    assert result["decision_rule"] == "alpha_driven"
    assert result["blocked_checks"] == []
    check_map = {item["key"]: item for item in result["switch_checks"]}
    assert check_map["candidate"]["status"] == "pass"
    assert check_map["score_margin"]["status"] == "pass"
    assert check_map["evidence"]["status"] == "pass"
    assert check_map["evidence"]["next_step"] is None


def test_meta_decision_blocks_worse_bucket_alpha_candidate(monkeypatch, tmp_path: Path):
    current = str((tmp_path / "current.py").resolve())
    target = str((tmp_path / "target.py").resolve())

    monkeypatch.setattr(
        "backtest.meta_decision._score_candidate",
        _fake_score_factory(
            {
                current: _candidate(current, live_score=0.40, performance_score=0.40, regime_bucket="normal"),
                target: _candidate(target, live_score=0.80, performance_score=0.80, regime_bucket="stressed"),
            }
        ),
    )

    result = run_meta_decision(
        as_of=date(2026, 2, 17),
        current_strategy=current,
        current_params={},
        candidates=[{"strategy": target, "params": {}}],
        decision_cadence="immediate",
        confirm_points=1,
        switch_margin=0.05,
        evidence_required=False,
        plan_mode="recommendation_only",
        decision_log_path=tmp_path / "meta_decision.jsonl",
    )

    assert result["recommended_target"] == current
    assert result["decision_rule"] == "hold"
    assert result["switch_allowed"] is False


def test_meta_decision_allows_fragility_driven_switch_with_conditioned_evidence(monkeypatch, tmp_path: Path):
    current = str((tmp_path / "current.py").resolve())
    safer = str((tmp_path / "safer.py").resolve())

    monkeypatch.setattr(
        "backtest.meta_decision._score_candidate",
        _fake_score_factory(
            {
                current: _candidate(current, live_score=0.60, performance_score=0.60, regime_bucket="stressed"),
                safer: _candidate(safer, live_score=0.57, performance_score=0.57, regime_bucket="normal", vol_63=0.12),
            }
        ),
    )
    monkeypatch.setattr(
        "backtest.meta_decision.find_latest_evidence_artifact",
        lambda current_strategy, target_strategy: {
            "artifact_id": "artifact-2",
            "artifact_path": "/tmp/artifact-2.json",
            "created_at": "2026-02-10T00:00:00+00:00",
            "summary": {
                "oos_cagr_edge_pp": 1.5,
                "oos_hit_rate": 0.6,
                "oos_degradation_pct": 20.0,
                "oos_dd_delta_pp": 2.0,
                "num_windows": 8,
            },
            "unconditional_summary": {
                "oos_cagr_edge_pp": 1.5,
                "oos_hit_rate": 0.6,
                "oos_degradation_pct": 20.0,
                "oos_dd_delta_pp": 2.0,
                "num_windows": 8,
            },
            "conditioned_summary": {
                "ausgewogen": {
                    "normal": {"num_windows": 0},
                    "fragile": {"num_windows": 0},
                    "stressed": {
                        "oos_cagr_edge_pp": 1.2,
                        "oos_hit_rate": 0.75,
                        "oos_degradation_pct": 10.0,
                        "oos_dd_delta_pp": 1.0,
                        "num_windows": 5,
                    },
                    "insufficient_history": {"num_windows": 0},
                }
            },
            "gates": {"pass": True, "reasons": []},
        },
    )

    result = run_meta_decision(
        as_of=date(2026, 2, 17),
        current_strategy=current,
        current_params={},
        candidates=[{"strategy": safer, "params": {}}],
        decision_cadence="immediate",
        confirm_points=2,
        switch_margin=0.10,
        evidence_required=True,
        plan_mode="recommendation_only",
        decision_log_path=tmp_path / "meta_decision.jsonl",
    )

    assert result["recommended_target"] == safer
    assert result["decision_rule"] == "fragility_driven"
    assert result["conditioned_evidence_status"] == "pass"
    assert result["switch_allowed"] is True
    assert result["performance_gap"] == pytest.approx(-0.03)


def test_meta_decision_blocks_fragility_switch_when_conditioned_windows_too_small(monkeypatch, tmp_path: Path):
    current = str((tmp_path / "current.py").resolve())
    safer = str((tmp_path / "safer.py").resolve())

    monkeypatch.setattr(
        "backtest.meta_decision._score_candidate",
        _fake_score_factory(
            {
                current: _candidate(current, live_score=0.60, performance_score=0.60, regime_bucket="fragile"),
                safer: _candidate(safer, live_score=0.58, performance_score=0.58, regime_bucket="normal"),
            }
        ),
    )
    monkeypatch.setattr(
        "backtest.meta_decision.find_latest_evidence_artifact",
        lambda current_strategy, target_strategy: {
            "artifact_id": "artifact-3",
            "artifact_path": "/tmp/artifact-3.json",
            "created_at": "2026-02-10T00:00:00+00:00",
            "summary": {
                "oos_cagr_edge_pp": 1.5,
                "oos_hit_rate": 0.6,
                "oos_degradation_pct": 20.0,
                "oos_dd_delta_pp": 2.0,
                "num_windows": 8,
            },
            "unconditional_summary": {
                "oos_cagr_edge_pp": 1.5,
                "oos_hit_rate": 0.6,
                "oos_degradation_pct": 20.0,
                "oos_dd_delta_pp": 2.0,
                "num_windows": 8,
            },
            "conditioned_summary": {
                "ausgewogen": {
                    "normal": {"num_windows": 0},
                    "fragile": {
                        "oos_cagr_edge_pp": 1.2,
                        "oos_hit_rate": 1.0,
                        "oos_degradation_pct": 5.0,
                        "oos_dd_delta_pp": 1.0,
                        "num_windows": 2,
                    },
                    "stressed": {"num_windows": 0},
                    "insufficient_history": {"num_windows": 0},
                }
            },
            "gates": {"pass": True, "reasons": []},
        },
    )

    result = run_meta_decision(
        as_of=date(2026, 2, 17),
        current_strategy=current,
        current_params={},
        candidates=[{"strategy": safer, "params": {}}],
        decision_cadence="immediate",
        confirm_points=1,
        switch_margin=0.10,
        evidence_required=True,
        conditioned_min_windows=4,
        plan_mode="recommendation_only",
        decision_log_path=tmp_path / "meta_decision.jsonl",
    )

    assert result["decision_rule"] == "fragility_driven"
    assert result["conditioned_evidence_status"] == "fail"
    assert result["switch_allowed"] is False
    assert result["executed_action"] == "hold_current"


def test_meta_decision_regime_mode_none_reproduces_old_alpha_behavior(monkeypatch, tmp_path: Path):
    current = str((tmp_path / "current.py").resolve())
    target = str((tmp_path / "target.py").resolve())

    monkeypatch.setattr(
        "backtest.meta_decision._score_candidate",
        _fake_score_factory(
            {
                current: _candidate(current, live_score=0.40, performance_score=0.40, regime_bucket="normal"),
                target: _candidate(target, live_score=0.80, performance_score=0.80, regime_bucket="stressed"),
            }
        ),
    )
    monkeypatch.setattr(
        "backtest.meta_decision.find_latest_evidence_artifact",
        lambda current_strategy, target_strategy: {
            "artifact_id": "artifact-4",
            "artifact_path": "/tmp/artifact-4.json",
            "created_at": "2026-02-10T00:00:00+00:00",
            "summary": {
                "oos_cagr_edge_pp": 2.0,
                "oos_hit_rate": 0.6,
                "oos_degradation_pct": 20.0,
                "oos_dd_delta_pp": 2.0,
                "num_windows": 8,
            },
            "gates": {"pass": True, "reasons": []},
        },
    )

    result = run_meta_decision(
        as_of=date(2026, 2, 17),
        current_strategy=current,
        current_params={},
        candidates=[{"strategy": target, "params": {}}],
        decision_cadence="immediate",
        confirm_points=1,
        switch_margin=0.05,
        evidence_required=True,
        regime_mode="none",
        plan_mode="recommendation_only",
        decision_log_path=tmp_path / "meta_decision.jsonl",
    )

    assert result["recommended_target"] == target
    assert result["decision_rule"] == "alpha_driven"
    assert result["switch_allowed"] is True


# ---------------------------------------------------------------------------
# Phase B: analyst-evidence pre-ranking check.
# ---------------------------------------------------------------------------


def test_analyst_evidence_check_not_run_when_weight_zero(monkeypatch, tmp_path: Path):
    """With analyst_score_weight=0 the analyst evidence gate is NA, never blocks."""
    current = str((tmp_path / "current.py").resolve())
    target = str((tmp_path / "target.py").resolve())

    monkeypatch.setattr(
        "backtest.meta_decision._score_candidate",
        _fake_score_factory(
            {
                current: _candidate(current, live_score=0.40, performance_score=0.40),
                target: _candidate(target, live_score=0.80, performance_score=0.80),
            }
        ),
    )
    monkeypatch.setattr(
        "backtest.meta_decision.find_latest_evidence_artifact",
        lambda current_strategy, target_strategy: {
            "artifact_id": "a1",
            "artifact_path": "/tmp/a1.json",
            "created_at": "2026-02-10T00:00:00+00:00",
            "summary": {
                "oos_cagr_edge_pp": 2.0,
                "oos_hit_rate": 0.6,
                "oos_degradation_pct": 20.0,
                "oos_dd_delta_pp": 2.0,
                "num_windows": 8,
            },
            "gates": {"pass": True, "reasons": []},
        },
    )
    # assess_analyst_evidence must NOT be called.
    calls: list[int] = []

    def _no_call(**kwargs):
        calls.append(1)
        return ("missing", [], None)

    monkeypatch.setattr("backtest.meta_decision.assess_analyst_evidence", _no_call)

    result = run_meta_decision(
        as_of=date(2026, 2, 17),
        current_strategy=current,
        current_params={},
        candidates=[{"strategy": target, "params": {}}],
        decision_cadence="immediate",
        confirm_points=1,
        switch_margin=0.05,
        evidence_required=True,
        gate_fail_action="hold_current",
        plan_mode="recommendation_only",
        decision_log_path=tmp_path / "meta_decision.jsonl",
        # No analyst provider, weight 0 -> NA.
        external_features_provider=None,
        analyst_score_weight=0.0,
    )
    assert calls == []  # function not called
    check_map = {item["key"]: item for item in result["switch_checks"]}
    assert check_map["analyst_evidence"]["status"] == "not_applicable"


def test_analyst_evidence_fail_zeroes_weight_but_does_not_abort(monkeypatch, tmp_path: Path):
    """When analyst evidence FAILs, effective_weight=0 and rest still runs."""
    current = str((tmp_path / "current.py").resolve())
    target = str((tmp_path / "target.py").resolve())

    monkeypatch.setattr(
        "backtest.meta_decision._score_candidate",
        _fake_score_factory(
            {
                current: _candidate(current, live_score=0.40),
                target: _candidate(target, live_score=0.80),
            }
        ),
    )
    monkeypatch.setattr(
        "backtest.meta_decision.find_latest_evidence_artifact",
        lambda current_strategy, target_strategy: {
            "artifact_id": "a1",
            "artifact_path": "/tmp/a1.json",
            "created_at": "2026-02-10T00:00:00+00:00",
            "summary": {
                "oos_cagr_edge_pp": 2.0,
                "oos_hit_rate": 0.6,
                "oos_degradation_pct": 20.0,
                "oos_dd_delta_pp": 2.0,
                "num_windows": 8,
            },
            "gates": {"pass": True, "reasons": []},
        },
    )
    monkeypatch.setattr(
        "backtest.meta_decision.assess_analyst_evidence",
        lambda **kwargs: ("fail", ["edge too small"], {"profile": "ausgewogen"}),
    )

    class _DummyProvider:
        def snapshot(self, as_of, tickers=None):
            return None

    result = run_meta_decision(
        as_of=date(2026, 2, 17),
        current_strategy=current,
        current_params={},
        candidates=[{"strategy": target, "params": {}}],
        decision_cadence="immediate",
        confirm_points=1,
        switch_margin=0.05,
        evidence_required=True,
        gate_fail_action="hold_current",
        plan_mode="recommendation_only",
        decision_log_path=tmp_path / "meta_decision.jsonl",
        external_features_provider=_DummyProvider(),
        analyst_score_weight=0.3,
    )
    # Switch still allowed via classic evidence path; analyst gate flagged as FAIL.
    check_map = {item["key"]: item for item in result["switch_checks"]}
    assert check_map["analyst_evidence"]["status"] == "fail"
    # blocked_checks now contains "Analyst Evidence" — switch may continue
    # because the analyst effective weight collapses to 0.
    assert "Analyst Evidence" in result["blocked_checks"]


def test_analyst_evidence_pass_sets_effective_weight(monkeypatch, tmp_path: Path):
    current = str((tmp_path / "current.py").resolve())
    target = str((tmp_path / "target.py").resolve())

    captured: dict[str, float] = {}

    def _capturing_score(*args, **kwargs):
        captured["effective_weight"] = kwargs.get("analyst_score_effective_weight", -1)
        captured["status"] = kwargs.get("analyst_evidence_status", "?")
        strategy_path = kwargs.get("strategy_path") or (args[0] if args else current)
        return _candidate(strategy_path, live_score=0.5)

    monkeypatch.setattr("backtest.meta_decision._score_candidate", _capturing_score)
    monkeypatch.setattr(
        "backtest.meta_decision.find_latest_evidence_artifact",
        lambda current_strategy, target_strategy: None,
    )
    monkeypatch.setattr(
        "backtest.meta_decision.assess_analyst_evidence",
        lambda **kwargs: ("pass", [], {"profile": "ausgewogen"}),
    )

    class _DummyProvider:
        def snapshot(self, as_of, tickers=None):
            return None

    run_meta_decision(
        as_of=date(2026, 2, 17),
        current_strategy=current,
        current_params={},
        candidates=[{"strategy": target, "params": {}}],
        decision_cadence="immediate",
        confirm_points=1,
        switch_margin=0.05,
        evidence_required=False,
        plan_mode="recommendation_only",
        decision_log_path=tmp_path / "meta_decision.jsonl",
        external_features_provider=_DummyProvider(),
        analyst_score_weight=0.25,
        analyst_datasets=("ds_test",),
    )
    assert captured["effective_weight"] == pytest.approx(0.25)
    assert captured["status"] == "pass"


def test_analyst_conditioned_status_reported_when_summary_available(monkeypatch, tmp_path: Path):
    """T-0099: With analyst_score_weight > 0 and a summary that contains a
    conditioned bucket entry, every CandidateScore receives the
    conditioned status and a switch_check is appended."""

    current = str((tmp_path / "current.py").resolve())
    target = str((tmp_path / "target.py").resolve())

    monkeypatch.setattr(
        "backtest.meta_decision._score_candidate",
        _fake_score_factory(
            {
                current: _candidate(current, live_score=0.45, regime_bucket="normal"),
                target: _candidate(target, live_score=0.65, regime_bucket="normal"),
            }
        ),
    )
    monkeypatch.setattr(
        "backtest.meta_decision.find_latest_evidence_artifact",
        lambda current_strategy, target_strategy: None,
    )
    summary = {
        "profile": "aggressiv",
        "thresholds": {"min_windows": 6, "min_cagr_edge_pp": 0.25, "min_hit_rate": 0.50},
        "conditioned": {
            "normal": {
                "num_windows": 10,
                "hit_rate": 0.70,
                "mean_edge": 0.012,
                "cagr_edge_pp": 14.4,
            }
        },
    }
    monkeypatch.setattr(
        "backtest.meta_decision.assess_analyst_evidence",
        lambda **kwargs: ("pass", [], summary),
    )

    class _DummyProvider:
        def snapshot(self, as_of, tickers=None):
            return None

    result = run_meta_decision(
        as_of=date(2026, 2, 17),
        current_strategy=current,
        current_params={},
        candidates=[{"strategy": target, "params": {}}],
        decision_cadence="immediate",
        confirm_points=1,
        switch_margin=0.05,
        evidence_required=False,
        plan_mode="recommendation_only",
        decision_log_path=tmp_path / "meta_decision.jsonl",
        external_features_provider=_DummyProvider(),
        analyst_score_weight=0.3,
        evidence_profile="aggressiv",
    )

    # Each candidate row carries the conditioned status.
    for row in result["candidates"]:
        assert row["analyst_evidence_conditioned_status"] == "pass"

    # switch_checks has a new entry for the conditioned gate.
    check_map = {item["key"]: item for item in result["switch_checks"]}
    assert "analyst_conditioned_evidence" in check_map
    assert check_map["analyst_conditioned_evidence"]["status"] == "pass"
    assert "windows=10" in check_map["analyst_conditioned_evidence"]["detail"]


def test_analyst_conditioned_require_true_downgrades_weight_on_fail(
    monkeypatch, tmp_path: Path
):
    """T-0099 hard-gate path: analyst_require_conditioned_evidence=True
    forces effective_weight back to 0 when the conditioned bucket fails,
    and reports status=fail in switch_checks."""

    current = str((tmp_path / "current.py").resolve())
    target = str((tmp_path / "target.py").resolve())

    captured_weights: list[float] = []

    def _capturing_score(*args, **kwargs):
        captured_weights.append(float(kwargs.get("analyst_score_effective_weight", 0.0)))
        strategy_path = kwargs.get("strategy_path") or (args[0] if args else current)
        # We have to return distinct objects so the dict-style _fake_score
        # cannot help here.
        regime = "fragile" if Path(strategy_path).resolve() == Path(current).resolve() else "normal"
        return _candidate(strategy_path, live_score=0.5, regime_bucket=regime)

    monkeypatch.setattr("backtest.meta_decision._score_candidate", _capturing_score)
    monkeypatch.setattr(
        "backtest.meta_decision.find_latest_evidence_artifact",
        lambda current_strategy, target_strategy: None,
    )
    # Unconditional PASS, but conditioned bucket "fragile" missing -> fail.
    summary = {
        "profile": "ausgewogen",
        "thresholds": {"min_windows": 8, "min_cagr_edge_pp": 0.75, "min_hit_rate": 0.54},
        "conditioned": {
            "normal": {
                "num_windows": 12,
                "hit_rate": 0.66,
                "mean_edge": 0.01,
                "cagr_edge_pp": 12.0,
            }
        },
    }
    monkeypatch.setattr(
        "backtest.meta_decision.assess_analyst_evidence",
        lambda **kwargs: ("pass", [], summary),
    )

    class _DummyProvider:
        def snapshot(self, as_of, tickers=None):
            return None

    result = run_meta_decision(
        as_of=date(2026, 2, 17),
        current_strategy=current,
        current_params={},
        candidates=[{"strategy": target, "params": {}}],
        decision_cadence="immediate",
        confirm_points=1,
        switch_margin=0.05,
        evidence_required=False,
        plan_mode="recommendation_only",
        decision_log_path=tmp_path / "meta_decision.jsonl",
        external_features_provider=_DummyProvider(),
        analyst_score_weight=0.3,
        analyst_require_conditioned_evidence=True,
    )

    # We expect two scoring passes: first with weight 0.3, then a downgrade
    # to 0.0 once the conditioned gate is observed to fail.
    assert any(w == pytest.approx(0.3) for w in captured_weights)
    assert captured_weights[-1] == pytest.approx(0.0)

    check_map = {item["key"]: item for item in result["switch_checks"]}
    assert check_map["analyst_conditioned_evidence"]["status"] == "fail"
    assert "Need conditioned analyst evidence" in (
        check_map["analyst_conditioned_evidence"]["next_step"] or ""
    )
