import json

from backtest.cli import (
    create_parser,
    _resolve_execution_lag,
    _apply_preset_profile_defaults,
    _resolve_exposure_policy,
    _write_portfolio_updates,
)


def test_run_parser_accepts_execution_and_liquidity_options():
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "strategies/dummy.py",
            "--execution-lag-days",
            "2",
            "--max-volume-participation",
            "0.1",
            "--min-daily-dollar-volume",
            "500000",
            "--liquidity-on-missing-volume",
            "skip",
        ]
    )

    assert args.execution_lag_days == 2
    assert args.max_volume_participation == 0.1
    assert args.min_daily_dollar_volume == 500000.0
    assert args.liquidity_on_missing_volume == "skip"


def test_run_parser_rebalance_frequency_defaults_to_strategy_file():
    parser = create_parser()

    args = parser.parse_args(["run", "strategies/dummy.py"])

    assert args.rebalance_frequency is None


def test_compare_parser_rebalance_frequency_defaults_to_strategy_files():
    parser = create_parser()

    args = parser.parse_args(
        [
            "compare",
            "strategies/a.py",
            "strategies/b.py",
        ]
    )

    assert args.rebalance_frequency is None


def test_meta_promotion_parser_defaults_to_governance_artifact_settings():
    parser = create_parser()

    args = parser.parse_args(["meta-promotion"])

    assert args.strategies == []
    assert args.baseline == "strategies/levered_etf_momentum_sticky.py"
    assert args.metric_basis == "net_liquidation"
    assert args.tail_risk_gate_basis == "daily"
    assert args.brokers is None
    assert args.output_dir == "results/meta_promotion"
    assert args.soxl_proxy is False


def test_meta_promotion_parser_accepts_soxl_proxy_mode():
    parser = create_parser()

    args = parser.parse_args(["meta-promotion", "--soxl-proxy"])

    assert args.soxl_proxy is True


def test_meta_promotion_parser_accepts_rebalance_tail_gate_reference_mode():
    parser = create_parser()

    args = parser.parse_args(["meta-promotion", "--tail-risk-gate-basis", "rebalance"])

    assert args.tail_risk_gate_basis == "rebalance"


def test_run_parser_accepts_explicit_rebalance_frequency_override():
    parser = create_parser()

    args = parser.parse_args(
        [
            "run",
            "strategies/dummy.py",
            "--rebalance-frequency",
            "weekly",
        ]
    )

    assert args.rebalance_frequency == "weekly"


def test_t_plus_one_shortcut_overrides_zero_lag():
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "strategies/dummy.py",
            "--execution-lag-days",
            "0",
            "--t-plus-one",
        ]
    )

    assert _resolve_execution_lag(args) == 1


def test_preset_profile_applies_execution_and_risk_defaults():
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "strategies/dummy.py",
            "--preset-profile",
            "realistic",
        ]
    )
    _apply_preset_profile_defaults(args)

    assert args.execution_lag_days == 1
    assert args.max_volume_participation == 0.10
    assert args.min_daily_dollar_volume == 250000.0
    assert args.liquidity_on_missing_volume == "skip"
    assert args.max_position == 0.35
    assert args.turnover_budget == 0.35


def test_preset_profile_keeps_explicit_non_default_overrides():
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "strategies/dummy.py",
            "--preset-profile",
            "defensive",
            "--execution-lag-days",
            "2",
            "--max-volume-participation",
            "0.2",
            "--max-position",
            "0.5",
        ]
    )
    _apply_preset_profile_defaults(args)

    assert args.execution_lag_days == 2
    assert args.max_volume_participation == 0.2
    assert args.max_position == 0.5
    # Still profile-driven because value remained default.
    assert args.turnover_budget == 0.20


def test_signals_parser_accepts_drift_tolerance():
    parser = create_parser()
    args = parser.parse_args(
        [
            "signals",
            "strategies/dummy.py",
            "--drift-tolerance",
            "0.02",
        ]
    )

    assert args.drift_tolerance == 0.02


def test_live_update_portfolio_parser_accepts_share_updates():
    parser = create_parser()
    args = parser.parse_args(
        [
            "live",
            "update-portfolio",
            "--portfolio",
            "data/manual/portfolio_tr.json",
            "--position",
            "3SEM.L=91.25",
            "--stand",
            "2026-06-08",
        ]
    )

    assert args.portfolio == "data/manual/portfolio_tr.json"
    assert args.position == ["3SEM.L=91.25"]
    assert args.stand == "2026-06-08"


def test_live_plan_parser_blocks_price_warnings_by_default_with_explicit_override():
    parser = create_parser()
    args = parser.parse_args(
        [
            "live",
            "plan",
            "--signals-report",
            "results/weekly_signals/2026-06-08.json",
            "--broker",
            "trade_republic_brief",
            "--allow-price-warnings",
        ]
    )

    assert args.allow_price_warnings is True


def test_write_portfolio_updates_preserves_manual_positionen_format(tmp_path):
    portfolio_path = tmp_path / "portfolio_tr.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "broker": "Trade Republic",
                "stand": "2026-06-06",
                "positionen": [
                    {
                        "name": "Semi 3x",
                        "price_ticker": "3SEM.L",
                        "waehrung": "EUR",
                        "shares": 25.0,
                        "rolle": "risk_3x_semi",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _write_portfolio_updates(
        portfolio_path,
        {"3SEM.L": 91.25},
        stand="2026-06-08",
    )

    payload = json.loads(portfolio_path.read_text(encoding="utf-8"))
    assert payload["stand"] == "2026-06-08"
    assert payload["positionen"][0]["price_ticker"] == "3SEM.L"
    assert payload["positionen"][0]["shares"] == 91.25
    assert "positions" not in payload


def test_run_parser_accepts_exposure_policy_options():
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "strategies/dummy.py",
            "--exposure-policy-enable",
            "--exposure-policy-profile",
            "trade_republic",
            "--exposure-policy-core-asset",
            "A0YEDG",
            "--exposure-level1-ret-5d-floor",
            "-0.10",
            "--exposure-release-confirmation-periods",
            "3",
        ]
    )

    policy = _resolve_exposure_policy(args)
    assert policy["enabled"] is True
    assert policy["profile"] == "trade_republic"
    assert policy["core_asset"] == "A0YEDG"
    assert policy["level1_ret_5d_floor"] == -0.10
    assert policy["release_confirmation_periods"] == 3


def test_exposure_policy_file_profile_is_not_overwritten_by_parser_default(tmp_path):
    config_path = tmp_path / "exposure_policy.json"
    config_path.write_text(json.dumps({"enabled": True, "profile": "us"}))

    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "strategies/dummy.py",
            "--exposure-policy-file",
            str(config_path),
        ]
    )

    policy = _resolve_exposure_policy(args)
    assert policy["profile"] == "us"


def test_signals_parser_accepts_meta_evidence_options():
    parser = create_parser()
    args = parser.parse_args(
        [
            "signals",
            "strategies/dummy.py",
            "--meta-enable",
            "--meta-candidate",
            "strategies/buy_and_hold.py",
            "--meta-scoring",
            "hybrid",
            "--meta-confirm-points",
            "3",
            "--meta-switch-margin",
            "0.15",
            "--meta-evidence-profile",
            "ausgewogen",
            "--meta-evidence-max-age-days",
            "45",
            "--meta-gate-fail-action",
            "hold_current",
            "--meta-regime-mode",
            "strategy_fragility",
            "--meta-regime-profile",
            "defensiv",
            "--meta-alpha-tie-band",
            "0.02",
            "--meta-stress-alpha-tolerance",
            "0.03",
            "--meta-conditioned-min-windows",
            "6",
        ]
    )

    assert args.meta_enable is True
    assert args.meta_candidates == ["strategies/buy_and_hold.py"]
    assert args.meta_confirm_points == 3
    assert args.meta_switch_margin == 0.15
    assert args.meta_evidence_profile == "ausgewogen"
    assert args.meta_evidence_max_age_days == 45
    assert args.meta_gate_fail_action == "hold_current"
    assert args.meta_regime_mode == "strategy_fragility"
    assert args.meta_regime_profile == "defensiv"
    assert args.meta_alpha_tie_band == 0.02
    assert args.meta_stress_alpha_tolerance == 0.03
    assert args.meta_conditioned_min_windows == 6


def test_meta_evidence_parser_accepts_required_fields():
    parser = create_parser()
    args = parser.parse_args(
        [
            "meta-evidence",
            "--current-strategy",
            "strategies/a.py",
            "--target-strategy",
            "strategies/b.py",
            "--profile",
            "custom",
            "--custom-threshold",
            "min_windows=4",
            "--custom-threshold",
            "min_cagr_edge_pp=1.5",
            "--train-years",
            "3",
            "--test-years",
            "1",
            "--step-months",
            "6",
            "--tuning-enabled",
            "--grid-confirm-points",
            "1,2",
            "--grid-switch-margin",
            "0.05,0.1",
            "--max-combinations",
            "80",
            "--top-k",
            "5",
        ]
    )

    assert args.current_strategy == "strategies/a.py"
    assert args.target_strategy == "strategies/b.py"
    assert args.evidence_profile == "custom"
    assert args.custom_thresholds == ["min_windows=4", "min_cagr_edge_pp=1.5"]
    assert args.train_years == 3.0
    assert args.test_years == 1.0
    assert args.step_months == 6
    assert args.tuning_enabled is True
    assert args.grid_confirm_points == "1,2"
    assert args.grid_switch_margin == "0.05,0.1"
    assert args.max_combinations == 80
    assert args.top_k == 5


def test_meta_bootstrap_parser_accepts_required_fields():
    parser = create_parser()
    args = parser.parse_args(
        [
            "meta-bootstrap",
            "--strategy-a",
            "strategies/a.py",
            "--strategy-b",
            "strategies/b.py",
            "--a-param",
            "safe_asset=SPY",
            "--b-param",
            "safe_asset=BIL",
            "--profile",
            "ausgewogen",
            "--fallback-cagr-tie-band-pp",
            "0.8",
            "--fallback-tie-breaker",
            "maxdd",
        ]
    )

    assert args.strategy_a == "strategies/a.py"
    assert args.strategy_b == "strategies/b.py"
    assert args.strategy_a_params == ["safe_asset=SPY"]
    assert args.strategy_b_params == ["safe_asset=BIL"]
    assert args.evidence_profile == "ausgewogen"
    assert args.fallback_cagr_tie_band_pp == 0.8
    assert args.fallback_tie_breaker == "maxdd"


def test_data_provenance_add_parser_accepts_required_fields():
    parser = create_parser()
    args = parser.parse_args(
        [
            "data",
            "provenance",
            "add",
            "data/manual/sample.csv",
            "--dataset",
            "fundamentals_sp500",
            "--source",
            "SeekingAlpha",
            "--quality-tag",
            "manual",
            "--as-of-date",
            "2026-02-06",
        ]
    )

    assert args.data_command == "provenance"
    assert args.provenance_command == "add"
    assert args.file_path == "data/manual/sample.csv"
    assert args.dataset == "fundamentals_sp500"
    assert args.source == "SeekingAlpha"
    assert args.quality_tag == "manual"
    assert args.as_of_date == "2026-02-06"


def test_data_provenance_verify_parser_accepts_skip_hash():
    parser = create_parser()
    args = parser.parse_args(
        [
            "data",
            "provenance",
            "verify",
            "--skip-hash",
        ]
    )

    assert args.data_command == "provenance"
    assert args.provenance_command == "verify"
    assert args.skip_hash is True


def test_batch_optimize_parser_accepts_nested_walk_forward_options():
    parser = create_parser()
    args = parser.parse_args(
        [
            "batch-optimize",
            "strategies/dummy.py",
            "--walk-forward",
            "--walk-forward-nested",
            "--train-years",
            "5",
            "--test-years",
            "1",
            "--step-months",
            "12",
            "--anchored",
            "--inner-train-years",
            "2",
            "--inner-test-years",
            "0.5",
            "--inner-step-months",
            "3",
            "--inner-anchored",
        ]
    )

    assert args.walk_forward is True
    assert args.walk_forward_nested is True
    assert args.train_years == 5.0
    assert args.test_years == 1.0
    assert args.step_months == 12
    assert args.anchored is True
    assert args.inner_train_years == 2.0
    assert args.inner_test_years == 0.5
    assert args.inner_step_months == 3
    assert args.inner_anchored is True
