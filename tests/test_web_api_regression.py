import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from backtest.data import PriceData
from backtest.web.app import create_app


def _fake_yahoo(
    tickers,
    start,
    end=None,
    currency="EUR",
    align="ffill",
    skip_failed=False,
    load_dividends=False,
    load_volumes=False,
    validate=True,
):
    """Deterministic in-memory market data for web API regression tests."""
    _ = (currency, align, skip_failed, validate)
    tickers = list(tickers)
    start_dt = pd.to_datetime(start or "2020-01-01")
    end_dt = pd.to_datetime(end or "2024-12-31")
    if end_dt <= start_dt:
        end_dt = start_dt + pd.Timedelta(days=365)

    idx = pd.bdate_range(start=start_dt, end=end_dt)
    if len(idx) < 60:
        idx = pd.bdate_range(start=start_dt, periods=120)

    prices = {}
    for i, ticker in enumerate(tickers):
        steps = np.arange(len(idx), dtype=float)
        trend = 1.0 + 0.0004 * steps
        seasonal = 1.0 + 0.01 * np.sin((steps / 12.0) + i)
        level = 75.0 + i * 10.0
        series = level * trend * seasonal
        prices[ticker] = np.maximum(series, 1.0)

    price_df = pd.DataFrame(prices, index=idx)

    volumes = None
    if load_volumes:
        volumes = pd.DataFrame(
            {
                ticker: 1_000_000 + i * 100_000 + np.arange(len(idx))
                for i, ticker in enumerate(tickers)
            },
            index=idx,
        ).astype(float)

    dividends = None
    if load_dividends:
        dividends = pd.DataFrame(0.0, index=idx, columns=tickers)
        quarter_mask = idx.month.isin([3, 6, 9, 12]) & idx.is_month_end
        if quarter_mask.any():
            dividends.loc[idx[quarter_mask], :] = 0.12

    return PriceData(
        prices=price_df,
        currency={ticker: "USD" for ticker in tickers},
        volumes=volumes,
        dividends=dividends,
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("backtest.data.DataLoader.yahoo", _fake_yahoo)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def manual_data_client(tmp_path, monkeypatch):
    monkeypatch.setattr("backtest.data.DataLoader.yahoo", _fake_yahoo)
    monkeypatch.setenv("BACKTEST_PROVENANCE_PATH", str(tmp_path / "provenance.json"))
    manual_file = tmp_path / "seekingalpha_export.csv"
    manual_file.write_text("ticker,score\nAAPL,0.9\nMSFT,0.8\n")

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client, manual_file


def _strategy_paths(client: TestClient):
    response = client.get("/api/v1/strategies")
    assert response.status_code == 200
    strategies = response.json()
    assert strategies, "Expected at least one strategy from /api/v1/strategies"

    by_name = {entry["file_name"]: entry["file_path"] for entry in strategies}
    buy_hold = by_name.get("buy_and_hold.py", strategies[0]["file_path"])
    classic_6040 = by_name.get("classic_60_40.py", buy_hold)
    return buy_hold, classic_6040


def _common_payload_fields():
    return {
        "initial_capital": 10_000,
        "costs_pct": 0.001,
        "tax_enabled": True,
        "tax_exemption": 1_000,
        "metric_basis": "net_liquidation",
        "drip_enabled": False,
        "skip_failed": False,
        "execution_lag_days": 1,
        "max_volume_participation": 0.10,
        "min_daily_dollar_volume": 10_000,
        "liquidity_on_missing_volume": "allow",
        "max_position": 0.8,
        "turnover_budget": 0.5,
        "sector_caps": {"equity": 0.9, "bonds": 0.8},
        "ticker_sectors": {"SPY": "equity", "AGG": "bonds", "BND": "bonds"},
        "drawdown_brake_threshold": 0.2,
        "drawdown_brake_cash_target": 0.4,
        "drawdown_brake_release": 0.1,
    }


def test_run_endpoint_handles_execution_and_risk_fields(client: TestClient):
    buy_hold, _ = _strategy_paths(client)
    payload = {
        "strategy": buy_hold,
        "start_date": "2020-01-02",
        "end_date": "2022-12-30",
        "benchmark": "SPY",
        "rebalance_frequency": "monthly",
        **_common_payload_fields(),
    }

    response = client.post("/api/v1/run", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["strategy_name"]
    assert data["final_value"] > 0
    assert "metrics" in data
    assert "constraint_impact" in data
    assert "liquidity" in data["constraint_impact"]
    assert "risk_overlay" in data["constraint_impact"]


def test_signals_endpoint_handles_exposure_policy(client: TestClient):
    buy_hold, _ = _strategy_paths(client)
    payload = {
        "strategy": buy_hold,
        "params": {"allocation": {"3SEM.L": 1.0}},
        "signal_date": "2022-12-30",
        "portfolio": {"positions": {}, "cash": 10_000},
        "exposure_policy": {
            "enabled": True,
            "profile": "trade_republic",
            "level1_ret_5d_floor": 1.0,
            "level2_proxy_ret_21d_floor": -1.0,
        },
    }

    response = client.post("/api/v1/signals", json=payload)
    assert response.status_code == 200, response.text
    report = response.json()["report"]
    assert report["exposure_policy"]["raw_strategy_target"] == {"3SEM.L": 1.0}
    assert report["exposure_policy"]["policy_adjusted_target"] == {"VVSM.DE": 1.0}
    assert any(row["ticker"] == "VVSM.DE" and row["action"] == "BUY" for row in report["buys"])


def test_compare_endpoint_handles_execution_and_risk_fields(client: TestClient):
    buy_hold, classic_6040 = _strategy_paths(client)
    payload = {
        "strategies": [
            {"strategy": buy_hold, "params": {}},
            {"strategy": classic_6040, "params": {}},
        ],
        "start_date": "2020-01-02",
        "end_date": "2022-12-30",
        "benchmark": "SPY",
        **_common_payload_fields(),
    }

    response = client.post("/api/v1/compare", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["rows"]) >= 2
    assert data["metric_basis"] == "net_liquidation"


def test_sweep_endpoint_handles_execution_and_risk_fields(client: TestClient):
    buy_hold, _ = _strategy_paths(client)
    payload = {
        "strategies": [buy_hold],
        "mode": "rolling",
        "window_length": "6m",
        "from_date": "2020-01-02",
        "to_date": "2020-12-31",
        "end_date": "2022-12-30",
        "start_grid": "yearly",
        "step": 1,
        "benchmark": "SPY",
        **_common_payload_fields(),
    }

    response = client.post("/api/v1/sweep", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["num_windows"] >= 1
    assert len(data["summaries"]) >= 1


def test_optimize_endpoint_handles_execution_and_risk_fields(client: TestClient):
    buy_hold, _ = _strategy_paths(client)
    payload = {
        "strategy": buy_hold,
        "param_grid": [],
        "rebalance_frequencies": ["monthly"],
        "metric": "sharpe_ratio",
        "minimize": False,
        "top_n": 5,
        "start_date": "2020-01-02",
        "end_date": "2022-12-30",
        "walk_forward": False,
        **_common_payload_fields(),
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["total_combinations"] >= 1
    assert len(data["results"]) >= 1


def test_optimize_walk_forward_respects_step_and_anchored(client: TestClient):
    buy_hold, _ = _strategy_paths(client)
    payload = {
        "strategy": buy_hold,
        "param_grid": [],
        "rebalance_frequencies": ["monthly"],
        "metric": "sharpe_ratio",
        "minimize": False,
        "top_n": 5,
        "start_date": "2020-01-02",
        "end_date": "2022-12-30",
        "walk_forward": True,
        "wf_train_years": 1.0,
        "wf_test_years": 1.0,
        "wf_step_months": 6,
        "wf_anchored": True,
        **_common_payload_fields(),
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["walk_forward_result"] is not None
    assert data["walk_forward_result"]["step_months"] == 6
    assert data["walk_forward_result"]["anchored"] is True


def test_optimize_walk_forward_uses_requested_metric_for_best_params(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    from backtest.research.walk_forward import WalkForwardResult, WalkForwardWindow

    class FakeWalkForwardAnalysis:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

        def run(self, data, config, progress=False):
            _ = (data, config, progress)
            window_1 = WalkForwardWindow(
                train_start=pd.Timestamp("2020-01-02").to_pydatetime(),
                train_end=pd.Timestamp("2020-12-31").to_pydatetime(),
                test_start=pd.Timestamp("2021-01-01").to_pydatetime(),
                test_end=pd.Timestamp("2021-12-31").to_pydatetime(),
                best_params={"lookback_months": 3},
                train_metrics={"sharpe_ratio": 1.0},
                test_metrics={
                    "sharpe_ratio": 2.2,
                    "cagr": 0.10,
                    "sortino_ratio": 1.8,
                    "volatility": 0.20,
                    "max_drawdown": -0.30,
                    "calmar_ratio": 0.33,
                    "total_return": 0.10,
                },
            )
            window_2 = WalkForwardWindow(
                train_start=pd.Timestamp("2021-01-01").to_pydatetime(),
                train_end=pd.Timestamp("2021-12-31").to_pydatetime(),
                test_start=pd.Timestamp("2022-01-03").to_pydatetime(),
                test_end=pd.Timestamp("2022-12-30").to_pydatetime(),
                best_params={"lookback_months": 12},
                train_metrics={"sharpe_ratio": 0.8},
                test_metrics={
                    "sharpe_ratio": 1.1,
                    "cagr": 0.35,
                    "sortino_ratio": 1.2,
                    "volatility": 0.25,
                    "max_drawdown": -0.40,
                    "calmar_ratio": 0.87,
                    "total_return": 0.35,
                },
            )
            return WalkForwardResult(windows=[window_1, window_2])

    monkeypatch.setattr("backtest.research.walk_forward.WalkForwardAnalysis", FakeWalkForwardAnalysis)

    buy_hold, _ = _strategy_paths(client)
    payload = {
        "strategy": buy_hold,
        "param_grid": [],
        "rebalance_frequencies": ["monthly"],
        "metric": "cagr",
        "minimize": False,
        "top_n": 5,
        "start_date": "2020-01-02",
        "end_date": "2022-12-30",
        "walk_forward": True,
        "wf_train_years": 1.0,
        "wf_test_years": 1.0,
        **_common_payload_fields(),
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["best_metric_value"] == pytest.approx(0.35)
    assert data["best_params"] == {"lookback_months": 12}
    assert data["walk_forward_result"]["best_params"] == {"lookback_months": 12}
    assert data["walk_forward_result"]["best_oos_sharpe"] == pytest.approx(2.2)


def test_optimize_walk_forward_nested_respects_inner_settings(client: TestClient):
    buy_hold, _ = _strategy_paths(client)
    payload = {
        "strategy": buy_hold,
        "param_grid": [],
        "rebalance_frequencies": ["monthly"],
        "metric": "sharpe_ratio",
        "minimize": False,
        "top_n": 5,
        "start_date": "2020-01-02",
        "end_date": "2024-12-30",
        "walk_forward": True,
        "walk_forward_nested": True,
        "wf_train_years": 2.0,
        "wf_test_years": 1.0,
        "wf_step_months": 6,
        "wf_anchored": False,
        "wf_inner_train_years": 1.0,
        "wf_inner_test_years": 0.5,
        "wf_inner_step_months": 3,
        "wf_inner_anchored": True,
        **_common_payload_fields(),
    }

    response = client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    wf = data["walk_forward_result"]
    assert wf is not None
    assert wf["nested"] is True
    assert wf["mode"] == "nested"
    assert wf["inner_train_months"] == 12
    assert wf["inner_test_months"] == 6
    assert wf["inner_step_months"] == 3
    assert wf["inner_anchored"] is True
    assert len(wf["windows"]) >= 1


def test_optimize_page_shows_nested_walk_forward_controls(client: TestClient):
    response = client.get("/optimize")
    assert response.status_code == 200
    assert "Nested Walk-Forward" in response.text


def test_batch_optimize_endpoint_handles_execution_and_risk_fields(client: TestClient):
    buy_hold, _ = _strategy_paths(client)
    payload = {
        "strategies": [buy_hold],
        "rebalance_frequencies": ["monthly"],
        "metric": "sharpe_ratio",
        "minimize": False,
        "start_date": "2020-01-02",
        "end_date": "2022-12-30",
        "walk_forward": False,
        "wf_train_years": 2.0,
        "wf_test_years": 1.0,
        **_common_payload_fields(),
    }

    response = client.post("/api/v1/batch-optimize", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["total_strategies"] == 1
    assert len(data["results"]) >= 1


def test_batch_optimize_walk_forward_respects_step_and_anchored(client: TestClient):
    buy_hold, _ = _strategy_paths(client)
    payload = {
        "strategies": [buy_hold],
        "rebalance_frequencies": ["monthly"],
        "metric": "sharpe_ratio",
        "minimize": False,
        "start_date": "2020-01-02",
        "end_date": "2022-12-30",
        "walk_forward": True,
        "wf_train_years": 1.0,
        "wf_test_years": 1.0,
        "wf_step_months": 6,
        "wf_anchored": True,
        **_common_payload_fields(),
    }

    response = client.post("/api/v1/batch-optimize", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["walk_forward_enabled"] is True
    assert data["wf_step_months"] == 6
    assert data["wf_anchored"] is True


def test_batch_optimize_walk_forward_nested_respects_inner_settings(client: TestClient):
    buy_hold, _ = _strategy_paths(client)
    payload = {
        "strategies": [buy_hold],
        "rebalance_frequencies": ["monthly"],
        "metric": "sharpe_ratio",
        "minimize": False,
        "start_date": "2020-01-02",
        "end_date": "2024-12-30",
        "walk_forward": True,
        "walk_forward_nested": True,
        "wf_train_years": 2.0,
        "wf_test_years": 1.0,
        "wf_step_months": 6,
        "wf_anchored": False,
        "wf_inner_train_years": 1.0,
        "wf_inner_test_years": 0.5,
        "wf_inner_step_months": 3,
        "wf_inner_anchored": True,
        **_common_payload_fields(),
    }

    response = client.post("/api/v1/batch-optimize", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["walk_forward_enabled"] is True
    assert data["walk_forward_nested"] is True
    assert data["wf_inner_train_years"] == 1.0
    assert data["wf_inner_test_years"] == 0.5
    assert data["wf_inner_step_months"] == 3
    assert data["wf_inner_anchored"] is True
    assert len(data["results"]) >= 1
    wf = data["results"][0]["walk_forward"]
    assert wf["nested"] is True
    assert wf["mode"] == "nested"


def test_manual_data_page_renders(client: TestClient):
    response = client.get("/manual-data")
    assert response.status_code == 200
    assert "Manual Data Provenance" in response.text


def test_signals_page_renders(client: TestClient):
    response = client.get("/signals")
    assert response.status_code == 200
    assert "Live Trading Signals" in response.text
    assert "Workflow right here" in response.text
    assert "Bootstrap Window" in response.text
    assert "meta_enabled: true" in response.text
    assert "Needs next:" in response.text


def test_signals_endpoint_returns_order_and_drift_report(client: TestClient):
    buy_hold, _ = _strategy_paths(client)
    payload = {
        "strategy": buy_hold,
        "params": {},
        "signal_date": "2022-12-30",
        "rebalance_frequency": "monthly",
        "drift_tolerance": 0.005,
        "skip_failed": False,
        "portfolio": {
            "positions": {},
            "cash": 10_000.0,
            "last_rebalance": None,
        },
    }

    response = client.post("/api/v1/signals", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    report = body["report"]

    assert report["strategy_name"]
    assert report["summary"]["orders"]["actionable"] >= 1
    assert "drift_reconciliation" in report
    assert len(report["orders"]) >= 1


def test_signals_endpoint_returns_meta_decision_block(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    buy_hold, _ = _strategy_paths(client)

    monkeypatch.setattr(
        "backtest.meta_decision.run_meta_decision",
        lambda **kwargs: {
            "enabled": True,
            "current_strategy": kwargs["current_strategy"],
            "recommended_target": kwargs["current_strategy"],
            "score_margin": 0.0,
            "performance_gap": 0.0,
            "switch_allowed": False,
            "executed_action": "hold_current",
            "decision_rule": "hold",
            "evidence_status": "missing",
            "conditioned_evidence_status": None,
            "conditioned_windows": None,
            "current_regime_bucket": "normal",
            "target_regime_bucket": "normal",
            "current_regime_reasons": ["No fragility flags triggered"],
            "target_regime_reasons": ["No fragility flags triggered"],
            "evidence_summary": None,
            "evidence_reasons": ["No evidence artifact found"],
            "evidence_artifact_id": None,
            "live_reasons": ["Current strategy remains top-ranked"],
            "candidates": [],
        },
    )

    payload = {
        "strategy": buy_hold,
        "params": {},
        "signal_date": "2022-12-30",
        "rebalance_frequency": "monthly",
        "drift_tolerance": 0.005,
        "skip_failed": False,
        "portfolio": {
            "positions": {},
            "cash": 10_000.0,
            "last_rebalance": None,
        },
        "meta_decision": {
            "enabled": True,
            "candidates": [{"strategy": buy_hold, "params": {}}],
            "evidence_required": True,
            "regime_mode": "strategy_fragility",
            "regime_profile": "ausgewogen",
            "alpha_tie_band": 0.03,
            "stress_alpha_tolerance": 0.05,
            "conditioned_min_windows": 4,
        },
    }

    response = client.post("/api/v1/signals", json=payload)
    assert response.status_code == 200, response.text
    report = response.json()["report"]
    assert report["meta_decision"] is not None
    assert report["meta_decision"]["executed_action"] == "hold_current"
    assert report["meta_decision"]["evidence_status"] == "missing"
    assert report["meta_decision"]["decision_rule"] == "hold"


def test_meta_evidence_endpoints(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    fake_artifact = {
        "artifact_id": "artifact-xyz",
        "pair_id": "a__to__b",
        "as_of": "2026-02-17",
        "created_at": "2026-02-17T00:00:00+00:00",
        "summary": {
            "num_windows": 8,
            "oos_cagr_edge_pp": 1.2,
            "oos_hit_rate": 0.6,
            "oos_degradation_pct": 25.0,
            "oos_dd_delta_pp": 2.0,
        },
        "gates": {"pass": True, "checks": {}, "reasons": [], "thresholds": {}},
        "artifact_path": "/tmp/artifact-xyz.json",
    }

    monkeypatch.setattr(
        "backtest.meta_evidence.run_meta_evidence_analysis",
        lambda **kwargs: fake_artifact,
    )
    monkeypatch.setattr(
        "backtest.meta_evidence.find_latest_evidence_artifact",
        lambda current_strategy, target_strategy: fake_artifact,
    )
    monkeypatch.setattr(
        "backtest.meta_evidence.find_evidence_artifact_by_id",
        lambda artifact_id: fake_artifact if artifact_id == "artifact-xyz" else None,
    )

    run_payload = {
        "current_strategy": "/tmp/current.py",
        "target_strategy": "/tmp/target.py",
        "current_params": {},
        "target_params": {},
        "evidence_profile": "ausgewogen",
        "tuning_enabled": True,
        "grid_confirm_points": [1, 2],
        "grid_switch_margin": [0.05, 0.10],
        "max_combinations": 120,
        "top_k": 2,
    }
    run_response = client.post("/api/v1/meta-evidence/run", json=run_payload)
    assert run_response.status_code == 200, run_response.text
    assert run_response.json()["artifact"]["artifact_id"] == "artifact-xyz"
    assert run_response.json()["artifact"]["tuning"]["mode"] == "2-stage-smart"

    latest_response = client.get(
        "/api/v1/meta-evidence/latest",
        params={"current_strategy": "/tmp/current.py", "target_strategy": "/tmp/target.py"},
    )
    assert latest_response.status_code == 200, latest_response.text
    assert latest_response.json()["artifact"]["pair_id"] == "a__to__b"

    by_id_response = client.get("/api/v1/meta-evidence/artifact-xyz")
    assert by_id_response.status_code == 200, by_id_response.text
    assert by_id_response.json()["artifact"]["artifact_id"] == "artifact-xyz"


def test_meta_bootstrap_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    fake_artifact = {
        "artifact_id": "bootstrap-xyz",
        "pair_id": "a__vs__b",
        "as_of": "2026-02-17",
        "strategy_a": {"path": "/tmp/a.py", "params": {}},
        "strategy_b": {"path": "/tmp/b.py", "params": {}},
        "evidence": {
            "a_to_b": {"pass": False, "oos_cagr_edge_pp": -1.2, "oos_hit_rate": 0.45},
            "b_to_a": {"pass": False, "oos_cagr_edge_pp": 1.2, "oos_hit_rate": 0.55},
        },
        "fallback": {
            "cagr_edge_b_minus_a_pp": 1.4,
            "tie_band_pp": 1.0,
            "tie_breaker": "maxdd",
        },
        "decision": {
            "recommended_start_strategy": "/tmp/b.py",
            "decision_rule": "fallback_cagr",
            "reasons": ["No unilateral evidence pass; selected higher full-period CAGR (edge +1.40pp)"],
        },
        "artifact_path": "/tmp/bootstrap-xyz.json",
    }

    monkeypatch.setattr(
        "backtest.meta_bootstrap.run_meta_bootstrap_decision",
        lambda **kwargs: fake_artifact,
    )

    payload = {
        "strategy_a": "/tmp/a.py",
        "strategy_b": "/tmp/b.py",
        "strategy_a_params": {},
        "strategy_b_params": {},
        "evidence_profile": "ausgewogen",
        "fallback_cagr_tie_band_pp": 1.0,
        "fallback_tie_breaker": "maxdd",
    }
    response = client.post("/api/v1/meta-bootstrap/run", json=payload)
    assert response.status_code == 200, response.text
    artifact = response.json()["artifact"]
    assert artifact["artifact_id"] == "bootstrap-xyz"
    assert artifact["decision"]["recommended_start_strategy"] == "/tmp/b.py"
    assert artifact["decision"]["decision_rule"] == "fallback_cagr"


def test_manual_provenance_endpoints_create_list_verify(manual_data_client):
    client, manual_file = manual_data_client

    create_payload = {
        "file_path": str(manual_file),
        "dataset": "fundamentals_sp500",
        "source": "SeekingAlpha",
        "quality_tag": "manual",
        "as_of_date": "2026-02-06",
        "import_method": "manual_csv_export",
        "license_tos_note": "Manual export from personal SeekingAlpha access.",
        "source_url": "https://seekingalpha.com/",
        "notes": "Smoke test entry",
    }

    create_response = client.post("/api/v1/data/manual/provenance", json=create_payload)
    assert create_response.status_code == 200, create_response.text
    created = create_response.json()
    entry_id = created["entry_id"]
    assert created["dataset"] == "fundamentals_sp500"
    assert created["source"] == "SeekingAlpha"
    assert created["row_count"] == 2
    assert created["column_count"] == 2

    list_response = client.get("/api/v1/data/manual/provenance")
    assert list_response.status_code == 200, list_response.text
    listed = list_response.json()
    assert listed["total"] == 1
    assert listed["entries"][0]["entry_id"] == entry_id

    single_response = client.get(f"/api/v1/data/manual/provenance/{entry_id}")
    assert single_response.status_code == 200, single_response.text
    single = single_response.json()
    assert single["entry_id"] == entry_id
    assert single["checksum_sha256"] == created["checksum_sha256"]

    verify_response = client.get("/api/v1/data/manual/provenance/verify")
    assert verify_response.status_code == 200, verify_response.text
    verify = verify_response.json()
    assert verify["total_entries"] == 1
    assert verify["ok_entries"] == 1
    assert verify["issue_count"] == 0

    # Changing the file should trigger checksum mismatch.
    manual_file.write_text("ticker,score\nAAPL,0.95\nMSFT,0.85\n")
    verify_after_change = client.get("/api/v1/data/manual/provenance/verify")
    assert verify_after_change.status_code == 200, verify_after_change.text
    changed = verify_after_change.json()
    assert changed["total_entries"] == 1
    assert changed["ok_entries"] == 0
    assert changed["issue_count"] == 1
    assert changed["issues"][0]["entry_id"] == entry_id
    assert changed["issues"][0]["status"] == "checksum_mismatch"
