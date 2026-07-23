"""Web API + page route tests for meta playbook (meta promotion governance)."""

import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from backtest.web.app import create_app


def _fake_artifact(artifact_id: str = "fake-123") -> dict:
    return {
        "artifact_id": artifact_id,
        "generated_at": "2026-05-16T12:00:00+00:00",
        "config": {
            "baseline_path": "strategies/levered_etf_momentum_sticky.py",
            "research_proxy_mode": "live",
            "start": "2024-01-01",
            "end": "2024-06-30",
            "metric_basis": "net_liquidation",
        },
        "strategies": [
            {
                "path": "strategies/levered_etf_momentum_sticky.py",
                "role": "baseline",
                "metrics": {
                    "strategy_name": "Sticky/Levered",
                    "cagr": 0.20,
                    "max_drawdown": -0.30,
                    "sharpe": 0.8,
                    "sortino": 1.0,
                    "calmar": 0.67,
                    "rebalance_frequency": "weekly",
                },
                "broker_mapping": {
                    "trade_republic": {"ok": True, "status": "ok", "assets": []},
                },
            },
            {
                "path": "strategies/sticky_levered_vol_targeted.py",
                "role": "candidate",
                "metrics": {
                    "strategy_name": "Sticky/Levered VolTarget",
                    "cagr": 0.28,
                    "max_drawdown": -0.20,
                    "sharpe": 1.1,
                    "sortino": 1.3,
                    "calmar": 1.4,
                    "rebalance_frequency": "weekly",
                },
                "vs_baseline": {
                    "baseline_name": "Sticky/Levered",
                    "cagr_delta_pp": 8.0,
                    "maxdd_delta_pp": 10.0,
                    "sharpe_delta": 0.3,
                    "rolling_3y": {"win_rate": 1.0, "windows": 4, "cagr_win_rate": 1.0, "maxdd_win_rate": 1.0, "worst_maxdd_delta_pp": 0.0},
                    "rolling_5y": {"win_rate": 1.0, "windows": 2, "cagr_win_rate": 1.0, "maxdd_win_rate": 1.0, "worst_maxdd_delta_pp": 0.0},
                },
                "broker_mapping": {
                    "trade_republic": {"ok": True, "status": "ok", "assets": []},
                },
            },
        ],
        "paths": {
            "json": "results/meta_promotion/20260516/fake-123/promotion_report.json",
            "markdown": "results/meta_promotion/20260516/fake-123/promotion_report.md",
        },
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with a mocked run_meta_promotion_report + isolated artifact dir.

    DEFAULT_PROMOTION_DIR is a relative path ("results/meta_promotion") and is
    resolved against cwd via .resolve() — chdir into tmp_path therefore cleanly
    isolates the discovery helpers without patching the module default.
    """
    monkeypatch.chdir(tmp_path)
    promotion_dir = tmp_path / "results" / "meta_promotion"

    captured = {"calls": []}

    def fake_run(**kwargs):
        captured["calls"].append(kwargs)
        artifact = _fake_artifact("run-abc")
        artifact_dir = promotion_dir / "20260516" / "run-abc"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        json_path = artifact_dir / "promotion_report.json"
        md_path = artifact_dir / "promotion_report.md"
        artifact["paths"] = {"json": str(json_path), "markdown": str(md_path)}
        json_path.write_text(json.dumps(artifact), encoding="utf-8")
        md_path.write_text("# Run abc\n\nSticky/Levered vs. Sticky/Levered VolTarget\n", encoding="utf-8")
        return artifact

    monkeypatch.setattr("backtest.meta_promotion.run_meta_promotion_report", fake_run)

    app = create_app()
    with TestClient(app) as test_client:
        test_client.captured = captured  # type: ignore[attr-defined]
        test_client.promotion_dir = promotion_dir  # type: ignore[attr-defined]
        yield test_client


def test_playbook_page_renders_with_navigation(client: TestClient):
    response = client.get("/playbook")

    assert response.status_code == 200
    assert "Meta-Playbook" in response.text
    # Nav entry in base.html (desktop) links to /playbook
    assert 'href="/playbook"' in response.text
    # Form submit button present
    assert "Generate promotion report" in response.text


def test_meta_promotion_run_returns_artifact_and_passes_flags(client: TestClient):
    body = {
        "research_proxy_mode": "soxl_proxy",
        "start": "2024-01-01",
        "end": "2024-06-30",
        "metric_basis": "gross",
        "tax_enabled": False,
        "brokers": ["trade_republic"],
        "tail_risk_gate_basis": "rebalance",
    }
    response = client.post("/api/v1/meta-promotion/run", json=body)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["artifact"]["artifact_id"] == "run-abc"
    assert payload["artifact"]["strategies"][0]["role"] == "baseline"
    # API translates soxl_proxy -> SOXL_PROXY_STRATEGIES + SOXL_PROXY_BASELINE.
    call = client.captured["calls"][-1]  # type: ignore[attr-defined]
    assert call["research_proxy_mode"] == "soxl_proxy"
    assert call["metric_basis"] == "gross"
    assert call["tax_enabled"] is False
    assert call["brokers"] == ["trade_republic"]
    assert call["tail_risk_gate_basis"] == "rebalance"
    assert call["baseline_path"].startswith("builtin:sticky_levered_soxl_proxy")


def test_meta_promotion_list_then_get_by_id(client: TestClient):
    # First run -> writes artifact under promotion_dir
    client.post("/api/v1/meta-promotion/run", json={})

    # List
    list_resp = client.get("/api/v1/meta-promotion/list")
    assert list_resp.status_code == 200
    artifacts = list_resp.json()["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0]["artifact_id"] == "run-abc"
    assert artifacts[0]["candidate_count"] == 1

    # Latest
    latest_resp = client.get("/api/v1/meta-promotion/latest")
    assert latest_resp.status_code == 200
    assert latest_resp.json()["artifact"]["artifact_id"] == "run-abc"

    # By-ID
    by_id_resp = client.get("/api/v1/meta-promotion/run-abc")
    assert by_id_resp.status_code == 200
    assert by_id_resp.json()["artifact"]["artifact_id"] == "run-abc"

    # Markdown
    md_resp = client.get("/api/v1/meta-promotion/run-abc/markdown")
    assert md_resp.status_code == 200
    assert "Run abc" in md_resp.text


def test_meta_promotion_missing_id_returns_404(client: TestClient):
    not_found = client.get("/api/v1/meta-promotion/this-does-not-exist")
    assert not_found.status_code == 404

    md_not_found = client.get("/api/v1/meta-promotion/this-does-not-exist/markdown")
    assert md_not_found.status_code == 404
