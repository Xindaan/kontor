from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.data import PriceData
from backtest.meta_evidence import (
    assess_conditioned_evidence_status,
    assess_evidence_status,
    run_meta_evidence_analysis,
)


def _write_strategy(path: Path, ticker: str) -> None:
    path.write_text(
        "\n".join(
            [
                "from datetime import date",
                "import pandas as pd",
                "from backtest.strategy import Strategy, Allocation",
                "",
                "class _StaticStrategy(Strategy):",
                f"    assets = ['{ticker}']",
                "    rebalance_frequency = 'monthly'",
                f"    name = 'Static {ticker}'",
                "",
                "    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:",
                "        _ = (current_date, data)",
                f"        return Allocation({{'{ticker}': 1.0}})",
                "",
                "strategy = _StaticStrategy()",
            ]
        )
    )


def _fake_yahoo(
    tickers,
    start,
    end=None,
    currency="EUR",
    align="ffill",
    skip_failed=False,
):
    _ = (currency, align, skip_failed)
    tickers = list(tickers)
    idx = pd.bdate_range(pd.to_datetime(start), pd.to_datetime(end or "2024-12-31"))
    steps = np.arange(len(idx), dtype=float)

    prices = {}
    for ticker in tickers:
        if ticker == "AAA":
            base = 100.0 * (1.0 + 0.0002 * steps)
        elif ticker == "BBB":
            base = 100.0 * (1.0 + 0.0006 * steps)
        else:
            base = 100.0 * (1.0 + 0.0003 * steps)
        prices[ticker] = np.maximum(base, 1.0)

    return PriceData(
        prices=pd.DataFrame(prices, index=idx),
        currency={ticker: "USD" for ticker in tickers},
    )


def test_meta_evidence_custom_thresholds_pass_and_fail(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("backtest.data.DataLoader.yahoo", _fake_yahoo)

    current_strategy = tmp_path / "current_strategy.py"
    target_strategy = tmp_path / "target_strategy.py"
    _write_strategy(current_strategy, "AAA")
    _write_strategy(target_strategy, "BBB")

    pass_artifact = run_meta_evidence_analysis(
        current_strategy=current_strategy,
        target_strategy=target_strategy,
        as_of=date(2024, 12, 31),
        evidence_profile="custom",
        custom_thresholds={
            "min_windows": 1,
            "min_cagr_edge_pp": -100.0,
            "min_hit_rate": 0.0,
            "max_degradation_pct": 999.0,
            "max_dd_worsening_pp": 999.0,
        },
        train_years=1.0,
        test_years=0.5,
        step_months=6,
        start_date="2020-01-01",
        save_artifact=True,
    )
    assert pass_artifact["gates"]["pass"] is True
    assert Path(pass_artifact["artifact_path"]).exists()

    fail_artifact = run_meta_evidence_analysis(
        current_strategy=current_strategy,
        target_strategy=target_strategy,
        as_of=date(2024, 12, 31),
        evidence_profile="custom",
        custom_thresholds={
            "min_windows": 1,
            "min_cagr_edge_pp": 500.0,
            "min_hit_rate": 1.0,
            "max_degradation_pct": 0.1,
            "max_dd_worsening_pp": 0.1,
        },
        train_years=1.0,
        test_years=0.5,
        step_months=6,
        start_date="2020-01-01",
        save_artifact=False,
    )
    assert fail_artifact["gates"]["pass"] is False
    assert len(fail_artifact["gates"]["reasons"]) >= 1


def test_assess_evidence_status_missing_and_stale():
    status, reasons, age = assess_evidence_status(
        artifact=None,
        evidence_max_age_days=30,
        as_of=date(2026, 2, 17),
    )
    assert status == "missing"
    assert "No evidence artifact found" in reasons[0]
    assert age is None

    stale_artifact = {
        "created_at": "2025-01-01T00:00:00+00:00",
        "gates": {"pass": True, "reasons": []},
    }
    status, reasons, age = assess_evidence_status(
        artifact=stale_artifact,
        evidence_max_age_days=30,
        as_of=date(2026, 2, 17),
    )
    assert status == "stale"
    assert age is not None and age > 30
    assert any("stale" in reason.lower() for reason in reasons)


def test_meta_evidence_builds_conditioned_summary(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("backtest.data.DataLoader.yahoo", _fake_yahoo)

    current_strategy = tmp_path / "current_strategy.py"
    target_strategy = tmp_path / "target_strategy.py"
    _write_strategy(current_strategy, "AAA")
    _write_strategy(target_strategy, "BBB")

    counter = {"value": 0}

    def fake_classify_current_regime(equity_curve):
        _ = equity_curve
        counter["value"] += 1
        bucket = "fragile" if counter["value"] % 2 else "normal"
        return {
            "status": "ok",
            "reference_days": 400,
            "available_feature_days": 400,
            "history_reason": None,
            "metrics": {},
            "percentiles": {},
            "buckets": {
                "defensiv": bucket,
                "ausgewogen": bucket,
                "aggressiv": bucket,
            },
            "reasons": {
                "defensiv": [bucket],
                "ausgewogen": [bucket],
                "aggressiv": [bucket],
            },
            "flags": {
                "defensiv": {},
                "ausgewogen": {},
                "aggressiv": {},
            },
        }

    monkeypatch.setattr("backtest.meta_evidence._classify_current_regime", fake_classify_current_regime)

    artifact = run_meta_evidence_analysis(
        current_strategy=current_strategy,
        target_strategy=target_strategy,
        as_of=date(2024, 12, 31),
        evidence_profile="ausgewogen",
        train_years=1.0,
        test_years=0.5,
        step_months=6,
        start_date="2020-01-01",
        save_artifact=False,
    )

    conditioned = artifact["conditioned_summary"]["ausgewogen"]
    total_conditioned = (
        conditioned["normal"]["num_windows"]
        + conditioned["fragile"]["num_windows"]
        + conditioned["stressed"]["num_windows"]
        + conditioned["insufficient_history"]["num_windows"]
    )

    assert artifact["unconditional_summary"]["num_windows"] == artifact["summary"]["num_windows"]
    assert total_conditioned == artifact["summary"]["num_windows"]
    assert conditioned["normal"]["num_windows"] > 0
    assert conditioned["fragile"]["num_windows"] > 0


def test_assess_conditioned_evidence_status_respects_min_windows():
    artifact = {
        "conditioned_summary": {
            "ausgewogen": {
                "fragile": {
                    "num_windows": 2,
                    "oos_cagr_edge_pp": 1.5,
                    "oos_hit_rate": 1.0,
                    "oos_degradation_pct": 5.0,
                    "oos_dd_delta_pp": 1.0,
                }
            }
        }
    }

    status, reasons, num_windows, summary = assess_conditioned_evidence_status(
        artifact=artifact,
        current_bucket="fragile",
        evidence_profile="ausgewogen",
        conditioned_min_windows=4,
    )

    assert status == "fail"
    assert num_windows == 2
    assert summary is not None
    assert any("Too few OOS windows" in reason for reason in reasons)
