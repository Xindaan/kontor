import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.meta_promotion import (
    SOXL_PROXY_BASELINE,
    broker_mapping_status,
    compare_rolling,
    find_latest_promotion_artifact,
    find_promotion_artifact_by_id,
    list_promotion_artifacts,
    load_strategy_from_path,
    read_promotion_markdown_by_id,
    run_meta_promotion_report,
    rolling_windows,
)
from backtest.data import PriceData
from backtest.strategy import Allocation, Strategy
from strategies.levered_etf_momentum_sticky import LeveredETFMomentumSticky


class _MappedStrategy(Strategy):
    name = "mapped"
    assets = ["3SEM.L", "SXR8.DE"]
    execution_proxy_map = {"3SEM.L": "VVSM.DE"}

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        return Allocation({"3SEM.L": 1.0})


class _SemiStrategy(Strategy):
    name = "semi"
    assets = ["3SEM.L", "SXR8.DE"]

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        return Allocation({"3SEM.L": 1.0})


class _SOXLResearchStrategy(Strategy):
    name = "soxl"
    assets = ["SOXL", "SXR8.DE"]

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        return Allocation({"SOXL": 1.0})


def test_broker_mapping_accepts_explicit_maxblue_proxy():
    status = broker_mapping_status(_MappedStrategy(), ["maxblue"])

    assert status["maxblue"]["ok"] is True
    semi_row = next(row for row in status["maxblue"]["assets"] if row["signal_asset"] == "3SEM.L")
    assert semi_row == {
        "signal_asset": "3SEM.L",
        "live_signal_asset": "3SEM.L",
        "execution_asset": "VVSM.DE",
        "status": "proxy",
    }


def test_broker_mapping_uses_default_maxblue_semiconductor_proxy():
    status = broker_mapping_status(_SemiStrategy(), ["maxblue"])

    assert status["maxblue"]["ok"] is True
    semi_row = next(row for row in status["maxblue"]["assets"] if row["signal_asset"] == "3SEM.L")
    assert semi_row["execution_asset"] == "VVSM.DE"
    assert semi_row["status"] == "proxy"


def test_rolling_compare_reports_win_rates_and_worst_delta():
    dates = pd.bdate_range("2020-01-01", periods=252 * 4)
    steps = np.arange(len(dates))
    baseline = pd.Series(100.0 * (1.0005 ** steps), index=dates)
    candidate = pd.Series(100.0 * (1.0007 ** steps), index=dates)

    baseline_windows = rolling_windows(baseline, 3, step_months=3)
    candidate_windows = rolling_windows(candidate, 3, step_months=3)
    comparison = compare_rolling(candidate_windows, baseline_windows)

    assert comparison["windows"] >= 2
    assert comparison["cagr_win_rate"] == 1.0
    assert comparison["maxdd_win_rate"] == 0.0
    assert comparison["worst_maxdd_delta_pp"] == 0.0


def test_rolling_windows_use_calendar_years_for_monthly_curves():
    dates = pd.date_range("2020-01-31", periods=61, freq="ME")
    curve = pd.Series(100.0 + np.arange(len(dates)), index=dates)

    windows = rolling_windows(curve, 3)

    assert len(windows) > 0
    assert windows[0].start == "2020-01-31"


def test_builtin_soxl_proxy_baseline_uses_long_history_us_proxies():
    strategy = load_strategy_from_path(SOXL_PROXY_BASELINE)

    assert "SOXL" in strategy.assets
    assert "3SEM.L" not in strategy.assets
    assert "SOXL-Proxy" in strategy.name


def _ucits_strategy():
    """A strategy configured on young UCITS ETPs (as a European broker would trade)."""
    from strategies.levered_etf_momentum_sticky import LeveredETFMomentumSticky

    return LeveredETFMomentumSticky(candidates=["QQQ3.L", "3LUS.L", "3SEM.L"])


def test_soxl_proxy_mode_remaps_young_ucits_candidates_to_long_history_proxy():
    """In soxl_proxy mode a UCITS strategy must be remapped onto long-history
    US proxies, so it runs on the same basis as the SOXL-proxy baseline.
    Without the remap it would lose ~14 years of semiconductor history
    (3SEM.L only has data from ~2024).
    """
    from backtest.meta_promotion import (
        _backtest_substitute_map,
        _remap_strategy_candidates,
    )

    strategy = _ucits_strategy()
    assert "3SEM.L" in strategy.candidates
    assert "3SEM.L" in strategy.assets
    assert "SOXL" not in strategy.assets

    substitute = _backtest_substitute_map("soxl_proxy")
    assert substitute.get("3SEM.L") == "SOXL"

    _remap_strategy_candidates(strategy, substitute)

    assert "SOXL" in strategy.candidates
    assert "3SEM.L" not in strategy.candidates
    assert "SOXL" in strategy.assets
    assert "3SEM.L" not in strategy.assets


def test_live_mode_does_not_remap_strategies():
    """In live mode no remap happens — a strategy keeps its real universe."""
    from backtest.meta_promotion import (
        _backtest_substitute_map,
        _remap_strategy_candidates,
    )

    strategy = _ucits_strategy()
    substitute = _backtest_substitute_map("live")
    assert substitute == {}

    _remap_strategy_candidates(strategy, substitute)

    assert "3SEM.L" in strategy.candidates
    assert "3SEM.L" in strategy.assets
    assert "SOXL" not in strategy.assets


def test_soxl_proxy_substitutes_cover_young_ucits_with_long_history_proxies():
    """In soxl_proxy mode, all young UCITS tickers should get a long-
    history substitute. 1:1 substitutes (same index/same
    leverage) and approximate substitutes (factor/region mismatch) are
    both included — only tickers that are genuinely substitution-resistant
    (3XFE.L, 3XEE.L) are left out.
    """
    from backtest.meta_promotion import _backtest_substitute_map

    substitute = _backtest_substitute_map("soxl_proxy")

    # 1:1 substitutes: same index, same leverage
    assert substitute["3SEM.L"] == "SOXL"
    assert substitute["VVSM.DE"] == "SOXX"
    assert substitute["EXXT.DE"] == "QQQ"
    # Approximate substitutes: documented factor/region mismatch
    assert substitute["QDVF.DE"] == "IVE"
    assert substitute["QDVH.DE"] == "MTUM"
    # Deliberately NOT substituted
    assert "3XFE.L" not in substitute
    assert "3XEE.L" not in substitute


def test_substitute_map_isolated_from_broker_audit():
    """Backtest substitutes must NOT change the broker audit semantics:
    e.g. QQQ is directly tradable via L&S on TR and should
    not be marked as a proxy for EXXT.DE in the broker audit,
    even though it is interchangeable with EXXT.DE in the backtest.
    """
    from backtest.meta_promotion import RESEARCH_PROXY_MAPS

    # RESEARCH_PROXY_MAPS should have ONLY the SOXL entry (broker-relevant)
    assert RESEARCH_PROXY_MAPS["soxl_proxy"] == {"SOXL": "3SEM.L"}
    # Ensure that QQQ/SOXX/IVE/MTUM are NOT in RESEARCH_PROXY_MAPS
    assert "QQQ" not in RESEARCH_PROXY_MAPS["soxl_proxy"]
    assert "SOXX" not in RESEARCH_PROXY_MAPS["soxl_proxy"]
    assert "IVE" not in RESEARCH_PROXY_MAPS["soxl_proxy"]
    assert "MTUM" not in RESEARCH_PROXY_MAPS["soxl_proxy"]


def test_soxl_research_proxy_maps_to_live_broker_execution_assets():
    status = broker_mapping_status(
        _SOXLResearchStrategy(),
        ["trade_republic", "maxblue"],
        research_proxy_mode="soxl_proxy",
    )

    tr_row = next(row for row in status["trade_republic"]["assets"] if row["signal_asset"] == "SOXL")
    maxblue_row = next(row for row in status["maxblue"]["assets"] if row["signal_asset"] == "SOXL")

    assert status["trade_republic"]["ok"] is True
    assert tr_row["live_signal_asset"] == "3SEM.L"
    assert tr_row["execution_asset"] == "3SEM.L"
    assert tr_row["status"] == "proxy"
    assert status["maxblue"]["ok"] is True
    assert maxblue_row["live_signal_asset"] == "3SEM.L"
    assert maxblue_row["execution_asset"] == "VVSM.DE"
    assert maxblue_row["status"] == "proxy"


def test_meta_promotion_report_writes_artifacts_and_respects_strategy_frequency(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.py"
    weekly = tmp_path / "weekly.py"
    strategy_code = """
from datetime import date
import pandas as pd
from backtest.strategy import Allocation, Strategy

class TestStrategy(Strategy):
    name = "{name}"
    assets = ["AAA"]
    rebalance_frequency = "{frequency}"
    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        return Allocation({{"AAA": 1.0}})

strategy = TestStrategy()
"""
    baseline.write_text(strategy_code.format(name="[Production] Baseline", frequency="monthly"))
    weekly.write_text(strategy_code.format(name="[Pilot] Weekly", frequency="weekly"))

    dates = pd.bdate_range("2020-01-01", periods=320)
    fake_data = PriceData(
        prices=pd.DataFrame({"AAA": 100.0 + np.arange(len(dates))}, index=dates),
        currency={"AAA": "EUR"},
    )

    def fake_yahoo(**kwargs):
        return fake_data

    monkeypatch.setattr("backtest.meta_promotion.DataLoader.yahoo", fake_yahoo)

    payload = run_meta_promotion_report(
        strategy_paths=[str(baseline), str(weekly)],
        baseline_path=str(baseline),
        start="2020-01-01",
        output_dir=tmp_path / "reports",
        tax_enabled=False,
        metric_basis="gross",
        brokers=["trade_republic"],
        validate=False,
    )

    assert payload["strategies"][0]["metrics"]["rebalance_frequency"] == "monthly"
    assert payload["strategies"][1]["metrics"]["rebalance_frequency"] == "weekly"
    assert payload["config"]["tail_risk_gate_basis"] == "daily"
    assert payload["strategies"][0]["tail_risk_gate_basis"] == "daily"
    assert "max_drawdown_rebalance" in payload["strategies"][0]["metrics"]
    assert "max_drawdown_daily" in payload["strategies"][0]["metrics"]
    assert "rolling_3y_rebalance" in payload["strategies"][0]
    assert "rolling_3y_daily" in payload["strategies"][0]
    assert (tmp_path / "reports").exists()
    assert Path(payload["paths"]["json"]).exists()
    assert Path(payload["paths"]["markdown"]).exists()


def test_meta_promotion_tail_gate_defaults_to_daily_mtm(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.py"
    candidate = tmp_path / "candidate.py"
    strategy_code = """
from datetime import date
import pandas as pd
from backtest.strategy import Allocation, Strategy

class TestStrategy(Strategy):
    name = "{name}"
    assets = ["{asset}"]
    rebalance_frequency = "yearly"
    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        return Allocation({{"{asset}": 1.0}})

strategy = TestStrategy()
"""
    baseline.write_text(strategy_code.format(name="[Production] Baseline", asset="AAA"))
    candidate.write_text(strategy_code.format(name="[Pilot] Candidate", asset="BBB"))

    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    baseline_prices = pd.Series(100.0, index=dates)
    baseline_prices.loc[pd.Timestamp("2022-06-15")] = 50.0
    candidate_prices = pd.Series(100.0, index=dates)
    fake_data = PriceData(
        prices=pd.DataFrame({"AAA": baseline_prices, "BBB": candidate_prices}, index=dates),
        currency={"AAA": "EUR", "BBB": "EUR"},
    )

    def fake_yahoo(**kwargs):
        return fake_data

    monkeypatch.setattr("backtest.meta_promotion.DataLoader.yahoo", fake_yahoo)

    payload = run_meta_promotion_report(
        strategy_paths=[str(baseline), str(candidate)],
        baseline_path=str(baseline),
        start="2020-01-01",
        output_dir=tmp_path / "reports",
        tax_enabled=False,
        metric_basis="gross",
        brokers=["trade_republic"],
        validate=False,
    )

    baseline_row, candidate_row = payload["strategies"]
    diag = candidate_row["vs_baseline"]

    assert payload["config"]["tail_risk_gate_basis"] == "daily"
    assert diag["gate_basis"] == "daily"
    assert diag["maxdd_delta_pp"] == diag["maxdd_delta_daily_pp"]
    assert diag["maxdd_delta_daily_pp"] > 40.0
    assert abs(diag["maxdd_delta_rebalance_pp"]) < 1e-9
    assert candidate_row["metrics"]["max_drawdown_daily"] > baseline_row["metrics"]["max_drawdown_daily"]
    assert "rolling_3y_rebalance" in diag
    assert "rolling_3y_daily" in diag


def _write_fake_artifact(base_dir: Path, artifact_id: str, generated_at: str) -> Path:
    """Write a minimal but realistically-shaped promotion artifact for discovery tests."""
    artifact_dir = base_dir / generated_at[:10].replace("-", "") / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifact_dir / "promotion_report.json"
    md_path = artifact_dir / "promotion_report.md"
    payload = {
        "artifact_id": artifact_id,
        "generated_at": generated_at,
        "config": {
            "baseline_path": "strategies/foo.py",
            "research_proxy_mode": "live",
        },
        "strategies": [
            {"role": "baseline", "path": "strategies/foo.py"},
            {"role": "candidate", "path": "strategies/bar.py"},
        ],
        "paths": {"json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(f"# Promotion Report {artifact_id}\n", encoding="utf-8")
    return json_path


def test_find_latest_promotion_artifact_returns_newest(tmp_path):
    older = _write_fake_artifact(tmp_path, "older", "2026-05-15T10:00:00+00:00")
    newer = _write_fake_artifact(tmp_path, "newer", "2026-05-16T12:00:00+00:00")
    import os, time
    # Ensure mtime increases monotonically — otherwise identical on fast tests.
    os.utime(older, (time.time() - 60, time.time() - 60))
    os.utime(newer, (time.time(), time.time()))

    latest = find_latest_promotion_artifact(base_dir=tmp_path)

    assert latest is not None
    assert latest["artifact_id"] == "newer"


def test_find_latest_promotion_artifact_returns_none_for_empty_dir(tmp_path):
    assert find_latest_promotion_artifact(base_dir=tmp_path / "leer") is None


def test_find_promotion_artifact_by_id_round_trip(tmp_path):
    _write_fake_artifact(tmp_path, "abc123", "2026-05-16T10:00:00+00:00")

    found = find_promotion_artifact_by_id("abc123", base_dir=tmp_path)
    missing = find_promotion_artifact_by_id("does-not-exist", base_dir=tmp_path)

    assert found is not None and found["artifact_id"] == "abc123"
    assert missing is None


def test_list_promotion_artifacts_returns_metadata_only(tmp_path):
    _write_fake_artifact(tmp_path, "a", "2026-05-15T10:00:00+00:00")
    _write_fake_artifact(tmp_path, "b", "2026-05-16T10:00:00+00:00")

    rows = list_promotion_artifacts(base_dir=tmp_path, limit=10)

    assert len(rows) == 2
    ids = {row["artifact_id"] for row in rows}
    assert ids == {"a", "b"}
    # Metadata shape: no full strategy payload, but count + baseline.
    for row in rows:
        assert row["candidate_count"] == 1
        assert row["strategy_count"] == 2
        assert row["baseline_path"] == "strategies/foo.py"
        assert row["research_proxy_mode"] == "live"


def test_read_promotion_markdown_by_id_returns_text(tmp_path):
    _write_fake_artifact(tmp_path, "md1", "2026-05-16T10:00:00+00:00")

    text = read_promotion_markdown_by_id("md1", base_dir=tmp_path)
    missing = read_promotion_markdown_by_id("nope", base_dir=tmp_path)

    assert text is not None and "md1" in text
    assert missing is None
