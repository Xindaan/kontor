from datetime import date

import pandas as pd
import pytest

from backtest.backtester import BacktestConfig, Backtester
from backtest.data import PriceData
from backtest.risk.exposure_policy import ExposurePolicyEngine, required_assets_from_raw
from backtest.strategy import Allocation, Strategy


class _AlwaysA4(Strategy):
    name = "Always A4ANZ5"
    assets = ["3SEM.L", "VVSM.DE", "SXR8.DE"]
    rebalance_frequency = "daily"

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        _ = (current_date, data)
        return Allocation({"3SEM.L": 1.0})


def _frame(raw_tail, proxy_tail=None):
    dates = pd.bdate_range("2026-01-01", periods=len(raw_tail))
    if proxy_tail is None:
        proxy_tail = [100.0 + i * 0.2 for i in range(len(raw_tail))]
    return pd.DataFrame(
        {
            "3SEM.L": raw_tail,
            "VVSM.DE": proxy_tail,
            "SXR8.DE": [100.0 + i * 0.1 for i in range(len(raw_tail))],
        },
        index=dates,
    )


def _policy():
    return {
        "enabled": True,
        "profile": "trade_republic",
        "core_asset": "A0YEDG",
    }


def test_required_assets_from_raw_resolves_default_tr_profile():
    assets = set(required_assets_from_raw(_policy()))
    assert {"3SEM.L", "VVSM.DE", "SXR8.DE"}.issubset(assets)


def test_required_assets_include_us_and_maxblue_challenger_mappings():
    us_assets = set(required_assets_from_raw({"enabled": True, "profile": "us"}))
    maxblue_assets = set(required_assets_from_raw({"enabled": True, "profile": "maxblue"}))

    assert {"ERX", "XLE", "FAS", "XLF"}.issubset(us_assets)
    assert {"3XEE.L", "QDVF.DE", "3XFE.L", "QDVH.DE"}.issubset(maxblue_assets)


def test_disabled_policy_returns_allocation_unchanged():
    engine = ExposurePolicyEngine.from_raw(None)
    allocation = Allocation({"3SEM.L": 1.0})

    decision = engine.apply(allocation, _frame([100.0] * 30))

    assert decision.allocation.weights == {"3SEM.L": 1.0}
    assert decision.exposure_state == "disabled"


def test_normal_state_keeps_3x_allocation():
    engine = ExposurePolicyEngine.from_raw(_policy())
    prices = _frame([100.0 + i for i in range(30)])

    decision = engine.apply(Allocation({"3SEM.L": 1.0}), prices)

    assert decision.exposure_state == "normal"
    assert decision.allocation.weights == {"3SEM.L": 1.0}


def test_level1_shock_delevers_to_1x_proxy():
    engine = ExposurePolicyEngine.from_raw(_policy())
    raw = [100.0] * 24 + [100.0, 97.0, 94.0, 90.0, 87.0, 84.0]
    proxy = [100.0 + i * 0.2 for i in range(30)]

    decision = engine.apply(Allocation({"3SEM.L": 1.0}), _frame(raw, proxy))

    assert decision.exposure_state == "deleveraged_1x"
    assert decision.allocation.weights == {"VVSM.DE": 1.0}
    assert "Level-1 shock" in (decision.fallback_reason or "")


def test_level2_shock_uses_core_safe_asset():
    engine = ExposurePolicyEngine.from_raw(_policy())
    raw = [100.0] * 24 + [100.0, 96.0, 91.0, 86.0, 80.0, 74.0]
    proxy = [100.0] * 8 + [98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 86.0, 84.0, 82.0, 80.0, 78.0, 76.0, 74.0, 72.0, 70.0, 68.0, 66.0, 64.0, 62.0, 60.0, 58.0, 56.0]

    decision = engine.apply(Allocation({"3SEM.L": 1.0}), _frame(raw, proxy))

    assert decision.exposure_state == "safe"
    assert decision.allocation.weights == {"SXR8.DE": 1.0}


def test_release_requires_confirmation_periods():
    engine = ExposurePolicyEngine.from_raw(_policy())
    shock = [100.0] * 24 + [100.0, 97.0, 94.0, 90.0, 87.0, 84.0]
    recovery = [100.0 + i * 0.4 for i in range(30)]

    first = engine.apply(Allocation({"3SEM.L": 1.0}), _frame(shock))
    second = engine.apply(Allocation({"3SEM.L": 1.0}), _frame(recovery))
    third = engine.apply(Allocation({"3SEM.L": 1.0}), _frame(recovery))

    assert first.exposure_state == "deleveraged_1x"
    assert second.exposure_state == "deleveraged_1x"
    assert third.exposure_state == "normal"
    assert third.allocation.weights == {"3SEM.L": 1.0}


def test_missing_fallback_data_blocks_replacement_without_improvising():
    engine = ExposurePolicyEngine.from_raw(_policy())
    raw = [100.0] * 24 + [100.0, 97.0, 94.0, 90.0, 87.0, 84.0]
    prices = pd.DataFrame({"3SEM.L": raw}, index=pd.bdate_range("2026-01-01", periods=30))

    decision = engine.apply(Allocation({"3SEM.L": 1.0}), prices)

    assert decision.exposure_state == "blocked"
    assert decision.allocation.weights == {"3SEM.L": 1.0}
    assert "no loaded price data" in (decision.fallback_reason or "")


def test_backtester_without_policy_keeps_legacy_path_unchanged():
    dates = pd.bdate_range("2026-01-01", periods=35)
    prices = pd.DataFrame(
        {
            "3SEM.L": [100.0 + i for i in range(35)],
            "VVSM.DE": [100.0 + i * 0.2 for i in range(35)],
            "SXR8.DE": [100.0 + i * 0.1 for i in range(35)],
        },
        index=dates,
    )
    data = PriceData(prices=prices, currency={ticker: "USD" for ticker in prices.columns})
    config = BacktestConfig(
        initial_capital=10_000.0,
        benchmark=None,
        rebalance_frequency="daily",
        costs_pct=0.0,
        slippage_pct=0.0,
        spread_pct=0.0,
        tax_enabled=False,
        validate=False,
    )

    result = Backtester(_AlwaysA4(), data, config).run()

    assert result.raw_allocations is None
    assert result.exposure_policy_decisions is None
    assert "exposure_policy" in result.constraint_impact
    assert result.constraint_impact["exposure_policy"] == {}


def test_backtester_with_policy_tracks_raw_and_adjusted_allocations():
    dates = pd.bdate_range("2026-01-01", periods=35)
    raw = [100.0] * 29 + [100.0, 96.0, 92.0, 88.0, 84.0, 80.0]
    prices = pd.DataFrame(
        {
            "3SEM.L": raw,
            "VVSM.DE": [100.0 + i * 0.2 for i in range(35)],
            "SXR8.DE": [100.0 + i * 0.1 for i in range(35)],
        },
        index=dates,
    )
    data = PriceData(prices=prices, currency={ticker: "USD" for ticker in prices.columns})
    config = BacktestConfig(
        initial_capital=10_000.0,
        benchmark=None,
        rebalance_frequency="daily",
        costs_pct=0.0,
        slippage_pct=0.0,
        spread_pct=0.0,
        tax_enabled=False,
        exposure_policy=_policy(),
        validate=False,
    )

    result = Backtester(_AlwaysA4(), data, config).run()

    assert result.raw_allocations is not None
    assert result.exposure_policy_decisions is not None
    assert any("VVSM.DE" in row for row in result.allocations.to_dict(orient="records"))
    diagnostics = result.constraint_impact["exposure_policy"]
    assert diagnostics["changed_calls"] >= 1
    assert diagnostics["guard_activations"] >= 1
    comparison = diagnostics["pure_strategy_comparison"]
    assert comparison["pure_cagr"] != comparison["policy_cagr"]
    assert "max_drawdown_delta_pp" in comparison
    assert "recovery_days_improvement_pct" in comparison
