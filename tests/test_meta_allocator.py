from datetime import date

import pandas as pd
import pytest

from backtest.meta_allocator import MetaAllocatorStrategy


def _sample_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=300)
    steps = pd.Series(range(len(idx)), index=idx, dtype=float)

    spy = 100.0 + 0.3 * steps + 3.0 * (steps % 7)
    bnd = 80.0 + 0.05 * steps + 0.2 * (steps % 5)
    efa = 90.0 + 0.15 * steps + 2.0 * ((steps + 2) % 6)

    return pd.DataFrame(
        {
            "SPY": spy.values,
            "BND": bnd.values,
            "EFA": efa.values,
        },
        index=idx,
    )


def test_meta_allocator_equal_weight_combines_member_allocations():
    data = _sample_prices()
    strategy = MetaAllocatorStrategy(
        members_csv="classic_60_40,buy_and_hold",
        allocator="equal_weight",
        lookback_days=63,
    )

    alloc = strategy.signal(date(2021, 3, 1), data)

    assert alloc.get("SPY") > 0
    assert alloc.get("BND") > 0
    assert abs(sum(alloc.weights.values()) - 1.0) < 1e-9
    # 60/40 sleeve + SPY-only sleeve => SPY must dominate BND in equal mode.
    assert alloc.get("SPY") > alloc.get("BND")


def test_meta_allocator_risk_budgeted_tilts_toward_lower_risk_member():
    data = _sample_prices()
    equal = MetaAllocatorStrategy(
        members_csv="classic_60_40,buy_and_hold",
        allocator="equal_weight",
        lookback_days=63,
    )
    risk_budgeted = MetaAllocatorStrategy(
        members_csv="classic_60_40,buy_and_hold",
        allocator="risk_budgeted",
        lookback_days=63,
    )

    alloc_equal = equal.signal(date(2021, 3, 1), data)
    alloc_risk = risk_budgeted.signal(date(2021, 3, 1), data)

    assert abs(sum(alloc_risk.weights.values()) - 1.0) < 1e-9
    # Classic 60/40 has lower realized sleeve risk than SPY-only in this fixture,
    # so risk-budgeted blend should increase bond share vs equal mode.
    assert alloc_risk.get("BND") > alloc_equal.get("BND")


def test_meta_allocator_rejects_self_reference_member():
    with pytest.raises(ValueError):
        MetaAllocatorStrategy(
            members_csv="meta_allocator",
            allocator="equal_weight",
        )
