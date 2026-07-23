from pathlib import Path

from backtest.web.services.strategies import list_strategies


def test_strategy_discovery_prioritizes_production_then_pilot_then_research():
    """Sort order: Production -> Pilot -> Research -> Benchmark -> Experimental."""
    strategies = list_strategies()
    file_names = [Path(item["file_path"]).name for item in strategies]

    # Tier 1: the production default sits at the very top.
    assert file_names[0] == "levered_etf_momentum_sticky.py"

    # Tier 2: pilot / promotion candidates come directly after.
    assert file_names[1:3] == [
        "sticky_levered_vol_targeted.py",
        "sticky_levered_cascade.py",
    ]

    # Tier 3 (Research) starts with the audit-demoted SectorAware VolTarget.
    assert file_names[3] == "sticky_levered_vol_targeted_sector_aware.py"

    # Research variants stay in Tier 3 (before the benchmarks).
    for research_file in (
        "sticky_levered_tax_aware.py",
        "sticky_levered_entry_staged.py",
        "levered_etf_momentum_sticky_adaptive_v2.py",
    ):
        assert file_names.index(research_file) < file_names.index("buy_and_hold.py")

    # The regime vol gate is Research: after the pilot block, before the
    # benchmarks -- not dumped at the experimental end.
    pct120 = file_names.index("sticky_levered_vol_targeted_pct120.py")
    assert pct120 > file_names.index("sticky_levered_cascade.py")
    assert pct120 < file_names.index("buy_and_hold.py")

    # Tier 4 (Benchmarks) after Tier 3 (Research).
    assert file_names.index("buy_and_hold.py") > file_names.index(
        "levered_etf_momentum_sticky_adaptive_v2.py"
    )
    # Classic reference strategies live in the benchmark block.
    for benchmark_file in ("classic_60_40.py", "all_weather.py", "dual_momentum.py"):
        assert file_names.index(benchmark_file) > file_names.index("buy_and_hold.py")

    # Tier 5 (Experimental) below the benchmarks.
    assert file_names.index("levered_5x_momentum_guard.py") > file_names.index(
        "sector_rotation_momentum.py"
    )


def test_strategy_names_carry_tier_prefix():
    """Every visible strategy carries a tier prefix in its name."""
    valid_tiers = {"[Production]", "[Pilot]", "[Research]", "[Benchmark]", "[Experimental]"}
    for item in list_strategies():
        prefix = item["name"].split(" ", 1)[0]
        assert prefix in valid_tiers, (
            f"{item['file_name']}: name '{item['name']}' has no tier prefix; "
            f"expected one of: {valid_tiers}"
        )
