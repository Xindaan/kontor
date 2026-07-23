"""Tests for the AI infrastructure basket (T-0413).

Covers:
- Teilfreistellung tax map (additive, off-by-default, bit-identical when None;
  single stock without Teilfreistellung pays more tax).
- Segment loader + PIT/U-B universe builder (survivorship).
- MetricsCalculator.segment_exposure + segment_cap_report.
- AIInfraBasket strategy smoke test (M1/M2/M3, semi_cap enforcement).
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from backtest.strategy import Strategy, Allocation
from backtest.data import PriceData
from backtest.backtester import Backtester, BacktestConfig
from backtest.metrics import MetricsCalculator
from backtest.research import segments as S


# --------------------------------------------------------------------------- #
# Tax map
# --------------------------------------------------------------------------- #
class _SingleHold(Strategy):
    name = "SingleHold"
    assets = ["AAA"]

    def signal(self, d: date, data: pd.DataFrame) -> Allocation:
        return Allocation({"AAA": 1.0})


def _rising_data():
    dates = pd.date_range("2020-01-01", periods=24, freq="ME")
    prices = pd.DataFrame({"AAA": [100 + i * 5 for i in range(24)]}, index=dates)
    return PriceData(prices=prices, currency={"AAA": "USD"}, fx_rates=None)


def _run(equity_fund_map):
    cfg = BacktestConfig(
        benchmark=None, metric_basis="net_liquidation",
        actual_liquidation_at_end=False, equity_fund_map=equity_fund_map,
    )
    return Backtester(_SingleHold(), _rising_data(), cfg).run()


def test_instrument_tax_flags_resolver():
    bt = Backtester(_SingleHold(), _rising_data(), BacktestConfig(benchmark=None))
    # Map None -> legacy (ETF/fund, Teilfreistellung).
    assert bt._instrument_tax_flags("AAA") == ("general", True)
    # Map with single stock -> "equity", no Teilfreistellung.
    bt.config.equity_fund_map = {"AAA": False}
    assert bt._instrument_tax_flags("AAA") == ("equity", False)
    # Ticker not in map -> legacy fallback.
    assert bt._instrument_tax_flags("ZZZ") == ("general", True)


def test_equity_fund_map_none_is_bit_identical_to_true():
    """map=None and map={AAA:True} must produce identical after-tax curves."""
    none_run = _run(None)
    true_run = _run({"AAA": True})
    assert none_run.tax_summary.final_value_net_liquidation == pytest.approx(
        true_run.tax_summary.final_value_net_liquidation
    )
    assert none_run.tax_summary.tax_paid_liquidation == pytest.approx(
        true_run.tax_summary.tax_paid_liquidation
    )


def test_single_stock_pays_more_tax_than_equity_fund():
    """Single stock (no 30% Teilfreistellung) -> more tax, less net."""
    fund = _run(None)                      # 30% Teilfreistellung
    stock = _run({"AAA": False})           # full 26.375%
    assert stock.tax_summary.tax_paid_liquidation > fund.tax_summary.tax_paid_liquidation
    assert (
        stock.tax_summary.final_value_net_liquidation
        < fund.tax_summary.final_value_net_liquidation
    )


# --------------------------------------------------------------------------- #
# Segment loader + builder
# --------------------------------------------------------------------------- #
def test_segment_map_and_cap_groups():
    sm = S.load_segment_map()
    cg = S.load_cap_groups()
    assert len(sm) > 50
    assert sm["NVDA"] == "compute"
    assert cg["NVDA"] == "semi"
    assert cg["ETN"] == "diversifier"
    semi = set(S.segment_tickers(cap_group="semi"))
    div = set(S.segment_tickers(cap_group="diversifier"))
    assert "NVDA" in semi and "ETN" in div
    assert semi.isdisjoint(div)


def test_build_pit_universe_is_survivorship_complete(tmp_path):
    out = tmp_path / "ai_infra_pit.csv"
    df = S.build_ai_infra_pit_universe(out_csv=out)
    assert list(df.columns) == ["as_of", "ticker"]
    assert out.exists()
    # XLNX (acquired by AMD in 2022) must be present historically, then gone later.
    by_date = df.groupby("as_of")["ticker"].apply(set)
    d2021 = max(d for d in by_date.index if d <= "2021-06-01")
    d2024 = max(d for d in by_date.index if d <= "2024-06-01")
    assert "XLNX" in by_date[d2021]
    assert "XLNX" not in by_date[d2024]
    # Diversifier names are included via NDX/SP500 membership.
    assert "ETN" in by_date[d2024]


def test_build_ub_universe_is_static(tmp_path):
    out = tmp_path / "ub.csv"
    df = S.build_ai_infra_ub_universe(out_csv=out, tickers=["NVDA", "XLNX", "ETN"])
    by_date = df.groupby("as_of")["ticker"].apply(set)
    # Static membership: every snapshot has the full list.
    sample = list(by_date.index)[len(by_date) // 2]
    assert by_date[sample] == {"NVDA", "XLNX", "ETN"}


# --------------------------------------------------------------------------- #
# segment_exposure metric
# --------------------------------------------------------------------------- #
def test_segment_exposure_and_cap_report():
    alloc = pd.DataFrame(
        {"NVDA": [0.5, 0.2], "ETN": [0.3, 0.4], "cash": [0.2, 0.4]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
    )
    seg_map = {"NVDA": "compute", "ETN": "power"}
    exp = MetricsCalculator.segment_exposure(alloc, seg_map)
    assert exp.loc["2020-01-31", "compute"] == pytest.approx(0.5)
    assert exp.loc["2020-01-31", "power"] == pytest.approx(0.3)
    assert "cash" not in exp.columns  # include_cash default False

    breaches = S.segment_cap_report(alloc, seg_map, {"compute": 0.4})
    # 0.5 > 0.4 on 2020-01-31 -> breach; 0.2 <= 0.4 on 2020-02-29 -> no breach.
    assert len(breaches) == 1
    assert breaches.index[0] == pd.Timestamp("2020-01-31")


# --------------------------------------------------------------------------- #
# Strategy smoke test
# --------------------------------------------------------------------------- #
@pytest.fixture
def mini_universe(tmp_path):
    seg_csv = tmp_path / "seg.csv"
    seg_csv.write_text(
        "ticker,segment,cap_group,status,gics_hint\n"
        "S1,compute,semi,active,x\n"
        "S2,compute,semi,active,x\n"
        "S3,memory,semi,active,x\n"
        "S4,memory,semi,active,x\n"
        "D1,power,diversifier,active,x\n"
        "D2,reit,diversifier,active,x\n"
    )
    uni_csv = tmp_path / "uni.csv"
    rows = ["as_of,ticker"]
    for d in ["2019-01-01", "2020-01-01"]:
        for t in ["S1", "S2", "S3", "S4", "D1", "D2"]:
            rows.append(f"{d},{t}")
    uni_csv.write_text("\n".join(rows) + "\n")
    return str(seg_csv), str(uni_csv)


def _price_panel():
    # ~320 trading days; semis rising sharply, diversifiers moderately.
    idx = pd.date_range("2019-06-01", periods=320, freq="B")
    n = len(idx)
    data = {}
    for t, slope in [("S1", 1.5), ("S2", 1.4), ("S3", 1.3), ("S4", 1.2),
                     ("D1", 0.3), ("D2", 0.25), ("SPY", 0.2)]:
        data[t] = 100 + slope * np.arange(n)
    return pd.DataFrame(data, index=idx)


def test_strategy_smoke_methodics(mini_universe):
    seg_csv, uni_csv = mini_universe
    panel = _price_panel()
    cur = panel.index[-1].date()
    for methodic in ["M1", "M2", "M3"]:
        strat = S_AIInfra(
            universe_csv=uni_csv, segments_csv=seg_csv, methodic=methodic,
            semi_cap=None, min_names=2, top_n=6,
        )
        alloc = strat.signal(cur, panel)
        assert isinstance(alloc, Allocation)
        assert abs(sum(alloc.weights.values()) - 1.0) < 1e-6 or \
            sum(alloc.weights.values()) <= 1.0 + 1e-9
        assert all(w >= 0 for w in alloc.weights.values())


def test_strategy_semi_cap_enforced(mini_universe):
    seg_csv, uni_csv = mini_universe
    panel = _price_panel()
    cur = panel.index[-1].date()
    strat = S_AIInfra(
        universe_csv=uni_csv, segments_csv=seg_csv, methodic="M3",
        weighting="equal", semi_cap=0.35, min_names=2, top_n=6, max_weight=1.0,
    )
    alloc = strat.signal(cur, panel)
    cg = S.load_cap_groups(seg_csv)
    semi_weight = sum(w for t, w in alloc.weights.items() if cg.get(t) == "semi")
    assert semi_weight <= 0.35 + 1e-6
    # Diversifiers must have absorbed the excess.
    div_weight = sum(w for t, w in alloc.weights.items() if cg.get(t) == "diversifier")
    assert div_weight > 0


def test_strategy_falls_back_to_safe_asset_when_no_universe(mini_universe):
    seg_csv, _ = mini_universe
    panel = _price_panel()
    strat = S_AIInfra(
        universe_csv="data/universes/__does_not_exist__.csv",
        segments_csv=seg_csv, safe_asset="SPY",
    )
    alloc = strat.signal(panel.index[-1].date(), panel)
    assert alloc.weights == {"SPY": 1.0}


# Import at the end so the strategy module's sys.path manipulation doesn't
# affect the tests.
from strategies.ai_infra_basket import AIInfraBasket as S_AIInfra  # noqa: E402
