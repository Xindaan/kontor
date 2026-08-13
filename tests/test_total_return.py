"""Golden-truth tests for the total-return path.

Proves the bug class "`.prices` extraction after load_dividends=True":
the old path (returns from `.prices` alone) massively understates
distributing instruments (DBMF 2022: +12.8% instead of ~+21.5% TR) --
the golden tests here demonstrably fail on the old path (see
test_golden_dbmf_2022_*). Second dimension: `.prices` is native,
never EUR -- the EUR path is checked against hand-computed values.

Fixtures: tests/fixtures/total_return/, frozen 2026-07-14 from the
Yahoo cache (window 2021-12-15..2023-01-10). External anchors:
- DBMF 2022 total return (market-price basis) ~ +21.5%; official NAV TR
  +23.1% (the gap is price-vs-NAV, not methodology). Price-only: +12.8%.
- SPY 2022: price return -19.5%, total return -18.2% -> gap ~ 1.3pp
  (dividend yield).
No live Yahoo calls: all tests run offline against the fixtures.
"""

from pathlib import Path

import pandas as pd
import pytest

from backtest.data import PriceData

FIXTURES = Path(__file__).parent / "fixtures" / "total_return"


def _load(name: str) -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / name, index_col=0, parse_dates=True)
    return df


def _year_2022_return(series: pd.Series) -> float:
    start = series.asof(pd.Timestamp("2021-12-31"))
    end = series.asof(pd.Timestamp("2022-12-30"))
    return float(end / start - 1.0)


# ---------------------------------------------------------------------------
# Synthetic semantics tests
# ---------------------------------------------------------------------------

def test_reinvest_on_ex_date_synthetic():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=idx)
    dividends = pd.DataFrame({"A": [0.0, 1.0, 0.0]}, index=idx)
    data = PriceData(prices=prices, currency={"A": "EUR"}, dividends=dividends)

    tr = data.total_return_prices()["A"]
    # 1 EUR distribution on 100 EUR reinvested on the ex-date = +1%.
    assert tr.iloc[0] == pytest.approx(100.0)
    assert tr.iloc[1] == pytest.approx(101.0)
    assert tr.iloc[2] == pytest.approx(101.0)
    # Old path (price-only) doesn't see the distribution.
    assert prices["A"].iloc[2] == pytest.approx(100.0)


def test_identity_without_dividends():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"A": [100.0, 102.0]}, index=idx)
    data = PriceData(prices=prices, currency={"A": "EUR"})

    pd.testing.assert_frame_equal(data.total_return_prices(), prices)


def test_leading_nan_ticker_starts_at_first_price():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.DataFrame(
        {"A": [100.0, 100.0, 100.0], "B": [float("nan"), 50.0, 51.0]}, index=idx
    )
    dividends = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 1.0]}, index=idx)
    data = PriceData(prices=prices, currency={"A": "EUR", "B": "EUR"}, dividends=dividends)

    tr = data.total_return_prices()
    assert pd.isna(tr["B"].iloc[0])
    assert tr["B"].iloc[1] == pytest.approx(50.0)
    # 51 price + 1 dividend on a base of 50 -> +4%.
    assert tr["B"].iloc[2] == pytest.approx(52.0)


def test_eur_conversion_reinvests_in_eur():
    """Golden: USD prices + EURUSD -> hand-computed EUR TR.

    USD: 100 -> 102, dividend 1.25 USD on day 2, EURUSD constant at 1.25.
    EUR prices: 80 -> 81.6; EUR dividend: 1.00.
    EUR TR day 2 = 80 * (81.6 + 1.0) / 80 = 82.6.
    """
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"A": [100.0, 102.0]}, index=idx)
    dividends = pd.DataFrame({"A": [0.0, 1.25]}, index=idx)
    fx = pd.Series([1.25, 1.25], index=idx)
    data = PriceData(prices=prices, currency={"A": "USD"}, fx_rates=fx, dividends=dividends)

    tr_eur = data.total_return_prices(currency="EUR")["A"]
    assert tr_eur.iloc[0] == pytest.approx(80.0)
    assert tr_eur.iloc[1] == pytest.approx(82.6)

    # Cross-check: native TR series stays in USD (103.0 = 102 + 1.25
    # reinvested on 100 -> 100 * 103.25/100 = 103.25).
    tr_native = data.total_return_prices()["A"]
    assert tr_native.iloc[1] == pytest.approx(103.25)


def test_dividend_on_missing_price_day_lands_on_next_trading_day():
    """Ex-date missing from the price index (e.g. after align="intersection"):
    the payment must not be lost, it lands on the next day instead."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-04"])  # 01-03 missing
    prices = pd.DataFrame({"A": [100.0, 100.0]}, index=idx)
    dividends = pd.DataFrame({"A": [2.0]}, index=pd.to_datetime(["2024-01-03"]))
    data = PriceData(prices=prices, currency={"A": "EUR"}, dividends=dividends)

    tr = data.total_return_prices()["A"]
    assert tr.iloc[1] == pytest.approx(102.0)


def test_intraday_dividend_timestamps_are_normalized():
    """Yahoo tz conversion leaves intraday remnants (05:00) on
    ex-dates; __post_init__ normalizes to the calendar day."""
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    prices = pd.DataFrame({"A": [100.0, 100.0]}, index=idx)
    dividends = pd.DataFrame(
        {"A": [1.0]}, index=pd.to_datetime(["2024-01-03 05:00:00"])
    )
    data = PriceData(prices=prices, currency={"A": "EUR"}, dividends=dividends)

    assert data.dividends.index[0] == pd.Timestamp("2024-01-03")
    tr = data.total_return_prices()["A"]
    assert tr.iloc[1] == pytest.approx(101.0)


def test_unsupported_target_currency_raises():
    idx = pd.to_datetime(["2024-01-02"])
    data = PriceData(prices=pd.DataFrame({"A": [1.0]}, index=idx), currency={"A": "EUR"})
    with pytest.raises(ValueError, match="Unsupported target currency"):
        data.total_return_prices(currency="USD")


# ---------------------------------------------------------------------------
# Golden truth against frozen market data
# ---------------------------------------------------------------------------

def test_golden_dbmf_2022_total_return_vs_price_only():
    """Red-on-old proof: the old path (+12.8% price-only) sits >6pp
    below the real TR (~+21.5%); the new path hits the anchor."""
    unadj = _load("dbmf_unadj.csv")
    dividends = _load("dbmf_dividends.csv")
    data = PriceData(prices=unadj, currency={"DBMF": "USD"}, dividends=dividends)

    r_tr = _year_2022_return(data.total_return_prices()["DBMF"])
    r_price_only = _year_2022_return(unadj["DBMF"])

    # External anchor: DBMF 2022 TR (market-price basis) ~ +21.5%.
    assert 0.19 < r_tr < 0.24
    # The old path fails exactly at this threshold (+12.8%).
    assert r_price_only < 0.19
    assert r_tr - r_price_only > 0.06


def test_golden_dbmf_2022_converges_to_adjusted_close():
    """3-path convergence check: unadjusted price + dividend reinvestment
    must match the adjusted close."""
    unadj = _load("dbmf_unadj.csv")
    dividends = _load("dbmf_dividends.csv")
    adj = _load("dbmf_adjclose.csv")
    data = PriceData(prices=unadj, currency={"DBMF": "USD"}, dividends=dividends)

    r_tr = _year_2022_return(data.total_return_prices()["DBMF"])
    r_adj = _year_2022_return(adj["DBMF"])
    assert abs(r_tr - r_adj) < 0.005  # <= 0.5pp


def test_golden_spy_2022_tr_gap_matches_dividend_yield():
    """SPY 2022: TR - PR ~ dividend yield (~1.3pp), TR > PR."""
    unadj = _load("spy_unadj.csv")
    dividends = _load("spy_dividends.csv")
    data = PriceData(prices=unadj, currency={"SPY": "USD"}, dividends=dividends)

    r_tr = _year_2022_return(data.total_return_prices()["SPY"])
    r_pr = _year_2022_return(unadj["SPY"])
    gap = r_tr - r_pr
    assert r_tr > r_pr
    assert 0.008 < gap < 0.025
