"""Source-integrity regressions: FX chronology, session proof and provenance."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from backtest.data import CurrencyConverter, DataIntegrityError, DataLoader, PriceData, align_fx
from backtest.freshness import GAP_UNEXPLAINED, classify_gap


@pytest.fixture(autouse=True)
def isolated_loader(monkeypatch, tmp_path):
    DataLoader._MEMORY_CACHE.clear()
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)

    def unexpected(*a, **kw):
        raise AssertionError("network access is forbidden in source regression tests")
    monkeypatch.setattr("backtest.data.yf.download", unexpected)
    monkeypatch.setattr("backtest.data.yf.Ticker", unexpected)
    yield
    DataLoader._MEMORY_CACHE.clear()


def test_asof_preserves_quote_on_a_source_only_date():
    # The new quote is dated Sunday. Reindex-then-fill loses it completely.
    fx = pd.Series([1.0, 1.2], index=pd.to_datetime(["2020-05-29", "2020-05-31"]))
    target = pd.to_datetime(["2020-05-29", "2020-06-01"])
    assert align_fx(fx, target).tolist() == [1.0, 1.2]
    assert fx.reindex(target).ffill().iloc[-1] == 1.0  # the old failure


@pytest.mark.parametrize("method", ["price_data", "converter", "helper"])
def test_missing_prior_fx_is_never_backfilled(method):
    ix = pd.to_datetime(["2020-01-02", "2020-01-03"])
    prices = pd.Series([100., 101.], index=ix)
    fx = pd.Series([1.1], index=ix[1:])
    with pytest.raises(ValueError, match="no prior observation"):
        if method == "price_data":
            PriceData(prices.to_frame("SPY"), {"SPY": "USD"}, fx).in_eur()
        elif method == "converter":
            CurrencyConverter().convert(prices, "USD", "EUR", fx)
        else:
            align_fx(fx, ix)


def _history(monkeypatch, times, closes=None, end="2026-09-07 17:30", metadata=True,
             timezone="Europe/Berlin", duplicate=False):
    index = pd.DatetimeIndex(times, tz=timezone)
    hist = pd.DataFrame({"Close": closes or [119.5] * len(index), "Volume": 10.}, index=index)
    if duplicate:
        hist = pd.concat([hist, hist.iloc[[-1]]])
    schedule = pd.DataFrame({"start": [pd.Timestamp("2026-09-07 09:00", tz="Europe/Berlin")],
                             "end": [pd.Timestamp(end, tz="Europe/Berlin")]})
    calls = []

    class Ticker:
        history_metadata = ({"exchangeTimezoneName": "Europe/Berlin", "tradingPeriods": schedule}
                            if metadata else {})

        def history(self, **kwargs):
            calls.append(kwargs)
            return hist
    monkeypatch.setattr("backtest.data.yf.Ticker", lambda ticker: Ticker())
    return calls


def test_two_morning_bars_do_not_prove_a_completed_session(monkeypatch):
    _history(monkeypatch, ["2026-09-07 09:00", "2026-09-07 10:00"])
    assert DataLoader._intraday_sessions("X") == {}


@pytest.mark.parametrize("missing", ["metadata", "timezone", "duplicate"])
def test_ambiguous_intraday_evidence_is_rejected(monkeypatch, missing):
    _history(monkeypatch, ["2026-09-07 09:00", "2026-09-07 17:00"],
             metadata=missing != "metadata",
             timezone=None if missing == "timezone" else "Europe/Berlin",
             duplicate=missing == "duplicate")
    assert DataLoader._intraday_sessions("X") == {}


def test_sorted_final_regular_bar_and_adjustment_mode_are_used(monkeypatch):
    calls = _history(monkeypatch, ["2026-09-07 17:45", "2026-09-07 17:00", "2026-09-07 09:00"],
                     closes=[999., 119.5, 119.0])
    sessions = DataLoader._intraday_sessions("X", auto_adjust=False)
    close = sessions[dt.date(2026, 9, 7)]
    assert close.value == 119.5 and close.prints == 2
    assert close.covers_close()
    assert calls[0]["auto_adjust"] is False and calls[0]["prepost"] is False


def test_short_session_end_is_taken_from_the_dated_schedule(monkeypatch):
    _history(monkeypatch, ["2026-09-07 09:00", "2026-09-07 12:00"], end="2026-09-07 13:00")
    assert dt.date(2026, 9, 7) in DataLoader._intraday_sessions("X")


def _daily(monkeypatch, closes, volumes, days):
    calls = []

    def download(*a, **kw):
        calls.append(kw)
        return pd.DataFrame({"Close": closes, "Volume": volumes}, index=pd.to_datetime(days))
    monkeypatch.setattr("backtest.data.yf.download", download)
    return calls


def test_inner_close_gap_with_trading_evidence_blocks_even_skip_failed(monkeypatch, tmp_path):
    _daily(monkeypatch, [100., np.nan, 102.], [10., 20., 30.],
           ["2020-03-02", "2020-03-03", "2020-03-04"])
    with pytest.raises(DataIntegrityError, match="2020-03-03"):
        DataLoader.yahoo(["SPY"], "2020-03-01", "2020-03-05", currency="USD",
                         cache=True, skip_failed=True, validate=False)
    assert not list(tmp_path.glob("*.csv"))
    assert not DataLoader._MEMORY_CACHE


def test_repaired_tail_is_not_persisted_and_next_call_refetches(monkeypatch, tmp_path):
    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 8, 8)
    monkeypatch.setattr("backtest.data.datetime", Clock)
    calls = _daily(monkeypatch, [119.8, np.nan], [10., 20.], ["2026-09-04", "2026-09-07"])
    _history(monkeypatch, ["2026-09-07 09:00", "2026-09-07 17:00"])
    for _ in range(2):
        result = DataLoader.yahoo(["SPY"], "2026-09-01", "2026-09-09", currency="USD",
                                  cache=True, validate=False)
        assert result.prices.loc["2026-09-07", "SPY"] == pytest.approx(119.5)
        assert "provisional_close_source" in result.prices.attrs
    assert len(calls) == 2 and not DataLoader._MEMORY_CACHE
    assert not list(tmp_path.glob("*.csv"))


def test_fx_last_print_survives_cache_and_does_not_follow_fill(monkeypatch):
    _daily(monkeypatch, [100., 101., 102.], [10., 20., 30.],
           ["2020-03-02", "2020-03-03", "2020-03-04"])
    raw_fx = pd.Series([1.1, 1.2], index=pd.to_datetime(["2020-03-02", "2020-03-05"]))
    monkeypatch.setattr(DataLoader, "get_fx_rates", classmethod(lambda cls, *a, **kw: raw_fx))
    kwargs = dict(tickers=["SPY"], start="2020-03-01", end="2020-03-05", currency="EUR",
                  cache=False, validate=False)
    first = DataLoader.yahoo(**kwargs)
    second = DataLoader.yahoo(**kwargs)
    for result in (first, second):
        assert result.fx_rates.iloc[-1] == 1.1
        assert result.fx_last_print == {"USD": pd.Timestamp("2020-03-02")}
        assert result.last_print == {"SPY": pd.Timestamp("2020-03-04")}
        assert result.resample("daily").fx_last_print == result.fx_last_print
        assert result.filter_dates(end_date="2020-03-03").fx_last_print == result.fx_last_print
        assert result.filter_dates(end_date="2020-03-01").fx_last_print == {}


def test_all_sources_failed_returns_empty_frame_instead_of_crashing(monkeypatch):
    # The currency of a failed ticker still sits in the currency map; the FX
    # freshness check must not compare against the max of an EMPTY index (NaN).
    monkeypatch.setattr("backtest.data.yf.download", lambda *a, **kw: pd.DataFrame())
    raw_fx = pd.Series([1.1, 1.2], index=pd.to_datetime(["2020-03-02", "2020-03-05"]))
    monkeypatch.setattr(DataLoader, "get_fx_rates", classmethod(lambda cls, *a, **kw: raw_fx))
    result = DataLoader.yahoo(["SPY"], "2020-03-01", "2020-03-05", currency="EUR",
                              cache=False, skip_failed=True, validate=False)
    assert "SPY" not in result.prices.columns
    assert result.fx_last_print == {}


def test_later_print_does_not_turn_a_weekday_feed_outage_into_a_holiday():
    evidence = {"good": ["2026-09-01", "2026-09-02", "2026-09-03"],
                "broken": ["2026-09-01", "2026-09-03"]}
    report = classify_gap(evidence, since=dt.date(2026, 9, 1), today=dt.date(2026, 9, 4))
    assert report.case == GAP_UNEXPLAINED and report.retry_worthwhile
    assert report.days[0].closed == ()
    assert report.days[0].skipped == ("broken",)


def test_empty_source_is_not_dropped_from_gap_diagnosis():
    report = classify_gap({"good": ["2026-09-01"], "empty": []}, today=dt.date(2026, 9, 2))
    assert report.retry_worthwhile and report.sources == ("empty", "good")
