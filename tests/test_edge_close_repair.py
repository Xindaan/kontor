"""Unconsolidated daily close at the edge of a series, repaired from intraday data.

Behavior tests on constructed frames -- network-free, the intraday source is
stubbed. What is checked is the VALUE that comes out, not whether the code knows
a helper.

Motivating case: Yahoo can deliver the last session's daily bar with volume but
``close = NaN``; ``dropna()`` then drops the row and every consumer computes on
the previous session's prices. Whether the empty bar is in the frame at all
depends on the requested window end: one end yields a NaN row, one day later the
row is missing entirely. A repair that only fills an existing row would be green
and useless in the second case -- ``test_missing_row_is_added`` is that case.

All prices, volumes and dates below are invented.
"""

import datetime

import numpy as np
import pandas as pd
import pytest

from backtest.data import DataIntegrityError, DataLoader, IntradaySessionClose

TODAY = datetime.date(2026, 9, 8)          # Tuesday; Monday 07.09. is the gap day
MONDAY = pd.Timestamp("2026-09-07")
TUESDAY = pd.Timestamp("2026-09-08")
FRIDAY = pd.Timestamp("2026-09-04")


def _frames(closes, volumes, days):
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    close_frame = pd.DataFrame({"X": closes}, index=index)
    td = pd.DataFrame({("Close", "X"): closes, ("Volume", "X"): volumes}, index=index)
    td.columns = pd.MultiIndex.from_tuples(td.columns, names=["Price", "Ticker"])
    return close_frame, td


def _run(monkeypatch, closes, volumes, days, sessions, today=TODAY):
    monkeypatch.setattr(
        DataLoader, "_intraday_sessions",
        classmethod(lambda cls, ticker, **kw: {
            day: (value if isinstance(value, IntradaySessionClose) else
                  IntradaySessionClose(value[0], value[1],
                                       pd.Timestamp(str(day) + " 17:00", tz="Europe/Berlin"),
                                       pd.Timestamp(str(day) + " 17:30", tz="Europe/Berlin")))
            for day, value in sessions.items()}),
    )
    close_frame, td = _frames(closes, volumes, days)
    return DataLoader._repair_edge_close(close_frame, td, ["X"], today=today)["X"]


# The live window shape: the Monday row is missing ENTIRELY.
NO_ROW = dict(closes=[120.4, 119.8], volumes=[5000, 5100],
              days=["2026-09-03", "2026-09-04"])
# The other window shape: Monday row present but empty.
EMPTY_ROW = dict(closes=[120.4, 119.8, np.nan], volumes=[5000, 5100, 5300],
                 days=["2026-09-03", "2026-09-04", "2026-09-07"])
MONDAY_SESSION = {datetime.date(2026, 9, 7): (119.5, 8)}


# --- The two main cases ----------------------------------------------------

def test_missing_row_is_added(monkeypatch):
    series = _run(monkeypatch, sessions=MONDAY_SESSION, **NO_ROW)
    assert MONDAY in series.index
    assert series.loc[MONDAY] == pytest.approx(119.5)
    assert series.loc[FRIDAY] == pytest.approx(119.8)
    assert list(series.index) == sorted(series.index)


def test_empty_row_is_filled(monkeypatch):
    series = _run(monkeypatch, sessions=MONDAY_SESSION, **EMPTY_ROW)
    assert series.loc[MONDAY] == pytest.approx(119.5)


def test_running_bar_does_not_hide_the_missing_previous_day(monkeypatch):
    """Today is running and already carries a price -- Monday must still come."""
    series = _run(monkeypatch, closes=[119.8, np.nan, 119.7],
                  volumes=[5100, 5300, 2000],
                  days=["2026-09-04", "2026-09-07", "2026-09-08"],
                  sessions=MONDAY_SESSION)
    assert series.loc[MONDAY] == pytest.approx(119.5)


# --- Guard (1): only the edge ------------------------------------------------

def test_session_before_the_last_close_is_ignored(monkeypatch):
    series = _run(monkeypatch, sessions={datetime.date(2026, 9, 3): (999.0, 8)}, **NO_ROW)
    assert series.loc[pd.Timestamp("2026-09-03")] == pytest.approx(120.4)
    assert MONDAY not in series.index


def test_hole_in_the_middle_is_not_filled(monkeypatch):
    """A NaN BEFORE the last close is a source outage, not a consolidation lag."""
    series = _run(monkeypatch, closes=[120.4, np.nan, 119.6],
                  volumes=[5000, 5100, 5300],
                  days=["2026-09-01", "2026-09-02", "2026-09-04"],
                  sessions={datetime.date(2026, 9, 2): (119.8, 8)})
    assert pd.isna(series.loc[pd.Timestamp("2026-09-02")])


# --- Guard (2): the session must be over -------------------------------------

def test_running_session_is_not_entered_as_a_close(monkeypatch):
    series = _run(monkeypatch, sessions={datetime.date(2026, 9, 8): (119.7, 8)}, **NO_ROW)
    assert TUESDAY not in series.index


# --- Guard (3): there must have been trading ---------------------------------

def test_existing_bar_without_volume_is_not_filled(monkeypatch):
    """Volume 0 = placeholder bar (real holiday) -> do not fill."""
    series = _run(monkeypatch, closes=[120.4, 119.8, np.nan],
                  volumes=[5000, 5100, 0],
                  days=["2026-09-03", "2026-09-04", "2026-09-07"],
                  sessions=MONDAY_SESSION)
    assert pd.isna(series.loc[MONDAY])


# --- Guard (4): plausibility band ----------------------------------------------

def test_scale_error_is_rejected(monkeypatch):
    """Pence instead of pounds (factor 100) must not enter the series."""
    series = _run(monkeypatch, sessions={datetime.date(2026, 9, 7): (11950.0, 8)}, **NO_ROW)
    assert MONDAY not in series.index


def test_brutal_but_real_session_passes(monkeypatch):
    """-45 % days happen for 3x products and must pass -- otherwise the repair
    fails exactly on the most important day."""
    series = _run(monkeypatch, sessions={datetime.date(2026, 9, 7): (119.8 * 0.55, 8)},
                  **NO_ROW)
    assert series.loc[MONDAY] == pytest.approx(119.8 * 0.55)


# --- Guard (5): several prints -------------------------------------------------

def test_single_print_is_not_a_session(monkeypatch):
    series = _run(monkeypatch, sessions={datetime.date(2026, 9, 7): (119.5, 1)}, **NO_ROW)
    assert MONDAY not in series.index


# --- Guard (6): only the live edge -----------------------------------------------

def test_backtest_triggers_no_intraday_fetch(monkeypatch):
    """Series ends in 2024 -> no fetch, no repair (otherwise one call per ticker)."""
    called = []

    def _count(cls, ticker, **kwargs):
        called.append(ticker)
        return MONDAY_SESSION

    monkeypatch.setattr(DataLoader, "_intraday_sessions", classmethod(_count))
    close_frame, td = _frames([100.0, 101.0], [10, 10], ["2024-03-01", "2024-03-04"])
    out = DataLoader._repair_edge_close(close_frame, td, ["X"], today=TODAY)
    assert called == [], "a backtest triggered an intraday fetch"
    pd.testing.assert_frame_equal(out, close_frame)


def test_fresh_close_triggers_no_fetch(monkeypatch):
    """Last close = yesterday -> nothing missing, no fetch."""
    called = []

    def _count(cls, ticker, **kwargs):
        called.append(ticker)
        return {}

    monkeypatch.setattr(DataLoader, "_intraday_sessions", classmethod(_count))
    close_frame, td = _frames([119.8, 119.5], [5100, 5300], ["2026-09-04", "2026-09-07"])
    DataLoader._repair_edge_close(close_frame, td, ["X"], today=TODAY)
    assert called == []


# --- Silent source -------------------------------------------------------------

def test_without_intraday_the_gap_stays(monkeypatch):
    """Holiday: the source knows no session for the day -> the gap stays."""
    series = _run(monkeypatch, sessions={}, **NO_ROW)
    assert MONDAY not in series.index
    assert series.iloc[-1] == pytest.approx(119.8)


# --- The intraday session parser -------------------------------------------------

LONDON = "Europe/London"
SESSION_DAY = datetime.date(2026, 9, 15)
# An illiquid session: the final bar carries a price but no volume of its own.
BARS = [
    ("08:00", 51.20, 40), ("09:00", 51.05, 120), ("10:00", 51.40, 15),
    ("11:00", 51.45, 60), ("12:00", 51.60, 90), ("13:00", 51.90, 110),
    ("14:00", 51.55, 8), ("15:00", 51.00, 95), ("16:00", 51.20, 0),
]
CLOSE = 51.20
BARS_WITH_VOLUME = 8
PREVIOUS_CLOSE = 51.90


def _session(monkeypatch, bars=BARS, end="16:30"):
    """Point `yf.Ticker` at a constructed session (no network)."""
    index = pd.DatetimeIndex([f"2026-09-15 {t}" for t, _, _ in bars], tz=LONDON)
    hist = pd.DataFrame({"Close": [c for _, c, _ in bars],
                         "Volume": [float(v) for _, _, v in bars]}, index=index)
    schedule = pd.DataFrame({
        "start": [pd.Timestamp("2026-09-15 08:00", tz=LONDON)],
        "end": [pd.Timestamp(f"2026-09-15 {end}", tz=LONDON)],
    })

    class Ticker:
        history_metadata = {"exchangeTimezoneName": LONDON, "tradingPeriods": schedule}

        def history(self, **kwargs):
            return hist

    monkeypatch.setattr("backtest.data.yf.Ticker", lambda ticker: Ticker())


def _repair_with_sessions(monkeypatch, sessions):
    days = pd.DatetimeIndex(["2026-09-14", "2026-09-15"])
    close_frame = pd.DataFrame({"LINE.L": [PREVIOUS_CLOSE, float("nan")]}, index=days)
    ticker_data = pd.concat(
        {"Close": close_frame, "Volume": pd.DataFrame({"LINE.L": [900.0, 1.0]}, index=days)},
        axis=1,
    )
    monkeypatch.setattr(DataLoader, "_intraday_sessions",
                        classmethod(lambda cls, t, **kw: sessions))
    return DataLoader._repair_edge_close(
        close_frame, ticker_data, ["LINE.L"], today=datetime.date(2026, 9, 16))["LINE.L"]


def test_closing_bar_without_volume_does_not_drop_the_session(monkeypatch):
    """The session survives, with the price of the closing bar."""
    _session(monkeypatch)
    sessions = DataLoader._intraday_sessions("LINE.L")
    assert SESSION_DAY in sessions
    assert sessions[SESSION_DAY].value == pytest.approx(CLOSE)
    assert sessions[SESSION_DAY].covers_close()


def test_prints_counts_bars_with_volume_not_bars(monkeypatch):
    """`prints` stays the SESSION evidence: only bars with volume count."""
    _session(monkeypatch)
    assert DataLoader._intraday_sessions("LINE.L")[SESSION_DAY].prints == BARS_WITH_VOLUME
    assert BARS_WITH_VOLUME == sum(1 for _, _, v in BARS if v > 0)


def test_session_without_any_volume_is_not_repaired(monkeypatch):
    """"A closing bar without volume is normal" must not become "a session
    without volume is normal"."""
    _session(monkeypatch, bars=[(t, c, 0) for t, c, _ in BARS])
    sessions = DataLoader._intraday_sessions("LINE.L")
    assert sessions[SESSION_DAY].prints == 0
    repaired = _repair_with_sessions(monkeypatch, sessions)
    assert pd.isna(repaired.loc[pd.Timestamp("2026-09-15")])


def test_closing_bar_without_price_is_still_rejected(monkeypatch):
    _session(monkeypatch, bars=BARS[:-1] + [("16:00", float("nan"), 0)])
    assert DataLoader._intraday_sessions("LINE.L") == {}


def test_repaired_close_lands_in_the_daily_frame(monkeypatch):
    """Effect, not wiring: the Tuesday carries the real close afterwards."""
    _session(monkeypatch)
    sessions = DataLoader._intraday_sessions("LINE.L")
    repaired = _repair_with_sessions(monkeypatch, sessions)
    assert repaired.loc[pd.Timestamp("2026-09-15")] == pytest.approx(CLOSE)
    assert repaired.loc[pd.Timestamp("2026-09-14")] == pytest.approx(PREVIOUS_CLOSE)


# --- Inner gaps with trading evidence ----------------------------------------------

def test_inner_missing_close_with_volume_fails_loud():
    """A missing daily close INSIDE the series despite volume is corrupt data,
    not a holiday; it must not disappear through dropna/alignment."""
    close_frame, td = _frames([100.0, np.nan, 101.0], [10, 12, 11],
                              ["2024-03-01", "2024-03-04", "2024-03-05"])
    with pytest.raises(DataIntegrityError, match="2024-03-04"):
        DataLoader._check_inner_close_gaps(close_frame, td, ["X"])


def test_inner_row_without_any_evidence_passes():
    """An all-empty row (a venue holiday in a multi-market union) is not classified."""
    close_frame, td = _frames([100.0, np.nan, 101.0], [10, np.nan, 11],
                              ["2024-03-01", "2024-03-04", "2024-03-05"])
    DataLoader._check_inner_close_gaps(close_frame, td, ["X"])
