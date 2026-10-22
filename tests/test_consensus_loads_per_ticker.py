"""The consensus must load EVERY source individually.

Loading several tickers in ONE `DataLoader.yahoo` call uses
`align="intersection"`, which drops every row in which a single ticker is
missing. At the edge of the series that is dangerous: the edge-close repair
fills a missing daily close per ticker from intraday data. If it succeeds for A
but not for B, the row A=value / B=NaN is dropped again -- A silently loses its
repaired close just because B was loaded alongside. Bundling the calls is an
obvious "optimization"; this test pins the per-ticker form.
"""

import datetime

import pandas as pd

from backtest.data import PriceData
from backtest.portfolio_pricing import consensus_price_eur

IDX = pd.to_datetime(["2026-09-07", "2026-09-08", "2026-09-09"])
TODAY = datetime.date(2026, 9, 10)
FRAME = pd.DataFrame(
    {"SRC_A": [50.10, 50.02, 49.30],
     "SRC_B": [50.12, 50.04, float("nan")],
     "SRC_C": [50.08, 50.01, 49.28]},
    index=IDX,
)


def test_every_source_is_loaded_individually(monkeypatch):
    calls: list[list[str]] = []

    def fake_yahoo(**kwargs):
        tickers = list(kwargs["tickers"])
        calls.append(tickers)
        cols = [t for t in tickers if t in FRAME.columns]
        return PriceData(prices=FRAME[cols].copy(), currency={c: "EUR" for c in cols})

    monkeypatch.setattr("backtest.portfolio_pricing.DataLoader.yahoo", staticmethod(fake_yahoo))
    consensus_price_eur(["SRC_A", "SRC_B", "SRC_C"], start="2026-09-01", end="2026-09-10",
                        today=TODAY)

    assert calls, "no DataLoader.yahoo call observed"
    bundled = [c for c in calls if len(c) > 1]
    assert not bundled, (
        "consensus sources were loaded bundled %r -- align='intersection' then drops "
        "every row not all lines have, and a repaired edge close is lost silently."
        % (bundled,))
    assert {t for c in calls for t in c} >= {"SRC_A", "SRC_B", "SRC_C"}
