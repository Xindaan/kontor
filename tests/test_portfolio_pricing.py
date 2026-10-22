"""Tests for multi-source pricing."""

import datetime

import pandas as pd
import pytest

from backtest.data import PriceData
from backtest.portfolio_pricing import ConsensusPrice, consensus_price_eur


def _fake_loader(monkeypatch, frame: pd.DataFrame, currency=None):
    """Replace DataLoader.yahoo with a fixed price table."""
    def fake_yahoo(**kwargs):
        cols = [t for t in kwargs["tickers"] if t in frame.columns]
        return PriceData(prices=frame[cols].copy(),
                         currency={c: "EUR" for c in cols} if currency is None else currency)
    monkeypatch.setattr("backtest.portfolio_pricing.DataLoader.yahoo", staticmethod(fake_yahoo))


IDX = pd.to_datetime(["2026-07-08", "2026-07-09"])
# Freshness is part of the verdict: the fixtures end on Thu 07-09,
# so all legacy tests compute against Fri 07-10 as "today".
TODAY = datetime.date(2026, 7, 10)


class TestConsensusPrice:
    def test_two_agreeing_sources_are_ok(self, monkeypatch):
        _fake_loader(monkeypatch, pd.DataFrame(
            {"3SEM.DE": [150.0, 151.01], "3SEM.L": [150.2, 151.33]}, index=IDX))
        c = consensus_price_eur(["3SEM.DE", "3SEM.L"], "2026-05-01", "2026-07-10", today=TODAY)
        assert c.ok and not c.diverged and not c.single_source
        assert c.price_eur == pytest.approx((151.01 + 151.33) / 2)
        assert c.spread == pytest.approx(151.33 / 151.01 - 1)

    def test_diverging_sources_flag(self, monkeypatch):
        """A broken source (fake print) must not pass through."""
        _fake_loader(monkeypatch, pd.DataFrame(
            {"3SEM.DE": [150.0, 151.0], "3SEM.L": [150.2, 201.0]}, index=IDX))
        c = consensus_price_eur(["3SEM.DE", "3SEM.L"], "2026-05-01", "2026-07-10", today=TODAY)
        assert c.diverged and not c.ok
        assert c.spread > 0.01

    def test_spread_exactly_at_gate_is_ok(self, monkeypatch):
        """The gate is an upper bound, not an exclusion."""
        _fake_loader(monkeypatch, pd.DataFrame(
            {"A": [100.0, 100.0], "B": [101.0, 101.0]}, index=IDX))
        c = consensus_price_eur(["A", "B"], "2026-05-01", "2026-07-10", gate=0.01, today=TODAY)
        assert c.spread == pytest.approx(0.01)
        assert c.ok

    def test_single_source_flags_but_still_prices(self, monkeypatch):
        _fake_loader(monkeypatch, pd.DataFrame({"3SEM.DE": [150.0, 151.0]}, index=IDX))
        c = consensus_price_eur(["3SEM.DE", "3SEM.L"], "2026-05-01", "2026-07-10", today=TODAY)
        assert c.single_source and not c.ok
        assert c.price_eur == pytest.approx(151.0)

    def test_no_source_configured_raises(self, monkeypatch):
        _fake_loader(monkeypatch, pd.DataFrame({"X": [1.0, 1.0]}, index=IDX))
        with pytest.raises(ValueError, match="no valuation source"):
            consensus_price_eur([None, None], "2026-05-01", "2026-07-10", today=TODAY)

    def test_unreachable_sources_raise_instead_of_guessing(self, monkeypatch):
        _fake_loader(monkeypatch, pd.DataFrame({"X": [1.0, 1.0]}, index=IDX))
        with pytest.raises(ValueError, match="no EUR source reachable"):
            consensus_price_eur(["3SEM.DE", "3SEM.L"], "2026-05-01", "2026-07-10", today=TODAY)

    def test_median_of_three(self, monkeypatch):
        _fake_loader(monkeypatch, pd.DataFrame(
            {"A": [1.0, 100.0], "B": [1.0, 101.0], "C": [1.0, 300.0]}, index=IDX))
        c = consensus_price_eur(["A", "B", "C"], "2026-05-01", "2026-07-10", today=TODAY)
        assert c.price_eur == pytest.approx(101.0)   # Median ignores the outlier
        assert c.diverged                            # but it's still reported


class TestStagedGate:
    """3% warns, 10% blocks -- the noise floor sits at p90 2.3%."""

    def test_noise_floor_spread_is_ok(self, monkeypatch):
        """A 2% spread is noise from two thin lines and must not trigger anything.

        Under the old 1% gate this would have aborted the Monday rebalance.
        """
        _fake_loader(monkeypatch, pd.DataFrame(
            {"A": [100.0, 100.0], "B": [100.0, 102.0]}, index=IDX))
        c = consensus_price_eur(["A", "B"], "2026-05-01", "2026-07-10", today=TODAY)
        assert c.spread == pytest.approx(0.02)
        assert c.ok and not c.noisy and not c.diverged

    def test_warn_band_is_noisy_but_not_blocking(self, monkeypatch):
        _fake_loader(monkeypatch, pd.DataFrame(
            {"A": [100.0, 100.0], "B": [100.0, 105.0]}, index=IDX))
        c = consensus_price_eur(["A", "B"], "2026-05-01", "2026-07-10", today=TODAY)
        assert c.spread == pytest.approx(0.05)
        assert c.noisy and not c.diverged and not c.ok

    def test_currency_line_print_blocks(self, monkeypatch):
        """A USD-denominated line read as pence is off by ~1.34x -> well over 10%."""
        _fake_loader(monkeypatch, pd.DataFrame(
            {"A": [100.0, 100.0], "B": [100.0, 133.3]}, index=IDX))
        c = consensus_price_eur(["A", "B"], "2026-05-01", "2026-07-10", today=TODAY)
        assert c.diverged and not c.noisy and not c.ok

    def test_gate_alias_points_at_block_threshold(self, monkeypatch):
        _fake_loader(monkeypatch, pd.DataFrame(
            {"A": [1.0, 100.0], "B": [1.0, 100.0]}, index=IDX))
        c = consensus_price_eur(["A", "B"], "2026-05-01", "2026-07-10", today=TODAY)
        assert c.gate == c.block_gate == pytest.approx(0.10)
        assert c.warn_gate == pytest.approx(0.03)


class TestDescribe:
    def test_describe_marks_divergence(self):
        c = ConsensusPrice(151.0, {"a": 151.0, "b": 201.0}, 0.33, 0.01)
        assert "SPREAD" in c.describe()

    def test_describe_block_and_warn_are_distinguishable(self):
        block = ConsensusPrice(151.0, {"a": 151.0, "b": 201.0}, 0.33, 0.10, 0.03)
        warn = ConsensusPrice(151.0, {"a": 151.0, "b": 158.0}, 0.05, 0.10, 0.03)
        assert "BLOCK" in block.describe()
        assert "SPREAD" in warn.describe() and "BLOCK" not in warn.describe()

    def test_describe_marks_single_source(self):
        c = ConsensusPrice(151.0, {"a": 151.0}, 0.0, 0.01)
        assert "NUR 1 QUELLE" in c.describe()


class TestSessionDatum:
    """Consensus only on the most recent COMMON session date.

    A real cross-source case: a EUR UCITS ETF's last print lagged one day
    behind a USD-denominated LSE ETP line's same-day print (Xetra closed
    for the day, LSE still live) -> naively taking `iloc[-1]` per source
    produced a large "divergence" that was really just a session-date
    mismatch, not a real price divergence -- a false alarm for "don't
    rebalance today". On a Monday this would have blocked a legitimate
    rebalance.
    """

    def test_session_versatz_ist_kein_fehlalarm(self, monkeypatch):
        idx = pd.to_datetime(["2026-07-15", "2026-07-16", "2026-07-17"])
        _fake_loader(monkeypatch, pd.DataFrame(
            {"3SEM.DE": [113.10, 112.47, float("nan")],
             "3SEM.L": [113.25, 112.33, 96.02]}, index=idx))
        c = consensus_price_eur(["3SEM.DE", "3SEM.L"], "2026-05-01", "2026-07-18",
                                today=datetime.date(2026, 7, 17))
        assert c.as_of == "2026-07-16"     # last date on which BOTH printed
        assert c.price_eur == pytest.approx((112.47 + 112.33) / 2)
        assert c.ok and not c.diverged and not c.stale

    def test_common_mode_staleness_ist_kein_verdikt(self, monkeypatch):
        """Both sources frozen -> spread 0 -> used to be 'ok'. The more dangerous
        direction of the bug: two matching stale prices are not proof of
        anything."""
        idx = pd.to_datetime(["2026-07-13", "2026-07-14"])
        _fake_loader(monkeypatch, pd.DataFrame(
            {"A": [100.0, 100.5], "B": [100.1, 100.4]}, index=idx))
        c = consensus_price_eur(["A", "B"], "2026-05-01", "2026-07-18",
                                today=datetime.date(2026, 7, 17))
        assert c.stale and not c.ok
        assert not c.diverged and not c.noisy   # not a false alarm, simply NO verdict
        assert "KEIN VERDIKT" in c.describe()

    def test_stale_gilt_auch_fuer_single_source(self, monkeypatch):
        idx = pd.to_datetime(["2026-07-13", "2026-07-14"])
        _fake_loader(monkeypatch, pd.DataFrame({"A": [100.0, 100.5]}, index=idx))
        c = consensus_price_eur(["A", "B"], "2026-05-01", "2026-07-18",
                                today=datetime.date(2026, 7, 17))
        assert c.stale and c.single_source and not c.ok

    def test_kein_gemeinsames_datum_raises(self, monkeypatch):
        _fake_loader(monkeypatch, pd.DataFrame(
            {"A": [100.0, float("nan")], "B": [float("nan"), 101.0]},
            index=pd.to_datetime(["2026-07-15", "2026-07-16"])))
        with pytest.raises(ValueError, match="no session date"):
            consensus_price_eur(["A", "B"], "2026-05-01", "2026-07-18",
                                today=datetime.date(2026, 7, 17))

    def test_describe_traegt_stand_und_nachzuegler(self, monkeypatch):
        """A verdict without a visible price date was an invitation for errors."""
        idx = pd.to_datetime(["2026-07-16", "2026-07-17"])
        _fake_loader(monkeypatch, pd.DataFrame(
            {"3SEM.DE": [112.47, float("nan")], "3SEM.L": [112.33, 112.90]}, index=idx))
        c = consensus_price_eur(["3SEM.DE", "3SEM.L"], "2026-05-01", "2026-07-18",
                                today=datetime.date(2026, 7, 17))
        out = c.describe()
        assert "Stand 2026-07-16" in out
        assert "3SEM.L zuletzt 2026-07-17" in out
