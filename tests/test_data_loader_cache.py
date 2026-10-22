import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from backtest.data import DataLoader, PriceData


def _today_plus(days: int) -> str:
    return (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")


def test_yahoo_in_memory_cache_avoids_duplicate_downloads(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_download(
        tickers,
        start=None,
        end=None,
        progress=False,
        auto_adjust=True,
        group_by="column",
    ):
        _ = (tickers, start, end, progress, auto_adjust, group_by)
        calls["count"] += 1
        idx = pd.bdate_range("2020-01-01", periods=5)
        return pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=idx)

    monkeypatch.setattr("backtest.data.yf.download", fake_download)
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)
    DataLoader._MEMORY_CACHE.clear()

    first = DataLoader.yahoo(
        tickers=["SPY"],
        start="2020-01-01",
        end="2020-01-10",
        currency="USD",
        cache=False,
        validate=False,
    )
    second = DataLoader.yahoo(
        tickers=["SPY"],
        start="2020-01-01",
        end="2020-01-10",
        currency="USD",
        cache=False,
        validate=False,
    )

    assert calls["count"] == 1
    assert first is not second
    assert first.prices.equals(second.prices)

    # Defensive copy check: mutating one result must not affect cached copy.
    first.prices.iloc[0, 0] = -1
    assert second.prices.iloc[0, 0] != -1


def test_clear_cache_also_clears_memory_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)
    DataLoader._MEMORY_CACHE.clear()
    DataLoader._MEMORY_CACHE[("dummy",)] = PriceData(
        prices=pd.DataFrame({"SPY": [1.0]}, index=pd.to_datetime(["2020-01-01"])),
        currency={"SPY": "USD"},
    )

    deleted = DataLoader.clear_cache()
    assert deleted == 0
    assert DataLoader._MEMORY_CACHE == {}


def _make_fake_download(calls):
    """Fake yf.download that records calls and honours the [start, end] range."""

    def fake_download(
        tickers,
        start=None,
        end=None,
        progress=False,
        auto_adjust=True,
        group_by="column",
    ):
        _ = (tickers, progress, auto_adjust, group_by)
        calls.append({"tickers": tickers, "start": start, "end": end})
        idx = pd.bdate_range(start, end)
        return pd.DataFrame(
            {"Close": [100.0 + i for i in range(len(idx))]}, index=idx
        )

    return fake_download


def test_yahoo_disk_cache_keeps_one_file_per_ticker(monkeypatch, tmp_path):
    """Repeated pulls with a moving end date must not spawn one CSV per run."""
    calls = []
    monkeypatch.setattr("backtest.data.yf.download", _make_fake_download(calls))
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)

    for end in ("2020-02-01", "2020-03-01", "2020-04-01"):
        DataLoader._MEMORY_CACHE.clear()
        DataLoader.yahoo(
            tickers=["SPY"],
            start="2020-01-01",
            end=end,
            currency="USD",
            cache=True,
            validate=False,
        )

    csvs = sorted(p.name for p in tmp_path.glob("*.csv"))
    assert csvs == ["SPY.csv"]
    assert (tmp_path / "_cache_manifest.json").exists()
    assert len(calls) == 3  # each newer end is a cache miss


def test_yahoo_disk_cache_hit_for_covered_subrange(monkeypatch, tmp_path):
    """A sub-window of an already-cached range is served from disk."""
    calls = []
    monkeypatch.setattr("backtest.data.yf.download", _make_fake_download(calls))
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)

    DataLoader._MEMORY_CACHE.clear()
    DataLoader.yahoo(
        tickers=["SPY"],
        start="2020-01-01",
        end="2020-06-01",
        currency="USD",
        cache=True,
        validate=False,
    )
    assert len(calls) == 1

    DataLoader._MEMORY_CACHE.clear()
    result = DataLoader.yahoo(
        tickers=["SPY"],
        start="2020-02-01",
        end="2020-03-01",
        currency="USD",
        cache=True,
        validate=False,
    )
    assert len(calls) == 1  # covered sub-range -> no extra download
    assert result.prices.index.min() >= pd.Timestamp("2020-02-01")
    assert result.prices.index.max() < pd.Timestamp("2020-03-01")


def test_yahoo_disk_cache_refetches_union_range_for_newer_end(monkeypatch, tmp_path):
    """A newer end date triggers a re-fetch of the union range, one file."""
    calls = []
    monkeypatch.setattr("backtest.data.yf.download", _make_fake_download(calls))
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)

    DataLoader._MEMORY_CACHE.clear()
    DataLoader.yahoo(
        tickers=["SPY"],
        start="2020-03-01",
        end="2020-06-01",
        currency="USD",
        cache=True,
        validate=False,
    )
    DataLoader._MEMORY_CACHE.clear()
    DataLoader.yahoo(
        tickers=["SPY"],
        start="2020-01-01",
        end="2020-09-01",
        currency="USD",
        cache=True,
        validate=False,
    )

    assert len(calls) == 2  # newer/earlier window -> miss
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == ["SPY.csv"]
    # Second download covers the union of both requested windows.
    assert calls[1]["start"] <= "2020-01-01"
    assert calls[1]["end"] >= "2020-09-01"


def test_clear_cache_removes_manifest(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("backtest.data.yf.download", _make_fake_download(calls))
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)

    DataLoader._MEMORY_CACHE.clear()
    DataLoader.yahoo(
        tickers=["SPY"],
        start="2020-01-01",
        end="2020-03-01",
        currency="USD",
        cache=True,
        validate=False,
    )
    assert (tmp_path / "_cache_manifest.json").exists()

    deleted = DataLoader.clear_cache()
    assert deleted == 1  # one SPY.csv, manifest not counted
    assert not (tmp_path / "_cache_manifest.json").exists()
    assert list(tmp_path.glob("*.csv")) == []


def test_get_fx_rates_keeps_one_file_per_pair(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("backtest.data.yf.download", _make_fake_download(calls))
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)

    DataLoader.get_fx_rates("2020-01-01", "2020-06-01", pair="EURUSD=X", cache=True)
    DataLoader.get_fx_rates("2020-02-01", "2020-03-01", pair="EURUSD=X", cache=True)

    assert len(calls) == 1  # covered sub-range served from disk
    assert sorted(p.name for p in tmp_path.glob("fx_*.csv")) == ["fx_EURUSDX.csv"]


def test_yahoo_builds_gbp_fx_dataframe_for_eur_conversion(monkeypatch, tmp_path):
    def fake_download(
        tickers,
        start=None,
        end=None,
        progress=False,
        auto_adjust=True,
        group_by="column",
    ):
        _ = (tickers, start, end, progress, auto_adjust, group_by)
        idx = pd.bdate_range("2020-01-01", periods=5)
        return pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=idx)

    def fake_fx_rates(start, end, pair="EURUSD=X", cache=True):
        _ = (start, end, pair, cache)
        idx = pd.bdate_range("2020-01-01", periods=5)
        return pd.Series([0.85, 0.851, 0.852, 0.853, 0.854], index=idx, name="EURGBP=X")

    monkeypatch.setattr("backtest.data.yf.download", fake_download)
    monkeypatch.setattr(DataLoader, "get_fx_rates", fake_fx_rates)
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)
    DataLoader._MEMORY_CACHE.clear()

    data = DataLoader.yahoo(
        tickers=["HSBA.L"],
        start="2020-01-01",
        end="2020-01-10",
        currency="EUR",
        cache=False,
        validate=False,
    )

    assert data.currency["HSBA.L"] == "GBP"
    assert isinstance(data.fx_rates, pd.DataFrame)
    assert list(data.fx_rates.columns) == ["GBP"]


def test_record_cache_range_caps_future_end(monkeypatch, tmp_path):
    """A far-future requested end must never be recorded verbatim: it would
    make _range_covered report a permanent false hit and freeze the file."""
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)

    DataLoader._record_cache_range("SPY.csv", "2012-01-01", "2099-01-01")

    raw = json.loads((tmp_path / "_cache_manifest.json").read_text())
    assert raw["SPY.csv"]["start"] == "2012-01-01"
    assert raw["SPY.csv"]["end"] == _today_plus(1)  # capped at today + 1


def test_record_cache_range_recaps_poisoned_previous_end(monkeypatch, tmp_path):
    """Writing onto an already-poisoned entry self-heals it: the load-time
    sanitize drops the future end first, so the merge records the honest new
    end instead of carrying the poison forward via max()."""
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)
    (tmp_path / "_cache_manifest.json").write_text(
        json.dumps({"SPY.csv": {"start": "2012-01-01", "end": "2099-01-01"}})
    )

    DataLoader._record_cache_range("SPY.csv", "2012-01-01", "2020-01-01")

    raw = json.loads((tmp_path / "_cache_manifest.json").read_text())
    assert raw["SPY.csv"]["end"] == "2020-01-01"  # poison dropped, not max()'d


def test_load_manifest_drops_future_poisoned_end(monkeypatch, tmp_path):
    """On load, an impossible future end is dropped (start kept) so the next
    coverage check misses and the stale file gets re-fetched."""
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)
    (tmp_path / "_cache_manifest.json").write_text(
        json.dumps({"SPY.csv": {"start": "2012-01-01", "end": "2099-01-01"}})
    )

    manifest = DataLoader._load_manifest()

    assert manifest["SPY.csv"]["start"] == "2012-01-01"
    assert "end" not in manifest["SPY.csv"]  # poisoned end dropped


def test_yahoo_future_end_request_does_not_poison_manifest(monkeypatch, tmp_path):
    """A pull with a far-future end (e.g. a padded backtest) must not record a
    future end that would freeze the live cache for the next runs."""
    calls = []
    monkeypatch.setattr("backtest.data.yf.download", _make_fake_download(calls))
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)
    DataLoader._MEMORY_CACHE.clear()

    DataLoader.yahoo(
        tickers=["SPY"],
        start="2020-01-01",
        end=_today_plus(400),  # far-future padded end
        currency="USD",
        cache=True,
        validate=False,
    )

    raw = json.loads((tmp_path / "_cache_manifest.json").read_text())
    assert raw["SPY.csv"]["end"] <= _today_plus(1)


def test_yahoo_refetches_when_manifest_end_is_in_future(monkeypatch, tmp_path):
    """Regression: a manifest entry with a future end must NOT suppress a
    refresh when newer real data exists -- otherwise the file stays frozen
    (this inverted a live rebalance signal, 2026-06-15)."""
    calls = []
    monkeypatch.setattr("backtest.data.yf.download", _make_fake_download(calls))
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)
    DataLoader._MEMORY_CACHE.clear()

    # Stale on-disk file: data only through 2020-01-06.
    stale = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0, 103.0]},
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
    )
    stale.to_csv(tmp_path / "SPY.csv")
    # Poisoned manifest: claims coverage far into the future.
    (tmp_path / "_cache_manifest.json").write_text(
        json.dumps({"SPY.csv": {"start": "2020-01-01", "end": "2099-01-01"}})
    )

    result = DataLoader.yahoo(
        tickers=["SPY"],
        start="2020-01-01",
        end="2020-02-01",  # newer than the stale file's last date
        currency="USD",
        cache=True,
        validate=False,
    )

    assert len(calls) == 1  # poison no longer suppresses the refetch
    assert result.prices.index.max() >= pd.Timestamp("2020-01-30")
    # Healed: manifest end is no longer the poisoned future date.
    raw = json.loads((tmp_path / "_cache_manifest.json").read_text())
    assert raw["SPY.csv"]["end"] <= _today_plus(1)


def test_apply_known_bad_prices_overwrites_only_listed_days():
    """Yahoo's permanently wrong print on a pence-denominated LSE line is
    corrected in the fetch path."""
    idx = pd.to_datetime(["2025-10-23", "2025-10-24", "2025-10-27"])
    series = pd.Series([10607.5, 14554.4248, 11228.5], index=idx, name="3LUS.L")

    fixed = DataLoader._apply_known_bad_prices("3LUS.L", series)

    # Exactly reconstructed from the USD-denominated counterpart line (its
    # USD price divided by GBPUSD, times 100), no longer interpolated
    # from neighboring days (the old value was 0.02% off).
    # The bad print IS the USD line's price x 100 -- a line mix-up, not garbage.
    assert fixed.loc[pd.Timestamp("2025-10-24")] == 10920.62
    assert fixed.loc[pd.Timestamp("2025-10-23")] == 10607.5
    assert fixed.loc[pd.Timestamp("2025-10-27")] == 11228.5
    # Caller's series must not be mutated in place.
    assert series.loc[pd.Timestamp("2025-10-24")] == 14554.4248


def test_apply_known_bad_prices_leaves_other_tickers_and_windows_alone():
    """Unlisted tickers, and windows that miss the bad day, pass through unchanged."""
    idx = pd.to_datetime(["2025-10-24"])
    unlisted = pd.Series([14554.4248], index=idx, name="SPY")
    assert DataLoader._apply_known_bad_prices("SPY", unlisted).equals(unlisted)

    narrow = pd.Series([11403.0], index=pd.to_datetime(["2025-10-28"]), name="3LUS.L")
    assert DataLoader._apply_known_bad_prices("3LUS.L", narrow).equals(narrow)


def test_known_bad_prices_repair_the_2017_06_line_swap_on_both_lines():
    """2017-06-26/27 is the same pence-field bug -- on BOTH lines.

    Important: the USD-denominated line doesn't work as a reference here
    (as it usually does), because it is itself broken on exactly these
    days -- the printed ratio between the two lines is exactly 20.000
    instead of the real GBPUSD of ~1.27.
    """
    days = pd.to_datetime(["2017-06-23", "2017-06-26", "2017-06-27", "2017-06-28"])

    pence = pd.Series([1799.07, 2291.50, 2235.74, 1767.78], index=days, name="3LUS.L")
    fixed_pence = DataLoader._apply_known_bad_prices("3LUS.L", pence)
    assert fixed_pence.loc[pd.Timestamp("2017-06-26")] == 1798.07
    assert fixed_pence.loc[pd.Timestamp("2017-06-27")] == 1756.89
    # Neighbors untouched.
    assert fixed_pence.loc[pd.Timestamp("2017-06-23")] == 1799.07
    assert fixed_pence.loc[pd.Timestamp("2017-06-28")] == 1767.78

    usd = pd.Series([22.9155, 458.30, 447.15, 22.85], index=days, name="3USL.L")
    fixed_usd = DataLoader._apply_known_bad_prices("3USL.L", usd)
    assert fixed_usd.loc[pd.Timestamp("2017-06-26")] == 22.9150
    assert fixed_usd.loc[pd.Timestamp("2017-06-27")] == 22.3574
    assert fixed_usd.loc[pd.Timestamp("2017-06-23")] == 22.9155


def test_repaired_lines_agree_on_the_currency_relationship():
    """Guard for the failure class: after the repair, the two lines of the
    same ETP must again agree via GBPUSD.

    This exact invariant was the detector for the whole finding -- a
    ratio of 20.000 or 100 instead of ~1.27 is a line mix-up. Both lines
    close on the same exchange at the same second, so only FX timing
    remains as noise.
    """
    days = pd.to_datetime(["2017-06-26", "2017-06-27"])
    gbpusd = {"2017-06-26": 1.2744, "2017-06-27": 1.2726}

    pence = DataLoader._apply_known_bad_prices(
        "3LUS.L", pd.Series([2291.50, 2235.74], index=days)
    )
    usd = DataLoader._apply_known_bad_prices(
        "3USL.L", pd.Series([458.30, 447.15], index=days)
    )

    for day in days:
        implied = usd.loc[day] / pence.loc[day] * 100.0
        assert implied == pytest.approx(gbpusd[str(day.date())], rel=0.005), (
            f"{day.date()}: line ratio {implied:.3f} instead of GBPUSD "
            f"{gbpusd[str(day.date())]:.4f} -- currency-line mix-up?"
        )


# --- manifest end must not claim a not-yet-published active tail ------

def _make_capped_download(calls, last_available):
    """Fake yf.download that returns bars only through ``last_available`` --
    however far into the future the requested end is (models Yahoo not yet
    publishing today's bar, a data gap, or a weekend)."""

    def fake_download(tickers, start=None, end=None, progress=False,
                      auto_adjust=True, group_by="column"):
        _ = (tickers, progress, auto_adjust, group_by)
        calls.append({"tickers": tickers, "start": start, "end": end})
        capped_end = min(pd.Timestamp(end), pd.Timestamp(last_available))
        idx = pd.bdate_range(start, capped_end)
        return pd.DataFrame({"Close": [100.0 + i for i in range(len(idx))]}, index=idx)

    return fake_download


def test_record_cache_range_active_tail_not_over_claimed(monkeypatch, tmp_path):
    """An active ticker whose last real bar is a few days old must record its
    end at that bar + 1, NOT at the requested (today+1) end -- else a
    not-yet-published bar freezes the stale series (a real stop-miss incident)."""
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)

    DataLoader._record_cache_range(
        "EXFC_MU.csv", "2026-06-01", _today_plus(1),
        last_data_date=pd.Timestamp(_today_plus(-3)),
    )

    raw = json.loads((tmp_path / "_cache_manifest.json").read_text())
    assert raw["EXFC_MU.csv"]["end"] == _today_plus(-2)  # last bar + 1, not today+1


def test_record_cache_range_active_overrides_poisoned_prev_end(monkeypatch, tmp_path):
    """The active-tail cap must override a poisoned prev_end (wholesale
    overwrite replaced the file), not carry it forward via max()."""
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)
    (tmp_path / "_cache_manifest.json").write_text(
        json.dumps({"EXFC_MU.csv": {"start": "2026-06-01", "end": _today_plus(1)}})
    )

    DataLoader._record_cache_range(
        "EXFC_MU.csv", "2026-06-01", _today_plus(1),
        last_data_date=pd.Timestamp(_today_plus(-3)),
    )

    raw = json.loads((tmp_path / "_cache_manifest.json").read_text())
    assert raw["EXFC_MU.csv"]["end"] == _today_plus(-2)  # not today+1


def test_record_cache_range_settled_ticker_keeps_fetch_intent(monkeypatch, tmp_path):
    """A settled/delisted ticker (last bar years old) keeps the fetch-intent
    end, so its unchanging history is not re-downloaded on every run."""
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)

    DataLoader._record_cache_range(
        "ESRX_unadj.csv", "2015-01-01", "2019-01-01",
        last_data_date=pd.Timestamp("2018-12-21"),
    )

    raw = json.loads((tmp_path / "_cache_manifest.json").read_text())
    assert raw["ESRX_unadj.csv"]["end"] == "2019-01-01"  # not 2018-12-22


def test_yahoo_refetches_active_tail_once_new_bars_publish(monkeypatch, tmp_path):
    """End-to-end regression: a pull whose requested end is beyond the
    last available bar must record only up to that bar, then re-fetch once
    the newer bars publish -- instead of freezing a stale price all day."""
    calls = []
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)

    # Morning run: today's bar not published yet -> last available is 3 days ago.
    monkeypatch.setattr("backtest.data.yf.download",
                        _make_capped_download(calls, _today_plus(-3)))
    DataLoader._MEMORY_CACHE.clear()
    first = DataLoader.yahoo(tickers=["EXFC.MU"], start="2026-06-01",
                             end=_today_plus(1), currency="EUR",
                             cache=True, validate=False)
    stale_last = first.prices.index.max()
    # End recorded at the real last bar + 1 (whatever business day that lands
    # on), NOT today+1 -- the false claim that froze the stale series.
    expected_end = (stale_last + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = json.loads((tmp_path / "_cache_manifest.json").read_text())
    assert raw["EXFC_MU.csv"]["end"] == expected_end
    assert raw["EXFC_MU.csv"]["end"] < _today_plus(1)  # strictly below the poison value

    # Later run: the fresh bars are now available; the request must re-fetch.
    monkeypatch.setattr("backtest.data.yf.download",
                        _make_capped_download(calls, _today_plus(0)))
    DataLoader._MEMORY_CACHE.clear()
    second = DataLoader.yahoo(tickers=["EXFC.MU"], start="2026-06-01",
                              end=_today_plus(1), currency="EUR",
                              cache=True, validate=False)

    price_calls = [c for c in calls if c["tickers"] == ["EXFC.MU"]]
    assert len(price_calls) == 2  # re-fetched, not served stale from cache
    assert second.prices.index.max() > stale_last  # picked up the fresh tail


def test_reconcile_manifest_heals_active_and_covers_settled(monkeypatch, tmp_path):
    """reconcile_manifest_with_files: shrink an over-claiming ACTIVE entry to
    its real tail (forces refresh) but mark a SETTLED one covered (no churn)."""
    monkeypatch.setattr(DataLoader, "CACHE_DIR", tmp_path)

    # Active file: last bar 3 days ago, manifest poisoned to today+1.
    pd.DataFrame(
        {"EXFC.MU": [1.0, 2.0]},
        index=pd.to_datetime([_today_plus(-6), _today_plus(-3)]),
    ).to_csv(tmp_path / "EXFC_MU.csv")
    # Settled file: last bar years ago, manifest poisoned to today+1.
    pd.DataFrame(
        {"ESRX": [1.0, 2.0]},
        index=pd.to_datetime(["2018-12-20", "2018-12-21"]),
    ).to_csv(tmp_path / "ESRX_unadj.csv")
    (tmp_path / "_cache_manifest.json").write_text(json.dumps({
        "EXFC_MU.csv": {"start": "2026-06-01", "end": _today_plus(1)},
        "ESRX_unadj.csv": {"start": "2015-01-01", "end": _today_plus(1)},
    }))

    changed = DataLoader.reconcile_manifest_with_files()

    raw = json.loads((tmp_path / "_cache_manifest.json").read_text())
    assert raw["EXFC_MU.csv"]["end"] == _today_plus(-2)   # active -> real tail + 1
    assert raw["ESRX_unadj.csv"]["end"] == _today_plus(1)  # settled -> covered
    assert "EXFC_MU.csv" in changed
