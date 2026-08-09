"""
Data module - Loading and processing price data.

This module provides functionality to:
- Load price data from Yahoo Finance
- Load price data from CSV files
- Convert prices between currencies
- Cache downloaded data for faster access
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from backtest.assets import (
    TICKER_HISTORY_VALID_FROM,
    detect_currency,
    get_yahoo_ticker,
)
from backtest.utils import MONTH_END_FREQ


@dataclass
class PriceData:
    """
    Container for price data with currency information.

    WARNING -- two documented pitfalls when accessing `.prices` directly:
    - When loaded with load_dividends=True, the prices are UNADJUSTED
      (dividends are kept separately in `.dividends`). Never derive returns
      from `.prices` alone in that case -- use `total_return_prices()`.
    - `.prices` is always in the NATIVE currency of each ticker; the
      loader's currency parameter only attaches FX series. Use `in_eur()`
      or `total_return_prices(currency="EUR")` to get EUR series.

    Attributes:
        prices: DataFrame with DatetimeIndex and ticker columns.
                Values are close prices in the ticker's NATIVE currency;
                dividend-adjusted only when load_dividends=False.
        currency: Dictionary mapping tickers to their base currency
        fx_rates: Series/DataFrame with FX rates for conversion into target currency (optional)
        volumes: DataFrame with DatetimeIndex and ticker columns.
                 Values are traded shares per day (optional)
        dividends: DataFrame with DatetimeIndex and ticker columns.
                   Values are dividend per share (optional)
        macro: DataFrame with DatetimeIndex and macro series columns (optional)
        last_print: Dict ticker -> date of the last REAL print, captured BEFORE
                   align="ffill" runs. ffill silently carries an old price
                   forward to the end of the frame; after that, `.prices.iloc[-1]`
                   looks like today's price. Anyone deriving a verdict from
                   `.prices` MUST cross-check against this field -- a frozen
                   ticker once inverted a live signal this way.
                   With align="intersection" this is identical to the frame end.
    """
    prices: pd.DataFrame
    currency: Dict[str, str] = field(default_factory=dict)
    fx_rates: Optional[Union[pd.Series, pd.DataFrame]] = None
    volumes: Optional[pd.DataFrame] = None
    dividends: Optional[pd.DataFrame] = None
    macro: Optional[pd.DataFrame] = None
    last_print: Dict[str, pd.Timestamp] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure index is DatetimeIndex
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            self.prices.index = pd.to_datetime(self.prices.index)
        if self.prices.index.tz is not None:
            self.prices.index = self.prices.index.tz_convert(None)
        # Sort by date
        self.prices = self.prices.sort_index()
        # Ensure dividends index is DatetimeIndex if provided
        if self.dividends is not None:
            if not isinstance(self.dividends.index, pd.DatetimeIndex):
                self.dividends.index = pd.to_datetime(self.dividends.index)
            if self.dividends.index.tz is not None:
                self.dividends.index = self.dividends.index.tz_convert(None)
            # Ex-dates are calendar days: after tz conversion, Yahoo returns
            # intraday remnants (e.g. 05:00 from US/Eastern) that miss exact
            # index matches against the price index.
            normalized = self.dividends.index.normalize()
            if not normalized.equals(self.dividends.index):
                self.dividends = self.dividends.groupby(normalized).sum()
            self.dividends = self.dividends.sort_index()
        if self.macro is not None:
            if not isinstance(self.macro.index, pd.DatetimeIndex):
                self.macro.index = pd.to_datetime(self.macro.index)
            if self.macro.index.tz is not None:
                self.macro.index = self.macro.index.tz_convert(None)
            self.macro = self.macro.sort_index()
        if self.fx_rates is not None and isinstance(self.fx_rates.index, pd.DatetimeIndex):
            if self.fx_rates.index.tz is not None:
                self.fx_rates.index = self.fx_rates.index.tz_convert(None)
        if self.volumes is not None:
            if not isinstance(self.volumes.index, pd.DatetimeIndex):
                self.volumes.index = pd.to_datetime(self.volumes.index)
            if self.volumes.index.tz is not None:
                self.volumes.index = self.volumes.index.tz_convert(None)
            self.volumes = self.volumes.sort_index()

    @property
    def tickers(self) -> List[str]:
        """List of ticker symbols in the data."""
        return list(self.prices.columns)

    @property
    def start_date(self) -> datetime:
        """First date in the data."""
        return self.prices.index[0]

    @property
    def end_date(self) -> datetime:
        """Last date in the data."""
        return self.prices.index[-1]

    @property
    def monthly_dates(self) -> pd.DatetimeIndex:
        """Month-end dates in the data."""
        return self.prices.resample(MONTH_END_FREQ).last().index

    def in_eur(self) -> pd.DataFrame:
        """
        Convert all prices to EUR.

        IMPORTANT: `.prices` itself stays in each ticker's NATIVE currency --
        even when the loader was called with currency="EUR" (that only
        controls which FX series get attached). Anyone who wants EUR series
        must go through this method (the backtester does so internally).
        Labeling native frames as EUR (fx=1.0) would distort USD legs by the
        FX drift.

        Returns:
            DataFrame with all prices converted to EUR
        """
        return self._frame_in_eur(self.prices)

    def dividends_in_eur(self) -> Optional[pd.DataFrame]:
        """Convert each ticker's dividends to EUR, using the same FX logic as in_eur().

        Returns None if no dividends are loaded. The backtester runs the
        same conversion internally on its dividend path; deliberately not
        replaced there, to keep the blast radius of this fix small.
        """
        if self.dividends is None:
            return None
        return self._frame_in_eur(self.dividends)

    def _frame_in_eur(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convert a per-ticker frame (prices or dividends) to EUR."""
        if self.fx_rates is None:
            # Without an FX series, a conversion is only still defined if
            # everything is already quoted in EUR (then it's the identity).
            non_eur = {t: c for t, c in self.currency.items() if c != "EUR"}
            if non_eur:
                raise ValueError(
                    f"FX rates not available for EUR conversion: {sorted(non_eur)}"
                )
            return frame.copy()

        # Ensure FX rates cover all frame dates by reindexing and filling gaps
        # This handles cases where FX data has gaps or starts later than price data
        fx_aligned = self.fx_rates.reindex(frame.index).ffill().bfill()

        result = frame.copy()
        for ticker in result.columns:
            currency = self.currency.get(ticker, "USD")
            if currency == "USD":
                # USD to EUR: divide by EUR/USD rate
                rate_series = self._fx_series_for_currency(fx_aligned, "USD")
                result[ticker] = result[ticker] / rate_series
            elif currency == "GBP":
                # GBP to EUR: divide by EUR/GBP rate
                rate_series = self._fx_series_for_currency(fx_aligned, "GBP")
                result[ticker] = result[ticker] / rate_series
            elif currency == "GBp":
                # London pence line: 100 GBp = 1 GBP, then GBP to EUR
                rate_series = self._fx_series_for_currency(fx_aligned, "GBP")
                result[ticker] = result[ticker] / (100.0 * rate_series)
            elif currency != "EUR":
                raise ValueError(f"Unsupported currency: {currency}")
        return result

    def total_return_prices(self, currency: Optional[str] = None) -> pd.DataFrame:
        """Total-return price series: distributions reinvested on the ex-date.

        Background: with load_dividends=True, `.prices` is UNADJUSTED
        (split- but not dividend-adjusted); `.prices` alone understates
        distributing instruments -- for a high-yield fund in a strong
        distribution year, the price-only return can understate the total
        return by more than half. This method is the canonical path for
        research returns from a dividend-loaded PriceData.

        Args:
            currency: None = native currency per ticker (like `.prices`).
                      "EUR" = prices AND dividends are first converted to
                      EUR, then reinvested (never label native frames as
                      EUR).

        Returns:
            DataFrame like `.prices` (same index/columns, each column
            starting at its first valid price). Identical to `.prices` or
            `in_eur()` when no dividends are loaded.
        """
        if currency not in (None, "EUR"):
            raise ValueError(f"Unsupported target currency: {currency}")
        prices = self.in_eur() if currency == "EUR" else self.prices.copy()
        if self.dividends is None:
            return prices
        dividends = self.dividends_in_eur() if currency == "EUR" else self.dividends

        # Assign the payment to the first trading day >= the ex-date (not an
        # exact reindex): ex-dates can be missing from the price index, e.g.
        # after align="intersection" or due to a calendar offset -- otherwise
        # the distribution would silently get lost.
        div = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in dividends.columns:
            if col not in div.columns:
                continue
            payments = dividends[col]
            payments = payments[payments != 0.0].dropna()
            for ex_date, amount in payments.items():
                pos = prices.index.searchsorted(pd.Timestamp(ex_date))
                if pos < len(prices.index):
                    div.iloc[pos, div.columns.get_loc(col)] += float(amount)

        result = {}
        for col in prices.columns:
            p = prices[col]
            first = p.first_valid_index()
            if first is None:
                result[col] = p.copy()
                continue
            # Daily return including the distribution reinvested on the ex-date.
            ret = (p + div[col]) / p.shift(1) - 1.0
            ret = ret.loc[first:].fillna(0.0)
            tr = (1.0 + ret).cumprod() * float(p.loc[first])
            result[col] = tr.reindex(p.index)
        return pd.DataFrame(result, index=prices.index)

    @staticmethod
    def _fx_series_for_currency(
        fx_rates: Union[pd.Series, pd.DataFrame],
        source_currency: str,
    ) -> pd.Series:
        """Return the aligned FX series for a given source currency."""
        if isinstance(fx_rates, pd.DataFrame):
            if source_currency not in fx_rates.columns:
                raise ValueError(f"FX rates not available for currency: {source_currency}")
            return pd.to_numeric(fx_rates[source_currency], errors="coerce")
        # Backward compatibility: plain Series means EUR/USD.
        if source_currency != "USD":
            raise ValueError(f"FX rates not available for currency: {source_currency}")
        return pd.to_numeric(fx_rates, errors="coerce")

    def resample_monthly(self) -> "PriceData":
        """
        Resample data to monthly frequency (month-end).

        Returns:
            New PriceData with monthly prices
        """
        return self.resample("monthly")

    def resample(self, frequency: str) -> "PriceData":
        """
        Resample data to a specified frequency.

        Args:
            frequency: "daily", "weekly", "monthly", "quarterly", or "yearly"

        Returns:
            New PriceData with resampled prices
        """
        freq = frequency.lower()
        if freq == "daily":
            return PriceData(
                prices=self.prices.copy(),
                currency=self.currency.copy(),
                fx_rates=self.fx_rates.copy() if self.fx_rates is not None else None,
                volumes=self.volumes.copy() if self.volumes is not None else None,
                dividends=self.dividends.copy() if self.dividends is not None else None,
            )

        if freq == "monthly":
            rule = MONTH_END_FREQ
        elif freq == "weekly":
            rule = "W-FRI"
        elif freq == "quarterly":
            try:
                pd.date_range("2020-01-01", periods=2, freq="QE")
                rule = "QE"
            except ValueError:
                rule = "Q"
        elif freq == "yearly":
            try:
                pd.date_range("2020-01-01", periods=2, freq="YE")
                rule = "YE"
            except ValueError:
                rule = "A"
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        resampled_prices = self.prices.resample(rule).last()
        resampled_fx = self.fx_rates.resample(rule).last() if self.fx_rates is not None else None
        # Volumes are additive across periods.
        resampled_volumes = self.volumes.resample(rule).sum() if self.volumes is not None else None
        # Dividends are summed within each period (not resampled to last)
        resampled_dividends = self.dividends.resample(rule).sum() if self.dividends is not None else None
        resampled_macro = self.macro.resample(rule).last() if self.macro is not None else None
        return PriceData(
            prices=resampled_prices,
            currency=self.currency.copy(),
            fx_rates=resampled_fx,
            volumes=resampled_volumes,
            dividends=resampled_dividends,
            macro=resampled_macro,
        )

    def filter_dates(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> "PriceData":
        """
        Filter data to a date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            New PriceData filtered to date range
        """
        prices = self.prices
        if start_date is not None:
            prices = prices[prices.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            prices = prices[prices.index <= pd.Timestamp(end_date)]

        fx_rates = None
        if self.fx_rates is not None:
            fx_rates = self.fx_rates
            if start_date is not None:
                fx_rates = fx_rates[fx_rates.index >= pd.Timestamp(start_date)]
            if end_date is not None:
                fx_rates = fx_rates[fx_rates.index <= pd.Timestamp(end_date)]

        dividends = None
        if self.dividends is not None:
            dividends = self.dividends
            if start_date is not None:
                dividends = dividends[dividends.index >= pd.Timestamp(start_date)]
            if end_date is not None:
                dividends = dividends[dividends.index <= pd.Timestamp(end_date)]

        volumes = None
        if self.volumes is not None:
            volumes = self.volumes
            if start_date is not None:
                volumes = volumes[volumes.index >= pd.Timestamp(start_date)]
            if end_date is not None:
                volumes = volumes[volumes.index <= pd.Timestamp(end_date)]

        macro = None
        if self.macro is not None:
            macro = self.macro
            if start_date is not None:
                macro = macro[macro.index >= pd.Timestamp(start_date)]
            if end_date is not None:
                macro = macro[macro.index <= pd.Timestamp(end_date)]

        return PriceData(
            prices=prices.copy(),
            currency=self.currency.copy(),
            fx_rates=fx_rates.copy() if fx_rates is not None else None,
            volumes=volumes.copy() if volumes is not None else None,
            dividends=dividends.copy() if dividends is not None else None,
            macro=macro.copy() if macro is not None else None,
        )


class CurrencyConverter:
    """
    Converts prices between currencies.

    Currently supports USD <-> EUR conversions.
    """

    def __init__(self, base_currency: str = "EUR"):
        """
        Initialize converter.

        Args:
            base_currency: Target currency for conversions (default: EUR)
        """
        self.base_currency = base_currency
        self.fx_cache: Dict[str, pd.Series] = {}

    def convert(
        self,
        prices: pd.Series,
        from_currency: str,
        to_currency: str,
        fx_rates: pd.Series
    ) -> pd.Series:
        """
        Convert price series from one currency to another.

        Args:
            prices: Series of prices to convert
            from_currency: Source currency code
            to_currency: Target currency code
            fx_rates: Relevant FX rate series (e.g. EUR/USD or EUR/GBP)

        Returns:
            Converted price series

        Raises:
            ValueError: If conversion is not supported
        """
        if from_currency == to_currency:
            return prices.copy()

        # Align fx_rates to prices index
        fx_aligned = fx_rates.reindex(prices.index).ffill()

        if from_currency in {"USD", "GBP"} and to_currency == "EUR":
            return prices / fx_aligned
        elif from_currency == "GBp" and to_currency == "EUR":
            # fx_aligned is EUR/GBP here; 100 GBp = 1 GBP
            return prices / (100.0 * fx_aligned)
        elif from_currency == "EUR" and to_currency in {"USD", "GBP"}:
            return prices * fx_aligned
        elif from_currency == "EUR" and to_currency == "GBp":
            return prices * fx_aligned * 100.0
        else:
            raise ValueError(f"Unsupported conversion: {from_currency} -> {to_currency}")


class DataLoader:
    """
    Loads price data from various sources.

    Supports Yahoo Finance and CSV files. Optionally caches data locally.
    """

    # Default cache directory
    CACHE_DIR = Path("data")

    # Known-bad source prices. Yahoo serves these permanently wrong for
    # certain LSE tickers -- a refetch does not heal them (auto_adjust=True and
    # =False return identical values), so the correction has to live in the
    # fetch path: data/*.csv is gitignored and the loader overwrites it
    # wholesale on a range miss, which would silently restore the bad print.
    # Map: yahoo_ticker -> {"YYYY-MM-DD": corrected_close}
    KNOWN_BAD_PRICES: Dict[str, Dict[str, float]] = {
        "3LUS.L": {
            # A pence-denominated LSE ETP line: on 2025-10-24 Yahoo printed
            # 14554.4248 into the pence line. That is not noise -- it is its
            # USD-denominated twin line's price times 100 (the twin line
            # closed at 145.544250 that day). Reconstructed exactly instead
            # of interpolated:
            #   145.544250 USD / 1.33275 GBPUSD * 100 = 10920.62 GBp
            # (the earlier interpolation from the neighbours gave 10918.0,
            # 0.02% off).
            "2025-10-24": 10920.62,
            # 2017-06-26/27: the same USD-value-in-the-pence-field bug.
            # BOTH lines are wrong on these two days, so the USD twin line is
            # NOT a usable reference here -- the printed ratio (USD twin
            # price / pence price * 100) is exactly 20.000 on both days
            # instead of the real GBPUSD (1.2744 / 1.2726). Reconstructed as
            # printed/GBPUSD and cross-checked against an independent path
            # (previous close carried forward by 3x S&P):
            #   2291.50 / 1.2744 = 1798.07  vs  1800.78 from the index (0.15%)
            #   2235.74 / 1.2726 = 1756.89  vs  1754.53 from the index (0.13%)
            "2017-06-26": 1798.07,
            "2017-06-27": 1756.89,
        },
        # This is the same leveraged S&P 500 ETP as the pence line above,
        # quoted in USD instead. It is the line that actually trades (4 stale
        # prints vs 62 on the pence line, longest run 2 days vs 16), so it is
        # the better source for this ETP -- but it carries the same two
        # broken days, inflated by exactly 20x (22.9155 -> 458.30 = +1900%
        # while 3x S&P moved +0.09%). Reconstructed from the repaired pence
        # line above (fixed pence price * GBPUSD / 100); the printed/20
        # cross-check agrees to 0.00%.
        "3USL.L": {
            "2017-06-26": 22.9150,
            "2017-06-27": 22.3574,
        },
    }
    # NOT corrected, on purpose: the pence line's stale runs (62 prints that
    # repeat while the index moves, longest run 16 trading days -- e.g.
    # 2018-03-19..22 and 2018-08-24..09-04). Those are not bad prints, they are
    # "nobody traded this line today". Reconstructing them would invent trades
    # that never happened. Use the USD twin line (or the US proxies) for any
    # analysis of this ETP; the pence line is an execution ticker, not a
    # research series.
    # Coverage manifest: maps a cache filename to the date range that has
    # been fetched for it. One file per ticker is kept and the manifest
    # records how far back/forward that single file is known to cover, so
    # we never accumulate one CSV per (start, end) request.
    _CACHE_MANIFEST_NAME = "_cache_manifest.json"
    # In-process cache to avoid repeated disk/network work for identical requests.
    _MEMORY_CACHE: Dict[Tuple[Any, ...], PriceData] = {}
    _MAX_MEMORY_CACHE_ITEMS = 32

    @staticmethod
    def _clone_price_data(data: PriceData) -> PriceData:
        """Return a defensive copy so callers can't mutate cached objects."""
        return PriceData(
            prices=data.prices.copy(),
            currency=data.currency.copy(),
            fx_rates=data.fx_rates.copy() if data.fx_rates is not None else None,
            volumes=data.volumes.copy() if data.volumes is not None else None,
            dividends=data.dividends.copy() if data.dividends is not None else None,
            macro=data.macro.copy() if data.macro is not None else None,
        )

    @classmethod
    def _memory_key_yahoo(
        cls,
        yahoo_tickers: List[str],
        start: str,
        end: str,
        currency: str,
        align: str,
        skip_failed: bool,
        load_dividends: bool,
        load_volumes: bool,
        validate: bool,
    ) -> Tuple[Any, ...]:
        return (
            "yahoo",
            tuple(yahoo_tickers),
            start,
            end,
            currency,
            align.lower(),
            bool(skip_failed),
            bool(load_dividends),
            bool(load_volumes),
            bool(validate),
        )

    @classmethod
    def _store_memory_cache(cls, key: Tuple[Any, ...], data: PriceData) -> None:
        """Store an item in the in-memory cache with simple FIFO eviction."""
        cls._MEMORY_CACHE[key] = cls._clone_price_data(data)
        while len(cls._MEMORY_CACHE) > cls._MAX_MEMORY_CACHE_ITEMS:
            oldest_key = next(iter(cls._MEMORY_CACHE))
            cls._MEMORY_CACHE.pop(oldest_key, None)

    # --- Disk cache: one file per ticker + coverage manifest -------------

    @staticmethod
    def _mask_spliced_history(prices: pd.DataFrame) -> pd.DataFrame:
        """Drop the segment of a spliced series that predates its currency line.

        A pence-denominated LSE ETP line prints US cents before 2017-03-16 and
        pence after. Reading the head as pence misprices it by GBPUSD. Rather
        than silently serve it, blank it and say so -- deep history belongs to
        the clean USD-denominated twin line.
        """
        for ticker, valid_from in TICKER_HISTORY_VALID_FROM.items():
            if ticker not in prices.columns:
                continue
            stale = prices.index < pd.Timestamp(valid_from)
            dropped = int((stale & prices[ticker].notna()).sum())
            if not dropped:
                continue
            prices.loc[stale, ticker] = float("nan")
            warnings.warn(
                f"{ticker}: dropped {dropped} prices before {valid_from} -- that "
                f"segment is a different currency line (spliced series, T-0433). "
                f"Use the clean USD line for deep history.",
                UserWarning,
                stacklevel=2,
            )
        return prices

    @classmethod
    def _apply_known_bad_prices(cls, ticker: str, series: pd.Series) -> pd.Series:
        """Overwrite closes that Yahoo serves permanently wrong.

        Applied on both the cache-hit and the download path, so a corrected
        value survives the wholesale CSV rewrite that follows a range miss.
        """
        overrides = cls.KNOWN_BAD_PRICES.get(ticker)
        if not overrides:
            return series
        hits = {
            pd.Timestamp(day): value
            for day, value in overrides.items()
            if pd.Timestamp(day) in series.index
        }
        if not hits:
            return series
        series = series.copy()
        for timestamp, value in hits.items():
            series.loc[timestamp] = value
        return series

    @staticmethod
    def _cache_filename(key: str, suffix: str = "") -> str:
        """Stable cache filename for a ticker/series, independent of the
        requested date range. ``key`` may be a Yahoo ticker, an FX pair or
        a FRED series id; dots are replaced so the name is filesystem-safe.
        """
        return f"{key.replace('.', '_')}{suffix}.csv"

    @classmethod
    def _manifest_path(cls) -> Path:
        return cls.CACHE_DIR / cls._CACHE_MANIFEST_NAME

    @classmethod
    def _load_manifest(cls) -> Dict[str, Dict[str, str]]:
        """Load the cache coverage manifest (filename -> {start, end}).

        Self-heals poisoned entries: an ``end`` beyond ``_max_recordable_end``
        (today + 1) cannot correspond to real data, so it is dropped (keeping
        ``start``). That turns the next coverage check into a miss, forcing a
        union re-fetch that overwrites the stale file with full history -- the
        only way to thaw a file an earlier run froze by recording a far-future
        requested end (see ``_record_cache_range``).
        """
        path = cls._manifest_path()
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            # A corrupt manifest must not break loading; treat as empty
            # (every ticker is then re-fetched and the manifest rebuilt).
            return {}
        if not isinstance(data, dict):
            return {}
        cap = cls._max_recordable_end()
        for entry in data.values():
            if isinstance(entry, dict) and (entry.get("end") or "") > cap:
                entry.pop("end", None)
        return data

    @classmethod
    def _save_manifest(cls, manifest: Dict[str, Dict[str, str]]) -> None:
        """Persist the manifest atomically (temp file + rename)."""
        path = cls._manifest_path()
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        tmp.replace(path)

    @staticmethod
    def _range_covered(
        entry: Optional[Dict[str, str]], start: str, end: str
    ) -> bool:
        """True if a manifest entry's recorded range spans [start, end].

        ISO date strings (YYYY-MM-DD) compare lexically == chronologically.
        """
        if not entry or not start or not end:
            return False
        cached_start = entry.get("start")
        cached_end = entry.get("end")
        if not cached_start or not cached_end:
            return False
        return cached_start <= start and cached_end >= end

    @classmethod
    def _max_recordable_end(cls) -> str:
        """Upper bound for a recordable cache ``end``: tomorrow (today + 1).

        A request may ask for data far in the future (e.g. a backtest with a
        padded end date), but the cache can only ever hold data up to "now".
        Recording a future end would make ``_range_covered`` report a
        permanent false cache-hit and freeze the file until that date passes
        -- this once hid a 10-day-stale single-stock series and inverted a
        live rebalance signal. The end-exclusive price convention means
        today's bar is requested as ``end = today + 1``, so the cap is
        today + 1.
        """
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # A cached daily series whose last bar is within this many calendar days of
    # "today" is treated as ACTIVE: its tail may still grow, so the manifest end
    # must not claim coverage past the last real bar (else a not-yet-published
    # bar freezes a stale series). A last bar older than this is treated as
    # SETTLED/delisted: refetching only re-downloads unchanging history, so the
    # fetch-intent end is recorded instead to avoid re-fetching it every run.
    # 45d spans long holiday/low-liquidity gaps without mistaking a live ticker
    # for a dead one (delisted names in this cache are years, not weeks, stale).
    _ACTIVE_TAIL_GRACE_DAYS = 45

    @classmethod
    def _record_cache_range(
        cls,
        filename: str,
        start: str,
        end: str,
        last_data_date: Optional[Any] = None,
    ) -> None:
        """Merge [start, end] into the manifest entry for ``filename``.

        Read-modify-write so concurrent single-ticker writers only ever
        lose a range record (-> a harmless re-fetch), never corrupt data.

        The merged ``end`` is capped at ``_max_recordable_end`` so neither a
        future-dated request nor an already-poisoned manifest entry can claim
        coverage the file does not have; capping the merge (not just the new
        value) self-heals a manifest that was poisoned by an earlier run.

        ``last_data_date`` is the last index actually returned by the
        source. Dense daily series (price/volume/FX/FRED) are written to the
        single file wholesale, so for an ACTIVE ticker their coverage ends at
        that bar -- NOT at the requested ``end``. Recording the requested end
        claims coverage the file lacks and freezes a stale series until that end
        passes: the today+1 cap stops a *future* end, but not the case where the
        requested end equals today+1 while the last bar is older (today's bar not
        yet published, a data gap, a weekend). That once hid a large stop-loss
        breach on a single-stock position -- a scheduled run requested
        end=today+1, the source's last bar for that line was several days
        stale, and the manifest froze the pre-drawdown price.
        For an active ticker the recorded end is therefore the day after the last
        real bar and OVERRIDES ``prev_end`` (a wholesale overwrite replaces the
        file, so an older poisoned end must not survive the max-merge). For a
        SETTLED/delisted ticker (last bar older than the active grace) the
        fetch-intent end is kept instead, so its unchanging history is not
        re-downloaded every run. Omit ``last_data_date`` for sparse series
        (dividends): their last event is not their coverage end.
        """
        cap = cls._max_recordable_end()
        manifest = cls._load_manifest()
        entry = dict(manifest.get(filename) or {})
        prev_start = entry.get("start")
        prev_end = entry.get("end")
        entry["start"] = min(start, prev_start) if prev_start else start
        base_end = min(end, cap)
        if last_data_date is not None and not pd.isna(pd.Timestamp(last_data_date)):
            last_ts = pd.Timestamp(last_data_date).normalize()
            data_end = (last_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            today = pd.Timestamp(datetime.now().date())
            active = last_ts >= today - pd.Timedelta(days=cls._ACTIVE_TAIL_GRACE_DAYS)
            if active and data_end < base_end:
                # Active tail not fully published: claim only the real bars and
                # drop any poisoned prev_end so the next request re-fetches.
                entry["end"] = data_end
                manifest[filename] = entry
                cls._save_manifest(manifest)
                return
        merged_end = max(base_end, prev_end) if prev_end else base_end
        entry["end"] = min(merged_end, cap)
        manifest[filename] = entry
        cls._save_manifest(manifest)

    @classmethod
    def reconcile_manifest_with_files(cls, grace_days: Optional[int] = None) -> Dict[str, str]:
        """Repair manifest ends against the bars actually on disk.

        Fixes pre-existing poison that the write-path cannot reach on its own: a
        manifest entry claiming coverage past its file's last row makes
        ``_range_covered`` report a hit, so the file is never re-fetched and
        ``_record_cache_range`` never runs to self-heal. Per file:

        * ACTIVE (last bar within the grace window): shrink the end to the day
          after the last real bar, so the next request re-fetches the fresh tail.
        * SETTLED/delisted (older last bar): mark covered through now
          (``_max_recordable_end``), so its unchanging history is served from
          cache instead of re-fetched every run (Yahoo still serves the delisted
          history, so this is a churn choice, not a correctness one).

        Returns {filename: new_end} for the entries it changed. Offline (reads
        only ``data/``); idempotent. Dividend files (``*_dividends.csv``) are
        sparse -- their coverage is not the last event -- and are left untouched.
        """
        grace = cls._ACTIVE_TAIL_GRACE_DAYS if grace_days is None else grace_days
        cap = cls._max_recordable_end()
        today = pd.Timestamp(datetime.now().date())
        manifest = cls._load_manifest()
        changed: Dict[str, str] = {}
        for filename, entry in list(manifest.items()):
            recorded_end = (entry or {}).get("end")
            if not recorded_end or filename.endswith("_dividends.csv"):
                continue
            path = cls.CACHE_DIR / filename
            if not path.exists():
                continue
            try:
                disk = pd.read_csv(path)
                # Price/volume/FX files store the date in the index column;
                # FRED files carry it as the first data column.
                last = pd.to_datetime(disk[disk.columns[0]], errors="coerce").max()
            except (ValueError, KeyError, pd.errors.EmptyDataError, IndexError):
                continue
            if pd.isna(last):
                continue
            last_ts = pd.Timestamp(last).normalize()
            if last_ts >= today - pd.Timedelta(days=grace):
                target = (last_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # active
            else:
                target = cap  # settled/delisted -> covered through now
            if target != recorded_end:
                entry["end"] = target
                manifest[filename] = entry
                changed[filename] = target
        if changed:
            cls._save_manifest(manifest)
        return changed

    @classmethod
    def yahoo(
        cls,
        tickers: List[str],
        start: str = "2000-01-01",
        end: Optional[str] = None,
        currency: str = "EUR",
        cache: bool = True,
        align: str = "intersection",
        skip_failed: bool = True,
        load_dividends: bool = False,
        load_volumes: bool = False,
        validate: bool = True
    ) -> PriceData:
        """
        Load price data from Yahoo Finance.

        Args:
            tickers: List of ticker symbols (WKN, name, or Yahoo ticker)
            start: Start date (YYYY-MM-DD format)
            end: End date (None = today)
            currency: Target currency for the ATTACHED FX series (default: EUR).
                Does NOT convert the prices themselves -- `.prices` stays
                native; get EUR series via `in_eur()`/`total_return_prices("EUR")`.
            cache: Whether to cache downloaded data
            align: Alignment mode for missing data:
                - "intersection": drop rows with any NaN (default)
                - "ffill": forward-fill per ticker, keep longer history
            skip_failed: If True, skip tickers that fail to download instead of raising error
            load_dividends: If True, also load dividend data for DRIP simulation.
                WARNING: prices are then UNADJUSTED (split- but not
                dividend-adjusted); get returns via `total_return_prices()`,
                never from `.prices` alone.
            load_volumes: If True, also load/share volume data for liquidity guards
            validate: If True, emit warnings for data issues (default: True)

        Returns:
            PriceData. Without load_dividends: dividend-adjusted (total-return)
            close prices. With load_dividends: unadjusted prices + `.dividends`.
        """
        # Resolve all tickers to Yahoo format
        yahoo_tickers = [get_yahoo_ticker(t) for t in tickers]
        ticker_map = dict(zip(yahoo_tickers, tickers))

        # Determine date range
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        memory_key = cls._memory_key_yahoo(
            yahoo_tickers=yahoo_tickers,
            start=start,
            end=end,
            currency=currency,
            align=align,
            skip_failed=skip_failed,
            load_dividends=load_dividends,
            load_volumes=load_volumes,
            validate=validate,
        )
        cached = cls._MEMORY_CACHE.get(memory_key)
        if cached is not None:
            return cls._clone_price_data(cached)

        # Check cache first
        if cache:
            cls.CACHE_DIR.mkdir(exist_ok=True)

        # Download market data
        all_prices = {}
        all_volumes = {}
        all_dividends = {}  # Collect dividend data if requested
        currencies = {}
        failed_tickers = []  # Collect failed tickers for summary

        # Suppress yfinance warnings when skip_failed is enabled
        import logging
        import warnings as _warnings
        if skip_failed:
            logging.getLogger('yfinance').setLevel(logging.CRITICAL)
            _warnings.filterwarnings('ignore', module='yfinance')

        tickers_to_download_prices = []
        tickers_to_download_volumes = []
        # Use different cache files for dividend-mode (unadjusted prices) vs total-return mode
        cache_suffix = "_unadj" if load_dividends else ""
        # One CSV per ticker; the manifest says how far it already covers.
        manifest = cls._load_manifest() if cache else {}
        for yahoo_ticker in yahoo_tickers:
            price_name = cls._cache_filename(yahoo_ticker, cache_suffix)
            volume_name = cls._cache_filename(yahoo_ticker, "_vol")
            cache_file = cls.CACHE_DIR / price_name
            volume_cache_file = cls.CACHE_DIR / volume_name

            price_hit = (
                cache
                and cls._range_covered(manifest.get(price_name), start, end)
                and cache_file.exists()
            )
            if price_hit:
                df = pd.read_csv(cache_file, index_col=0)
                df.index = pd.to_datetime(df.index)
                if yahoo_ticker in df.columns:
                    all_prices[yahoo_ticker] = cls._apply_known_bad_prices(
                        yahoo_ticker, df[yahoo_ticker]
                    )
                else:
                    tickers_to_download_prices.append(yahoo_ticker)
            else:
                tickers_to_download_prices.append(yahoo_ticker)

            if load_volumes:
                volume_hit = (
                    cache
                    and cls._range_covered(manifest.get(volume_name), start, end)
                    and volume_cache_file.exists()
                )
                if volume_hit:
                    vdf = pd.read_csv(volume_cache_file, index_col=0)
                    vdf.index = pd.to_datetime(vdf.index)
                    if yahoo_ticker in vdf.columns:
                        all_volumes[yahoo_ticker] = vdf[yahoo_ticker]
                    else:
                        tickers_to_download_volumes.append(yahoo_ticker)
                else:
                    tickers_to_download_volumes.append(yahoo_ticker)

            currencies[yahoo_ticker] = detect_currency(yahoo_ticker)

        download_universe = sorted(set(tickers_to_download_prices + tickers_to_download_volumes))
        # Fetch the union of the requested window and any range already
        # recorded for these tickers. The download is then always a
        # superset of the existing per-ticker file, so it can simply be
        # overwritten -- no merge of differently-adjusted price vintages.
        download_start, download_end = start, end
        if cache:
            for yahoo_ticker in download_universe:
                for name in (
                    cls._cache_filename(yahoo_ticker, cache_suffix),
                    cls._cache_filename(yahoo_ticker, "_vol"),
                ):
                    entry = manifest.get(name)
                    if entry and entry.get("start"):
                        download_start = min(download_start, entry["start"])
                    if entry and entry.get("end"):
                        download_end = max(download_end, entry["end"])
        if download_universe:
            try:
                # When loading dividends (for DRIP/tax), use auto_adjust=False
                # to get split-adjusted but NOT dividend-adjusted prices.
                # This avoids double-counting dividends (once in price, once explicitly).
                ticker_data = yf.download(
                    download_universe,
                    start=download_start,
                    end=download_end,
                    progress=False,
                    auto_adjust=not load_dividends,
                    group_by="column",
                )
                if ticker_data.empty and tickers_to_download_prices:
                    raise ValueError("No data returned for requested tickers")

                close_frame = None
                volume_frame = None
                if isinstance(ticker_data.columns, pd.MultiIndex):
                    if "Close" in ticker_data.columns.get_level_values(0):
                        close_frame = ticker_data["Close"]
                        if isinstance(close_frame, pd.Series):
                            close_frame = close_frame.to_frame(name=download_universe[0])
                    if "Volume" in ticker_data.columns.get_level_values(0):
                        volume_frame = ticker_data["Volume"]
                        if isinstance(volume_frame, pd.Series):
                            volume_frame = volume_frame.to_frame(name=download_universe[0])
                else:
                    close_frame = pd.DataFrame({download_universe[0]: ticker_data["Close"]})
                    if "Volume" in ticker_data.columns:
                        volume_frame = pd.DataFrame({download_universe[0]: ticker_data["Volume"]})

                for yahoo_ticker in download_universe:
                    # Price series (required for all backtests)
                    if yahoo_ticker in tickers_to_download_prices:
                        if close_frame is None or yahoo_ticker not in close_frame.columns:
                            if skip_failed:
                                failed_tickers.append(yahoo_ticker)
                                continue
                            raise ValueError(f"No data returned for {yahoo_ticker}")

                        close_prices = close_frame[yahoo_ticker].dropna()
                        if close_prices.empty:
                            if skip_failed:
                                failed_tickers.append(yahoo_ticker)
                                continue
                            raise ValueError(f"No data returned for {yahoo_ticker}")

                        close_prices = cls._apply_known_bad_prices(yahoo_ticker, close_prices)
                        df = pd.DataFrame({yahoo_ticker: close_prices})
                        if cache:
                            price_name = cls._cache_filename(yahoo_ticker, cache_suffix)
                            df.to_csv(cls.CACHE_DIR / price_name)
                            cls._record_cache_range(
                                price_name, download_start, download_end,
                                last_data_date=close_prices.index.max(),
                            )
                        all_prices[yahoo_ticker] = close_prices

                    # Volume series (optional, for liquidity guards)
                    if load_volumes and yahoo_ticker in tickers_to_download_volumes:
                        if volume_frame is not None and yahoo_ticker in volume_frame.columns:
                            volume_series = pd.to_numeric(volume_frame[yahoo_ticker], errors="coerce").dropna()
                            if len(volume_series) > 0:
                                all_volumes[yahoo_ticker] = volume_series
                                if cache:
                                    volume_name = cls._cache_filename(yahoo_ticker, "_vol")
                                    pd.DataFrame({yahoo_ticker: volume_series}).to_csv(
                                        cls.CACHE_DIR / volume_name
                                    )
                                    cls._record_cache_range(
                                        volume_name, download_start, download_end,
                                        last_data_date=volume_series.index.max(),
                                    )
            except (ValueError, KeyError, requests.RequestException, pd.errors.EmptyDataError) as e:
                if tickers_to_download_prices:
                    if skip_failed:
                        failed_tickers.extend(tickers_to_download_prices)
                    else:
                        raise ValueError(f"Failed to download tickers: {e}")

        # Load dividend data if requested
        if load_dividends:
            for yahoo_ticker in yahoo_tickers:
                if yahoo_ticker in failed_tickers:
                    continue
                div_name = cls._cache_filename(yahoo_ticker, "_dividends")
                div_cache_file = cls.CACHE_DIR / div_name
                div_hit = (
                    cache
                    and cls._range_covered(manifest.get(div_name), start, end)
                    and div_cache_file.exists()
                )
                if div_hit:
                    div_df = pd.read_csv(div_cache_file, index_col=0)
                    div_df.index = pd.to_datetime(div_df.index)
                    if yahoo_ticker in div_df.columns:
                        series = div_df[yahoo_ticker]
                        windowed = series[
                            (series.index >= pd.Timestamp(start)) &
                            (series.index <= pd.Timestamp(end))
                        ]
                        if len(windowed) > 0:
                            all_dividends[yahoo_ticker] = windowed
                else:
                    try:
                        ticker_obj = yf.Ticker(yahoo_ticker)
                        dividends = ticker_obj.dividends
                        if dividends is not None and len(dividends) > 0:
                            if isinstance(dividends.index, pd.DatetimeIndex) and dividends.index.tz is not None:
                                if validate:
                                    print(f"  Normalizing dividend timezone for {yahoo_ticker} ({dividends.index.tz})")
                                dividends.index = dividends.index.tz_convert(None)
                            dividends.name = yahoo_ticker
                            if cache:
                                # Persist the full dividend history once;
                                # the manifest records its reach so the
                                # single file serves every future window.
                                # Coverage start = min(first dividend,
                                # requested start): if a ticker pays its
                                # first dividend only AFTER the window end
                                # (e.g. PYPL in 2025), index.min() > end
                                # would leave the manifest entry broken.
                                pd.DataFrame({yahoo_ticker: dividends}).to_csv(div_cache_file)
                                coverage_start = min(
                                    dividends.index.min(), pd.Timestamp(start)
                                )
                                cls._record_cache_range(
                                    div_name,
                                    coverage_start.strftime("%Y-%m-%d"),
                                    download_end,
                                )
                            windowed = dividends[
                                (dividends.index >= pd.Timestamp(start)) &
                                (dividends.index <= pd.Timestamp(end))
                            ]
                            if len(windowed) > 0:
                                all_dividends[yahoo_ticker] = windowed
                    except (ValueError, KeyError, requests.RequestException, AttributeError):
                        # Dividend data not available - continue without
                        pass

        # Print summary of failed tickers (if any)
        if failed_tickers:
            print(f"  Skipped {len(failed_tickers)} unavailable tickers: {', '.join(failed_tickers[:5])}"
                  + (f"... and {len(failed_tickers) - 5} more" if len(failed_tickers) > 5 else ""))

        # Combine all prices
        prices_df = pd.DataFrame(all_prices)

        # Per-ticker cache files (and union-range downloads) may now span a
        # wider range than requested; restrict to the requested window so
        # the result is identical to a fresh per-request fetch. `end` is
        # exclusive, matching yfinance's download semantics.
        if not prices_df.empty:
            prices_df = prices_df.sort_index()
            window = (
                (prices_df.index >= pd.Timestamp(start))
                & (prices_df.index < pd.Timestamp(end))
            )
            prices_df = prices_df.loc[window]

        # Analyze which ticker limits the start date BEFORE dropna
        ticker_starts = {}
        for col in prices_df.columns:
            first_valid = prices_df[col].first_valid_index()
            if first_valid is not None:
                ticker_starts[col] = first_valid

        # Record the last REAL print per ticker BEFORE align="ffill" makes it
        # unrecognizable. After that it's no longer reconstructable whether
        # the tail value is today's price or an old one carried forward.
        last_print = {}
        for col in prices_df.columns:
            last_valid = prices_df[col].last_valid_index()
            if last_valid is not None:
                last_print[col] = last_valid

        # Track limiting ticker info for warning after alignment decision
        limiting_ticker_info = None
        if ticker_starts:
            # Find the ticker with the latest start date (most limiting)
            limiting_ticker = max(ticker_starts, key=ticker_starts.get)
            limiting_date = ticker_starts[limiting_ticker]
            earliest_date = min(ticker_starts.values())

            if limiting_date > earliest_date:
                days_lost = (limiting_date - earliest_date).days
                limiting_ticker_info = {
                    "ticker": limiting_ticker,
                    "start_date": limiting_date,
                    "earliest_date": earliest_date,
                    "days_lost": days_lost,
                }

        align_mode = align.lower()
        if align_mode == "intersection":
            # Drop rows with any NaN values (align all series)
            prices_df = prices_df.dropna()
        elif align_mode == "ffill":
            # Forward-fill gaps per ticker to retain longer histories.
            # Leading NaNs (before first available price) are kept.
            prices_df = prices_df.ffill()
            prices_df = prices_df.dropna(how="all")
        else:
            raise ValueError(f"Unsupported alignment mode: {align}")

        # Emit warning about limiting ticker, tailored to alignment mode
        if validate and limiting_ticker_info:
            import warnings
            info = limiting_ticker_info
            ticker = info["ticker"]
            start_date = info["start_date"].strftime("%Y-%m-%d")
            earliest = info["earliest_date"].strftime("%Y-%m-%d")
            days_lost = info["days_lost"]

            if align_mode == "intersection":
                # With intersection mode, data is actually truncated
                warnings.warn(
                    f"Data range limited by {ticker} (starts {start_date}). "
                    f"Other tickers have data from {earliest} ({days_lost} days earlier). "
                    f"Consider using align='ffill' to keep longer history, "
                    f"or remove {ticker} to extend the backtest range.",
                    UserWarning
                )
            elif align_mode == "ffill":
                # With ffill mode, the ticker just joins later - less critical
                warnings.warn(
                    f"{ticker} has shorter history (starts {start_date}, "
                    f"{days_lost} days after other tickers). "
                    f"It will be included in strategy calculations once sufficient data is available.",
                    UserWarning
                )

        prices_df = cls._mask_spliced_history(prices_df)

        # Load FX rates if currency conversion is needed
        fx_rates = None
        if currency == "EUR":
            needed_fx_pairs = {}
            if any(c == "USD" for c in currencies.values()):
                needed_fx_pairs["USD"] = "EURUSD=X"
            # GBp (pence) needs the same EURGBP series as GBP -- just the 1/100 factor
            if any(c in ("GBP", "GBp") for c in currencies.values()):
                needed_fx_pairs["GBP"] = "EURGBP=X"

            if needed_fx_pairs:
                fx_series_map: Dict[str, pd.Series] = {}
                for source_currency, pair in needed_fx_pairs.items():
                    fx_series = cls.get_fx_rates(start, end, pair=pair, cache=cache)
                    fx_series = fx_series.reindex(prices_df.index).ffill().bfill()
                    fx_series_map[source_currency] = fx_series

                if len(fx_series_map) == 1 and "USD" in fx_series_map:
                    # Backward compatibility for existing USD-only callers/tests.
                    fx_rates = fx_series_map["USD"]
                else:
                    fx_rates = pd.DataFrame(fx_series_map, index=prices_df.index)

        # Build dividends DataFrame if we loaded dividend data
        dividends_df = None
        if load_dividends and all_dividends:
            dividends_df = pd.DataFrame(all_dividends)
            # Fill missing dividend values with 0 (most days have no dividend)
            dividends_df = dividends_df.fillna(0.0)

        # Build volume DataFrame if requested.
        volumes_df = None
        if load_volumes and all_volumes:
            volumes_df = pd.DataFrame(all_volumes)
            # Align to price index without forward filling volume.
            volumes_df = volumes_df.reindex(prices_df.index)

        result = PriceData(
            prices=prices_df,
            currency=currencies,
            fx_rates=fx_rates,
            volumes=volumes_df,
            dividends=dividends_df,
            last_print={t: d for t, d in last_print.items() if t in prices_df.columns},
        )
        cls._store_memory_cache(memory_key, result)
        return result

    @classmethod
    def fred_series(
        cls,
        series_id: str,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        cache: bool = True,
    ) -> pd.Series:
        """
        Load a single FRED series via the public CSV endpoint.

        Args:
            series_id: FRED series ID (e.g., "DGS10")
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD or None for today)
            cache: Whether to cache downloaded data

        Returns:
            Series indexed by date
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        if cache:
            cls.CACHE_DIR.mkdir(exist_ok=True)

        # One file per FRED series, range-independent; the manifest tracks
        # how far it covers so we never spawn one file per (start, end).
        fred_name = f"fred_{series_id}.csv"
        cache_file = cls.CACHE_DIR / fred_name
        manifest_entry = cls._load_manifest().get(fred_name) if cache else None

        if (
            cache
            and cls._range_covered(manifest_entry, start, end)
            and cache_file.exists()
        ):
            df = pd.read_csv(cache_file)
        else:
            # Fetch the union of the requested window and any prior
            # coverage, then overwrite the single file with that superset.
            fetch_start, fetch_end = start, end
            if manifest_entry:
                if manifest_entry.get("start"):
                    fetch_start = min(fetch_start, manifest_entry["start"])
                if manifest_entry.get("end"):
                    fetch_end = max(fetch_end, manifest_entry["end"])
            url = (
                "https://fred.stlouisfed.org/graph/fredgraph.csv"
                f"?id={series_id}&cosd={fetch_start}&coed={fetch_end}"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            if cache:
                df.to_csv(cache_file, index=False)
                cls._record_cache_range(
                    fred_name, fetch_start, fetch_end,
                    last_data_date=pd.to_datetime(df[df.columns[0]], errors="coerce").max(),
                )

        df.rename(columns={df.columns[0]: "date", df.columns[1]: series_id}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        series = pd.to_numeric(df[series_id], errors="coerce").dropna()
        # Cache file may span a wider range than requested; restrict it.
        series = series[
            (series.index >= pd.Timestamp(start))
            & (series.index <= pd.Timestamp(end))
        ]
        return series

    @classmethod
    def fred(
        cls,
        series_ids: List[str],
        start: str = "2000-01-01",
        end: Optional[str] = None,
        cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load multiple FRED series into a single DataFrame.

        Args:
            series_ids: List of FRED series IDs
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD or None)
            cache: Whether to cache downloaded data

        Returns:
            DataFrame indexed by date with series columns
        """
        data = {}
        for series_id in series_ids:
            data[series_id] = cls.fred_series(
                series_id=series_id,
                start=start,
                end=end,
                cache=cache,
            )
        return pd.DataFrame(data).sort_index()

    @classmethod
    def load_with_fred(
        cls,
        tickers: List[str],
        fred_series: Optional[List[str]] = None,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        currency: str = "EUR",
        cache: bool = True,
        align: str = "intersection",
        skip_failed: bool = True,
        load_dividends: bool = False,
        load_volumes: bool = False,
        validate: bool = True,
    ) -> PriceData:
        """
        Unified loader for price data and optional FRED macro series.

        Args:
            tickers: List of Yahoo Finance tickers
            fred_series: Optional list of FRED series IDs
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD or None)
            currency: Target currency for prices
            cache: Whether to cache downloaded data
            align: Price alignment mode
            skip_failed: Skip failed tickers
            load_dividends: Load dividend data
            load_volumes: Load traded volume series
            validate: Emit warnings for data issues

        Returns:
            PriceData with macro series populated when requested
        """
        price_data = cls.yahoo(
            tickers=tickers,
            start=start,
            end=end,
            currency=currency,
            cache=cache,
            align=align,
            skip_failed=skip_failed,
            load_dividends=load_dividends,
            load_volumes=load_volumes,
            validate=validate,
        )
        macro = None
        if fred_series:
            macro = cls.fred(
                series_ids=fred_series,
                start=start,
                end=end,
                cache=cache,
            )
        return PriceData(
            prices=price_data.prices,
            currency=price_data.currency,
            fx_rates=price_data.fx_rates,
            volumes=price_data.volumes,
            dividends=price_data.dividends,
            macro=macro,
        )

    @classmethod
    def get_fx_rates(
        cls,
        start: str,
        end: str,
        pair: str = "EURUSD=X",
        cache: bool = True
    ) -> pd.Series:
        """
        Load exchange rates from Yahoo Finance.

        Args:
            start: Start date
            end: End date
            pair: Currency pair ticker (default: EUR/USD)
            cache: Whether to cache data

        Returns:
            Series of exchange rates
        """
        # One file per FX pair, range-independent; the manifest tracks
        # how far it covers (see DataLoader._record_cache_range).
        fx_name = f"fx_{pair.replace('=', '')}.csv"
        cache_file = cls.CACHE_DIR / fx_name

        manifest_entry = cls._load_manifest().get(fx_name) if cache else None
        if (
            cache
            and cls._range_covered(manifest_entry, start, end)
            and cache_file.exists()
        ):
            df = pd.read_csv(cache_file, index_col=0)
            df.index = pd.to_datetime(df.index)
            return df["Close"]

        try:
            # Try downloading with extended date range if original range is too short
            # This handles cases where FX data isn't available for very short periods
            from datetime import datetime, timedelta

            # Fetch the union of the requested window and any prior
            # coverage, so the single fx file is overwritten with a
            # superset rather than spawning one file per (start, end).
            req_start, req_end = start, end
            if manifest_entry:
                if manifest_entry.get("start"):
                    req_start = min(req_start, manifest_entry["start"])
                if manifest_entry.get("end") and req_end:
                    req_end = max(req_end, manifest_entry["end"])

            start_dt = datetime.strptime(req_start, "%Y-%m-%d")
            end_dt = datetime.strptime(req_end, "%Y-%m-%d") if req_end else datetime.now()

            # If date range is less than 7 days, extend start date back
            if (end_dt - start_dt).days < 7:
                extended_start = (start_dt - timedelta(days=30)).strftime("%Y-%m-%d")
            else:
                extended_start = req_start

            fx_data = yf.download(
                pair,
                start=extended_start,
                end=req_end,
                progress=False,
                auto_adjust=True
            )
            covered_start, covered_end = extended_start, req_end
            if fx_data.empty:
                # Fallback: try to get recent FX data (last 30 days)
                fallback_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                fx_data = yf.download(
                    pair,
                    start=fallback_start,
                    progress=False,
                    auto_adjust=True
                )
                if fx_data.empty:
                    raise ValueError(f"No FX data returned for {pair}")
                # Partial fallback data: do not claim the wide range.
                covered_start, covered_end = fallback_start, None

            # Handle multi-level columns from newer yfinance versions
            if isinstance(fx_data.columns, pd.MultiIndex):
                close_prices = fx_data["Close"].iloc[:, 0]
            else:
                close_prices = fx_data["Close"]

            if cache:
                cls.CACHE_DIR.mkdir(exist_ok=True)
                pd.DataFrame({"Close": close_prices}).to_csv(cache_file)
                if covered_end:
                    cls._record_cache_range(
                        fx_name, covered_start, covered_end,
                        last_data_date=close_prices.index.max(),
                    )

            return close_prices
        except (ValueError, KeyError, requests.RequestException, pd.errors.EmptyDataError) as e:
            raise ValueError(f"Failed to download FX rates: {e}")

    @classmethod
    def csv(
        cls,
        path: Union[str, Path],
        date_column: str = "Date",
        date_format: str = "%Y-%m-%d",
        currency: Optional[Dict[str, str]] = None
    ) -> PriceData:
        """
        Load price data from a CSV file.

        The CSV should have a date column and one column per ticker
        with adjusted close prices.

        Args:
            path: Path to CSV file
            date_column: Name of the date column
            date_format: Date format string
            currency: Optional dict mapping tickers to currencies

        Returns:
            PriceData with loaded prices
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path, parse_dates=[date_column], date_format=date_format)
        df = df.set_index(date_column)
        df = df.sort_index()

        # Detect currencies if not provided
        if currency is None:
            currency = {col: detect_currency(col) for col in df.columns}

        return PriceData(prices=df, currency=currency)

    @classmethod
    def clear_cache(cls) -> int:
        """
        Clear all cached data files.

        Returns:
            Number of CSV files deleted (the coverage manifest is removed
            too but not counted, as it is bookkeeping rather than data).
        """
        count = 0
        for file in cls.CACHE_DIR.glob("*.csv"):
            file.unlink()
            count += 1
        manifest_path = cls._manifest_path()
        if manifest_path.exists():
            manifest_path.unlink()
        cls._MEMORY_CACHE.clear()
        return count

    @classmethod
    def list_cache(cls) -> List[str]:
        """
        List all cached data files.

        Returns:
            List of cached file names
        """
        if not cls.CACHE_DIR.exists():
            return []
        return [f.name for f in cls.CACHE_DIR.glob("*.csv")]
