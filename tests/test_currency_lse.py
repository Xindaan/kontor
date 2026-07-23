"""
Regression tests for the LSE currency-line error class.

Background: `.L` identifies the exchange, not the quoted currency.
The LSE lists several lines of the same ETP. Reading the suffix as GBP
blindly causes a miscalculation of GBPUSD (~1.34x) for USD lines and
of exactly 100x for pence lines (GBp). Both factors were reproduced
against live quotes on 2026-07-10.
"""

import warnings

import pandas as pd
import pytest

from backtest.assets import detect_currency
from backtest.data import DataLoader, PriceData


def _price_data(ticker: str, price: float, currency: str, eur_gbp: float) -> PriceData:
    idx = pd.to_datetime(["2026-07-09"])
    return PriceData(
        prices=pd.DataFrame({ticker: [price]}, index=idx),
        currency={ticker: currency},
        fx_rates=pd.DataFrame({"GBP": [eur_gbp]}, index=idx),
    )


class TestLseCurrencyLines:
    def test_pence_line_converts_via_hundred(self):
        """3LUS.L quotes in GBp: 13392 pence -> 133.92 GBP -> EUR."""
        pd_obj = _price_data("3LUS.L", 13392.0, "GBp", eur_gbp=0.8521)
        eur = pd_obj.in_eur()["3LUS.L"].iloc[0]
        assert eur == pytest.approx(13392.0 / 100.0 / 0.8521, rel=1e-9)
        # The old bug would have produced the 100x value:
        assert eur < 1000.0

    def test_gbp_line_unaffected(self):
        """Genuine GBP lines are left unchanged."""
        pd_obj = _price_data("VOD.L", 133.92, "GBP", eur_gbp=0.8521)
        eur = pd_obj.in_eur()["VOD.L"].iloc[0]
        assert eur == pytest.approx(133.92 / 0.8521, rel=1e-9)

    def test_detect_currency_drives_the_conversion(self):
        """The override must propagate through to in_eur()."""
        assert detect_currency("3LUS.L") == "GBp"
        assert detect_currency("3SEM.L") == "USD"
        assert detect_currency("VOD.L") == "GBP"

    def test_unsupported_currency_still_raises(self):
        """No silent pass-through for unknown currencies."""
        pd_obj = _price_data("7203.T", 2500.0, "JPY", eur_gbp=0.8521)
        with pytest.raises(ValueError, match="Unsupported currency"):
            pd_obj.in_eur()

    def test_usd_line_of_the_same_etp_is_known(self):
        """3USL.L is the same ETP as 3LUS.L, but the USD line."""
        assert detect_currency("3USL.L") == "USD"

    def test_eur_native_needs_no_fx(self):
        """in_eur() without an FX series is the identity when everything is EUR."""
        idx = pd.to_datetime(["2026-07-09"])
        obj = PriceData(prices=pd.DataFrame({"SXR8.DE": [708.8]}, index=idx),
                        currency={"SXR8.DE": "EUR"}, fx_rates=None)
        assert obj.in_eur()["SXR8.DE"].iloc[0] == pytest.approx(708.8)

    def test_missing_fx_still_raises_for_non_eur(self):
        idx = pd.to_datetime(["2026-07-09"])
        obj = PriceData(prices=pd.DataFrame({"3SEM.L": [172.85]}, index=idx),
                        currency={"3SEM.L": "USD"}, fx_rates=None)
        with pytest.raises(ValueError, match="FX rates not available"):
            obj.in_eur()


class TestSplicedHistoryGuard:
    """T-0433: 3LUS.L is the USD line before 2017-03-16, pence after."""

    def test_valid_from_is_registered(self):
        from backtest.assets import TICKER_HISTORY_VALID_FROM
        assert TICKER_HISTORY_VALID_FROM["3LUS.L"] == "2017-03-16"

    def test_pre_break_prices_are_dropped_with_warning(self):
        """The spliced part must not silently flow into a backtest."""
        idx = pd.to_datetime(["2017-03-14", "2017-03-16", "2017-03-20"])
        frame = pd.DataFrame({"3LUS.L": [2099.796, 1734.275, 1722.425]}, index=idx)

        with pytest.warns(UserWarning, match="spliced series"):
            out = DataLoader._mask_spliced_history(frame)

        assert pd.isna(out.loc[pd.Timestamp("2017-03-14"), "3LUS.L"])
        assert out.loc[pd.Timestamp("2017-03-16"), "3LUS.L"] == pytest.approx(1734.275)
        assert out.loc[pd.Timestamp("2017-03-20"), "3LUS.L"] == pytest.approx(1722.425)

    def test_clean_ticker_is_untouched_and_silent(self):
        idx = pd.to_datetime(["2017-03-14", "2017-03-20"])
        frame = pd.DataFrame({"3USL.L": [20.91, 21.28]}, index=idx)
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # any warning would be an error
            out = DataLoader._mask_spliced_history(frame)
        assert out["3USL.L"].notna().all()

    def test_no_warning_when_window_starts_after_the_break(self):
        """A window starting only after the break should not be warned needlessly."""
        idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
        frame = pd.DataFrame({"3LUS.L": [5000.0, 5100.0]}, index=idx)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = DataLoader._mask_spliced_history(frame)
        assert out["3LUS.L"].notna().all()
