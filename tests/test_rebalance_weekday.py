"""Weekday anchor for weekly rebalancing.

The default weekly rebalance falls on the last trading day of the ISO week
(effectively Friday). A live process that trades on Mondays is on a different
weekday, and the anchor day materially shifts underwater duration -- so the
weekday must be selectable and its edge cases must hold.
"""
import pandas as pd
import pytest

from backtest.rebalance import generate_rebalance_dates

IDX = pd.bdate_range("2024-01-01", "2024-03-31")   # business days only


class TestWeekdayAnchor:
    def test_default_stays_last_trading_day(self):
        d = pd.DatetimeIndex(generate_rebalance_dates(IDX, "weekly"))
        assert set(d.dayofweek) == {4}, "default is no longer Friday"

    @pytest.mark.parametrize("wd", [0, 2, 4])
    def test_anchor_hits_the_weekday(self, wd):
        d = pd.DatetimeIndex(generate_rebalance_dates(IDX, "weekly", weekday=wd))
        assert set(d.dayofweek) == {wd}

    def test_one_rebalance_per_week(self):
        d = pd.DatetimeIndex(generate_rebalance_dates(IDX, "weekly", weekday=0))
        iso = d.isocalendar()
        assert len(d) == len(set(zip(iso.year, iso.week)))

    def test_missing_anchor_day_slips_forward(self):
        """Monday holiday -> Tuesday, not 'skip the week' and not the prior week."""
        no_monday = IDX[~((IDX.dayofweek == 0) & (IDX.isocalendar().week == 3))]
        d = pd.DatetimeIndex(generate_rebalance_dates(no_monday, "weekly", weekday=0))
        week3 = [x for x in d if x.isocalendar().week == 3 and x.year == 2024]
        assert len(week3) == 1, "a week without its anchor day dropped out"
        assert week3[0].dayofweek == 1, "did not slip to the next open day"

    def test_invalid_weekday_fails_loud(self):
        with pytest.raises(ValueError, match="0 .Mon. to 4"):
            generate_rebalance_dates(IDX, "weekly", weekday=5)

    def test_weekday_ignored_for_non_weekly(self):
        # weekday only applies to weekly; monthly must be unaffected.
        a = pd.DatetimeIndex(generate_rebalance_dates(IDX, "monthly"))
        b = pd.DatetimeIndex(generate_rebalance_dates(IDX, "monthly", weekday=0))
        assert list(a) == list(b)
