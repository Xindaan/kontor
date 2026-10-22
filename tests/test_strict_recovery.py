"""Strict recovery measurement NEXT TO the classic drawdown duration.

Classic (``max_drawdown_duration``): the episode ends at the FIRST touch of the high.
Strict (``max_drawdown_duration_strict``): it ends on the LAST day below the high.
Both measure from the first day below the high, in calendar days.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import StrictRecovery, MetricsCalculator as M

# TestStrictVsClassic checks the LITERAL reading (confirm_days=0);
# TestConfirmation checks the default confirmation rule (252 samples).
LITERAL = dict(confirm_days=0)


def _curve(values, start="2000-01-03"):
    return pd.Series([float(v) for v in values], index=pd.bdate_range(start, periods=len(values)))


def _relapse_curve():
    """High 100 on day 0, 200 days at 50, TWO days touching 100, then 100 days at 90,
    then 101 for good. Classic ends after 200 trading days (the 90 phase is SHORTER,
    otherwise it would be the longest classic episode), strict after 302."""
    return _curve([100.0] + [50.0] * 200 + [100.0, 100.0] + [90.0] * 100 + [101.0] * 10)


class TestStrictVsClassic:
    def test_relapse_after_touch_only_extends_the_strict_measure(self):
        c = _relapse_curve()
        idx = c.index
        classic = M.max_drawdown_duration(c)
        strict = M.max_drawdown_duration_strict(c, **LITERAL)
        assert classic == (idx[200] - idx[1]).days
        assert strict == (idx[302] - idx[1]).days
        assert strict > classic
        assert 1.1 < strict / 365.25 < 1.2

    def test_two_day_touch_does_not_end_the_episode(self):
        ep = M.strict_recovery(_relapse_curve(), **LITERAL)
        assert isinstance(ep, StrictRecovery)
        idx = _relapse_curve().index
        assert ep.high == idx[0]
        assert ep.first_day_below == idx[1]
        assert ep.first_touch == idx[201]
        assert ep.last_day_below == idx[302]
        assert ep.calendar_days_first_touch == (idx[200] - idx[1]).days
        assert ep.calendar_days == (idx[302] - idx[1]).days

    def test_without_relapse_both_are_equal(self):
        c = _curve([100.0] + [50.0] * 754 + [101.0] * 5)
        assert M.max_drawdown_duration_strict(c, **LITERAL) == M.max_drawdown_duration(c) > 1000

    def test_open_episode_at_the_end_counts_to_the_last_day(self):
        c = _curve([100.0, 120.0] + [80.0] * 100)
        idx = c.index
        assert M.max_drawdown_duration_strict(c, **LITERAL) == (idx[-1] - idx[2]).days
        assert M.max_drawdown_duration(c) == (idx[-1] - idx[2]).days
        ep = M.strict_recovery(c, **LITERAL)
        assert ep.high == idx[1] and ep.first_touch is None
        assert ep.calendar_days_first_touch == ep.calendar_days

    def test_tie_with_the_high_is_not_underwater(self):
        c = _curve([100.0, 100.0, 100.0, 100.0])
        assert M.max_drawdown_duration_strict(c, **LITERAL) == 0
        assert M.strict_recovery(c, **LITERAL) is None

    def test_monotonic_increase_is_zero(self):
        c = _curve(np.linspace(100, 200, 50))
        assert M.max_drawdown_duration_strict(c, **LITERAL) == 0
        assert M.strict_recovery(c, **LITERAL) is None

    def test_too_short_is_zero(self):
        assert M.max_drawdown_duration_strict(_curve([100.0]), **LITERAL) == 0
        assert M.max_drawdown_duration_strict(pd.Series([], dtype=float), **LITERAL) == 0

    def test_later_episode_can_be_the_longest(self):
        c = _curve([100.0, 90.0, 110.0] + [100.0] * 400 + [111.0])
        idx = c.index
        ep = M.strict_recovery(c, **LITERAL)
        assert ep.high == idx[2]
        assert ep.calendar_days == (idx[402] - idx[3]).days
        assert ep.calendar_days == M.max_drawdown_duration(c)

    def test_anchor_is_the_high_with_the_longest_strict_episode(self):
        """Local high 100 (day 0), dip to 95 (days 1-9), rally to 120 (days 10-19, ABOVE
        the first high), crash to 60 (days 20-319), then 110 (days 320-389: above 100,
        below 120), then 121. The anchor must be chosen freely among earlier highs."""
        c = _curve([100.0] + [95.0] * 9 + [120.0] * 10 + [60.0] * 300 + [110.0] * 70 + [121.0])
        idx = c.index
        ep = M.strict_recovery(c, **LITERAL)
        assert ep.high == idx[19]                          # the higher high wins here
        assert ep.calendar_days == (idx[389] - idx[20]).days
        # Shorten the 110 phase: days 20..319 below 120 (300 days) against days 1..319
        # below 100 (319 days) -- the earlier, LOWER anchor gives the longer episode.
        c2 = _curve([100.0] + [95.0] * 9 + [120.0] * 10 + [60.0] * 300 + [121.0])
        ep2 = M.strict_recovery(c2, **LITERAL)
        assert ep2.high == c2.index[0]
        assert ep2.first_touch == c2.index[10]             # the rally is only a touch
        assert ep2.last_day_below == c2.index[319]
        assert ep2.calendar_days == (c2.index[319] - c2.index[1]).days
        assert M.max_drawdown_duration(c2) == (c2.index[319] - c2.index[20]).days
        assert ep2.calendar_days > M.max_drawdown_duration(c2)

    @pytest.mark.parametrize("seed", range(25))
    def test_strict_is_never_shorter_than_classic(self, seed):
        rng = np.random.default_rng(seed)
        c = _curve(100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, 1500))))
        assert M.max_drawdown_duration_strict(c, **LITERAL) >= M.max_drawdown_duration(c)

    @pytest.mark.parametrize("seed", range(10))
    def test_strict_matches_the_naive_reference(self, seed):
        """Reference O(n^2): for every high, find the last day below it by loop."""
        rng = np.random.default_rng(100 + seed)
        c = _curve(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, 400))))
        w = c.to_numpy()
        idx = c.index
        best = 0
        for p in range(len(w) - 1):
            high = w[:p + 1].max()
            if w[p] < high or w[p + 1] >= high:
                continue
            below = [j for j in range(p + 1, len(w)) if w[j] < high]
            best = max(best, (idx[below[-1]] - idx[p + 1]).days)
        assert M.max_drawdown_duration_strict(c, **LITERAL) == best


class TestWiring:
    def test_calculate_all_carries_both_fields(self):
        from backtest.backtester import BacktestConfig
        c = _relapse_curve()
        m = M.calculate_all(c, trades=[], config=BacktestConfig())
        assert m.max_drawdown_duration == M.max_drawdown_duration(c)
        assert m.max_drawdown_duration_strict == M.max_drawdown_duration_strict(c)
        assert m.max_drawdown_duration_strict > m.max_drawdown_duration


class TestConfirmation:
    """Default rule: 252 consecutive samples above the anchor = recovered."""

    def _cases(self, n_above):
        # High 100, 200 days at 50, n_above days at 110 (above the high), ONE day at 99,
        # then 120 for good.
        return _curve([100.0] + [50.0] * 200 + [110.0] * n_above + [99.0] + [120.0] * 30)

    def test_single_day_after_twelve_months_above_is_no_relapse(self):
        c = self._cases(300)
        idx = c.index
        assert M.max_drawdown_duration_strict(c, confirm_days=0) == (idx[501] - idx[1]).days
        assert M.max_drawdown_duration_strict(c) == (idx[200] - idx[1]).days
        assert M.max_drawdown_duration_strict(c) == M.max_drawdown_duration(c)

    def test_single_day_after_short_recovery_is_a_relapse(self):
        c = self._cases(100)                               # 100 < 252 days above
        idx = c.index
        assert M.max_drawdown_duration_strict(c) == (idx[301] - idx[1]).days
        assert M.max_drawdown_duration_strict(c) == M.max_drawdown_duration_strict(c, confirm_days=0)

    def test_confirmed_lies_between_classic_and_literal(self):
        for seed in range(15):
            rng = np.random.default_rng(500 + seed)
            c = _curve(100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, 2000))))
            classic = M.max_drawdown_duration(c)
            confirmed = M.max_drawdown_duration_strict(c)
            literal = M.max_drawdown_duration_strict(c, confirm_days=0)
            assert classic <= confirmed <= literal

    def test_real_relapse_stays_with_confirmation(self):
        c = _relapse_curve()                               # 2 days above, then 100 days below
        assert M.max_drawdown_duration_strict(c) == M.max_drawdown_duration_strict(c, confirm_days=0)

    def test_run_to_the_end_of_data_counts_as_confirmed(self):
        # High 100, 50 days below, then 100 days above to the end (< 252): recovered.
        c = _curve([100.0] + [50.0] * 50 + [110.0] * 100)
        idx = c.index
        assert M.max_drawdown_duration_strict(c) == (idx[50] - idx[1]).days
        assert M.max_drawdown_duration_strict(c, confirm_days=0) == (idx[50] - idx[1]).days

    def test_naive_reference_with_confirmation(self):
        def naive(c, n=252):
            w = c.to_numpy(float)
            idx = c.index
            cm = np.maximum.accumulate(w)
            below = w < cm
            best = 0
            for p in np.flatnonzero(~below[:-1] & below[1:]):
                hw = cm[p]
                run = 0
                end = None
                last_below = p + 1
                for j in range(p + 1, len(w)):
                    if w[j] >= hw:
                        run += 1
                        if run >= n:
                            end = j - n
                            break
                    else:
                        run = 0
                        last_below = j
                if end is None:
                    end = last_below             # a run to the end counts as confirmed
                best = max(best, (idx[end] - idx[p + 1]).days)
            return best
        for seed in range(8):
            rng = np.random.default_rng(900 + seed)
            c = _curve(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, 600))))
            assert M.max_drawdown_duration_strict(c, confirm_days=60) == naive(c, 60)
