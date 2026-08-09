"""No-trade band on |actual - target| -- the live rule in the backtester.

Without it the backtester snaps to target at every rebalance date (down to a
one-cent difference), which simulates a weekly churn that a live rule
("only trade when actual-vs-target drift breaches the gate") never produces.
rebalance_min_deviation gates that: below the band, the rebalance is skipped.
"""
import pandas as pd
import pytest

from backtest.backtester import BacktestConfig, Backtester
from backtest.data import PriceData
from strategies.buy_and_hold import BuyAndHold


def _data(n=260):
    idx = pd.bdate_range("2020-01-01", periods=n)
    # A rises, B flat -> the weights are guaranteed to drift apart.
    a = pd.Series([100.0 * (1.004 ** i) for i in range(n)], index=idx)
    b = pd.Series([100.0] * n, index=idx)
    df = pd.DataFrame({"A": a, "B": b})
    return PriceData(prices=df, currency={"A": "EUR", "B": "EUR"},
                     fx_rates=pd.Series(1.0, index=idx))


def _run(threshold):
    d = _data()
    cfg = BacktestConfig(initial_capital=100_000.0, costs_pct=0.0, slippage_pct=0.0,
                         currency="EUR", rebalance_frequency="weekly",
                         tax_enabled=False, validate=False,
                         rebalance_min_deviation=threshold)
    return Backtester(BuyAndHold({"A": 0.5, "B": 0.5}), d, cfg).run()


class TestNoTradeBand:
    def test_default_zero_changes_nothing(self):
        assert BacktestConfig().rebalance_min_deviation == 0.0

    def test_band_cuts_trades_sharply(self):
        without = _run(0.0)
        with_band = _run(0.20)
        assert len(with_band.trades) < len(without.trades), (
            "no-trade band did not reduce trades (%d vs %d)"
            % (len(with_band.trades), len(without.trades))
        )

    def test_huge_band_suppresses_drift_trading(self):
        """At 0.99 pure weight drift must no longer trigger a trade."""
        r = _run(0.99)
        after_entry = [t for t in r.trades if t.date > r.trades[0].date] if r.trades else []
        assert len(after_entry) == 0, (
            "traded despite a 99pp band: %d trades" % len(after_entry)
        )

    @pytest.mark.parametrize("threshold", [0.05, 0.10, 0.20])
    def test_wider_band_never_trades_more(self, threshold):
        narrow = _run(0.01)
        wide = _run(threshold)
        assert len(wide.trades) <= len(narrow.trades)
