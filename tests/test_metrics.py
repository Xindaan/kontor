"""Tests for the metrics module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from backtest.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Tests for the MetricsCalculator class."""

    @pytest.fixture
    def simple_equity_curve(self):
        """Create a simple equity curve for testing."""
        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        values = [100, 105, 102, 108, 110, 115, 112, 120, 125, 130, 128, 135]
        return pd.Series(values, index=dates)

    @pytest.fixture
    def constant_equity_curve(self):
        """Create a constant equity curve."""
        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        return pd.Series([100] * 12, index=dates)

    def test_total_return(self, simple_equity_curve):
        """Test total return calculation."""
        result = MetricsCalculator.total_return(simple_equity_curve)
        assert result == pytest.approx(0.35, rel=0.01)  # 35% return

    def test_total_return_constant(self, constant_equity_curve):
        """Test total return with no change."""
        result = MetricsCalculator.total_return(constant_equity_curve)
        assert result == pytest.approx(0.0)

    def test_cagr(self, simple_equity_curve):
        """Test CAGR calculation."""
        result = MetricsCalculator.cagr(simple_equity_curve)
        # Over ~1 year with 35% total return, CAGR should be ~35%
        assert result > 0.30
        assert result < 0.40

    def test_cagr_constant(self, constant_equity_curve):
        """Test CAGR with no growth."""
        result = MetricsCalculator.cagr(constant_equity_curve)
        assert result == pytest.approx(0.0, abs=0.001)

    def test_max_drawdown(self, simple_equity_curve):
        """Test max drawdown calculation."""
        result = MetricsCalculator.max_drawdown(simple_equity_curve)
        # Max drawdown should be negative
        assert result < 0
        # Should be around -5.6% (from 115 to 112)
        assert result > -0.10  # Not more than 10%

    def test_max_drawdown_constant(self, constant_equity_curve):
        """Test max drawdown with no change."""
        result = MetricsCalculator.max_drawdown(constant_equity_curve)
        assert result == pytest.approx(0.0)

    def test_volatility(self):
        """Test volatility calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01])
        result = MetricsCalculator.volatility(returns, annualize=False)
        assert result > 0
        assert result < 0.1  # Should be relatively small

    def test_volatility_annualized(self):
        """Test annualized volatility."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01])
        result = MetricsCalculator.volatility(returns, annualize=True)
        # Annualized should be larger (multiplied by sqrt(12))
        non_annualized = MetricsCalculator.volatility(returns, annualize=False)
        assert result > non_annualized

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Positive returns should give positive Sharpe
        returns = pd.Series([0.02, 0.01, 0.015, 0.02, 0.01, 0.015])
        result = MetricsCalculator.sharpe_ratio(returns, risk_free_rate=0.02)
        assert result > 0

        # Negative returns should give negative Sharpe
        returns = pd.Series([-0.02, -0.01, -0.015, -0.02, -0.01, -0.015])
        result = MetricsCalculator.sharpe_ratio(returns, risk_free_rate=0.02)
        assert result < 0

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        # Returns with only upside should give high Sortino
        returns = pd.Series([0.02, 0.01, 0.015, 0.02, 0.01, 0.015])
        result = MetricsCalculator.sortino_ratio(returns, risk_free_rate=0.02)
        # Should be positive
        assert result > 0 or result == float("inf")

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        result = MetricsCalculator.calmar_ratio(cagr=0.10, max_drawdown=-0.20)
        assert result == pytest.approx(0.5)

        result = MetricsCalculator.calmar_ratio(cagr=0.10, max_drawdown=-0.10)
        assert result == pytest.approx(1.0)

    def test_win_rate(self):
        """Test win rate calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02])
        result = MetricsCalculator.win_rate(returns)
        # 4 positive out of 6 = 66.7%
        assert result == pytest.approx(4/6)

    def test_monthly_returns(self, simple_equity_curve):
        """Test monthly returns calculation."""
        result = MetricsCalculator.monthly_returns(simple_equity_curve)
        assert len(result) == 11  # One less due to pct_change
        # First return should be 5% (100 -> 105)
        assert result.iloc[0] == pytest.approx(0.05)


def test_max_drawdown_duration_liefert_kalendertage_nicht_handelstage():
    """Regression guard: max_drawdown_duration returns CALENDAR DAYS
    (index[-1]-index[0]).days, NOT the number of trading days. Anyone
    converting that to years must divide by 365.25, never by 252.

    The bug: household-sizing scripts divided by 252 -> underwater
    1.45x too large -> a false threshold breach. This test freezes the
    semantics.
    """
    import pandas as pd
    from backtest.metrics import MetricsCalculator

    # Business-day index: ~756 trading days, but ~1096 CALENDAR days span.
    idx = pd.bdate_range('2000-01-03', periods=756)
    vals = [100.0] + [50.0] * 754 + [101.0]        # peak day 0, drawdown, recovery at the end
    curve = pd.Series(vals, index=idx)

    dur = MetricsCalculator.max_drawdown_duration(curve)

    # Must be CALENDAR days (~1050), NOT the ~754 trading days of the group.
    assert dur > 1000
    assert 2.5 < dur / 365.25 < 3.5                     # ~3 calendar years (correct)
    # The trap that produced the false breach: /252 turns this into >4 "years".
    assert dur / 252.0 > 4.0


def test_rebalance_reihenfolge_ist_deterministisch_nicht_hash_abhaengig():
    """Regression guard: buy AND sell order must be sorted.

    Both loops iterated over a `set` of tickers. Because each sale
    sequentially fills the loss-offset bucket and each buy sequentially
    consumes cash, the hash-dependent iteration order decided tax and
    fills -> the same configuration produced CAGR differing by 0.2-0.3pp
    depending on PYTHONHASHSEED. This test freezes the ordering.
    """
    src = open('src/backtest/backtester.py', encoding='utf-8').read()

    # Both ticker sets that drive sequential state must be sorted:
    # calculate_rebalance_trades (sell order -> loss bucket + cash) and
    # execute_buys (buy order -> who gets filled first when cash is scarce).
    assert src.count("sorted(set(target_values.keys()) | set(current_values.keys()))") >= 2, (
        "Sell AND buy sets must be sorted -- otherwise results are "
        "hash-dependent again."
    )
    assert "for ticker in set(target_values.keys())" not in src, (
        "Unsorted set iteration is back in the trade path."
    )


def test_steuerachsen_sind_entkoppelt_debt_etp_bekommt_allgemeinen_verlusttopf():
    """Regression guard: partial exemption and loss-bucket class are
    separate axes.

    The old bool `equity_fund_map` forced "no partial exemption => equity
    loss bucket". For 3x leveraged debt ETPs (structured as debt
    instruments) that's wrong: 0% partial exemption, but the GENERAL loss
    bucket (section 20 of the German income tax act restricts special
    loss offsetting to equity disposals).
    """
    import pandas as pd
    from backtest.backtester import BacktestConfig, Backtester
    from backtest.data import PriceData

    idx = pd.bdate_range('2024-01-02', periods=5)
    D = PriceData(prices=pd.DataFrame({'QQQ3': [100.0] * 5, 'SAFE': [100.0] * 5}, index=idx),
                  currency={'QQQ3': 'USD', 'SAFE': 'USD'},
                  fx_rates=pd.Series(1.0, index=idx))

    # Legacy: False -> ("equity", False), i.e. equity loss bucket.
    legacy = Backtester.__new__(Backtester)
    legacy.config = BacktestConfig(equity_fund_map={'QQQ3': False, 'SAFE': True})
    assert legacy._instrument_tax_flags('QQQ3') == ('equity', False)

    # New: explicitly decoupled -> 0% partial exemption, but general bucket.
    neu = Backtester.__new__(Backtester)
    neu.config = BacktestConfig(equity_fund_map={'QQQ3': False, 'SAFE': True},
                                tax_treatment_map={'QQQ3': ('general', False)})
    assert neu._instrument_tax_flags('QQQ3') == ('general', False)

    # Without a hit in the new map, the legacy path stays bit-identical.
    assert neu._instrument_tax_flags('SAFE') == ('general', True)


# ---------- follow-up fixes (independent cross-check) ----------

def test_nicht_handelbares_endsignal_verliert_keine_accruals():
    """Regression: the skip must ONLY block the trade.

    The first fix put `continue` BEFORE cash interest/dividends/valuation --
    that caused interest and dividends on the last, non-executable signal
    to be lost (independent cross-check: 10005.22 instead of 10007.83;
    DividendEvents 0 instead of 1).
    """
    import pandas as pd
    from backtest.backtester import BacktestConfig, Backtester
    from backtest.data import PriceData
    from strategies.buy_and_hold import BuyAndHold

    idx = pd.bdate_range('2024-01-02', periods=4)
    px = pd.DataFrame({'AAA': [100.0] * 4}, index=idx)

    # Cash interest must still run on the skipped final signal.
    r = Backtester(BuyAndHold({}), PriceData(prices=px.copy(), currency={'AAA': 'USD'}),
                   BacktestConfig(initial_capital=10_000.0, currency='USD',
                                  rebalance_frequency='daily', benchmark=None,
                                  tax_enabled=False, validate=False, cash_rate=0.10,
                                  execution_lag_days=1)).run()
    assert r.equity_curve.dropna().iloc[-1] > 10_007.0, "Cash interest lost on the final signal"

    # Dividend on the last day must be processed.
    div = pd.DataFrame({'AAA': [0.0, 0.0, 0.0, 1000.0]}, index=idx)
    r2 = Backtester(BuyAndHold({'AAA': 1.0}),
                    PriceData(prices=px.copy(), currency={'AAA': 'USD'}, dividends=div),
                    BacktestConfig(initial_capital=10_000.0, currency='USD',
                                   rebalance_frequency='daily', benchmark=None,
                                   tax_enabled=False, validate=False,
                                   execution_lag_days=1)).run()
    assert len(getattr(r2, 'dividend_events', []) or []) == 1, "Dividend lost on the final signal"


def test_fehlgeschlagener_verkauf_laesst_lots_unveraendert():
    """Atomicity: a genuine inconsistency must NOT destroy the FIFO lots.

    Previously the FIFO loop would start and empty the lots before the
    error was caught -> tax 0 AND holdings gone (the next inconsistency
    would then silently vanish as a NoLotsError).
    """
    import datetime
    from backtest.tax.de_tax_model import GermanTaxModel, InsufficientSharesError

    m = GermanTaxModel()
    m.record_purchase('X', shares=1.0, price_per_share=100.0,
                      purchase_date=datetime.date(2024, 1, 2))
    try:
        m.apply_sale('X', shares_sold=2.0, sale_price=110.0,
                     sale_date=datetime.date(2024, 6, 1))
        raise AssertionError("InsufficientSharesError was not raised")
    except InsufficientSharesError:
        pass
    assert sum(l.shares for l in m._lots['X']) == 1.0, "Lots were mutated despite the error"


def test_gespeichertes_is_equity_fund_false_bleibt_false():
    """`equity_fund_detected or True` turned a stored False back into True
    -> partial exemption applied to a single stock (70 instead of 100)."""
    import datetime
    from backtest.tax.de_tax_model import GermanTaxModel

    m = GermanTaxModel()
    m.record_purchase('AKTIE', shares=1.0, price_per_share=100.0,
                      purchase_date=datetime.date(2024, 1, 2), is_equity_fund=False)
    r = m.apply_sale('AKTIE', shares_sold=1.0, sale_price=200.0,
                     sale_date=datetime.date(2024, 6, 1))
    assert r.taxable_gain.taxable_gain == 100.0, "Partial exemption wrongly applied"
