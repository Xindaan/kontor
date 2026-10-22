"""Empty holdings must fail loud, not read as a full-size drift/BUY.

Failure class: an empty portfolio and a portfolio genuinely at weight 0 both
return {} from get_weights(). The drift computed against that looks like an
urgent BUY of the whole book -- a confident wrong signal, same class as a stale
price. has_holdings() distinguishes the two, and the signal report warns.
"""
from backtest.signals import Portfolio


class TestHasHoldings:
    def test_empty_portfolio_has_no_holdings(self):
        assert Portfolio(positions={}, cash=0.0).has_holdings() is False

    def test_all_zero_shares_has_no_holdings(self):
        assert Portfolio(positions={"A": 0.0, "B": 0.0}, cash=100.0).has_holdings() is False

    def test_any_positive_share_counts_as_holding(self):
        assert Portfolio(positions={"A": 0.0, "B": 3.0}, cash=0.0).has_holdings() is True

    def test_cash_alone_is_not_a_holding(self):
        # Cash without positions is still an unknown book for drift purposes.
        assert Portfolio(positions={}, cash=10_000.0).has_holdings() is False
