"""Equity-fund losses belong in the GENERAL loss pot.

Two axes that are easy to mix up in code because both hinge on the word "equity":

* **Loss pot** (Section 20 (6) sentence 4 EStG): the Aktienverlusttopf only takes
  losses from selling INDIVIDUAL STOCKS. Funds and ETFs -- including EQUITY
  funds -- go into the general pot, together with bonds and certificates.
* **Teilfreistellung** (Section 20 InvStG): 30 % applies to EQUITY FUNDS. An
  individual stock does not get it, neither does a debt ETP.

For an equity ETF the axes diverge: Teilfreistellung yes, equity pot no.
``('equity', True)`` locks ETF losses into a pot the law does not assign them to.
"""
import datetime as dt

import pytest

from backtest.tax import GermanTaxModel, tax_class


class TestTheThreeCases:
    def test_equity_fund_general_pot_with_partial_exemption(self):
        assert tax_class(equity_fund=True) == ('general', True)

    def test_single_stock_equity_pot_without_partial_exemption(self):
        assert tax_class(single_stock=True) == ('equity', False)

    def test_debt_etp_general_pot_without_partial_exemption(self):
        """3x leveraged products are debt securities, not funds."""
        assert tax_class() == ('general', False)

    def test_both_at_once_is_an_error(self):
        """Exactly this mix-up was the bug -- it must not be expressible."""
        with pytest.raises(ValueError, match='mutually exclusive'):
            tax_class(single_stock=True, equity_fund=True)

    def test_no_case_yields_equity_pot_with_partial_exemption(self):
        for kw in ({}, {'equity_fund': True}, {'single_stock': True}):
            assert tax_class(**kw) != ('equity', True)


class TestEffectOnTheLossPot:
    """Behavior, not just mapping: does the loss land in the right pot?"""

    def _run(self, klass):
        pot, pe = klass
        m = GermanTaxModel(tax_rate=0.26375, partial_exemption=0.30,
                           exemption_amount=0.0)
        m.record_purchase('ETF', 100.0, 100.0, dt.date(2020, 1, 1),
                          instrument_class=pot, is_equity_fund=pe)
        m.apply_sale('ETF', 100.0, 50.0, dt.date(2020, 6, 1),
                     instrument_class=pot, is_equity_fund=pe)
        return m

    def _etp_gain(self, m):
        m.record_purchase('ETP', 100.0, 100.0, dt.date(2021, 1, 1),
                          instrument_class='general', is_equity_fund=False)
        return m.apply_sale('ETP', 100.0, 130.0, dt.date(2021, 6, 1),
                            instrument_class='general', is_equity_fund=False)

    def test_etf_loss_lands_in_the_general_pot(self):
        m = self._run(tax_class(equity_fund=True))
        assert m.loss_pot_general > 0
        assert m.loss_pot_equity == pytest.approx(0.0)

    def test_and_then_covers_an_etp_gain(self):
        m = self._run(tax_class(equity_fund=True))
        r = self._etp_gain(m)
        assert r.loss_pot_offset > 0, (
            'The ETF loss does not cover the ETP gain -- it sits in the wrong pot')

    def test_with_the_old_mapping_it_would_be_locked_in(self):
        """Negative probe: the old classification locks the loss in."""
        m = self._run(('equity', True))
        assert m.loss_pot_equity > 0
        r = self._etp_gain(m)
        assert r.loss_pot_offset == pytest.approx(0.0)
        assert r.tax_due > 0
