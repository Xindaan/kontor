"""The more specialized loss pot is used first.

Previously the GENERAL pot was drawn first, even for an equity gain. Both orders
pay the same tax at the moment of the gain -- they differ in WHICH pot is left
over. The pots are not equivalent: the general pot covers any kind of gain, the
equity pot (Aktienverlusttopf, Section 20 (6) sentence 4 EStG) only equity
gains. Using up the flexible pot first keeps the one that may not fit later.

A pure debt-ETP run (no equity losses) is bit-identical to the old order,
because the equity pot stays empty.
"""
import datetime as dt

import pytest

from backtest.tax.de_tax_model import GermanTaxModel


def _model_with_two_pots():
    """Equity pot 5,000 EUR, general pot 5,000 EUR."""
    m = GermanTaxModel(tax_rate=0.26375, partial_exemption=0.30,
                       exemption_amount=0.0)
    m.record_purchase('STOCK', 100.0, 100.0, dt.date(2020, 1, 1),
                      instrument_class='equity', is_equity_fund=False)
    m.apply_sale('STOCK', 100.0, 50.0, dt.date(2020, 6, 1),
                 instrument_class='equity', is_equity_fund=False)
    m.record_purchase('ETP', 100.0, 100.0, dt.date(2020, 1, 1),
                      instrument_class='general', is_equity_fund=False)
    m.apply_sale('ETP', 100.0, 50.0, dt.date(2020, 6, 1),
                 instrument_class='general', is_equity_fund=False)
    return m


def _equity_gain(m, price_per_share=150.0):
    m.record_purchase('STOCK2', 100.0, 100.0, dt.date(2021, 1, 1),
                      instrument_class='equity', is_equity_fund=False)
    return m.apply_sale('STOCK2', 100.0, price_per_share, dt.date(2021, 6, 1),
                        instrument_class='equity', is_equity_fund=False)


class TestOrder:
    def test_starting_point_has_both_pots(self):
        m = _model_with_two_pots()
        assert m.loss_pot_equity == pytest.approx(5000.0)
        assert m.loss_pot_general == pytest.approx(5000.0)

    def test_equity_gain_draws_the_equity_pot(self):
        m = _model_with_two_pots()
        _equity_gain(m)
        assert m.loss_pot_equity == pytest.approx(0.0), (
            'The equity pot must be used first')
        assert m.loss_pot_general == pytest.approx(5000.0), (
            'The flexible pot must remain')

    def test_tax_at_the_moment_of_the_gain_is_the_same(self):
        """The difference is NOT in the immediate effect -- that is the point."""
        m = _model_with_two_pots()
        r = _equity_gain(m)
        assert r.tax_due == pytest.approx(0.0)
        assert r.loss_pot_offset == pytest.approx(5000.0)

    def test_the_difference_shows_in_the_following_year(self):
        """The actual damage of the old order, in euros."""
        m = _model_with_two_pots()
        _equity_gain(m)
        m.record_purchase('ETP2', 100.0, 100.0, dt.date(2022, 1, 1),
                          instrument_class='general', is_equity_fund=False)
        r = m.apply_sale('ETP2', 100.0, 150.0, dt.date(2022, 6, 1),
                         instrument_class='general', is_equity_fund=False)
        # The general pot is still there and fully covers the ETP gain.
        assert r.tax_due == pytest.approx(0.0), (
            'With the old order 1,318.75 EUR would have been due here')

    def test_general_gain_does_not_touch_the_equity_pot(self):
        """The one-way rule still holds: the equity pot only covers equity."""
        m = _model_with_two_pots()
        m.record_purchase('ETP3', 100.0, 100.0, dt.date(2021, 1, 1),
                          instrument_class='general', is_equity_fund=False)
        m.apply_sale('ETP3', 100.0, 250.0, dt.date(2021, 6, 1),
                     instrument_class='general', is_equity_fund=False)
        # 15,000 gain, but only the general pot (5,000) may help.
        assert m.loss_pot_general == pytest.approx(0.0)
        assert m.loss_pot_equity == pytest.approx(5000.0), (
            'A general gain touched the equity pot')

    def test_partial_cover_spills_into_the_second_pot(self):
        """If the equity pot is not enough, the general pot steps in."""
        m = _model_with_two_pots()
        # 8,000 equity gain against 5,000 equity pot + 5,000 general
        m.record_purchase('STOCK3', 100.0, 100.0, dt.date(2021, 1, 1),
                          instrument_class='equity', is_equity_fund=False)
        r = m.apply_sale('STOCK3', 100.0, 180.0, dt.date(2021, 6, 1),
                         instrument_class='equity', is_equity_fund=False)
        assert m.loss_pot_equity == pytest.approx(0.0)
        assert m.loss_pot_general == pytest.approx(2000.0)
        assert r.loss_pot_offset == pytest.approx(8000.0)
        assert r.tax_due == pytest.approx(0.0)


class TestPureDebtEtpRunIsUnchanged:
    """Without equity losses the equity pot stays empty: identical to before."""

    def test_without_equity_losses_nothing_changes(self):
        m = GermanTaxModel(tax_rate=0.26375, partial_exemption=0.30,
                           exemption_amount=0.0)
        m.record_purchase('ETP', 100.0, 100.0, dt.date(2020, 1, 1),
                          instrument_class='general', is_equity_fund=False)
        m.apply_sale('ETP', 100.0, 50.0, dt.date(2020, 6, 1),
                     instrument_class='general', is_equity_fund=False)
        assert m.loss_pot_equity == pytest.approx(0.0)
        m.record_purchase('ETP2', 100.0, 100.0, dt.date(2021, 1, 1),
                          instrument_class='general', is_equity_fund=False)
        r = m.apply_sale('ETP2', 100.0, 150.0, dt.date(2021, 6, 1),
                         instrument_class='general', is_equity_fund=False)
        assert r.loss_pot_offset == pytest.approx(5000.0)
        assert r.tax_due == pytest.approx(0.0)
