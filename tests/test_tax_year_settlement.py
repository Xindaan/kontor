"""Calendar-year reconciliation: BMF 14.05.2025 Rn. 212/229/230.

The expected balances are independent cash examples, including transaction-order
permutations. A later loss can refund the current year but never an earlier one.
"""
from datetime import date
from itertools import permutations

import pytest

from backtest.tax.de_tax_model import GermanTaxModel, TaxModelError


def sell_gain(model, ticker, gain, *, year=2024, kind='general', fund=False):
    model.record_purchase(ticker, 1, 10000, date(year, 1, 1),
                          instrument_class=kind, is_equity_fund=fund)
    return model.apply_sale(ticker, 1, 10000 + gain, date(year, 6, 1))


@pytest.mark.parametrize('events', list(permutations([
    ('equity', 1000), ('equity', -1000), ('general', -1000), ('dividend', 1500),
])))
def test_calendar_year_settlement_is_order_independent(events):
    model = GermanTaxModel(exemption_amount=0)
    tax_cash = 0.0
    for i, (kind, amount) in enumerate(events):
        result = (model.apply_dividend(amount, date(2024, 6, 1), is_equity_fund=False)
                  if kind == 'dividend' else sell_gain(model, str(i), amount, kind=kind))
        tax_cash += result.tax_due
    # Equity net zero; general income 1500 - 1000 = 500.
    assert tax_cash == pytest.approx(131.875)
    assert model.total_tax_paid == pytest.approx(131.875)
    assert model.loss_pot_equity == 0
    assert model.loss_pot_general == 0
    summary = model.get_annual_summary(2024)
    assert summary.tax_paid == pytest.approx(131.875)
    assert summary.taxable_income_after_netting == pytest.approx(500)


def test_later_equity_loss_restores_general_pot_used_against_equity_gain():
    model = GermanTaxModel(exemption_amount=0)
    sell_gain(model, 'old_loss', -1000, year=2023)
    assert sell_gain(model, 'share_gain', 1000, kind='equity').tax_due == 0
    assert model.loss_pot_general == 0
    sell_gain(model, 'share_loss', -1000, kind='equity')
    assert model.loss_pot_general == 1000
    assert model.loss_pot_equity == 0
    assert model.apply_dividend(1000, date(2024, 7, 1), False).tax_due == 0
    assert model.loss_pot_general == 0


def test_equity_losses_cannot_offset_dividends_or_fund_gains():
    model = GermanTaxModel(exemption_amount=0)
    sell_gain(model, 'stock', -1000, kind='equity')
    div = model.apply_dividend(1000, date(2024, 7, 1), False)
    fund = sell_gain(model, 'fund', 1000, fund=True)
    assert div.tax_due == pytest.approx(263.75)
    assert fund.tax_due == pytest.approx(184.625)
    assert model.loss_pot_equity == 1000
    assert model.loss_pot_general == 0


def test_fund_partial_exemption_applies_to_losses_and_dividends():
    model = GermanTaxModel(exemption_amount=0)
    sell_gain(model, 'fund', -2000, fund=True)
    assert model.loss_pot_general == 1400
    dividend = model.apply_dividend(1000, date(2024, 7, 1), True)
    assert dividend.loss_pot_offset == 700
    assert dividend.tax_due == 0
    assert model.loss_pot_general == 700
    assert sell_gain(model, 'debt', 1000).tax_due == pytest.approx(79.125)


def test_refund_restores_allowance_before_later_general_income():
    model = GermanTaxModel(exemption_amount=500)
    assert sell_gain(model, 'gain', 1000).tax_due == pytest.approx(131.875)
    refund = sell_gain(model, 'loss', -1000)
    assert refund.tax_due == pytest.approx(-131.875)
    assert refund.exemption_used == -500
    assert model._exemption_used[2024] == 0
    assert model.apply_dividend(600, date(2024, 7, 1), False).tax_due == pytest.approx(26.375)
    assert model.get_annual_summary(2024).exemption_used == 500
    assert model.get_annual_summary(2024).tax_paid == pytest.approx(26.375)


def test_year_end_carries_only_unabsorbed_losses_and_never_refunds_prior_year():
    model = GermanTaxModel(exemption_amount=0)
    sell_gain(model, 'gain', 1000, year=2023)
    assert sell_gain(model, 'loss', -1500, year=2024).tax_due == 0
    assert sell_gain(model, 'partial', 500, year=2024).tax_due == 0
    assert model.loss_pot_general == 1000
    assert sell_gain(model, 'next', 1500, year=2025).tax_due == pytest.approx(131.875)
    assert model.total_tax_paid == pytest.approx(395.625)
    assert model.get_annual_summary(2023).tax_paid == pytest.approx(263.75)


def test_virtual_clone_can_refund_current_year_without_mutating_original():
    model = GermanTaxModel(exemption_amount=0)
    sell_gain(model, 'gain', 1000)
    model.record_purchase('loss', 1, 2000, date(2024, 1, 1), is_equity_fund=False)
    before = model.to_dict()
    virtual = model.clone()
    assert virtual.apply_sale('loss', 1, 1000, date(2024, 12, 31)).tax_due == pytest.approx(-263.75)
    assert virtual.total_tax_paid == pytest.approx(0)
    assert model.to_dict() == before
    # A clone in the following year must retain the earlier year's tax.
    later = model.clone()
    assert later.apply_sale('loss', 1, 1000, date(2025, 1, 1)).tax_due == 0
    assert later.total_tax_paid == pytest.approx(263.75)
    assert later.loss_pot_general == 1000


def test_invalid_prior_year_and_class_fail_before_consuming_fifo_lots():
    model = GermanTaxModel(exemption_amount=0)
    sell_gain(model, 'gain', 1000, year=2025)
    model.record_purchase('holding', 1, 100, date(2024, 1, 1), is_equity_fund=False)
    before = model.to_dict()
    with pytest.raises(TaxModelError, match='chronologically'):
        model.apply_sale('holding', 1, 110, date(2024, 12, 31))
    assert model.to_dict() == before
    with pytest.raises(TaxModelError, match='instrument class'):
        model.apply_sale('holding', 1, 110, date(2025, 12, 31), instrument_class='invalid')
    assert model.to_dict() == before


def test_partial_year_reset_is_rejected_and_full_reset_clears_ledger():
    model = GermanTaxModel(exemption_amount=0)
    sell_gain(model, 'gain', 1000)
    before = model.to_dict()
    with pytest.raises(TaxModelError, match='reset_all'):
        model.reset_year(2024)
    assert model.to_dict() == before
    model.reset_all()
    assert model.to_dict()['year_ledgers'] == {}
    assert sell_gain(model, 'older', 1000, year=2020).tax_due == pytest.approx(263.75)
