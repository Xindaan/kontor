"""Only FIFO cost basis is implemented; anything else must fail loud.

The lot engine sells strictly FIFO. A non-FIFO cost_basis_method used to be
stored and then silently ignored, so a run requesting AVGCOST actually computed
FIFO and returned a wrong-but-plausible tax figure. The model now rejects it at
construction instead.
"""
import pytest

from backtest.tax.de_tax_model import GermanTaxModel


def test_fifo_constructs():
    m = GermanTaxModel(cost_basis_method="FIFO")
    assert m.cost_basis_method == "FIFO"


def test_default_is_fifo():
    assert GermanTaxModel().cost_basis_method == "FIFO"


def test_avgcost_is_rejected():
    with pytest.raises(ValueError, match="FIFO"):
        GermanTaxModel(cost_basis_method="AVGCOST")


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        GermanTaxModel(cost_basis_method="LIFO")
