"""
Tax models for Investment-Backtest.

Currently supports German tax model (Abgeltungssteuer) without Vorabpauschale.
"""

from backtest.tax.de_tax_model import (
    GermanTaxModel,
    tax_class,
    TaxLot,
    TaxableGain,
    TaxResult,
)

__all__ = [
    "tax_class",
    "InsufficientSharesError",
    "NoLotsError",
    "TaxModelError",
    "GermanTaxModel",
    "TaxLot",
    "TaxableGain",
    "TaxResult",
]

from .de_tax_model import InsufficientSharesError, NoLotsError, TaxModelError  # noqa: E402
