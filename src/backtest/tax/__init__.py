"""
Tax models for Investment-Backtest.

Currently supports German tax model (Abgeltungssteuer) without Vorabpauschale.
"""

from backtest.tax.de_tax_model import (
    GermanTaxModel,
    TaxLot,
    TaxableGain,
    TaxResult,
)

__all__ = [
    "InsufficientSharesError",
    "NoLotsError",
    "TaxModelError",
    "GermanTaxModel",
    "TaxLot",
    "TaxableGain",
    "TaxResult",
]

from .de_tax_model import InsufficientSharesError, NoLotsError, TaxModelError  # noqa: E402
