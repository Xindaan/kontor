"""Sticky Levered + vol targeting with a tax-lot-aware switch buffer (T-0411).

Friction/execution optimization, deterministic — no ML alpha.

Extends ``StickyLeveredVolTargeted`` with tax-aware switch timing: in the
single-position system, a momentum switch realizes the FULL gain of
the held 3x position (FIFO, one position -> all lots sold) and
immediately triggers Abgeltungssteuer (capital gains tax). If the
position sits on a large unrealized gain, the cost of a switch is
high: the tax becomes due immediately instead of continuing to
compound tax-deferred. The strategy then requires a proportionally
larger momentum lead before it triggers the taxable switch::

    effective_switch_buffer = switch_buffer + tax_buffer_factor * max(0, g)

where ``g`` = unrealized gain of the current 3x position
(``price / entry_price - 1``). With ``tax_buffer_factor=0`` the
strategy is bit-identical to ``StickyLeveredVolTargeted``.

Deliberately NOT touched: ``force_switch_floor`` and the baseline
drawdown exit. Those are risk exits, not momentum churn — they should
never be delayed by a tax argument.

Note on the year boundary: the German model (see ``de_tax_model.py``)
carries forward loss pots (Verlusttoepfe) indefinitely, and the
Sparer-Pauschbetrag (EUR 1000 annual tax-free allowance) is immaterial
against the switch gains of a growing 3x portfolio. A pure December
deferral would therefore only produce a ~week-long tax-deferral effect
on pre-tax momentum drag — the unrealized-gain buffer is the stronger,
state-dependent lever.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from strategies.sticky_levered_vol_targeted import StickyLeveredVolTargeted


class StickyLeveredTaxAware(StickyLeveredVolTargeted):
    """StickyLeveredVolTargeted with an unrealized-gain-weighted switch buffer."""

    name = "[Research] Sticky Levered + Vol-Targeting + Tax-Switch Buffer"

    def __init__(self, *args, tax_buffer_factor: float = 0.10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._base_switch_buffer = float(self.switch_buffer)
        self.tax_buffer_factor = float(tax_buffer_factor)
        self.params["tax_buffer_factor"] = self.tax_buffer_factor
        # Entry tracking for the currently held 3x position.
        self._entry_ticker: Optional[str] = None
        self._entry_price: Optional[float] = None

    @classmethod
    def get_param_grid(cls) -> Dict[str, List]:
        grid = dict(super().get_param_grid())
        grid["tax_buffer_factor"] = [0.0, 0.05, 0.10, 0.20, 0.40]
        return grid

    @staticmethod
    def _last_price(data: pd.DataFrame, ticker: Optional[str]) -> Optional[float]:
        if ticker is None or ticker not in data.columns:
            return None
        series = data[ticker].dropna()
        return float(series.iloc[-1]) if len(series) else None

    def _unrealized_gain(self, data: pd.DataFrame) -> float:
        """Unrealized gain of the held 3x position (0 if safe)."""
        if (
            self._entry_ticker is None
            or self._entry_price is None
            or self._entry_ticker == self.safe_asset
            or self._entry_price <= 0.0
        ):
            return 0.0
        px = self._last_price(data, self._entry_ticker)
        if px is None:
            return 0.0
        return px / self._entry_price - 1.0

    def _refresh_monthly_pick(self, current_date: date, data: pd.DataFrame) -> None:
        month_key = (current_date.year, current_date.month)
        if month_key == self._last_pick_month:
            return
        # Set the tax-aware buffer BEFORE the base momentum logic
        # (LeveredETFMomentumSticky.signal) makes its switch decision.
        gain = self._unrealized_gain(data)
        self.switch_buffer = (
            self._base_switch_buffer + self.tax_buffer_factor * max(0.0, gain)
        )
        super()._refresh_monthly_pick(current_date, data)
        # Update the entry price on a position change.
        if self._entry_ticker != self._picked:
            self._entry_ticker = self._picked
            self._entry_price = self._last_price(data, self._picked)


strategy = StickyLeveredTaxAware()
