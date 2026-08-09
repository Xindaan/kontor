"""Sticky Levered + vol targeting with staged position entry (T-0412).

Friction/execution optimization, deterministic — no ML alpha.

Extends ``StickyLeveredVolTargeted`` with **entry staging**: when the
monthly momentum pick switches to a NEW 3x position, the strategy does
not drive the risk weight to the vol-target goal in one step, but
ramps it up linearly over ``entry_stage_periods`` rebalance periods::

    ramp   = min(1, periods_since_entry / entry_stage_periods)
    w_risky_staged = w_risky_vol_target * ramp

The not-yet-invested part stays in ``SPY`` in the meantime. With
``entry_stage_periods=1`` the strategy is bit-identical to
``StickyLeveredVolTargeted``.

Hypothesis: a staged entry into volatile 3x ETFs lowers entry-timing
risk (no full entry at a local high). The staging acts as a second
damper on top of the vol-target overlay.

Deliberately ONLY entry staging, no exit staging: a momentum exit is
typically a risk reaction (trend broken) and should not be delayed.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from backtest.strategy import Allocation
from strategies.sticky_levered_vol_targeted import StickyLeveredVolTargeted


class StickyLeveredEntryStaged(StickyLeveredVolTargeted):
    """StickyLeveredVolTargeted with a linearly staged entry into new picks."""

    name = "[Research] Sticky Levered + Vol-Targeting + Entry-Staging"

    def __init__(self, *args, entry_stage_periods: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.entry_stage_periods = int(entry_stage_periods)
        self.params["entry_stage_periods"] = self.entry_stage_periods
        # Staging state: which ticker is currently being ramped in + for how long.
        self._staged_ticker: Optional[str] = None
        self._periods_held = 0

    @classmethod
    def get_param_grid(cls) -> Dict[str, List]:
        grid = dict(super().get_param_grid())
        grid["entry_stage_periods"] = [1, 2, 4, 8]
        return grid

    def signal(self, current_date: date, data: pd.DataFrame) -> Allocation:
        alloc = super().signal(current_date, data)
        if self.entry_stage_periods <= 1:
            return alloc

        weights = dict(alloc.weights)
        picked = self._picked or self.safe_asset

        # Safe phase: no staging, reset the state.
        if picked == self.safe_asset or picked not in weights:
            self._staged_ticker = None
            self._periods_held = 0
            return alloc

        # New risky position detected -> reset the ramp counter.
        if picked != self._staged_ticker:
            self._staged_ticker = picked
            self._periods_held = 0
        self._periods_held += 1

        ramp = min(1.0, self._periods_held / self.entry_stage_periods)
        if ramp >= 1.0:
            return alloc

        risky = weights.get(picked, 0.0) * ramp
        safe = 1.0 - risky
        if safe <= 1e-6:
            return Allocation({picked: 1.0})
        return Allocation({picked: risky, self.safe_asset: safe})


strategy = StickyLeveredEntryStaged()
