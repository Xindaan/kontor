"""
Example strategies for the backtest framework.
"""

from strategies.buy_and_hold import BuyAndHold
from strategies.levered_momentum_crash_guard import LeveredMomentumCrashGuard
from strategies.levered_momentum_crash_guard_lbh_challenger import (
    LeveredMomentumCrashGuardLBHChallenger,
)
from strategies.levered_etf_momentum_sticky import LeveredETFMomentumSticky
from strategies.levered_etf_momentum_sticky_adaptive_v2 import (
    LeveredETFMomentumStickyAdaptiveV2,
)
from strategies.levered_5x_momentum_guard import Levered5xMomentumGuard
from strategies.sticky_levered_vol_targeted import StickyLeveredVolTargeted
from strategies.sticky_levered_cascade import StickyLeveredCascade
from strategies.sticky_levered_entry_staged import StickyLeveredEntryStaged
from strategies.sticky_levered_tax_aware import StickyLeveredTaxAware
from strategies.sticky_levered_vol_targeted_sector_aware import (
    StickyLeveredVolTargetedSectorAware,
)
from strategies.ai_infra_basket import AIInfraBasket

__all__ = [
    "BuyAndHold",
    "LeveredMomentumCrashGuard",
    "LeveredMomentumCrashGuardLBHChallenger",
    "LeveredETFMomentumSticky",
    "LeveredETFMomentumStickyAdaptiveV2",
    "Levered5xMomentumGuard",
    "StickyLeveredVolTargeted",
    "StickyLeveredCascade",
    "StickyLeveredEntryStaged",
    "StickyLeveredTaxAware",
    "StickyLeveredVolTargetedSectorAware",
    "AIInfraBasket",
]
