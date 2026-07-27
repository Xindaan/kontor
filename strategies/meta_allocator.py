"""
Meta Allocator Strategy

Combines multiple child strategies into one portfolio allocation.

Default members:
- classic_60_40
- dual_momentum
- inverse_vol_risk_parity
"""

from backtest.meta_allocator import MetaAllocatorStrategy


class MetaAllocator(MetaAllocatorStrategy):
    """Strategy wrapper for the reusable MetaAllocatorStrategy."""

    name = "[Research] Meta Allocator"


strategy = MetaAllocator()
