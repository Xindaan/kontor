"""Commission cap (commission_max) and the German broker fee profiles.

A real retail schedule brackets the percentage tier with both a floor and a
ceiling. The cap was previously absent: a large order paid the uncapped
percentage, overstating costs. commission_max closes that, and the fix has to
survive the config round-trip (the field used to be dropped silently).
"""
import pytest

from backtest.costs.transaction_cost_model import TransactionCostModel
from backtest.config.run_config import CostConfig


class TestCommissionCap:
    def test_no_cap_by_default(self):
        m = TransactionCostModel(commission_pct=0.0025, commission_min=8.90)
        # 100k * 0.25% = 250, well above the floor, no ceiling -> 250.
        assert m.calculate(100_000.0).commission == pytest.approx(250.0)

    def test_cap_applies_above_ceiling(self):
        m = TransactionCostModel(commission_pct=0.0025, commission_min=8.90,
                                 commission_max=58.90)
        # 100k * 0.25% = 250 -> capped to 58.90.
        assert m.calculate(100_000.0).commission == pytest.approx(58.90)

    def test_floor_still_wins_on_small_orders(self):
        m = TransactionCostModel(commission_pct=0.0025, commission_min=8.90,
                                 commission_max=58.90)
        # 1000 * 0.25% = 2.50 -> floored to 8.90 (cap irrelevant).
        assert m.calculate(1_000.0).commission == pytest.approx(8.90)

    def test_cap_in_the_tier_passes_through(self):
        m = TransactionCostModel(commission_pct=0.0025, commission_min=8.90,
                                 commission_max=58.90)
        # 10k * 0.25% = 25 -> between floor and ceiling, unchanged.
        assert m.calculate(10_000.0).commission == pytest.approx(25.0)


class TestBrokerProfiles:
    def test_deutsche_bank_maxblue_schedule(self):
        m = TransactionCostModel.deutsche_bank_maxblue()
        assert m.calculate(1_000.0).commission == pytest.approx(8.90)     # floor
        assert m.calculate(10_000.0).commission == pytest.approx(25.0)    # tier
        assert m.calculate(100_000.0).commission == pytest.approx(58.90)  # cap

    def test_trade_republic_is_flat(self):
        m = TransactionCostModel.trade_republic()
        for value in (500.0, 10_000.0, 250_000.0):
            assert m.calculate(value).commission == pytest.approx(1.0)


class TestConfigRoundTrip:
    def test_commission_max_survives_to_dict(self):
        m = TransactionCostModel(commission_max=58.90)
        assert m.to_dict()["commission_max"] == pytest.approx(58.90)

    def test_commission_max_carried_by_cost_config(self):
        # The field used to be dropped silently on the config path.
        cc = CostConfig(commission_pct=0.0025, commission_min=8.90, commission_max=58.90)
        assert cc.to_dict()["commission_max"] == pytest.approx(58.90)
        assert CostConfig.from_dict(cc.to_dict()).commission_max == pytest.approx(58.90)

    def test_from_config_reads_commission_max(self):
        # from_config's supported non-RunConfig input is a plain dict (the
        # CostConfig branch has a pre-existing eager-eval quirk unrelated to this
        # field). commission_max must survive the dict path.
        m = TransactionCostModel.from_config({"commission_pct": 0.0025, "commission_max": 58.90})
        assert m.commission_max == pytest.approx(58.90)
        assert m.calculate(100_000.0).commission == pytest.approx(58.90)
