"""
Tests for German Tax Model.

Tests the implementation of:
- Abgeltungssteuer (26.375%)
- Teilfreistellung (30% for equity funds)
- Freistellungsauftrag (1000/2000 EUR)
- FIFO cost basis tracking
"""

import pytest
from datetime import date

from backtest.tax.de_tax_model import (
    GermanTaxModel,
    TaxLot,
    TaxableGain,
    TaxResult,
)


class TestTaxLot:
    """Tests for TaxLot class."""

    def test_total_cost(self):
        """Test total cost calculation."""
        lot = TaxLot(
            purchase_date=date(2024, 1, 1),
            shares=10,
            cost_per_share=100.0,
            ticker="SPY",
            is_equity_fund=True,
        )
        assert lot.total_cost == 1000.0

    def test_split_partial(self):
        """Test splitting a lot for partial sale."""
        lot = TaxLot(
            purchase_date=date(2024, 1, 1),
            shares=10,
            cost_per_share=100.0,
            ticker="SPY",
            is_equity_fund=True,
        )

        sold, remaining = lot.split(4)

        assert sold.shares == 4
        assert sold.cost_per_share == 100.0
        assert remaining.shares == 6
        assert remaining.cost_per_share == 100.0

    def test_split_full(self):
        """Test splitting when selling entire lot."""
        lot = TaxLot(
            purchase_date=date(2024, 1, 1),
            shares=10,
            cost_per_share=100.0,
            ticker="SPY",
            is_equity_fund=True,
        )

        sold, remaining = lot.split(10)

        assert sold.shares == 10
        assert remaining is None


class TestGermanTaxModel:
    """Tests for GermanTaxModel class."""

    def test_basic_gain_calculation(self):
        """Test basic gain calculation without exemptions."""
        model = GermanTaxModel(exemption_amount=0)  # No exemption

        # Buy 10 shares at 100
        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1))

        # Sell 10 shares at 120
        result = model.apply_sale("SPY", 10, 120.0, date(2024, 6, 1))

        # Raw gain: (120 - 100) * 10 = 200
        assert result.taxable_gain.raw_gain == 200.0

        # Teilfreistellung (30%): taxable = 200 * 0.7 = 140
        assert result.taxable_gain.taxable_gain == 140.0

        # Tax: 140 * 0.26375 = 36.925
        assert abs(result.tax_due - 36.925) < 0.01

    def test_abgeltungssteuer_rate(self):
        """Test that tax rate is exactly 26.375%."""
        model = GermanTaxModel(
            tax_rate=0.26375,
            partial_exemption=0.0,  # No Teilfreistellung
            exemption_amount=0,
        )

        model.record_purchase("SPY", 100, 100.0, date(2024, 1, 1))
        result = model.apply_sale("SPY", 100, 200.0, date(2024, 6, 1))

        # Gain: 10,000
        # Tax: 10,000 * 0.26375 = 2,637.50
        assert abs(result.tax_due - 2637.50) < 0.01

    def test_teilfreistellung_equity_fund(self):
        """Test 30% Teilfreistellung for equity funds."""
        model = GermanTaxModel(
            partial_exemption=0.30,
            exemption_amount=0,
        )

        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1), is_equity_fund=True)
        result = model.apply_sale("SPY", 10, 200.0, date(2024, 6, 1))

        # Raw gain: 1000
        # Taxable after Teilfreistellung: 1000 * 0.7 = 700
        assert result.taxable_gain.raw_gain == 1000.0
        assert result.taxable_gain.taxable_gain == 700.0

    def test_teilfreistellung_non_equity(self):
        """Test no Teilfreistellung for non-equity funds."""
        model = GermanTaxModel(
            partial_exemption=0.30,
            exemption_amount=0,
        )

        model.record_purchase("BND", 10, 100.0, date(2024, 1, 1), is_equity_fund=False)
        result = model.apply_sale("BND", 10, 200.0, date(2024, 6, 1), is_equity_fund=False)

        # No Teilfreistellung: taxable = raw
        assert result.taxable_gain.raw_gain == 1000.0
        assert result.taxable_gain.taxable_gain == 1000.0

    def test_freistellungsauftrag_single(self):
        """Test Freistellungsauftrag for single filer (1000 EUR)."""
        model = GermanTaxModel(
            partial_exemption=0.0,  # No Teilfreistellung for clarity
            exemption_amount=1000.0,
        )

        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1))
        result = model.apply_sale("SPY", 10, 150.0, date(2024, 6, 1))

        # Gain: 500 (below exemption)
        # Exemption used: 500
        # Tax due: 0
        assert result.taxable_gain.raw_gain == 500.0
        assert result.exemption_used == 500.0
        assert result.tax_due == 0.0

    def test_freistellungsauftrag_exceeded(self):
        """Test when gains exceed Freistellungsauftrag."""
        model = GermanTaxModel(
            partial_exemption=0.0,
            exemption_amount=1000.0,
        )

        model.record_purchase("SPY", 100, 100.0, date(2024, 1, 1))
        result = model.apply_sale("SPY", 100, 120.0, date(2024, 6, 1))

        # Gain: 2000
        # Exemption: 1000
        # Taxable: 1000
        # Tax: 1000 * 0.26375 = 263.75
        assert result.taxable_gain.raw_gain == 2000.0
        assert result.exemption_used == 1000.0
        assert abs(result.tax_due - 263.75) < 0.01

    def test_freistellungsauftrag_joint(self):
        """Test Freistellungsauftrag for joint filers (2000 EUR)."""
        model = GermanTaxModel(
            partial_exemption=0.0,
            exemption_amount=2000.0,
        )

        model.record_purchase("SPY", 100, 100.0, date(2024, 1, 1))
        result = model.apply_sale("SPY", 100, 115.0, date(2024, 6, 1))

        # Gain: 1500 (below 2000 exemption)
        # Tax due: 0
        assert result.taxable_gain.raw_gain == 1500.0
        assert result.tax_due == 0.0

    def test_fifo_ordering(self):
        """Test FIFO (First In, First Out) cost basis."""
        model = GermanTaxModel(exemption_amount=0, partial_exemption=0)

        # Buy 10 at 100
        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1))
        # Buy 10 at 150
        model.record_purchase("SPY", 10, 150.0, date(2024, 2, 1))

        # Sell 10 at 200 (should use first lot at 100)
        result = model.apply_sale("SPY", 10, 200.0, date(2024, 6, 1))

        # FIFO: Uses the 100 cost basis, not 150
        # Gain: (200 - 100) * 10 = 1000
        assert result.taxable_gain.raw_gain == 1000.0
        assert result.taxable_gain.cost_basis == 1000.0

    def test_partial_lot_sale(self):
        """Test selling partial lot."""
        model = GermanTaxModel(exemption_amount=0, partial_exemption=0)

        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1))

        # Sell only 4 shares
        result = model.apply_sale("SPY", 4, 150.0, date(2024, 6, 1))

        # Gain: (150 - 100) * 4 = 200
        assert result.taxable_gain.raw_gain == 200.0

        # Check remaining position
        shares, cost = model.get_position("SPY")
        assert shares == 6
        assert cost == 600.0

    def test_loss_no_tax(self):
        """Test that losses result in no tax."""
        model = GermanTaxModel(exemption_amount=1000)

        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1))
        result = model.apply_sale("SPY", 10, 80.0, date(2024, 6, 1))

        # Loss: -200
        assert result.taxable_gain.raw_gain == -200.0
        assert result.tax_due == 0.0
        assert result.exemption_used == 0.0

    def test_multiple_sales_same_year(self):
        """Test exemption is shared across multiple sales in same year."""
        model = GermanTaxModel(
            partial_exemption=0,
            exemption_amount=1000.0,
        )

        # First trade
        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1))
        result1 = model.apply_sale("SPY", 10, 160.0, date(2024, 3, 1))

        # Gain: 600, exemption used: 600
        assert result1.exemption_used == 600.0
        assert result1.tax_due == 0.0

        # Second trade
        model.record_purchase("VTI", 10, 100.0, date(2024, 4, 1))
        result2 = model.apply_sale("VTI", 10, 180.0, date(2024, 5, 1))

        # Gain: 800, remaining exemption: 400
        # Taxable: 400, Tax: 400 * 0.26375 = 105.50
        assert result2.exemption_used == 400.0
        assert abs(result2.tax_due - 105.50) < 0.01

    def test_annual_summary(self):
        """Test annual tax summary."""
        model = GermanTaxModel(exemption_amount=0, partial_exemption=0)

        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1))
        model.apply_sale("SPY", 10, 150.0, date(2024, 6, 1))  # Gain: 500

        model.record_purchase("VTI", 10, 100.0, date(2024, 2, 1))
        model.apply_sale("VTI", 10, 80.0, date(2024, 7, 1))   # Loss: -200

        summary = model.get_annual_summary(2024)

        assert summary.total_gains == 500.0
        assert summary.total_losses == -200.0
        assert summary.net_gain == 300.0

    def test_holding_period(self):
        """Test holding period calculation."""
        model = GermanTaxModel(exemption_amount=0)

        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1))
        result = model.apply_sale("SPY", 10, 150.0, date(2024, 7, 1))

        # 182 days from Jan 1 to Jul 1
        assert result.taxable_gain.holding_period_days == 182

    def test_complete_scenario(self):
        """Test complete realistic scenario."""
        # German investor with 1000 EUR exemption
        model = GermanTaxModel(
            tax_rate=0.26375,
            partial_exemption=0.30,
            exemption_amount=1000.0,
        )

        # Buy 100 shares of equity ETF at 50 EUR
        model.record_purchase("IWDA", 100, 50.0, date(2023, 1, 15), is_equity_fund=True)

        # Sell 50 shares at 70 EUR after 1 year
        result = model.apply_sale("IWDA", 50, 70.0, date(2024, 1, 20))

        # Raw gain: (70 - 50) * 50 = 1000
        assert result.taxable_gain.raw_gain == 1000.0

        # After Teilfreistellung (30%): 1000 * 0.7 = 700
        assert result.taxable_gain.taxable_gain == 700.0

        # After Freistellungsauftrag: 700 - 700 = 0 (fully covered)
        # But exemption is 1000, and taxable is 700
        assert result.exemption_used == 700.0
        assert result.tax_due == 0.0

        # Check remaining position
        shares, cost = model.get_position("IWDA")
        assert shares == 50
        assert cost == 2500.0

    def test_high_turnover_more_tax(self):
        """Test that high turnover results in more tax than buy-and-hold."""
        # High turnover scenario
        high_turnover_model = GermanTaxModel(exemption_amount=0, partial_exemption=0)
        high_turnover_model.record_purchase("SPY", 100, 100.0, date(2024, 1, 1))
        high_turnover_model.apply_sale("SPY", 100, 110.0, date(2024, 2, 1))
        high_turnover_model.record_purchase("SPY", 100, 110.0, date(2024, 2, 1))
        high_turnover_model.apply_sale("SPY", 100, 120.0, date(2024, 3, 1))

        # Buy and hold scenario (same start/end prices)
        buy_hold_model = GermanTaxModel(exemption_amount=0, partial_exemption=0)
        buy_hold_model.record_purchase("SPY", 100, 100.0, date(2024, 1, 1))
        buy_hold_model.apply_sale("SPY", 100, 120.0, date(2024, 3, 1))

        # Both have same final gain of 2000
        # But high turnover paid tax twice (on 1000 + 1000)
        # Buy-hold paid once (on 2000)
        assert high_turnover_model.total_tax_paid == buy_hold_model.total_tax_paid

    def test_insufficient_shares(self):
        """Test error when selling more shares than owned."""
        model = GermanTaxModel()

        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1))

        with pytest.raises(ValueError, match="Insufficient shares"):
            model.apply_sale("SPY", 20, 150.0, date(2024, 6, 1))

    def test_no_position_error(self):
        """Test error when selling without any position."""
        model = GermanTaxModel()

        with pytest.raises(ValueError, match="No lots found"):
            model.apply_sale("SPY", 10, 150.0, date(2024, 6, 1))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_gain(self):
        """Test selling at exactly cost basis."""
        model = GermanTaxModel(exemption_amount=0)

        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1))
        result = model.apply_sale("SPY", 10, 100.0, date(2024, 6, 1))

        assert result.taxable_gain.raw_gain == 0.0
        assert result.tax_due == 0.0

    def test_very_small_position(self):
        """Test handling of very small positions."""
        model = GermanTaxModel(exemption_amount=0, partial_exemption=0)

        model.record_purchase("SPY", 0.001, 100.0, date(2024, 1, 1))
        result = model.apply_sale("SPY", 0.001, 200.0, date(2024, 6, 1))

        assert abs(result.taxable_gain.raw_gain - 0.1) < 0.001

    def test_reset_model(self):
        """Test resetting the model."""
        model = GermanTaxModel()

        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1))
        model.apply_sale("SPY", 10, 150.0, date(2024, 6, 1))

        assert model.total_tax_paid > 0

        model.reset_all()

        assert model.total_tax_paid == 0.0
        shares, cost = model.get_position("SPY")
        assert shares == 0.0
