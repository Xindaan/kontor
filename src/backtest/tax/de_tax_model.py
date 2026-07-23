"""
German Tax Model (Abgeltungssteuer) - With Loss Pot (Verlustvortrag).

This module implements German capital gains taxation for backtesting:

1. Abgeltungssteuer: 26.375% (25% + 5.5% Soli) on realized gains
2. Teilfreistellung: 30% tax-free for equity funds
3. Freistellungsauftrag: First 1,000/2,000 EUR tax-free
4. Verlustvortrag: Loss pots that carry forward across years:
   - Aktienverlusttopf (equity loss pot): For individual stocks
   - Allgemeiner Verlusttopf (general loss pot): For ETFs/Funds

NOT IMPLEMENTED (explicitly excluded):
- Vorabpauschale (advance flat tax)
- Church tax (Kirchensteuer)
- Termingeschäfte-Verlusttopf (derivatives loss pot)

Loss Pot Rules (§20 Abs. 6 EStG):
- Equity losses can ONLY offset equity gains (Aktienverlusttopf)
- General losses (ETFs/Funds) can offset ALL gains including equity gains
- Loss pots carry forward indefinitely (no yearly reset)
- Allowance (Freistellungsauftrag) is applied AFTER loss pot netting

Cost Basis Method: FIFO (First In, First Out)
- When selling, the oldest lots are sold first
- This is the standard method required by German tax law

Usage:
    from backtest.tax import GermanTaxModel

    tax_model = GermanTaxModel(
        tax_rate=0.26375,           # 25% + 5.5% Soli
        partial_exemption=0.30,      # 30% for equity funds
        exemption_amount=1000.0,     # Freistellungsauftrag
    )

    # Apply tax to a sale
    result = tax_model.apply_sale(
        ticker="SPY",
        shares_sold=10,
        sale_price=100.0,
        sale_date=date(2024, 1, 15),
        instrument_class="general",  # "equity" or "general"
    )
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Literal, Optional, Tuple
from collections import deque


# Instrument classes for loss pot assignment
InstrumentClass = Literal["equity", "general"]


class TaxModelError(ValueError):
    """Base class for tax model errors. Inherits from ValueError (backward compatibility:
    existing `except ValueError` clauses still catch both subclasses)."""


class NoLotsError(TaxModelError):
    """No holdings for this ticker in the tax model — LEGITIMATE (e.g. first rebalance).
    Callers may silently swallow this."""


class InsufficientSharesError(TaxModelError):
    """Sale exceeds the tracked holdings — a genuine bookkeeping INCONSISTENCY.
    If this is swallowed, tax is lost. Callers must report it,
    not ignore it."""


@dataclass
class TaxLot:
    """
    Represents a purchase lot for cost basis tracking.

    Each lot tracks:
    - When and at what price shares were purchased
    - How many shares remain (not yet sold)
    - Instrument class for tax treatment

    Instrument Classes:
    - "equity": Individual stocks (Aktien) - losses go to Aktienverlusttopf
    - "general": ETFs, Funds, Certificates - losses go to allgemeiner Verlusttopf

    FIFO order: Oldest lots are sold first.
    """
    purchase_date: date
    shares: float
    cost_per_share: float
    ticker: str
    instrument_class: InstrumentClass = "general"  # Default: ETFs/Funds
    is_equity_fund: bool = True  # Legacy: for Teilfreistellung (30%)

    @property
    def total_cost(self) -> float:
        """Total cost basis of this lot."""
        return self.shares * self.cost_per_share

    def split(self, shares_to_sell: float) -> Tuple["TaxLot", Optional["TaxLot"]]:
        """
        Split this lot when partially selling.

        Args:
            shares_to_sell: Number of shares to sell from this lot

        Returns:
            Tuple of (sold_lot, remaining_lot or None)
        """
        if shares_to_sell >= self.shares:
            # Sell entire lot
            return self, None

        # Split the lot
        sold_lot = TaxLot(
            purchase_date=self.purchase_date,
            shares=shares_to_sell,
            cost_per_share=self.cost_per_share,
            ticker=self.ticker,
            instrument_class=self.instrument_class,
            is_equity_fund=self.is_equity_fund,
        )
        remaining_lot = TaxLot(
            purchase_date=self.purchase_date,
            shares=self.shares - shares_to_sell,
            cost_per_share=self.cost_per_share,
            ticker=self.ticker,
            instrument_class=self.instrument_class,
            is_equity_fund=self.is_equity_fund,
        )
        return sold_lot, remaining_lot


@dataclass
class TaxableGain:
    """
    Represents a taxable gain from a sale.

    Attributes:
        ticker: Asset ticker
        sale_date: Date of sale
        shares_sold: Number of shares sold
        sale_proceeds: Total sale amount
        cost_basis: Total cost basis of sold shares
        raw_gain: Gain before any exemptions
        taxable_gain: Gain after Teilfreistellung
        holding_period_days: Average holding period
        instrument_class: "equity" or "general" for loss pot assignment
    """
    ticker: str
    sale_date: date
    shares_sold: float
    sale_proceeds: float
    cost_basis: float
    raw_gain: float
    taxable_gain: float  # After Teilfreistellung
    holding_period_days: int
    instrument_class: InstrumentClass = "general"
    is_equity_fund: bool = True  # Legacy: for Teilfreistellung

    @property
    def is_gain(self) -> bool:
        """True if this is a gain (positive)."""
        return self.raw_gain > 0

    @property
    def is_loss(self) -> bool:
        """True if this is a loss (negative)."""
        return self.raw_gain < 0


@dataclass
class TaxResult:
    """
    Result of applying tax to a sale.

    Attributes:
        taxable_gain: The taxable gain object
        loss_pot_offset: Amount offset from loss pot
        gain_after_loss_pot: Gain after loss pot netting
        exemption_used: Amount of Freistellungsauftrag used
        taxable_income: Final taxable amount after all deductions
        tax_due: Final tax amount to be paid
        added_to_loss_pot: Amount added to loss pot (if loss)
    """
    taxable_gain: TaxableGain
    loss_pot_offset: float = 0.0  # Amount netted from loss pot
    gain_after_loss_pot: float = 0.0  # After loss pot netting
    exemption_used: float = 0.0  # Freistellungsauftrag used
    taxable_income: float = 0.0  # Final taxable amount
    tax_due: float = 0.0  # Tax to pay
    added_to_loss_pot: float = 0.0  # If loss, amount added to pot

    # Legacy fields for compatibility
    @property
    def tax_before_exemption(self) -> float:
        """Legacy: tax before Freistellungsauftrag."""
        return self.gain_after_loss_pot * 0.26375 if self.gain_after_loss_pot > 0 else 0.0

    @property
    def effective_tax_rate(self) -> float:
        """Effective tax rate on the raw gain."""
        if self.taxable_gain.raw_gain <= 0:
            return 0.0
        return self.tax_due / self.taxable_gain.raw_gain


@dataclass
class AnnualTaxSummary:
    """
    Summary of taxes for a calendar year.

    Attributes:
        year: Calendar year
        total_gains: Total realized gains (before netting)
        total_losses: Total realized losses (before netting)
        loss_pot_used: Amount offset from loss pots
        loss_pot_added: Amount added to loss pots
        taxable_income_after_netting: Taxable income after loss pot netting
        exemption_used: Freistellungsauftrag used
        tax_paid: Total tax paid
    """
    year: int
    total_gains: float = 0.0
    total_losses: float = 0.0
    loss_pot_used: float = 0.0
    loss_pot_added: float = 0.0
    taxable_income_after_netting: float = 0.0
    exemption_used: float = 0.0
    tax_paid: float = 0.0

    @property
    def net_gain(self) -> float:
        """Net gain after losses (legacy compatibility)."""
        return self.total_gains + self.total_losses

    @property
    def taxable_amount(self) -> float:
        """Legacy: taxable amount."""
        return self.taxable_income_after_netting

    @property
    def effective_rate(self) -> float:
        """Effective tax rate on taxable income after netting."""
        if self.taxable_income_after_netting <= 0:
            return 0.0
        return self.tax_paid / self.taxable_income_after_netting

    @property
    def effective_rate_on_net(self) -> float:
        """Effective tax rate on net realized gains (diagnostic)."""
        net = self.total_gains + self.total_losses
        if net <= 0:
            return 0.0
        return self.tax_paid / net


class GermanTaxModel:
    """
    German tax model implementing Abgeltungssteuer with Verlustvortrag.

    Features:
    - 26.375% tax rate (25% + 5.5% Solidaritaetszuschlag)
    - 30% Teilfreistellung for equity funds
    - Freistellungsauftrag (tax-free allowance)
    - FIFO cost basis tracking
    - Loss pots (Verlustvortrag) for loss carry-forward:
      - Aktienverlusttopf: For individual stocks
      - Allgemeiner Verlusttopf: For ETFs/Funds

    NOT included:
    - Vorabpauschale
    - Church tax

    Usage:
        model = GermanTaxModel(exemption_amount=1000.0)

        # Record a purchase
        model.record_purchase("SPY", 10, 100.0, date(2024, 1, 1), instrument_class="general")

        # Process a sale
        result = model.apply_sale("SPY", 5, 120.0, date(2024, 6, 1))

        print(f"Tax due: {result.tax_due:.2f} EUR")
        print(f"Loss pot equity: {model.loss_pot_equity:.2f}")
        print(f"Loss pot general: {model.loss_pot_general:.2f}")
    """

    def __init__(
        self,
        tax_rate: float = 0.26375,
        partial_exemption: float = 0.30,
        exemption_amount: float = 0.0,
        cost_basis_method: Literal["FIFO", "AVGCOST"] = "FIFO",
    ):
        """
        Initialize German tax model.

        Args:
            tax_rate: Tax rate on realized gains (default: 26.375%)
            partial_exemption: Teilfreistellung for equity funds (default: 30%)
            exemption_amount: Annual Freistellungsauftrag (default: 0 EUR)
            cost_basis_method: FIFO (default) or AVGCOST
        """
        self.tax_rate = tax_rate
        self.partial_exemption = partial_exemption
        self.annual_exemption = exemption_amount
        self.cost_basis_method = cost_basis_method

        # Lot tracking per ticker (FIFO order using deque)
        self._lots: Dict[str, deque] = {}

        # Track exemption usage per year (resets each year)
        self._exemption_used: Dict[int, float] = {}

        # Track all gains/losses per year
        self._annual_gains: Dict[int, List[TaxableGain]] = {}
        self._annual_tax_results: Dict[int, List[TaxResult]] = {}

        # Total tax paid
        self.total_tax_paid: float = 0.0

        # Loss pots (Verlustvortrag) - carry forward indefinitely
        # Aktienverlusttopf: Only equity losses, can only offset equity gains
        self.loss_pot_equity: float = 0.0
        # Allgemeiner Verlusttopf: ETF/Fund losses, can offset any gains
        self.loss_pot_general: float = 0.0

        # Tracking for reporting
        self._total_loss_pot_used: float = 0.0
        self._total_loss_pot_added: float = 0.0

    def record_purchase(
        self,
        ticker: str,
        shares: float,
        price_per_share: float,
        purchase_date: date,
        instrument_class: InstrumentClass = "general",
        is_equity_fund: bool = True,
    ) -> TaxLot:
        """
        Record a purchase for cost basis tracking.

        Args:
            ticker: Asset ticker
            shares: Number of shares purchased
            price_per_share: Price per share
            purchase_date: Date of purchase
            instrument_class: "equity" (stocks) or "general" (ETFs/Funds)
            is_equity_fund: True if this is an equity fund (for Teilfreistellung)

        Returns:
            The created TaxLot
        """
        lot = TaxLot(
            purchase_date=purchase_date,
            shares=shares,
            cost_per_share=price_per_share,
            ticker=ticker,
            instrument_class=instrument_class,
            is_equity_fund=is_equity_fund,
        )

        if ticker not in self._lots:
            self._lots[ticker] = deque()

        self._lots[ticker].append(lot)
        return lot

    def apply_sale(
        self,
        ticker: str,
        shares_sold: float,
        sale_price: float,
        sale_date: date,
        instrument_class: Optional[InstrumentClass] = None,
        is_equity_fund: Optional[bool] = None,
    ) -> TaxResult:
        """
        Apply tax to a sale transaction.

        Uses FIFO to determine which lots are sold.
        Applies loss pot netting and Freistellungsauftrag.

        Args:
            ticker: Asset ticker
            shares_sold: Number of shares sold
            sale_price: Sale price per share
            sale_date: Date of sale
            instrument_class: Override instrument class (uses lot's class if None)
            is_equity_fund: Override equity fund status (uses lot's status if None)

        Returns:
            TaxResult with tax calculation details

        Raises:
            ValueError: If trying to sell more shares than owned
        """
        if ticker not in self._lots or not self._lots[ticker]:
            raise NoLotsError(f"No lots found for {ticker}")

        # Check holdings BEFORE any mutation. Previously the FIFO loop would run first,
        # emptying/splitting lots, and only AFTERWARDS discover that the share count
        # wasn't sufficient -- the caller got a warning, but the lots had already been
        # destroyed and the tax was 0. The next inconsistency could then silently
        # disappear as a NoLotsError.
        _tol = max(1e-9, abs(shares_sold) * 1e-12)
        _available = sum(lot.shares for lot in self._lots[ticker])
        if shares_sold - _available > _tol:
            raise InsufficientSharesError(
                f"Insufficient shares for {ticker}: "
                f"tried to sell {shares_sold}, but only had {_available} "
                f"(Bestand unveraendert -- kein Lot mutiert)"
            )

        sale_proceeds = shares_sold * sale_price
        remaining_to_sell = shares_sold
        total_cost_basis = 0.0
        total_holding_days = 0.0
        lots_sold = []
        detected_instrument_class: Optional[InstrumentClass] = None
        equity_fund_detected = None

        # FIFO: Sell from oldest lots first
        while remaining_to_sell > max(1e-9, abs(shares_sold) * 1e-12) and self._lots[ticker]:
            lot = self._lots[ticker][0]

            if detected_instrument_class is None:
                detected_instrument_class = lot.instrument_class
            if equity_fund_detected is None:
                equity_fund_detected = lot.is_equity_fund

            if lot.shares <= remaining_to_sell:
                # Sell entire lot
                total_cost_basis += lot.total_cost
                holding_days = (sale_date - lot.purchase_date).days
                total_holding_days += holding_days * lot.shares
                remaining_to_sell -= lot.shares
                lots_sold.append(lot)
                self._lots[ticker].popleft()
            else:
                # Partial lot sale
                sold_lot, remaining_lot = lot.split(remaining_to_sell)
                total_cost_basis += sold_lot.total_cost
                holding_days = (sale_date - sold_lot.purchase_date).days
                total_holding_days += holding_days * sold_lot.shares
                lots_sold.append(sold_lot)
                self._lots[ticker][0] = remaining_lot
                remaining_to_sell = 0

        # RELATIVE tolerance: the old absolute 1e-9 threshold was too tight for the
        # synthetically very large share counts in leveraged backtests -- accumulated
        # float residuals exceeded it, the sale was wrongly flagged as inconsistent,
        # and the broad `except ValueError` swallowed the tax along with it (4 cases
        # in the canonical run).
        share_tolerance = max(1e-9, abs(shares_sold) * 1e-12)
        if remaining_to_sell > share_tolerance:
            raise InsufficientSharesError(
                f"Insufficient shares for {ticker}: "
                f"tried to sell {shares_sold}, but only had {shares_sold - remaining_to_sell}"
            )

        # Calculate gain
        raw_gain = sale_proceeds - total_cost_basis
        avg_holding_days = int(total_holding_days / shares_sold) if shares_sold > 0 else 0

        # Determine instrument class (use override or detected from lots)
        inst_class = instrument_class if instrument_class is not None else (detected_instrument_class or "general")

        # Determine if equity fund (use override or detected from lots)
        is_eq_fund = (is_equity_fund if is_equity_fund is not None
                      # `or True` used to turn a STORED False back into True,
                      # applying Teilfreistellung to an individual stock.
                      else (True if equity_fund_detected is None else equity_fund_detected))

        # Apply Teilfreistellung for equity funds (symmetrically on gains AND losses)
        # Per §20 Abs. 1 S. 6 InvStG: both gains and losses are reduced by 30%
        if is_eq_fund:
            taxable_gain = raw_gain * (1 - self.partial_exemption)
        else:
            taxable_gain = raw_gain

        gain_record = TaxableGain(
            ticker=ticker,
            sale_date=sale_date,
            shares_sold=shares_sold,
            sale_proceeds=sale_proceeds,
            cost_basis=total_cost_basis,
            raw_gain=raw_gain,
            taxable_gain=taxable_gain,
            holding_period_days=avg_holding_days,
            instrument_class=inst_class,
            is_equity_fund=is_eq_fund,
        )

        # Calculate tax with loss pot netting
        tax_result = self._calculate_tax_with_loss_pots(gain_record)

        # Record for annual tracking
        year = sale_date.year
        if year not in self._annual_gains:
            self._annual_gains[year] = []
        self._annual_gains[year].append(gain_record)

        if year not in self._annual_tax_results:
            self._annual_tax_results[year] = []
        self._annual_tax_results[year].append(tax_result)

        # Update total tax paid
        self.total_tax_paid += tax_result.tax_due

        return tax_result

    def apply_dividend(
        self,
        amount: float,
        payout_date: date,
        is_equity_fund: bool = True,
    ) -> TaxResult:
        """
        Apply tax to a dividend payout.

        Dividends are treated as capital income and subject to the same
        tax rate and allowance. Loss pots are not applied to dividends.

        Args:
            amount: Gross dividend amount
            payout_date: Date of dividend payment
            is_equity_fund: Whether Teilfreistellung applies (30% for equity funds)

        Returns:
            TaxResult with tax calculation details
        """
        year = payout_date.year
        if year not in self._exemption_used:
            self._exemption_used[year] = 0.0

        if amount <= 0:
            gain_record = TaxableGain(
                ticker="DIVIDEND",
                sale_date=payout_date,
                shares_sold=0.0,
                sale_proceeds=0.0,
                cost_basis=0.0,
                raw_gain=0.0,
                taxable_gain=0.0,
                holding_period_days=0,
                instrument_class="general",
                is_equity_fund=is_equity_fund,
            )
            return TaxResult(taxable_gain=gain_record)

        taxable_gain = amount * (1 - self.partial_exemption) if is_equity_fund else amount

        remaining_exemption = self.annual_exemption - self._exemption_used[year]
        exemption_to_use = min(remaining_exemption, taxable_gain)

        if exemption_to_use > 0:
            self._exemption_used[year] += exemption_to_use

        taxable_income = max(0.0, taxable_gain - exemption_to_use)
        tax_due = taxable_income * self.tax_rate

        gain_record = TaxableGain(
            ticker="DIVIDEND",
            sale_date=payout_date,
            shares_sold=0.0,
            sale_proceeds=amount,
            cost_basis=0.0,
            raw_gain=amount,
            taxable_gain=taxable_gain,
            holding_period_days=0,
            instrument_class="general",
            is_equity_fund=is_equity_fund,
        )

        tax_result = TaxResult(
            taxable_gain=gain_record,
            loss_pot_offset=0.0,
            gain_after_loss_pot=taxable_gain,
            exemption_used=exemption_to_use,
            taxable_income=taxable_income,
            tax_due=tax_due,
            added_to_loss_pot=0.0,
        )

        if year not in self._annual_gains:
            self._annual_gains[year] = []
        self._annual_gains[year].append(gain_record)

        self.total_tax_paid += tax_due
        if year not in self._annual_tax_results:
            self._annual_tax_results[year] = []
        self._annual_tax_results[year].append(tax_result)
        return tax_result

    def _calculate_tax_with_loss_pots(self, gain: TaxableGain) -> TaxResult:
        """
        Calculate tax with loss pot (Verlustvortrag) netting.

        Order of operations:
        1. If loss: Add to appropriate loss pot, tax_due = 0
        2. If gain: First offset against loss pots, then allowance, then tax

        Loss pot rules (§20 Abs. 6 EStG):
        - Equity losses (Aktienverlusttopf) can ONLY offset equity gains
        - General losses (allg. Verlusttopf) can offset ANY gains (equity or general)

        Netting order for gains:
        1. Use general loss pot first (can offset any gain)
        2. Use equity loss pot (only for equity gains)
        3. Apply Freistellungsauftrag
        4. Tax the remainder

        Args:
            gain: TaxableGain to calculate tax on

        Returns:
            TaxResult with full netting details
        """
        year = gain.sale_date.year

        # Initialize exemption tracking for year
        if year not in self._exemption_used:
            self._exemption_used[year] = 0.0

        # Case 1: Loss - add to appropriate loss pot
        if gain.taxable_gain < 0:
            loss_amount = abs(gain.taxable_gain)

            if gain.instrument_class == "equity":
                self.loss_pot_equity += loss_amount
            else:
                self.loss_pot_general += loss_amount

            self._total_loss_pot_added += loss_amount

            return TaxResult(
                taxable_gain=gain,
                loss_pot_offset=0.0,
                gain_after_loss_pot=0.0,
                exemption_used=0.0,
                taxable_income=0.0,
                tax_due=0.0,
                added_to_loss_pot=loss_amount,
            )

        # Case 2: No gain (break-even)
        if gain.taxable_gain == 0:
            return TaxResult(
                taxable_gain=gain,
                loss_pot_offset=0.0,
                gain_after_loss_pot=0.0,
                exemption_used=0.0,
                taxable_income=0.0,
                tax_due=0.0,
                added_to_loss_pot=0.0,
            )

        # Case 3: Gain - apply loss pots, then allowance, then tax
        remaining_gain = gain.taxable_gain
        total_loss_pot_offset = 0.0

        # Step 1: Use general loss pot (can offset any gain type)
        if self.loss_pot_general > 0 and remaining_gain > 0:
            offset = min(self.loss_pot_general, remaining_gain)
            self.loss_pot_general -= offset
            remaining_gain -= offset
            total_loss_pot_offset += offset

        # Step 2: Use equity loss pot (only for equity gains)
        if gain.instrument_class == "equity" and self.loss_pot_equity > 0 and remaining_gain > 0:
            offset = min(self.loss_pot_equity, remaining_gain)
            self.loss_pot_equity -= offset
            remaining_gain -= offset
            total_loss_pot_offset += offset

        if total_loss_pot_offset > 0:
            self._total_loss_pot_used += total_loss_pot_offset

        gain_after_loss_pot = remaining_gain

        # Step 3: Apply Freistellungsauftrag
        remaining_exemption = self.annual_exemption - self._exemption_used[year]
        exemption_to_use = min(remaining_exemption, gain_after_loss_pot)

        if exemption_to_use > 0:
            self._exemption_used[year] += exemption_to_use
            remaining_gain -= exemption_to_use

        taxable_income = max(0, remaining_gain)

        # Step 4: Calculate tax
        tax_due = taxable_income * self.tax_rate

        return TaxResult(
            taxable_gain=gain,
            loss_pot_offset=total_loss_pot_offset,
            gain_after_loss_pot=gain_after_loss_pot,
            exemption_used=exemption_to_use,
            taxable_income=taxable_income,
            tax_due=tax_due,
            added_to_loss_pot=0.0,
        )

    def get_annual_summary(self, year: int) -> AnnualTaxSummary:
        """
        Get tax summary for a specific year.

        Args:
            year: Calendar year

        Returns:
            AnnualTaxSummary with all tax info for the year
        """
        gains = self._annual_gains.get(year, [])
        tax_results = self._annual_tax_results.get(year, [])

        total_gains = sum(g.raw_gain for g in gains if g.raw_gain > 0)
        total_losses = sum(g.raw_gain for g in gains if g.raw_gain < 0)

        # Sum up from tax results for accurate loss pot tracking
        loss_pot_used = sum(r.loss_pot_offset for r in tax_results)
        loss_pot_added = sum(r.added_to_loss_pot for r in tax_results)
        taxable_income_after_netting = sum(r.taxable_income for r in tax_results)
        exemption_used = self._exemption_used.get(year, 0.0)
        tax_paid = sum(r.tax_due for r in tax_results)

        return AnnualTaxSummary(
            year=year,
            total_gains=total_gains,
            total_losses=total_losses,
            loss_pot_used=loss_pot_used,
            loss_pot_added=loss_pot_added,
            taxable_income_after_netting=taxable_income_after_netting,
            exemption_used=exemption_used,
            tax_paid=tax_paid,
        )

    def get_position(self, ticker: str) -> Tuple[float, float]:
        """
        Get current position and cost basis for a ticker.

        Args:
            ticker: Asset ticker

        Returns:
            Tuple of (total_shares, total_cost_basis)
        """
        if ticker not in self._lots:
            return 0.0, 0.0

        total_shares = sum(lot.shares for lot in self._lots[ticker])
        total_cost = sum(lot.total_cost for lot in self._lots[ticker])
        return total_shares, total_cost

    def get_unrealized_gain(self, ticker: str, current_price: float) -> float:
        """
        Calculate unrealized gain for a position.

        Args:
            ticker: Asset ticker
            current_price: Current price per share

        Returns:
            Unrealized gain (positive) or loss (negative)
        """
        shares, cost_basis = self.get_position(ticker)
        if shares == 0:
            return 0.0
        current_value = shares * current_price
        return current_value - cost_basis

    def reset_year(self, year: int) -> None:
        """
        Reset tracking for a specific year.

        Useful for simulation restarts.

        Args:
            year: Year to reset
        """
        self._exemption_used.pop(year, None)
        self._annual_gains.pop(year, None)
        self._annual_tax_results.pop(year, None)

    def reset_all(self) -> None:
        """Reset all tracking data."""
        self._lots.clear()
        self._exemption_used.clear()
        self._annual_gains.clear()
        self._annual_tax_results.clear()
        self.total_tax_paid = 0.0
        self.loss_pot_equity = 0.0
        self.loss_pot_general = 0.0
        self._total_loss_pot_used = 0.0
        self._total_loss_pot_added = 0.0

    def to_dict(self) -> dict:
        """Convert model state to dictionary.

        Now also includes the LOTS. Without them the dump was incomplete:
        loss pots and tax paid were serialized, but not the cost basis --
        a model restored from that dump would have rejected every subsequent sale
        for lack of a cost basis (``NoLotsError``). For provenance dumps this
        means the tax state is only now fully documented.
        """
        return {
            "tax_rate": self.tax_rate,
            "partial_exemption": self.partial_exemption,
            "annual_exemption": self.annual_exemption,
            "cost_basis_method": self.cost_basis_method,
            "total_tax_paid": self.total_tax_paid,
            "exemption_used": dict(self._exemption_used),
            "loss_pot_equity": self.loss_pot_equity,
            "loss_pot_general": self.loss_pot_general,
            "total_loss_pot_used": self._total_loss_pot_used,
            "total_loss_pot_added": self._total_loss_pot_added,
            "lots": {
                ticker: [
                    {
                        "purchase_date": lot.purchase_date.isoformat(),
                        "shares": lot.shares,
                        "cost_per_share": lot.cost_per_share,
                        "ticker": lot.ticker,
                        "instrument_class": lot.instrument_class,
                        "is_equity_fund": lot.is_equity_fund,
                    }
                    for lot in lots
                ]
                for ticker, lots in sorted(self._lots.items())
                if lots
            },
        }

    @classmethod
    def lots_from_dict(cls, state: dict) -> Dict[str, deque]:
        """Reconstruct the lot structure from a ``to_dict`` dump.

        Deliberately a standalone helper rather than a full ``from_dict``: the
        model has further runtime state, and a half-kept round-trip promise
        would be worse than none. Whoever needs the lots (provenance checks,
        state comparison) gets exactly those here -- verifiable against the dump.
        """
        wieder: Dict[str, deque] = {}
        for ticker, lots in (state.get("lots") or {}).items():
            wieder[ticker] = deque(
                TaxLot(
                    purchase_date=date.fromisoformat(lot["purchase_date"]),
                    shares=lot["shares"],
                    cost_per_share=lot["cost_per_share"],
                    ticker=lot["ticker"],
                    instrument_class=lot.get("instrument_class", "general"),
                    is_equity_fund=lot.get("is_equity_fund", True),
                )
                for lot in lots
            )
        return wieder

    def clone(self) -> "GermanTaxModel":
        """
        Create a shallow clone of the tax model for virtual calculations.

        This is more efficient than deepcopy() for scenarios like virtual
        liquidation calculations where we need a copy of the current state
        but don't need to preserve the full history.

        The clone shares the same configuration but has independent:
        - Loss pots
        - Tax lots (shallow copied - the deques are new but lots are shared)
        - Exemption tracking

        Returns:
            A new GermanTaxModel with copied state
        """
        clone = GermanTaxModel(
            tax_rate=self.tax_rate,
            partial_exemption=self.partial_exemption,
            exemption_amount=self.annual_exemption,
            cost_basis_method=self.cost_basis_method,
        )

        # Copy loss pots
        clone.loss_pot_equity = self.loss_pot_equity
        clone.loss_pot_general = self.loss_pot_general

        # Copy tax lots - create new deques with copies of TaxLot objects
        # TaxLots are dataclasses, so we need to copy them to avoid mutation
        for ticker, lots in self._lots.items():
            clone._lots[ticker] = deque(
                TaxLot(
                    purchase_date=lot.purchase_date,
                    shares=lot.shares,
                    cost_per_share=lot.cost_per_share,
                    ticker=lot.ticker,
                    instrument_class=lot.instrument_class,
                    is_equity_fund=lot.is_equity_fund,
                )
                for lot in lots
            )

        # Copy exemption tracking
        clone._exemption_used = dict(self._exemption_used)

        # Copy totals
        clone.total_tax_paid = self.total_tax_paid
        clone._total_loss_pot_used = self._total_loss_pot_used
        clone._total_loss_pot_added = self._total_loss_pot_added

        # Note: We don't copy _annual_gains and _annual_tax_results
        # as they're typically not needed for virtual calculations

        return clone
