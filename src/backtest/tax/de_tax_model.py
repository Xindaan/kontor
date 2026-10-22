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
- Equity gains are netted against the equity pot FIRST; the general pot only
  covers what remains (so the general pot is not used up while equity losses sit idle)
- Losses from equity ETFs/funds belong to the GENERAL pot (see ``tax_class``)
- Settlement is per calendar year: a later loss in the same year refunds tax
  withheld on earlier gains (negative ``tax_due``); no carryback into prior years
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

from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Dict, List, Literal, Optional, Tuple
from collections import deque


# Instrument classes for loss pot assignment
InstrumentClass = Literal["equity", "general"]


def tax_class(*, single_stock: bool = False,
              equity_fund: bool = False) -> Tuple[InstrumentClass, bool]:
    """Tax treatment of a security as ``(loss pot, Teilfreistellung)``.

    The single place where this mapping lives. Two axes that are easy to mix
    up in code because both hinge on the word "equity":

    * **Loss pot** (Section 20 (6) sentence 4 EStG): the equity loss pot
      (Aktienverlusttopf) only takes losses from selling INDIVIDUAL STOCKS.
      Losses from funds and ETFs -- including EQUITY funds -- go into the
      general pot, together with bonds and certificates. A loss in the equity
      pot is worthless unless equity gains follow.
    * **Teilfreistellung** (Section 20 InvStG): 30 % applies to EQUITY FUNDS.
      An individual stock does NOT get it, and neither does a 3x debt ETP (it
      is a debt security, not a fund).

    For an equity ETF the two axes therefore diverge: Teilfreistellung yes,
    equity pot no. Encoding it as ``('equity', True)`` locks ETF losses into a
    pot the law does not assign them to.

    Args:
        single_stock: directly held stock (equity loss pot, no
            Teilfreistellung).
        equity_fund: equity ETF/fund (general pot, 30 % Teilfreistellung).

    Neither: debt ETP, bond, certificate -- general pot, no Teilfreistellung.

    Do not replace calls with a literal; a second ``('equity', True)`` in the
    code is the next copy of this decision.
    """
    if single_stock and equity_fund:
        raise ValueError(
            "single_stock and equity_fund are mutually exclusive: an individual "
            "stock is not a fund.")
    if single_stock:
        return ("equity", False)
    if equity_fund:
        return ("general", True)
    return ("general", False)


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
        tax_due: Signed cash-tax adjustment (negative means refund)
        added_to_loss_pot: Amount added to loss pot (if loss)
    """
    taxable_gain: TaxableGain
    loss_pot_offset: float = 0.0  # Amount netted from loss pot
    gain_after_loss_pot: float = 0.0  # After loss pot netting
    exemption_used: float = 0.0  # Signed allowance adjustment; negative restores it
    taxable_income: float = 0.0  # Signed change in annual taxable income
    tax_due: float = 0.0  # Tax payment, or refund if negative
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


@dataclass
class TaxYearLedger:
    """Calendar-year settlement, before and after each capital-income event.

    Amounts already include any partial exemption. Equity losses may offset
    equity sale gains only; the general pot also offsets dividends/interest.
    Opening pots are fixed for the year, so a later equity loss can restore a
    general pot previously used against an equity gain (BMF Rn. 229).
    """
    opening_equity_loss: float = 0.0
    opening_general_loss: float = 0.0
    equity_income: float = 0.0
    general_income: float = 0.0
    income_after_losses: float = 0.0
    taxable_income: float = 0.0
    exemption_used: float = 0.0
    tax_due: float = 0.0


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
        cost_basis_method: str = "FIFO",  # only "FIFO" is implemented; others raise
    ):
        """
        Initialize German tax model.

        Args:
            tax_rate: Tax rate on realized gains (default: 26.375%)
            partial_exemption: Teilfreistellung for equity funds (default: 30%)
            exemption_amount: Annual Freistellungsauftrag (default: 0 EUR)
            cost_basis_method: FIFO (default). AVGCOST is NOT implemented.

        Raises:
            ValueError: if cost_basis_method is anything other than "FIFO".
                The lot engine sells strictly FIFO; a non-FIFO value used to be
                stored and then silently ignored, so a run requesting AVGCOST
                actually computed FIFO. Rejecting it fails loud instead of
                returning a wrong-but-plausible tax figure.
        """
        if cost_basis_method != "FIFO":
            raise ValueError(
                "cost_basis_method=%r is not implemented; only 'FIFO' is supported. "
                "The lot engine sells FIFO and any other value would be silently "
                "ignored." % (cost_basis_method,)
            )
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
        self._year_ledgers: Dict[int, TaxYearLedger] = {}
        self._active_year: Optional[int] = None

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
        if instrument_class not in ("equity", "general"):
            raise TaxModelError(f"Unknown instrument class: {instrument_class!r}")
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

        # Reject historical-year edits before mutating FIFO lots. Replaying an
        # earlier year would also change every following year's opening pots.
        self._validate_event_year(sale_date.year)
        if instrument_class is not None and instrument_class not in ("equity", "general"):
            raise TaxModelError(f"Unknown instrument class: {instrument_class!r}")

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
        """Settle a dividend against the general pot and the year's allowance.

        Equity-sale losses do not offset dividends. Amounts are already in the
        portfolio currency; foreign withholding is outside this model.
        """
        amount = max(0.0, amount)
        gain = TaxableGain(
            ticker="DIVIDEND", sale_date=payout_date, shares_sold=0.0,
            sale_proceeds=amount, cost_basis=0.0, raw_gain=amount,
            taxable_gain=amount * (1 - self.partial_exemption) if is_equity_fund else amount,
            holding_period_days=0, instrument_class="general",
            is_equity_fund=is_equity_fund,
        )
        if amount == 0:
            return TaxResult(taxable_gain=gain)
        result = self._calculate_tax_with_loss_pots(gain)
        year = payout_date.year
        self._annual_gains.setdefault(year, []).append(gain)
        self._annual_tax_results.setdefault(year, []).append(result)
        self.total_tax_paid += result.tax_due
        return result

    def _validate_event_year(self, year: int) -> None:
        if self._active_year is not None and year < self._active_year:
            raise TaxModelError(
                f"Capital-income event in {year} precedes settled year "
                f"{self._active_year}; replay the model chronologically instead."
            )

    def _calculate_tax_with_loss_pots(self, gain: TaxableGain) -> TaxResult:
        """Reconcile the whole current year and return the cash-tax adjustment.

        §43a(3) EStG and BMF 14.05.2025 Rn. 212, 229, 230: a later loss
        offsets earlier gains of the same year, can restore a general pot used
        against equity gains, and restores previously used allowance. Only the
        unabsorbed losses carry forward; no carryback into a prior year.

        Settlement timing is an explicit model convention: immediately after
        each event. ``tax_due``, ``taxable_income`` and ``exemption_used`` are
        signed adjustments; a negative tax_due is a cash refund. This also
        makes virtual liquidation independent of symbol/order of the sales.
        """
        year = gain.sale_date.year
        self._validate_event_year(year)
        if gain.instrument_class not in ("equity", "general"):
            raise TaxModelError(f"Unknown instrument class: {gain.instrument_class!r}")
        if year not in self._year_ledgers:
            self._year_ledgers[year] = TaxYearLedger(
                opening_equity_loss=self.loss_pot_equity,
                opening_general_loss=self.loss_pot_general,
            )
        self._active_year = year
        ledger = self._year_ledgers[year]
        old_tax = ledger.tax_due
        old_taxable = ledger.taxable_income
        old_income = ledger.income_after_losses
        old_exemption = ledger.exemption_used
        old_pots = self.loss_pot_equity + self.loss_pot_general

        if gain.instrument_class == "equity":
            ledger.equity_income += gain.taxable_gain
        else:
            ledger.general_income += gain.taxable_gain

        # Specialized losses first; excess equity losses never offset general
        # income. Recomputing from opening balances also restores an earlier
        # general offset when an equity loss arrives later in the same year.
        equity_balance = ledger.equity_income - ledger.opening_equity_loss
        general_balance = ledger.general_income - ledger.opening_general_loss
        self.loss_pot_equity = max(0.0, -equity_balance)
        combined_general = general_balance + max(0.0, equity_balance)
        self.loss_pot_general = max(0.0, -combined_general)
        ledger.income_after_losses = max(0.0, combined_general)
        ledger.exemption_used = min(self.annual_exemption, ledger.income_after_losses)
        ledger.taxable_income = max(0.0, ledger.income_after_losses - self.annual_exemption)
        ledger.tax_due = ledger.taxable_income * self.tax_rate
        self._exemption_used[year] = ledger.exemption_used

        new_pots = self.loss_pot_equity + self.loss_pot_general
        used = max(0.0, old_pots - new_pots)
        added = max(0.0, new_pots - old_pots)
        self._total_loss_pot_used += used
        self._total_loss_pot_added += added
        return TaxResult(
            taxable_gain=gain,
            loss_pot_offset=used,
            gain_after_loss_pot=ledger.income_after_losses - old_income,
            exemption_used=ledger.exemption_used - old_exemption,
            taxable_income=ledger.taxable_income - old_taxable,
            tax_due=ledger.tax_due - old_tax,
            added_to_loss_pot=added,
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
        """Reject partial rollback of a settled year; restart/replay instead.

        Dropping only its report/allowance would leave consumed FIFO lots and
        following-year loss carry unchanged. ``reset_all`` is the atomic reset.
        """
        if year in self._year_ledgers:
            raise TaxModelError("Cannot reset a settled tax year independently; use reset_all and replay.")

    def reset_all(self) -> None:
        """Reset all tracking data."""
        self._lots.clear()
        self._exemption_used.clear()
        self._annual_gains.clear()
        self._annual_tax_results.clear()
        self._year_ledgers.clear()
        self._active_year = None
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
            "settlement_timing": "per_event_calendar_year",
            "active_year": self._active_year,
            "year_ledgers": {year: asdict(ledger) for year, ledger in sorted(self._year_ledgers.items())},
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

        # Virtual sales must retain prior income/tax of the current year to
        # value a refund, while the real ledger stays untouched.
        clone._year_ledgers = {year: replace(ledger) for year, ledger in self._year_ledgers.items()}
        clone._active_year = self._active_year

        # Copy totals
        clone.total_tax_paid = self.total_tax_paid
        clone._total_loss_pot_used = self._total_loss_pot_used
        clone._total_loss_pot_added = self._total_loss_pot_added

        # Note: We don't copy _annual_gains and _annual_tax_results
        # as they're typically not needed for virtual calculations

        return clone
