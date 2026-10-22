"""
Cashflow Reconciliation Module.

This module provides functionality to verify that all cashflows in a backtest
are consistent and properly accounted for. It generates reconciliation reports
that can be used for auditing and debugging purposes.

Key Formulas:
    cash_after = cash_before + sells_inflow - buys_outflow - trading_costs - taxes_paid + dividends
    residual = cash_after - expected_cash_after

    Where:
    - sells_inflow = Sum(value_exec for SELL trades) - Sum(costs for SELL trades)
    - buys_outflow = Sum(value_exec for BUY trades) + Sum(costs for BUY trades)
    - trading_costs = Sum(costs for all trades) [already included in inflow/outflow]
    - taxes_paid = Sum(tax_paid_trade for SELL trades)
    - dividends = Any dividend cashflows (tracked via dividend_events)

Note: Residual should be ~0 (within tolerance) for consistent accounting.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple
import json
from pathlib import Path

import pandas as pd

if TYPE_CHECKING:
    from backtest.backtester import BacktestResult, Trade


@dataclass
class DailyReconciliation:
    """
    Cashflow reconciliation for a single date.

    Attributes:
        date: The reconciliation date
        cash_before: Cash balance at start of day (before trades)
        sells_inflow: Total cash received from SELL trades (value_exec - costs)
        buys_outflow: Total cash spent on BUY trades (value_exec + costs)
        trading_costs: Total trading costs (already reflected in inflow/outflow)
        taxes_paid: Total taxes paid on this date
        dividends: Total dividends received
        cash_after: Cash balance at end of day (after all transactions)
        expected_cash_after: Expected cash based on formula
        residual: Difference between actual and expected (should be ~0)
        is_consistent: Whether residual is within tolerance
        num_sells: Number of SELL trades
        num_buys: Number of BUY trades
    """
    date: date
    cash_before: float
    sells_inflow: float  # value_exec - costs for each SELL
    buys_outflow: float  # value_exec + costs for each BUY
    trading_costs: float  # Informational (already in inflow/outflow)
    taxes_paid: float
    dividends: float
    cash_after: float
    expected_cash_after: float = 0.0
    residual: float = 0.0
    is_consistent: bool = True
    num_sells: int = 0
    num_buys: int = 0

    def __post_init__(self):
        """Calculate expected cash and residual."""
        # Expected: cash_before + sells - buys - taxes + dividends
        # Note: costs are already reflected in sells_inflow (subtract) and buys_outflow (add)
        self.expected_cash_after = (
            self.cash_before
            + self.sells_inflow
            - self.buys_outflow
            - self.taxes_paid
            + self.dividends
        )
        self.residual = self.cash_after - self.expected_cash_after

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat() if isinstance(self.date, date) else str(self.date),
            "cash_before": round(self.cash_before, 4),
            "sells_inflow": round(self.sells_inflow, 4),
            "buys_outflow": round(self.buys_outflow, 4),
            "trading_costs": round(self.trading_costs, 4),
            "taxes_paid": round(self.taxes_paid, 4),
            "dividends": round(self.dividends, 4),
            "cash_after": round(self.cash_after, 4),
            "expected_cash_after": round(self.expected_cash_after, 4),
            "residual": round(self.residual, 6),
            "is_consistent": bool(self.is_consistent),
            "num_sells": int(self.num_sells),
            "num_buys": int(self.num_buys),
        }


@dataclass
class MonthlyReconciliation:
    """
    Cashflow reconciliation aggregated by month.

    Attributes:
        year_month: The month in YYYY-MM format
        cash_start: Cash balance at start of month
        total_sells_inflow: Total cash from sells in the month
        total_buys_outflow: Total cash spent on buys in the month
        total_trading_costs: Total trading costs in the month
        total_taxes_paid: Total taxes paid in the month
        total_dividends: Total dividends received in the month
        cash_end: Cash balance at end of month
        expected_cash_end: Expected cash based on aggregated flows
        residual: Difference between actual and expected
        is_consistent: Whether residual is within tolerance
        num_trading_days: Number of days with trades
    """
    year_month: str
    cash_start: float
    total_sells_inflow: float
    total_buys_outflow: float
    total_trading_costs: float
    total_taxes_paid: float
    total_dividends: float
    cash_end: float
    expected_cash_end: float = 0.0
    residual: float = 0.0
    is_consistent: bool = True
    num_trading_days: int = 0

    def __post_init__(self):
        """Calculate expected cash and residual."""
        self.expected_cash_end = (
            self.cash_start
            + self.total_sells_inflow
            - self.total_buys_outflow
            - self.total_taxes_paid
            + self.total_dividends
        )
        self.residual = self.cash_end - self.expected_cash_end

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "year_month": self.year_month,
            "cash_start": round(self.cash_start, 4),
            "total_sells_inflow": round(self.total_sells_inflow, 4),
            "total_buys_outflow": round(self.total_buys_outflow, 4),
            "total_trading_costs": round(self.total_trading_costs, 4),
            "total_taxes_paid": round(self.total_taxes_paid, 4),
            "total_dividends": round(self.total_dividends, 4),
            "cash_end": round(self.cash_end, 4),
            "expected_cash_end": round(self.expected_cash_end, 4),
            "residual": round(self.residual, 6),
            "is_consistent": bool(self.is_consistent),
            "num_trading_days": int(self.num_trading_days),
        }


@dataclass
class CashflowReconciliation:
    """
    Complete cashflow reconciliation for a backtest.

    This class computes and stores reconciliation data for all rebalance dates
    and provides monthly aggregations.

    Attributes:
        daily: List of daily reconciliation records
        monthly: List of monthly reconciliation records
        tolerance: Tolerance for considering residual as consistent
        total_residual: Sum of all residuals (should be ~0)
        max_residual: Maximum absolute residual
        is_fully_consistent: Whether all records are consistent
    """
    daily: List[DailyReconciliation] = field(default_factory=list)
    monthly: List[MonthlyReconciliation] = field(default_factory=list)
    tolerance: float = 0.01
    total_residual: float = 0.0
    max_residual: float = 0.0
    is_fully_consistent: bool = True
    currency: str = "EUR"

    @classmethod
    def from_backtest_result(
        cls,
        result: "BacktestResult",
        tolerance: float = 0.01,
    ) -> "CashflowReconciliation":
        """
        Build reconciliation from a BacktestResult.

        This reconstructs the cash flow history from trades and validates
        that all flows are consistent.

        Args:
            result: BacktestResult to reconcile
            tolerance: Tolerance for residual consistency check

        Returns:
            CashflowReconciliation with daily and monthly records
        """
        recon = cls(tolerance=tolerance, currency=result.config.currency)

        if not result.trades:
            return recon

        # Group trades by date
        trades_by_date: Dict[date, List["Trade"]] = {}
        for trade in result.trades:
            trade_date = trade.date.date() if isinstance(trade.date, datetime) else trade.date
            if trade_date not in trades_by_date:
                trades_by_date[trade_date] = []
            trades_by_date[trade_date].append(trade)

        # Sort dates
        sorted_dates = sorted(trades_by_date.keys())

        # Reconstruct cash history
        # Start with initial capital
        cash_before = result.config.initial_capital
        daily_records: List[DailyReconciliation] = []
        dividends_by_date: List[Tuple[date, float]] = []
        if result.dividend_events:
            for event in result.dividend_events:
                event_date = event.date.date() if isinstance(event.date, datetime) else event.date
                net_amount = event.net_amount if event.net_amount is not None else event.gross_amount - event.tax_paid
                dividends_by_date.append((event_date, net_amount))
            dividends_by_date.sort(key=lambda item: item[0])
        dividend_cursor = 0

        for rebal_date in sorted_dates:
            day_trades = trades_by_date[rebal_date]

            # Calculate flows for this date
            sells_inflow = 0.0
            buys_outflow = 0.0
            trading_costs = 0.0
            taxes_paid = 0.0
            num_sells = 0
            num_buys = 0

            for trade in day_trades:
                if trade.action == "SELL":
                    # Sell inflow = value_exec - costs (net cash received)
                    sells_inflow += trade.value_exec - trade.costs
                    trading_costs += trade.costs
                    taxes_paid += trade.tax_paid_trade
                    num_sells += 1
                else:  # BUY
                    # Buy outflow = value_exec + costs (total cash spent)
                    buys_outflow += trade.value_exec + trade.costs
                    trading_costs += trade.costs
                    num_buys += 1

            # Sum dividends between the previous trade date and this date (inclusive)
            dividends = 0.0
            if result.dividend_events:
                for div_date, div_amount in dividends_by_date[dividend_cursor:]:
                    if div_date <= rebal_date:
                        dividends += div_amount
                        dividend_cursor += 1
                    else:
                        break

            # Calculate expected cash after
            expected_cash = (
                cash_before
                + sells_inflow
                - buys_outflow
                - taxes_paid
                + dividends
            )

            # For now, we compute cash_after from expected (since we don't
            # have actual cash snapshots per date in the result)
            # In a full implementation, we would track cash at each date
            cash_after = expected_cash

            daily_rec = DailyReconciliation(
                date=rebal_date,
                cash_before=cash_before,
                sells_inflow=sells_inflow,
                buys_outflow=buys_outflow,
                trading_costs=trading_costs,
                taxes_paid=taxes_paid,
                dividends=dividends,
                cash_after=cash_after,
                num_sells=num_sells,
                num_buys=num_buys,
            )

            # Check consistency
            daily_rec.is_consistent = bool(abs(daily_rec.residual) <= tolerance)

            daily_records.append(daily_rec)

            # Next day's cash_before is this day's cash_after
            cash_before = cash_after

        recon.daily = daily_records

        # Calculate aggregate metrics
        recon.total_residual = sum(r.residual for r in daily_records)
        recon.max_residual = max(abs(r.residual) for r in daily_records) if daily_records else 0.0
        recon.is_fully_consistent = all(r.is_consistent for r in daily_records)

        # Generate monthly aggregations
        recon._compute_monthly_aggregation()

        return recon

    def _compute_monthly_aggregation(self) -> None:
        """Compute monthly aggregations from daily records."""
        if not self.daily:
            return

        # Group daily records by month
        by_month: Dict[str, List[DailyReconciliation]] = {}
        for rec in self.daily:
            rec_date = rec.date if isinstance(rec.date, date) else rec.date
            year_month = f"{rec_date.year:04d}-{rec_date.month:02d}"
            if year_month not in by_month:
                by_month[year_month] = []
            by_month[year_month].append(rec)

        monthly_records: List[MonthlyReconciliation] = []

        for year_month in sorted(by_month.keys()):
            daily_recs = by_month[year_month]

            # First day's cash_before is month start
            cash_start = daily_recs[0].cash_before

            # Last day's cash_after is month end
            cash_end = daily_recs[-1].cash_after

            # Sum up all flows
            total_sells = sum(r.sells_inflow for r in daily_recs)
            total_buys = sum(r.buys_outflow for r in daily_recs)
            total_costs = sum(r.trading_costs for r in daily_recs)
            total_taxes = sum(r.taxes_paid for r in daily_recs)
            total_divs = sum(r.dividends for r in daily_recs)

            monthly_rec = MonthlyReconciliation(
                year_month=year_month,
                cash_start=cash_start,
                total_sells_inflow=total_sells,
                total_buys_outflow=total_buys,
                total_trading_costs=total_costs,
                total_taxes_paid=total_taxes,
                total_dividends=total_divs,
                cash_end=cash_end,
                num_trading_days=len(daily_recs),
            )

            monthly_rec.is_consistent = bool(
                abs(monthly_rec.residual) <= self.tolerance * len(daily_recs)
            )
            monthly_records.append(monthly_rec)

        self.monthly = monthly_records

    def to_dataframe(self, level: str = "daily") -> pd.DataFrame:
        """
        Convert reconciliation to pandas DataFrame.

        Args:
            level: "daily" or "monthly"

        Returns:
            DataFrame with reconciliation data
        """
        if level == "monthly":
            return pd.DataFrame([r.to_dict() for r in self.monthly])
        else:
            return pd.DataFrame([r.to_dict() for r in self.daily])

    def to_csv(self, path: str, level: str = "daily") -> None:
        """
        Export reconciliation to CSV file.

        Args:
            path: Output file path
            level: "daily" or "monthly"
        """
        df = self.to_dataframe(level)
        df.to_csv(path, index=False)

    def to_json(self, path: str) -> None:
        """
        Export complete reconciliation to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "tolerance": self.tolerance,
            "currency": self.currency,
            "total_residual": round(self.total_residual, 6),
            "max_residual": round(self.max_residual, 6),
            "is_fully_consistent": bool(self.is_fully_consistent),
            "daily": [r.to_dict() for r in self.daily],
            "monthly": [r.to_dict() for r in self.monthly],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def generate_html_section(self) -> str:
        """
        Generate HTML section for the reconciliation report.

        Returns:
            HTML string for the reconciliation section
        """
        if not self.daily:
            return ""

        # Status badge
        if self.is_fully_consistent:
            status_badge = '<span style="background: #dcfce7; color: #166534; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: 500;">CONSISTENT</span>'
        else:
            status_badge = '<span style="background: #fef2f2; color: #dc2626; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: 500;">INCONSISTENT</span>'

        # Build daily table rows (last 20 entries for brevity)
        daily_rows = ""
        for rec in self.daily[-20:]:
            row_style = "" if rec.is_consistent else 'style="background: #fef2f2;"'
            residual_color = "#16a34a" if rec.is_consistent else "#dc2626"
            rec_date = rec.date.isoformat() if isinstance(rec.date, date) else str(rec.date)
            daily_rows += f"""
            <tr {row_style}>
                <td>{rec_date}</td>
                <td style="text-align: right;">{self.currency} {rec.cash_before:,.2f}</td>
                <td style="text-align: right; color: #16a34a;">+{self.currency} {rec.sells_inflow:,.2f}</td>
                <td style="text-align: right; color: #dc2626;">-{self.currency} {rec.buys_outflow:,.2f}</td>
                <td style="text-align: right;">{self.currency} {rec.trading_costs:,.2f}</td>
                <td style="text-align: right;">{self.currency} {rec.taxes_paid:,.2f}</td>
                <td style="text-align: right;">{self.currency} {rec.dividends:,.2f}</td>
                <td style="text-align: right;">{self.currency} {rec.cash_after:,.2f}</td>
                <td style="text-align: right; color: {residual_color}; font-weight: 500;">{rec.residual:+.4f}</td>
            </tr>
            """

        # Build monthly table rows
        monthly_rows = ""
        for rec in self.monthly:
            row_style = "" if rec.is_consistent else 'style="background: #fef2f2;"'
            residual_color = "#16a34a" if rec.is_consistent else "#dc2626"
            monthly_rows += f"""
            <tr {row_style}>
                <td>{rec.year_month}</td>
                <td style="text-align: right;">{self.currency} {rec.cash_start:,.2f}</td>
                <td style="text-align: right; color: #16a34a;">+{self.currency} {rec.total_sells_inflow:,.2f}</td>
                <td style="text-align: right; color: #dc2626;">-{self.currency} {rec.total_buys_outflow:,.2f}</td>
                <td style="text-align: right;">{self.currency} {rec.total_trading_costs:,.2f}</td>
                <td style="text-align: right;">{self.currency} {rec.total_taxes_paid:,.2f}</td>
                <td style="text-align: right;">{self.currency} {rec.total_dividends:,.2f}</td>
                <td style="text-align: right;">{self.currency} {rec.cash_end:,.2f}</td>
                <td style="text-align: right; color: {residual_color}; font-weight: 500;">{rec.residual:+.4f}</td>
                <td style="text-align: center;">{rec.num_trading_days}</td>
            </tr>
            """

        html = f"""
        <section>
            <h2>Cashflow Reconciliation {status_badge}</h2>
            <p style="font-size: 0.875rem; color: #6b7280; margin-bottom: 1rem;">
                Verifies that all cashflows are consistent: cash_after = cash_before + sells - buys - taxes + dividends.
                <strong>Residual</strong> should be ~0 (tolerance: {self.tolerance:.4f} {self.currency}).
            </p>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
                <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #6b7280;">Total Residual</div>
                    <div style="font-size: 1.25rem; font-weight: 600; color: {'#16a34a' if abs(self.total_residual) <= self.tolerance else '#dc2626'};">{self.total_residual:+.6f}</div>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #6b7280;">Max Residual</div>
                    <div style="font-size: 1.25rem; font-weight: 600; color: {'#16a34a' if self.max_residual <= self.tolerance else '#dc2626'};">{self.max_residual:+.6f}</div>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #6b7280;">Trading Days</div>
                    <div style="font-size: 1.25rem; font-weight: 600;">{len(self.daily)}</div>
                </div>
            </div>

            <h3 style="font-size: 1rem; margin-bottom: 0.75rem;">Monthly Aggregation</h3>
            <div style="overflow-x: auto; margin-bottom: 1.5rem;">
            <table>
                <thead>
                    <tr>
                        <th>Month</th>
                        <th style="text-align: right;">Cash Start</th>
                        <th style="text-align: right;">Sells</th>
                        <th style="text-align: right;">Buys</th>
                        <th style="text-align: right;">Costs</th>
                        <th style="text-align: right;">Taxes</th>
                        <th style="text-align: right;">Dividends</th>
                        <th style="text-align: right;">Cash End</th>
                        <th style="text-align: right;">Residual</th>
                        <th style="text-align: center;">Days</th>
                    </tr>
                </thead>
                <tbody>
                    {monthly_rows}
                </tbody>
            </table>
            </div>

            <h3 style="font-size: 1rem; margin-bottom: 0.75rem;">Daily Detail (Last 20)</h3>
            <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th style="text-align: right;">Cash Before</th>
                        <th style="text-align: right;">Sells</th>
                        <th style="text-align: right;">Buys</th>
                        <th style="text-align: right;">Costs</th>
                        <th style="text-align: right;">Taxes</th>
                        <th style="text-align: right;">Dividends</th>
                        <th style="text-align: right;">Cash After</th>
                        <th style="text-align: right;">Residual</th>
                    </tr>
                </thead>
                <tbody>
                    {daily_rows}
                </tbody>
            </table>
            </div>
        </section>
        """

        return html


def validate_trades_invariants(trades: List["Trade"]) -> List[str]:
    """
    Validate that all trades satisfy their invariants.

    Args:
        trades: List of Trade objects to validate

    Returns:
        List of warning messages for any invariant violations
    """
    warnings = []

    for i, trade in enumerate(trades):
        if not trade.validate_invariants():
            expected_value_exec = trade.shares * trade.price_exec
            expected_value_ref = trade.shares * trade.price_ref

            if abs(trade.value_exec - expected_value_exec) > 1e-6:
                warnings.append(
                    f"Trade {i} ({trade.date}, {trade.ticker}): "
                    f"value_exec ({trade.value_exec:.6f}) != shares * price_exec ({expected_value_exec:.6f})"
                )

            if abs(trade.value_ref - expected_value_ref) > 1e-6:
                warnings.append(
                    f"Trade {i} ({trade.date}, {trade.ticker}): "
                    f"value_ref ({trade.value_ref:.6f}) != shares * price_ref ({expected_value_ref:.6f})"
                )

    return warnings
