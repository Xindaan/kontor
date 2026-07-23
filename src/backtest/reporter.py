"""
Reporter module - Generate HTML and JSON reports.

This module provides functionality to generate beautiful HTML reports
with interactive Plotly charts, as well as machine-readable JSON exports.

Report Schema v2 includes:
- Run Metadata Block (configuration, git, environment)
- Manifest Block (Configuration, Assumptions)
- Metrics Table with benchmark comparison
- Cost Breakdown (commission, spread, slippage)
- Equity Curve + Drawdown Curve
- Monthly Returns Heatmap
- Trade Summary
- Cashflow Reconciliation (validates cash consistency)
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from backtest.utils import MONTH_END_FREQ
from backtest.reconciliation import CashflowReconciliation

if TYPE_CHECKING:
    from backtest.backtester import BacktestResult
    from backtest.manifest import RunManifest
    from backtest.metadata import RunMetadata


class Reporter:
    """
    Generates reports from backtest results.

    Supports HTML reports with interactive charts and JSON exports.
    Report Schema v2 includes manifest block, cost breakdown, benchmark metrics,
    and cashflow reconciliation for audit purposes.
    """

    def __init__(
        self,
        result: "BacktestResult",
        manifest: Optional["RunManifest"] = None,
        metadata: Optional["RunMetadata"] = None,
    ):
        """
        Initialize reporter.

        Args:
            result: BacktestResult to generate reports from
            manifest: Optional RunManifest for reproducibility info
            metadata: Optional RunMetadata for auditability
        """
        self.result = result
        self.manifest = manifest
        self.metadata = metadata
        # Compute cashflow reconciliation
        self._reconciliation: Optional[CashflowReconciliation] = None

    @property
    def reconciliation(self) -> CashflowReconciliation:
        """
        Lazily compute and return the cashflow reconciliation.

        Returns:
            CashflowReconciliation object with daily and monthly records
        """
        if self._reconciliation is None:
            self._reconciliation = CashflowReconciliation.from_backtest_result(
                self.result, tolerance=0.01
            )
        return self._reconciliation

    def summary(self) -> str:
        """
        Generate a text summary for terminal output.

        Returns:
            Formatted string summary
        """
        return self.result.summary()

    def to_html(self, path: str) -> None:
        """
        Generate an HTML report with interactive charts.

        Args:
            path: Output file path
        """
        import plotly.graph_objects as go

        result = self.result
        m = result.metrics

        # Base layout settings (avoiding template= which triggers expensive deep copy)
        base_layout = dict(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Arial, sans-serif", size=12, color="#1f2937"),
            xaxis=dict(gridcolor="#e5e7eb", linecolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", linecolor="#e5e7eb"),
        )

        # Create equity curve chart
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            mode="lines",
            name="Portfolio",
            line=dict(color="#2563eb", width=2)
        ))

        if result.benchmark_curve is not None:
            equity_fig.add_trace(go.Scatter(
                x=result.benchmark_curve.index,
                y=result.benchmark_curve.values,
                mode="lines",
                name=f"Benchmark ({result.config.benchmark})",
                line=dict(color="#9ca3af", width=1.5, dash="dash")
            ))

        equity_fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title=f"Value ({result.config.currency})",
            hovermode="x unified",
            **base_layout,
        )

        # Create drawdown chart
        rolling_max = result.equity_curve.cummax()
        drawdown = (result.equity_curve - rolling_max) / rolling_max * 100

        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#dc2626", width=1),
            fillcolor="rgba(220, 38, 38, 0.3)"
        ))
        dd_fig.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            **base_layout,
        )

        # Create monthly returns heatmap
        # Forward-fill the equity curve before resampling to handle quarterly/sparse data
        equity_ffill = result.equity_curve.resample('D').last().ffill()
        monthly_returns = equity_ffill.resample(MONTH_END_FREQ).last().pct_change().dropna()
        monthly_df = pd.DataFrame({
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
            "return": monthly_returns.values * 100
        })
        pivot = monthly_df.pivot(index="year", columns="month", values="return")

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        heatmap_fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=month_names[:len(pivot.columns)],
            y=pivot.index,
            colorscale=[
                [0, "#dc2626"],
                [0.5, "#ffffff"],
                [1, "#16a34a"]
            ],
            zmid=0,
            text=[[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.1f}%<extra></extra>"
        ))
        heatmap_fig.update_layout(
            title="Monthly Returns (%)",
            xaxis_title="Month",
            yaxis_title="Year",
            **base_layout,
        )

        # Generate HTML
        html_content = self._generate_html(
            equity_fig.to_html(full_html=False, include_plotlyjs=False),
            dd_fig.to_html(full_html=False, include_plotlyjs=False),
            heatmap_fig.to_html(full_html=False, include_plotlyjs=False)
        )

        Path(path).write_text(html_content)

    def _generate_html(
        self,
        equity_chart: str,
        drawdown_chart: str,
        heatmap_chart: str
    ) -> str:
        """Generate the full HTML document with Report Schema v2."""
        result = self.result
        m = result.metrics

        # Format trades table with semantic pricing fields
        trades_rows = ""
        for t in result.trades[-50:]:  # Last 50 trades
            color = "#16a34a" if t.action == "BUY" else "#dc2626"
            # Show tax for SELL trades (including €0.00 if no tax due), "-" for BUY
            tax_cell = f"{result.config.currency} {t.tax_paid_trade:.2f}" if t.action == "SELL" else "-"
            trades_rows += f"""
            <tr>
                <td>{t.date.strftime('%Y-%m-%d')}</td>
                <td>{t.ticker}</td>
                <td style="color: {color}; font-weight: 500;">{t.action}</td>
                <td>{t.shares:.4f}</td>
                <td>{result.config.currency} {t.price_ref:.4f}</td>
                <td style="font-weight: 500;">{result.config.currency} {t.price_exec:.4f}</td>
                <td>{result.config.currency} {t.value_ref:.2f}</td>
                <td style="font-weight: 500;">{result.config.currency} {t.value_exec:.2f}</td>
                <td>{result.config.currency} {t.costs:.2f}</td>
                <td>{tax_cell}</td>
            </tr>
            """

        # Generate manifest block if available
        manifest_html = ""
        manifest_css = ""
        if self.manifest is not None:
            manifest_html = self.manifest.html_summary()
            manifest_css = self.manifest.manifest_css()

        # Generate metadata block if available
        metadata_html = ""
        metadata_css = ""
        if self.metadata is not None:
            from backtest.metadata import generate_metadata_html, get_metadata_css
            metadata_html = generate_metadata_html(self.metadata)
            metadata_css = get_metadata_css()

        # Generate cashflow reconciliation HTML
        reconciliation_html = self.reconciliation.generate_html_section()

        # Benchmark metrics section
        benchmark_html = ""
        if m.tracking_difference is not None:
            benchmark_html = f"""
            <div class="metric-card">
                <h3>Tracking Diff</h3>
                <div class="value {'positive' if m.tracking_difference > 0 else 'negative'}">{m.tracking_difference:+.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Alpha</h3>
                <div class="value {'positive' if (m.alpha or 0) > 0 else 'negative'}">{(m.alpha or 0):+.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Beta</h3>
                <div class="value">{(m.beta or 0):.2f}</div>
            </div>
            <div class="metric-card">
                <h3>Info Ratio</h3>
                <div class="value">{(m.information_ratio or 0):.2f}</div>
            </div>
            """

        # Tax status header
        t = result.tax_summary
        tax_status = "ON" if (t and t.tax_enabled) else "OFF"
        basis = result.headline_metric_basis.upper()
        tax_mode = t.tax_accounting_mode if t else "cash_effective"

        tax_header_html = f"""
        <div class="tax-header" style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
                <div><strong>Taxes:</strong> {tax_status}</div>
                <div><strong>Return Basis:</strong> {basis}</div>
                <div><strong>Tax Accounting Mode:</strong> {tax_mode}</div>
            </div>
        </div>
        """

        # Gross vs Net comparison table (if tax is enabled)
        gross_net_html = ""
        if t and t.tax_enabled and result.metrics_gross and result.metrics_net:
            mg = result.metrics_gross
            mn = result.metrics_net
            gross_net_html = f"""
        <section>
            <h2>Gross vs Net Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th style="text-align: right;">Gross</th>
                        <th style="text-align: right;">Net</th>
                        <th style="text-align: right;">Tax Drag</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Final Value</td>
                        <td style="text-align: right;">{result.config.currency} {result.equity_curve_gross.iloc[-1]:,.0f}</td>
                        <td style="text-align: right;">{result.config.currency} {result.equity_curve_net.iloc[-1]:,.0f}</td>
                        <td style="text-align: right; color: #dc2626;">{t.tax_drag_final_pct:.1f}%</td>
                    </tr>
                    <tr>
                        <td>CAGR</td>
                        <td style="text-align: right;" class="{'positive' if mg.cagr > 0 else 'negative'}">{mg.cagr:.1%}</td>
                        <td style="text-align: right;" class="{'positive' if mn.cagr > 0 else 'negative'}">{mn.cagr:.1%}</td>
                        <td style="text-align: right; color: #dc2626;">{t.tax_drag_cagr_pp:.1f}pp</td>
                    </tr>
                    <tr>
                        <td>Volatility</td>
                        <td style="text-align: right;">{mg.volatility:.1%}</td>
                        <td style="text-align: right;">{mn.volatility:.1%}</td>
                        <td style="text-align: right;">-</td>
                    </tr>
                    <tr>
                        <td>Sharpe Ratio</td>
                        <td style="text-align: right;">{mg.sharpe_ratio:.2f}</td>
                        <td style="text-align: right;">{mn.sharpe_ratio:.2f}</td>
                        <td style="text-align: right;">-</td>
                    </tr>
                    <tr>
                        <td>Max Drawdown</td>
                        <td style="text-align: right;" class="negative">{mg.max_drawdown:.1%}</td>
                        <td style="text-align: right;" class="negative">{mn.max_drawdown:.1%}</td>
                        <td style="text-align: right;">-</td>
                    </tr>
                    <tr>
                        <td>Calmar Ratio</td>
                        <td style="text-align: right;">{mg.calmar_ratio:.2f}</td>
                        <td style="text-align: right;">{mn.calmar_ratio:.2f}</td>
                        <td style="text-align: right;">-</td>
                    </tr>
                </tbody>
            </table>
        </section>
            """

        # Tax breakdown section (if available)
        tax_html = ""
        if t and t.tax_enabled:
            tax_html = f"""
        <p class="section-title">Tax Impact (German Model)</p>
        <div class="grid">
            <div class="metric-card">
                <h3>Total Tax Paid</h3>
                <div class="value negative">{result.config.currency} {t.total_tax_paid:,.0f}</div>
            </div>
            <div class="metric-card">
                <h3>Realized Gains</h3>
                <div class="value positive">{result.config.currency} {t.total_realized_gains:,.0f}</div>
            </div>
            <div class="metric-card">
                <h3>Realized Losses</h3>
                <div class="value negative">{result.config.currency} {t.total_realized_losses:,.0f}</div>
            </div>
            <div class="metric-card">
                <h3>Net Realized</h3>
                <div class="value {'positive' if t.net_realized_gain > 0 else 'negative'}">{result.config.currency} {t.net_realized_gain:,.0f}</div>
            </div>
            <div class="metric-card">
                <h3>Exemption Used</h3>
                <div class="value">{result.config.currency} {t.exemption_used:,.0f}</div>
            </div>
            <div class="metric-card">
                <h3>Effective Tax Rate</h3>
                <div class="value">{t.effective_tax_rate:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Tax Drag (CAGR)</h3>
                <div class="value negative">{t.tax_drag_cagr_pp:.1f}pp</div>
            </div>
            <div class="metric-card">
                <h3>Tax Drag (Final)</h3>
                <div class="value negative">{t.tax_drag_final_pct:.1f}%</div>
            </div>
        </div>
            """

        # Calculate years for costs p.a.
        years = (result.equity_curve.index[-1] - result.equity_curve.index[0]).days / 365.25

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report: {result.strategy.name}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #1f2937;
            background: #f9fafb;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }}
        header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        header p {{
            opacity: 0.9;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            font-size: 0.875rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        .metric-card .value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #1f2937;
        }}
        .metric-card .value.positive {{
            color: #16a34a;
        }}
        .metric-card .value.negative {{
            color: #dc2626;
        }}
        section {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        section h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: #1f2937;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background: #f9fafb;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        tr:hover {{
            background: #f9fafb;
        }}
        footer {{
            text-align: center;
            padding: 2rem;
            color: #6b7280;
            font-size: 0.875rem;
        }}
        .chart-container {{
            width: 100%;
            min-height: 400px;
        }}
        .section-title {{
            font-size: 1rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e5e7eb;
        }}
        {manifest_css}
        {metadata_css}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{result.strategy.name}</h1>
            <p>Backtest: {result.equity_curve.index[0].strftime('%Y-%m-%d')} &rarr; {result.equity_curve.index[-1].strftime('%Y-%m-%d')} | Starting Capital: {result.config.currency} {result.config.initial_capital:,.0f}</p>
        </header>

        {metadata_html}
        {manifest_html}

        {tax_header_html}

        <p class="section-title">Performance Metrics</p>
        <div class="grid">
            <div class="metric-card">
                <h3>CAGR</h3>
                <div class="value {'positive' if m.cagr > 0 else 'negative'}">{m.cagr:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Total Return</h3>
                <div class="value {'positive' if m.total_return > 0 else 'negative'}">{m.total_return:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Volatility</h3>
                <div class="value">{m.volatility:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Sharpe Ratio</h3>
                <div class="value {'positive' if m.sharpe_ratio > 0.5 else ''}">{m.sharpe_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <h3>Max Drawdown</h3>
                <div class="value negative">{m.max_drawdown:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Calmar Ratio</h3>
                <div class="value">{m.calmar_ratio:.2f}</div>
            </div>
            {benchmark_html}
        </div>

        <section>
            <h2>Equity Curve</h2>
            <div class="chart-container">
                {equity_chart}
            </div>
        </section>

        <section>
            <h2>Drawdown</h2>
            <div class="chart-container">
                {drawdown_chart}
            </div>
        </section>

        <section>
            <h2>Monthly Returns</h2>
            <div class="chart-container">
                {heatmap_chart}
            </div>
        </section>

        {gross_net_html}

        <p class="section-title">Statistics</p>
        <div class="grid">
            <div class="metric-card">
                <h3>Sortino Ratio</h3>
                <div class="value">{m.sortino_ratio:.2f}</div>
            </div>
            <div class="metric-card">
                <h3>Win Rate (Monthly)</h3>
                <div class="value">{m.win_rate_monthly:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Best Month</h3>
                <div class="value positive">{m.best_month:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Worst Month</h3>
                <div class="value negative">{m.worst_month:.1%}</div>
            </div>
        </div>

        <p class="section-title">Trading & Costs</p>
        <div class="grid">
            <div class="metric-card">
                <h3>Total Trades</h3>
                <div class="value">{m.num_trades}</div>
            </div>
            <div class="metric-card">
                <h3>Annual Turnover</h3>
                <div class="value">{m.turnover_annual:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Total Costs</h3>
                <div class="value">{result.config.currency} {m.total_costs:,.0f}</div>
            </div>
            <div class="metric-card">
                <h3>Costs p.a.</h3>
                <div class="value">{result.config.currency} {m.costs_per_year:,.0f}</div>
            </div>
            <div class="metric-card">
                <h3>Costs % of Final</h3>
                <div class="value">{m.costs_pct_of_final:.2f}%</div>
            </div>
        </div>

        {tax_html}

        <section>
            <h2>Trade Log (Last 50)</h2>
            <p style="font-size: 0.875rem; color: #6b7280; margin-bottom: 1rem;">
                <strong>Price Ref</strong>: Market price before execution friction |
                <strong>Price Exec</strong>: Actual execution price (cash-effective) |
                <strong>Value Exec</strong>: Cash-effective trade value = Shares × Price Exec
            </p>
            <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Ticker</th>
                        <th>Action</th>
                        <th>Shares</th>
                        <th>Price Ref</th>
                        <th>Price Exec</th>
                        <th>Value Ref</th>
                        <th>Value Exec</th>
                        <th>Costs</th>
                        <th>Tax</th>
                    </tr>
                </thead>
                <tbody>
                    {trades_rows}
                </tbody>
            </table>
            </div>
        </section>

        {reconciliation_html}

        <section>
            <h2>Definitions (Glossary)</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem;">
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #1e40af;">CAGR</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Compound Annual Growth Rate. The annualized return that would have been required to grow the initial investment to its final value over the period.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #1e40af;">Volatility</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Annualized standard deviation of returns, measuring the dispersion of returns around the mean. Higher volatility indicates higher risk.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #1e40af;">Sharpe Ratio</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Risk-adjusted return: (Return - Risk-Free Rate) / Volatility. Higher is better. Assumes 2% risk-free rate.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #1e40af;">Sortino Ratio</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Like Sharpe, but uses only downside deviation (negative returns) instead of total volatility. Penalizes only harmful volatility.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #1e40af;">Max Drawdown</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Maximum peak-to-trough decline during the investment period. Measures worst-case loss scenario.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #1e40af;">Calmar Ratio</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">CAGR divided by absolute Max Drawdown. Measures return per unit of drawdown risk. Higher is better.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #1e40af;">Tax Drag</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">The reduction in returns due to taxation. Calculated as CAGR_gross - CAGR_net in percentage points.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #1e40af;">Metric Basis</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;"><strong>Gross:</strong> Before tax. <strong>Net Realized:</strong> After realized taxes only. <strong>Net Liquidation:</strong> After all taxes including virtual end-of-period liquidation tax on unrealized gains.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #1e40af;">Price Ref / Price Exec</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;"><strong>Price Ref:</strong> Reference (market) price before slippage. <strong>Price Exec:</strong> Actual execution price after slippage. For BUY: Price Exec = Price Ref × (1 + slippage%). For SELL: Price Exec = Price Ref × (1 - slippage%).</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #1e40af;">Value Ref / Value Exec</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;"><strong>Value Ref:</strong> Shares × Price Ref (notional value). <strong>Value Exec:</strong> Shares × Price Exec (cash-effective value). The difference explains why Shares × Price ≠ Value in older reports.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #1e40af;">Cashflow Reconciliation</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Verifies cash consistency: cash_after = cash_before + sells_inflow - buys_outflow - taxes + dividends. Residual should be ~0 (within tolerance). Non-zero residuals indicate accounting errors.</p>
                </div>
            </div>
        </section>

        <footer>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Framework: backtest v0.1</p>
        </footer>
    </div>
</body>
</html>"""

    def to_json(self, path: str) -> None:
        """
        Export results to JSON file.

        Args:
            path: Output file path
        """
        self.result.to_json(path)

    def export_reconciliation(
        self,
        output_dir: str,
        prefix: str = "cashflow_recon",
    ) -> dict:
        """
        Export cashflow reconciliation data to CSV and JSON files.

        Args:
            output_dir: Directory to write output files
            prefix: Filename prefix (default: cashflow_recon)

        Returns:
            Dict with paths to generated files
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Export daily CSV
        daily_csv = output_path / f"{prefix}_daily.csv"
        self.reconciliation.to_csv(str(daily_csv), level="daily")
        paths["daily_csv"] = str(daily_csv)

        # Export monthly CSV
        monthly_csv = output_path / f"{prefix}_monthly.csv"
        self.reconciliation.to_csv(str(monthly_csv), level="monthly")
        paths["monthly_csv"] = str(monthly_csv)

        # Export full JSON
        full_json = output_path / f"{prefix}.json"
        self.reconciliation.to_json(str(full_json))
        paths["json"] = str(full_json)

        return paths
