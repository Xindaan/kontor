"""
Comparator module - Compare multiple strategies.

This module provides functionality to run multiple strategies on the
same data and generate comparative reports.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from backtest.strategy import Strategy
from backtest.data import PriceData
from backtest.backtester import Backtester, BacktestConfig, BacktestResult

if TYPE_CHECKING:
    from backtest.metadata import RunMetadata


@dataclass
class ComparisonResult:
    """
    Results of comparing multiple strategies.

    Attributes:
        results: List of BacktestResult objects, one per strategy
        benchmark_result: Optional benchmark backtest result
        metadata: Optional run metadata for auditability
    """
    results: List[BacktestResult]
    benchmark_result: Optional[BacktestResult] = None
    metadata: Optional["RunMetadata"] = None

    def summary_table(self, show_gross_net: bool = True) -> pd.DataFrame:
        """
        Generate a comparison table of all strategies.

        Args:
            show_gross_net: If True and tax enabled, show separate gross/net columns

        Returns:
            DataFrame with strategies as rows and metrics as columns
        """
        data = []

        # Get benchmark CAGR for excess calculations
        benchmark_cagr = None
        if self.benchmark_result:
            bm_t = self.benchmark_result.tax_summary
            if bm_t and bm_t.tax_enabled:
                benchmark_cagr = bm_t.cagr_for_metric_basis
            else:
                benchmark_cagr = self.benchmark_result.metrics.cagr

        for r in self.results:
            m = r.metrics
            mg = r.metrics_gross
            mn = r.metrics_net
            t = r.tax_summary

            row = {
                "Strategy": r.strategy.name,
            }

            # If tax is enabled and we have both metrics, show gross/net
            if show_gross_net and t and t.tax_enabled and mg and mn:
                # Use CAGR from TaxSummary which correctly accounts for metric_basis
                cagr_net = t.cagr_for_metric_basis
                tax_drag = t.tax_drag_for_metric_basis_pp

                # For net_liquidation mode, show total tax including virtual liquidation
                if t.metric_basis == "net_liquidation":
                    total_tax = t.total_tax_paid + t.tax_paid_liquidation
                    tax_label = "Tax (incl. Liq)"
                    cagr_label = "CAGR (Net Liq)"
                else:
                    total_tax = t.total_tax_paid
                    tax_label = "Tax Paid"
                    cagr_label = "CAGR (Net)"

                # Calculate excess vs benchmark
                excess_cagr = (cagr_net - benchmark_cagr) if benchmark_cagr is not None else None

                row.update({
                    "CAGR (Gross)": f"{t.cagr_gross:.1%}",
                    cagr_label: f"{cagr_net:.1%}",
                    "Tax Drag": f"{tax_drag:.1f}pp",
                    "Excess": f"{excess_cagr:+.1%}" if excess_cagr is not None else "N/A",
                    "Sharpe (Net)": f"{mn.sharpe_ratio:.2f}",
                    "Sortino (Net)": f"{mn.sortino_ratio:.2f}",
                    "Calmar (Net)": f"{mn.calmar_ratio:.2f}",
                    "Max DD (Net)": f"{mn.max_drawdown:.1%}",
                    tax_label: f"€{total_tax:,.0f}",
                })
            else:
                # Calculate excess vs benchmark
                excess_cagr = (m.cagr - benchmark_cagr) if benchmark_cagr is not None else None

                row.update({
                    "CAGR": f"{m.cagr:.1%}",
                    "Excess": f"{excess_cagr:+.1%}" if excess_cagr is not None else "N/A",
                    "Volatility": f"{m.volatility:.1%}",
                    "Sharpe": f"{m.sharpe_ratio:.2f}",
                    "Sortino": f"{m.sortino_ratio:.2f}",
                    "Max DD": f"{m.max_drawdown:.1%}",
                    "Calmar": f"{m.calmar_ratio:.2f}",
                    "Win Rate": f"{m.win_rate_monthly:.1%}",
                    "Trades": m.num_trades,
                    "Costs": f"€{m.total_costs:,.0f}",
                })

            data.append(row)

        # Add benchmark row if available
        if self.benchmark_result:
            bm = self.benchmark_result
            bm_m = bm.metrics
            bm_t = bm.tax_summary
            bm_row = {"Strategy": f"📊 Benchmark ({bm.strategy.name})"}

            if show_gross_net and bm_t and bm_t.tax_enabled:
                bm_cagr_net = bm_t.cagr_for_metric_basis
                bm_tax_drag = bm_t.tax_drag_for_metric_basis_pp
                if bm_t.metric_basis == "net_liquidation":
                    bm_total_tax = bm_t.total_tax_paid + bm_t.tax_paid_liquidation
                    tax_label = "Tax (incl. Liq)"
                    cagr_label = "CAGR (Net Liq)"
                else:
                    bm_total_tax = bm_t.total_tax_paid
                    tax_label = "Tax Paid"
                    cagr_label = "CAGR (Net)"
                bm_mn = bm.metrics_net or bm_m
                bm_row.update({
                    "CAGR (Gross)": f"{bm_t.cagr_gross:.1%}",
                    cagr_label: f"{bm_cagr_net:.1%}",
                    "Tax Drag": f"{bm_tax_drag:.1f}pp",
                    "Excess": "—",
                    "Sharpe (Net)": f"{bm_mn.sharpe_ratio:.2f}",
                    "Sortino (Net)": f"{bm_mn.sortino_ratio:.2f}",
                    "Calmar (Net)": f"{bm_mn.calmar_ratio:.2f}",
                    "Max DD (Net)": f"{bm_mn.max_drawdown:.1%}",
                    tax_label: f"€{bm_total_tax:,.0f}",
                })
            else:
                bm_row.update({
                    "CAGR": f"{bm_m.cagr:.1%}",
                    "Excess": "—",
                    "Volatility": f"{bm_m.volatility:.1%}",
                    "Sharpe": f"{bm_m.sharpe_ratio:.2f}",
                    "Sortino": f"{bm_m.sortino_ratio:.2f}",
                    "Max DD": f"{bm_m.max_drawdown:.1%}",
                    "Calmar": f"{bm_m.calmar_ratio:.2f}",
                    "Win Rate": f"{bm_m.win_rate_monthly:.1%}",
                    "Trades": bm_m.num_trades,
                    "Costs": f"€{bm_m.total_costs:,.0f}",
                })
            data.append(bm_row)

        return pd.DataFrame(data)

    def summary(self) -> str:
        """
        Generate a text summary for terminal output.

        Returns:
            Formatted string comparison
        """
        df = self.summary_table()

        lines = [
            "",
            "═" * 100,
            " STRATEGY COMPARISON",
            "═" * 100,
            "",
        ]

        # Format as aligned table
        col_widths = {col: max(len(col), df[col].astype(str).str.len().max()) + 2
                      for col in df.columns}

        # Header
        header = "".join(f"{col:<{col_widths[col]}}" for col in df.columns)
        lines.append(header)
        lines.append("─" * len(header))

        # Rows
        for _, row in df.iterrows():
            line = "".join(f"{str(row[col]):<{col_widths[col]}}" for col in df.columns)
            lines.append(line)

        lines.append("")
        lines.append("═" * 100)
        return "\n".join(lines)

    def plot_equity_curves(self) -> go.Figure:
        """
        Create a Plotly figure with all equity curves.

        Returns:
            Plotly Figure object
        """
        # Base layout settings (avoiding template= which triggers expensive deep copy)
        base_layout = dict(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Arial, sans-serif", size=12, color="#1f2937"),
            xaxis=dict(gridcolor="#e5e7eb", linecolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", linecolor="#e5e7eb"),
        )

        fig = go.Figure()

        colors = [
            "#2563eb", "#dc2626", "#16a34a", "#9333ea",
            "#ea580c", "#0891b2", "#4f46e5", "#be123c"
        ]

        for i, result in enumerate(self.results):
            # Normalize to starting value of 100 for comparison
            normalized = result.equity_curve / result.equity_curve.iloc[0] * 100

            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized.values,
                mode="lines",
                name=result.strategy.name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))

        # Add benchmark if available
        if self.benchmark_result:
            bm_normalized = self.benchmark_result.equity_curve / self.benchmark_result.equity_curve.iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=bm_normalized.index,
                y=bm_normalized.values,
                mode="lines",
                name=f"📊 Benchmark ({self.benchmark_result.strategy.name})",
                line=dict(color="#6b7280", width=2, dash="dash")
            ))

        fig.update_layout(
            title="Equity Curves (Normalized to 100)",
            xaxis_title="Date",
            yaxis_title="Value (Starting = 100)",
            hovermode="x unified",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            **base_layout,
        )

        return fig

    def to_html(self, path: str) -> None:
        """
        Generate an HTML comparison report.

        Args:
            path: Output file path
        """
        # Base layout settings (avoiding template= which triggers expensive deep copy)
        base_layout = dict(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Arial, sans-serif", size=12, color="#1f2937"),
            xaxis=dict(gridcolor="#e5e7eb", linecolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", linecolor="#e5e7eb"),
        )

        # Create equity curves chart
        equity_fig = self.plot_equity_curves()

        # Create drawdown comparison chart
        dd_fig = go.Figure()
        colors = [
            "#2563eb", "#dc2626", "#16a34a", "#9333ea",
            "#ea580c", "#0891b2", "#4f46e5", "#be123c"
        ]

        for i, result in enumerate(self.results):
            rolling_max = result.equity_curve.cummax()
            drawdown = (result.equity_curve - rolling_max) / rolling_max * 100

            dd_fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name=result.strategy.name,
                line=dict(color=colors[i % len(colors)], width=1.5)
            ))

        dd_fig.update_layout(
            title="Drawdown Comparison",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            **base_layout,
        )

        # Create metrics bar chart
        metrics_fig = go.Figure()
        strategies = [r.strategy.name for r in self.results]
        cagrs = [r.metrics.cagr * 100 for r in self.results]
        sharpes = [r.metrics.sharpe_ratio for r in self.results]

        metrics_fig.add_trace(go.Bar(
            name="CAGR (%)",
            x=strategies,
            y=cagrs,
            marker_color="#2563eb"
        ))

        metrics_fig.update_layout(
            title="CAGR Comparison",
            yaxis_title="CAGR (%)",
            **base_layout,
        )

        # Generate comparison table HTML
        df = self.summary_table()
        table_html = df.to_html(classes="comparison-table", index=False)

        # Generate metadata block if available
        metadata_html = ""
        metadata_css = ""
        metadata_json = ""
        if self.metadata is not None:
            from backtest.metadata import generate_metadata_html, get_metadata_css
            metadata_html = generate_metadata_html(self.metadata)
            metadata_css = get_metadata_css()
            metadata_json = self.metadata.to_json(indent=2)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strategy Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #1f2937;
            background: #f9fafb;
        }}
        .container {{
            max-width: 1400px;
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
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .comparison-table th, .comparison-table td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        .comparison-table th {{
            background: #f9fafb;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            font-size: 0.75rem;
        }}
        .comparison-table tr:hover {{
            background: #f9fafb;
        }}
        .chart-container {{
            width: 100%;
            min-height: 450px;
        }}
        footer {{
            text-align: center;
            padding: 2rem;
            color: #6b7280;
            font-size: 0.875rem;
        }}
        {metadata_css}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Strategy Comparison</h1>
            <p>Comparing {len(self.results)} strategies | {self.results[0].equity_curve.index[0].strftime('%Y-%m-%d')} &rarr; {self.results[0].equity_curve.index[-1].strftime('%Y-%m-%d')}</p>
        </header>

        {metadata_html}

        <section>
            <h2>Performance Metrics</h2>
            {table_html}
        </section>

        <section>
            <h2>Equity Curves (Normalized)</h2>
            <div class="chart-container">
                {equity_fig.to_html(full_html=False, include_plotlyjs=False)}
            </div>
        </section>

        <section>
            <h2>Drawdown Comparison</h2>
            <div class="chart-container">
                {dd_fig.to_html(full_html=False, include_plotlyjs=False)}
            </div>
        </section>

        <section>
            <h2>CAGR Comparison</h2>
            <div class="chart-container">
                {metrics_fig.to_html(full_html=False, include_plotlyjs=False)}
            </div>
        </section>

        <section>
            <h2>Definitions (Glossary)</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem;">
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #2563eb;">CAGR</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Compound Annual Growth Rate. The annualized return that would have been required to grow the initial investment to its final value over the period.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #2563eb;">Volatility</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Annualized standard deviation of returns, measuring the dispersion of returns around the mean. Higher volatility indicates higher risk.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #2563eb;">Sharpe Ratio</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Risk-adjusted return: (Return - Risk-Free Rate) / Volatility. Higher is better. Assumes 2% risk-free rate.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #2563eb;">Sortino Ratio</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Like Sharpe, but uses only downside deviation (negative returns) instead of total volatility. Penalizes only harmful volatility.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #2563eb;">Max Drawdown</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">Maximum peak-to-trough decline during the investment period. Measures worst-case loss scenario.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #2563eb;">Calmar Ratio</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">CAGR divided by absolute Max Drawdown. Measures return per unit of drawdown risk. Higher is better.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #2563eb;">Tax Drag</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;">The reduction in returns due to taxation. Calculated as CAGR_gross - CAGR_net in percentage points.</p>
                </div>
                <div>
                    <h4 style="margin-bottom: 0.5rem; color: #2563eb;">Metric Basis</h4>
                    <p style="font-size: 0.9rem; color: #6b7280;"><strong>Gross:</strong> Before tax. <strong>Net Realized:</strong> After realized taxes only. <strong>Net Liquidation:</strong> After all taxes including virtual end-of-period liquidation tax on unrealized gains.</p>
                </div>
            </div>
        </section>

        <footer>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Framework: backtest v0.1</p>
        </footer>
    </div>
</body>
</html>"""

        Path(path).write_text(html_content)


class Comparator:
    """
    Compares multiple strategies on the same dataset.
    """

    def __init__(
        self,
        strategies: List[Strategy],
        data: PriceData,
        config: Optional[BacktestConfig] = None,
        benchmark_ticker: Optional[str] = None
    ):
        """
        Initialize comparator.

        Args:
            strategies: List of strategies to compare
            data: Historical price data
            config: Backtest configuration (shared by all strategies)
            benchmark_ticker: Optional ticker for benchmark comparison (e.g., "URTH")
        """
        self.strategies = strategies
        self.data = data
        self.config = config or BacktestConfig()
        self.benchmark_ticker = benchmark_ticker

    def run(self, progress: bool = True) -> ComparisonResult:
        """
        Run backtests for all strategies.

        Args:
            progress: If True, print progress messages

        Returns:
            ComparisonResult with all backtest results
        """
        results = []

        for i, strategy in enumerate(self.strategies, 1):
            if progress:
                print(f"  [{i}/{len(self.strategies)}] Running {strategy.name}...")

            # Backtester resolves a None rebalance_frequency from the strategy
            # and writes the resolved value back to its config for reporting.
            # Use a per-run config copy so mixed-frequency comparisons do not
            # let the first strategy's cadence leak into the next one.
            backtester = Backtester(strategy, self.data, replace(self.config))
            result = backtester.run()
            results.append(result)

        # Compute benchmark if ticker is specified and available in data
        benchmark_result = None
        if self.benchmark_ticker and self.benchmark_ticker in self.data.prices.columns:
            if progress:
                print(f"  [Benchmark] Running {self.benchmark_ticker}...")
            benchmark_result = self._compute_benchmark()

        return ComparisonResult(results=results, benchmark_result=benchmark_result)

    def _compute_benchmark(self) -> Optional[BacktestResult]:
        """Compute benchmark backtest (Buy & Hold on benchmark_ticker)."""
        from backtest.strategy import Strategy, Allocation

        class BenchmarkStrategy(Strategy):
            def __init__(self, ticker: str):
                self.name = f"Benchmark_{ticker}"
                self.assets = [ticker]
                self._ticker = ticker

            def signal(self, date, data):
                return Allocation({self._ticker: 1.0})

        try:
            benchmark_strategy = BenchmarkStrategy(self.benchmark_ticker)
            backtester = Backtester(benchmark_strategy, self.data, replace(self.config))
            return backtester.run()
        except Exception as e:
            if self.config.validate:
                import warnings
                warnings.warn(f"Benchmark computation failed: {e}")
            return None
