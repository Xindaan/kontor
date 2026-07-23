"""Chart generation service using existing Reporter functionality."""

from typing import Any, Dict, Optional

import plotly.graph_objects as go

from backtest.utils import MONTH_END_FREQ


def generate_charts(result) -> Dict[str, Any]:
    """Generate Plotly chart configurations from a BacktestResult.

    Returns a dict with 'equity' and 'drawdown' chart JSON configs.
    """
    charts = {}

    # Equity curve chart
    charts["equity"] = _create_equity_chart(result)

    # Drawdown chart
    charts["drawdown"] = _create_drawdown_chart(result)

    return charts


def _create_equity_chart(result) -> Dict[str, Any]:
    """Create equity curve chart configuration."""
    fig = go.Figure()

    # Strategy equity curve
    fig.add_trace(go.Scatter(
        x=[d.isoformat() for d in result.equity_curve.index],
        y=result.equity_curve.values.tolist(),
        mode="lines",
        name=result.strategy.name,
        line=dict(color="#2563eb", width=2),
    ))

    # Benchmark curve if available
    if result.benchmark_curve is not None:
        fig.add_trace(go.Scatter(
            x=[d.isoformat() for d in result.benchmark_curve.index],
            y=result.benchmark_curve.values.tolist(),
            mode="lines",
            name="Benchmark",
            line=dict(color="#9ca3af", width=1, dash="dash"),
        ))

    fig.update_layout(
        margin=dict(t=20, r=20, b=40, l=60),
        xaxis_title="",
        yaxis_title="Portfolio Value (EUR)",
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
    )

    return fig.to_dict()


def _create_drawdown_chart(result) -> Dict[str, Any]:
    """Create drawdown chart configuration."""
    # Calculate drawdown series
    equity = result.equity_curve
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[d.isoformat() for d in drawdown.index],
        y=(drawdown.values * 100).tolist(),
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="#dc2626", width=1),
        fillcolor="rgba(220, 38, 38, 0.3)",
    ))

    fig.update_layout(
        margin=dict(t=20, r=20, b=40, l=60),
        xaxis_title="",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        yaxis=dict(
            ticksuffix="%",
            range=[drawdown.min() * 100 * 1.1, 0],
        ),
    )

    return fig.to_dict()


def _create_monthly_returns_heatmap(result) -> Optional[Dict[str, Any]]:
    """Create monthly returns heatmap configuration.

    This is a more complex chart that requires monthly resampling.
    Only generated if there's enough data.
    """
    import pandas as pd

    equity = result.equity_curve

    # Need at least 12 months of data
    if len(equity) < 252:
        return None

    # Calculate monthly returns
    monthly = equity.resample(MONTH_END_FREQ).last()
    monthly_returns = monthly.pct_change().dropna()

    # Create year/month matrix
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    years = monthly_returns.index.year.unique()
    months = range(1, 13)

    # Build heatmap data
    z = []
    for year in years:
        row = []
        for month in months:
            mask = (monthly_returns.index.year == year) & (monthly_returns.index.month == month)
            if mask.any():
                row.append(float(monthly_returns[mask].values[0] * 100))
            else:
                row.append(None)
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        y=[str(y) for y in years],
        colorscale=[
            [0, "#dc2626"],
            [0.5, "#ffffff"],
            [1, "#16a34a"],
        ],
        zmid=0,
        text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        margin=dict(t=20, r=20, b=40, l=60),
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )

    return fig.to_dict()
