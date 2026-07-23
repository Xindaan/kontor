"""
Run Metadata module - Collect and build metadata for reproducible backtests.

This module provides functionality to:
- Collect environment metadata (Python version, platform)
- Collect git metadata (commit, branch, dirty state)
- Build comprehensive run metadata from backtest configuration
- Embed metadata in HTML reports for auditability

The metadata structure is designed to be:
- Machine-readable (JSON)
- Human-readable (summary table)
- Complete (all parameters that affect results)
"""

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backtest.backtester import BacktestConfig, BacktestResult


@dataclass
class GitInfo:
    """Git repository information."""
    commit: Optional[str] = None
    branch: Optional[str] = None
    dirty: Optional[bool] = None
    available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commit": self.commit,
            "branch": self.branch,
            "dirty": self.dirty,
            "available": self.available,
        }


@dataclass
class EnvironmentInfo:
    """Environment information."""
    python_version: str = ""
    platform_os: str = ""
    platform_machine: str = ""
    platform_release: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python": {"version": self.python_version},
            "platform": {
                "os": self.platform_os,
                "machine": self.platform_machine,
                "release": self.platform_release,
            },
        }


@dataclass
class BacktestInfo:
    """Backtest configuration information."""
    start: str = ""
    end: str = ""
    initial_capital: float = 0.0
    currency: str = "EUR"
    cash_rate: float = 0.0
    rebalance_frequency_cli: Optional[str] = None
    rebalance_frequency_effective: str = "monthly"
    warmup_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "initial_capital": self.initial_capital,
            "currency": self.currency,
            "cash_rate": self.cash_rate,
            "rebalance_frequency_cli": self.rebalance_frequency_cli,
            "rebalance_frequency_effective": self.rebalance_frequency_effective,
            "warmup_days": self.warmup_days,
        }


@dataclass
class DataInfo:
    """Data source information."""
    provider: str = "yfinance"
    auto_adjust: bool = True
    price_field: str = "Close"
    missing_data_policy: str = "ffill"
    download_start: str = ""
    download_end: str = ""
    assets_requested: List[str] = field(default_factory=list)
    assets_downloaded: List[str] = field(default_factory=list)
    assets_failed: List[str] = field(default_factory=list)
    total_trading_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "auto_adjust": self.auto_adjust,
            "price_field": self.price_field,
            "missing_data_policy": self.missing_data_policy,
            "download_start": self.download_start,
            "download_end": self.download_end,
            "assets_requested": self.assets_requested,
            "assets_downloaded": self.assets_downloaded,
            "assets_failed": self.assets_failed,
            "total_trading_days": self.total_trading_days,
        }


@dataclass
class CostInfo:
    """Cost model information."""
    enabled: bool = True
    costs_pct: float = 0.001
    slippage_pct: float = 0.0005
    spread_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "costs_pct": self.costs_pct,
            "slippage_pct": self.slippage_pct,
            "spread_pct": self.spread_pct,
            "total_pct": self.costs_pct + self.slippage_pct + self.spread_pct,
        }


@dataclass
class TaxInfo:
    """Tax model information."""
    enabled: bool = True
    model: str = "DE_basic"
    tax_rate: float = 0.26375
    partial_exemption: float = 0.30
    exemption_amount: float = 1000.0
    notes: str = "cash-effective on realized gains"
    # New fields for gross/net reporting
    tax_accounting_mode: str = "cash_effective"
    headline_metric_basis: str = "net"  # "gross" or "net"
    tax_drag_definition: str = "tax_drag_cagr_pp = (cagr_gross - cagr_net) * 100"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "tax_accounting_mode": self.tax_accounting_mode,
            "headline_metric_basis": self.headline_metric_basis,
            "tax_drag_definition": self.tax_drag_definition,
            "params": {
                "tax_rate": self.tax_rate,
                "partial_exemption": self.partial_exemption,
                "exemption_amount": self.exemption_amount,
            },
            "notes": self.notes,
        }


@dataclass
class StrategyInfo:
    """Strategy information."""
    name: str = ""
    file: str = ""
    class_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    assets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file,
            "class": self.class_name,
            "params": self.params,
            "assets": self.assets,
        }


@dataclass
class RunMetadata:
    """Complete run metadata for a backtest."""
    run_id: str = ""
    mode: str = "single"  # "single" or "compare"
    timestamp_utc: str = ""
    timezone: str = "UTC"

    git: GitInfo = field(default_factory=GitInfo)
    environment: EnvironmentInfo = field(default_factory=EnvironmentInfo)
    backtest: BacktestInfo = field(default_factory=BacktestInfo)
    data: DataInfo = field(default_factory=DataInfo)
    costs: CostInfo = field(default_factory=CostInfo)
    tax: TaxInfo = field(default_factory=TaxInfo)
    strategy: Optional[StrategyInfo] = None

    # For compare mode
    compare_strategies: List[StrategyInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "run_id": self.run_id,
            "mode": self.mode,
            "timestamp_utc": self.timestamp_utc,
            "timezone": self.timezone,
            "git": self.git.to_dict(),
            **self.environment.to_dict(),
            "backtest": self.backtest.to_dict(),
            "data": self.data.to_dict(),
            "costs": self.costs.to_dict(),
            "tax": self.tax.to_dict(),
        }

        if self.mode == "single" and self.strategy:
            result["strategy"] = self.strategy.to_dict()
        elif self.mode == "compare":
            result["compare"] = {
                "strategies": [s.to_dict() for s in self.compare_strategies]
            }

        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


def collect_git_metadata(repo_root: Optional[Path] = None) -> GitInfo:
    """
    Collect git repository information.

    Args:
        repo_root: Path to repository root (uses cwd if None)

    Returns:
        GitInfo with commit, branch, and dirty state
    """
    info = GitInfo()

    if repo_root is None:
        repo_root = Path.cwd()

    try:
        # Check if git is available
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return info

        info.available = True

        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info.commit = result.stdout.strip()[:12]  # Short hash

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info.branch = result.stdout.strip()

        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info.dirty = bool(result.stdout.strip())

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return info


def collect_environment_metadata() -> EnvironmentInfo:
    """
    Collect environment information.

    Returns:
        EnvironmentInfo with Python and platform details
    """
    return EnvironmentInfo(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform_os=platform.system(),
        platform_machine=platform.machine(),
        platform_release=platform.release(),
    )


def build_run_metadata(
    config: "BacktestConfig",
    data_info: Optional[DataInfo] = None,
    strategy_info: Optional[StrategyInfo] = None,
    mode: str = "single",
    compare_strategies: Optional[List[StrategyInfo]] = None,
    cli_rebalance_frequency: Optional[str] = None,
    cli_start: Optional[str] = None,
    cli_end: Optional[str] = None,
) -> RunMetadata:
    """
    Build complete run metadata from backtest configuration.

    Args:
        config: BacktestConfig used for the backtest
        data_info: Optional data source information
        strategy_info: Strategy information (for single mode)
        mode: "single" or "compare"
        compare_strategies: List of strategy infos (for compare mode)
        cli_rebalance_frequency: Rebalance frequency from CLI (if set)
        cli_start: Start date from CLI
        cli_end: End date from CLI

    Returns:
        Complete RunMetadata object
    """
    now = datetime.now(timezone.utc)

    metadata = RunMetadata(
        run_id=now.strftime("%Y%m%d_%H%M%S"),
        mode=mode,
        timestamp_utc=now.isoformat(),
        timezone="UTC",
        git=collect_git_metadata(),
        environment=collect_environment_metadata(),
        backtest=BacktestInfo(
            start=cli_start or "",
            end=cli_end or now.strftime("%Y-%m-%d"),
            initial_capital=config.initial_capital,
            currency=config.currency,
            cash_rate=config.cash_rate,
            rebalance_frequency_cli=cli_rebalance_frequency,
            rebalance_frequency_effective=config.rebalance_frequency,
            warmup_days=0,
        ),
        data=data_info or DataInfo(),
        costs=CostInfo(
            enabled=config.costs_pct > 0 or config.slippage_pct > 0,
            costs_pct=config.costs_pct,
            slippage_pct=config.slippage_pct,
            spread_pct=config.spread_pct,
        ),
        tax=TaxInfo(
            enabled=config.tax_enabled,
            model="DE_basic" if config.tax_enabled else "none",
            tax_rate=config.tax_rate,
            partial_exemption=config.tax_partial_exemption,
            exemption_amount=config.tax_exemption_amount,
        ),
        strategy=strategy_info,
        compare_strategies=compare_strategies or [],
    )

    return metadata


def build_strategy_info(
    strategy,
    file_path: Optional[str] = None,
) -> StrategyInfo:
    """
    Build strategy info from a strategy object.

    Args:
        strategy: Strategy instance
        file_path: Path to strategy file

    Returns:
        StrategyInfo object
    """
    return StrategyInfo(
        name=getattr(strategy, "name", strategy.__class__.__name__),
        file=file_path or "",
        class_name=strategy.__class__.__name__,
        params=getattr(strategy, "params", {}),
        assets=getattr(strategy, "assets", []),
    )


def build_data_info(
    data,
    requested_assets: Optional[List[str]] = None,
    failed_assets: Optional[List[str]] = None,
) -> DataInfo:
    """
    Build data info from PriceData.

    Args:
        data: PriceData object
        requested_assets: List of originally requested assets
        failed_assets: List of assets that failed to download

    Returns:
        DataInfo object
    """
    return DataInfo(
        provider="yfinance",
        auto_adjust=True,
        price_field="Close",
        missing_data_policy="ffill",
        download_start=data.start_date.strftime("%Y-%m-%d") if hasattr(data, 'start_date') else "",
        download_end=data.end_date.strftime("%Y-%m-%d") if hasattr(data, 'end_date') else "",
        assets_requested=requested_assets or list(data.prices.columns),
        assets_downloaded=list(data.prices.columns),
        assets_failed=failed_assets or [],
        total_trading_days=len(data.prices),
    )


def generate_metadata_html(metadata: RunMetadata) -> str:
    """
    Generate HTML for metadata display.

    Args:
        metadata: RunMetadata object

    Returns:
        HTML string for embedding in report
    """
    m = metadata
    bt = m.backtest
    tax = m.tax
    costs = m.costs
    data = m.data

    # Build strategy info string
    if m.mode == "single" and m.strategy:
        strategy_html = f"""
        <tr><td>Strategy</td><td><strong>{m.strategy.name}</strong></td></tr>
        <tr><td>File</td><td><code>{m.strategy.file or 'N/A'}</code></td></tr>
        <tr><td>Class</td><td><code>{m.strategy.class_name}</code></td></tr>
        <tr><td>Assets</td><td>{', '.join(m.strategy.assets[:5])}{'...' if len(m.strategy.assets) > 5 else ''}</td></tr>
        """
        if m.strategy.params:
            params_str = ', '.join(f"{k}={v}" for k, v in list(m.strategy.params.items())[:5])
            strategy_html += f"<tr><td>Params</td><td><code>{params_str}</code></td></tr>"
    elif m.mode == "compare":
        strategies_list = ', '.join(s.name for s in m.compare_strategies)
        strategy_html = f"""
        <tr><td>Mode</td><td><strong>Compare</strong></td></tr>
        <tr><td>Strategies</td><td>{strategies_list}</td></tr>
        """
    else:
        strategy_html = ""

    # Git info
    git_html = ""
    if m.git.available:
        dirty_badge = ' <span style="color: #dc2626;">(dirty)</span>' if m.git.dirty else ""
        git_html = f"""
        <tr><td>Git Commit</td><td><code>{m.git.commit or 'N/A'}</code>{dirty_badge}</td></tr>
        <tr><td>Git Branch</td><td><code>{m.git.branch or 'N/A'}</code></td></tr>
        """

    return f"""
    <section class="run-metadata">
        <details open>
            <summary><h2 style="display: inline;">Run Metadata</h2></summary>
            <div class="metadata-grid">
                <table class="metadata-table">
                    <tbody>
                        <tr><td>Run ID</td><td><code>{m.run_id}</code></td></tr>
                        <tr><td>Timestamp</td><td>{m.timestamp_utc[:19].replace('T', ' ')} UTC</td></tr>
                        {git_html}
                        {strategy_html}
                    </tbody>
                </table>
                <table class="metadata-table">
                    <tbody>
                        <tr><td>Start</td><td><strong>{bt.start}</strong></td></tr>
                        <tr><td>End</td><td><strong>{bt.end}</strong></td></tr>
                        <tr><td>Capital</td><td>{bt.currency} {bt.initial_capital:,.0f}</td></tr>
                        <tr><td>Rebalance</td><td><strong>{bt.rebalance_frequency_effective}</strong>{' (CLI)' if bt.rebalance_frequency_cli else ''}</td></tr>
                        <tr><td>Data Provider</td><td>{data.provider}</td></tr>
                        <tr><td>Trading Days</td><td>{data.total_trading_days:,}</td></tr>
                    </tbody>
                </table>
                <table class="metadata-table">
                    <tbody>
                        <tr><td>Tax Enabled</td><td><strong>{'Yes' if tax.enabled else 'No'}</strong></td></tr>
                        {'<tr><td>Tax Model</td><td>' + tax.model + '</td></tr>' if tax.enabled else ''}
                        {'<tr><td>Tax Rate</td><td>' + f"{tax.tax_rate:.2%}" + '</td></tr>' if tax.enabled else ''}
                        {'<tr><td>Exemption</td><td>€' + f"{tax.exemption_amount:,.0f}" + '</td></tr>' if tax.enabled else ''}
                        <tr><td>Costs (per trade)</td><td>{costs.costs_pct:.2%}</td></tr>
                        <tr><td>Slippage</td><td>{costs.slippage_pct:.2%}</td></tr>
                    </tbody>
                </table>
            </div>
            <details>
                <summary>Full Metadata (JSON)</summary>
                <pre style="background: #f1f5f9; padding: 1rem; border-radius: 4px; overflow-x: auto; font-size: 0.75rem;">{m.to_json(indent=2)}</pre>
            </details>
        </details>
    </section>
    <script type="application/json" id="run-metadata">
{m.to_json(indent=2)}
    </script>
    """


def get_metadata_css() -> str:
    """Get CSS for metadata display."""
    return """
        .run-metadata {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 2rem;
        }
        .run-metadata h2 {
            font-size: 1rem;
            margin: 0;
            color: #475569;
        }
        .run-metadata summary {
            cursor: pointer;
            user-select: none;
        }
        .run-metadata details[open] > summary {
            margin-bottom: 1rem;
        }
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .metadata-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }
        .metadata-table td {
            padding: 0.375rem 0.5rem;
            border-bottom: 1px solid #e2e8f0;
        }
        .metadata-table td:first-child {
            color: #64748b;
            width: 40%;
        }
        .metadata-table code {
            background: #e2e8f0;
            padding: 0.125rem 0.375rem;
            border-radius: 3px;
            font-size: 0.8125rem;
        }
    """
