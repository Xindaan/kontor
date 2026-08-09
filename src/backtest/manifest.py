"""
Run Manifest - Reproducibility and Audit Trail.

This module provides functionality to create run manifests that capture
all information needed to reproduce a backtest run.

A manifest includes:
- Configuration parameters
- Strategy details
- Data source metadata
- Git commit information
- Timestamps
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import hashlib
import json
import subprocess

if TYPE_CHECKING:
    from backtest.config.run_config import RunConfig
    from backtest.strategy import Strategy
    from backtest.data import PriceData


@dataclass
class DataSnapshot:
    """
    Metadata about the data used in a backtest run.

    Attributes:
        source: Data source (e.g., "yahoo", "csv")
        tickers: List of tickers loaded
        start_date: First date in data
        end_date: Last date in data
        data_points: Number of data points
        file_hash: Hash of data files (if applicable)
        last_updated: When data was downloaded/cached
    """
    source: str
    tickers: List[str]
    start_date: str
    end_date: str
    data_points: int
    file_hash: Optional[str] = None
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_price_data(cls, data: "PriceData", source: str = "yahoo") -> "DataSnapshot":
        """Create snapshot from PriceData object."""
        return cls(
            source=source,
            tickers=list(data.prices.columns),
            start_date=data.prices.index[0].strftime("%Y-%m-%d"),
            end_date=data.prices.index[-1].strftime("%Y-%m-%d"),
            data_points=len(data.prices),
            last_updated=datetime.now().isoformat(),
        )


@dataclass
class GitInfo:
    """
    Git repository information for reproducibility.

    Attributes:
        commit_hash: Current commit SHA
        branch: Current branch name
        dirty: True if there are uncommitted changes
        remote_url: Remote repository URL (if available)
    """
    commit_hash: str
    branch: str
    dirty: bool
    remote_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def capture(cls) -> "GitInfo":
        """
        Capture current git state.

        Returns GitInfo with commit hash, branch, and dirty flag.
        Returns unknown values if not in a git repository.
        """
        try:
            # Get commit hash
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            commit_hash = commit.stdout.strip() if commit.returncode == 0 else "unknown"

            # Get branch name
            branch_cmd = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            branch = branch_cmd.stdout.strip() if branch_cmd.returncode == 0 else "unknown"

            # Check for uncommitted changes
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5
            )
            dirty = bool(status.stdout.strip()) if status.returncode == 0 else True

            # Get remote URL (optional)
            remote = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=5
            )
            remote_url = remote.stdout.strip() if remote.returncode == 0 else None

            return cls(
                commit_hash=commit_hash,
                branch=branch,
                dirty=dirty,
                remote_url=remote_url
            )
        except Exception:
            return cls(
                commit_hash="unknown",
                branch="unknown",
                dirty=True,
                remote_url=None
            )


@dataclass
class StrategyInfo:
    """
    Strategy metadata for the manifest.

    Attributes:
        name: Strategy display name
        class_name: Python class name
        params: Strategy parameters
        assets: Required assets
        rebalance_frequency: Strategy's rebalance frequency
    """
    name: str
    class_name: str
    params: Dict[str, Any]
    assets: List[str]
    rebalance_frequency: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_strategy(cls, strategy: "Strategy") -> "StrategyInfo":
        """Create StrategyInfo from Strategy object."""
        return cls(
            name=getattr(strategy, 'name', strategy.__class__.__name__),
            class_name=strategy.__class__.__name__,
            params=getattr(strategy, 'params', {}),
            assets=getattr(strategy, 'assets', []),
            rebalance_frequency=getattr(strategy, 'rebalance_frequency', 'monthly'),
        )


@dataclass
class RunManifest:
    """
    Complete manifest for a backtest run.

    Contains all information needed to reproduce the run:
    - Configuration
    - Strategy details
    - Data snapshot
    - Git information
    - Timestamps

    The manifest is written to JSON alongside HTML reports
    and its summary is displayed in the HTML report header.
    """
    # Run identification
    run_id: str
    timestamp: str

    # Configuration
    config: Dict[str, Any]

    # Strategy
    strategy: StrategyInfo

    # Data
    data: DataSnapshot

    # Environment
    git: GitInfo

    # Results summary (added after run completes)
    results_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "strategy": self.strategy.to_dict(),
            "data": self.data.to_dict(),
            "git": self.git.to_dict(),
            "results_summary": self.results_summary,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        """Save manifest to JSON file."""
        Path(path).write_text(self.to_json())

    @classmethod
    def from_json(cls, json_str: str) -> "RunManifest":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            config=data["config"],
            strategy=StrategyInfo(**data["strategy"]),
            data=DataSnapshot(**data["data"]),
            git=GitInfo(**data["git"]),
            results_summary=data.get("results_summary"),
        )

    @classmethod
    def from_file(cls, path: str) -> "RunManifest":
        """Load from JSON file."""
        return cls.from_json(Path(path).read_text())

    @classmethod
    def create(
        cls,
        config: "RunConfig",
        strategy: "Strategy",
        data: "PriceData",
        data_source: str = "yahoo"
    ) -> "RunManifest":
        """
        Create a new manifest for a backtest run.

        Args:
            config: RunConfig used for the run
            strategy: Strategy being tested
            data: Price data used
            data_source: Data source identifier

        Returns:
            New RunManifest instance
        """
        timestamp = datetime.now()
        run_id = generate_run_id(config, strategy, timestamp)

        return cls(
            run_id=run_id,
            timestamp=timestamp.isoformat(),
            config=config.to_dict(),
            strategy=StrategyInfo.from_strategy(strategy),
            data=DataSnapshot.from_price_data(data, data_source),
            git=GitInfo.capture(),
        )

    def add_results(self, metrics: Dict[str, Any]) -> None:
        """Add results summary to manifest after run completes."""
        self.results_summary = metrics

    def html_summary(self) -> str:
        """Generate HTML summary block for inclusion in reports."""
        config = self.config
        strategy = self.strategy
        git = self.git

        git_status = f"{git.commit_hash[:8]}"
        if git.dirty:
            git_status += " (dirty)"

        return f"""
        <div class="manifest-block">
            <h3>Run Manifest</h3>
            <div class="manifest-grid">
                <div class="manifest-section">
                    <h4>Configuration</h4>
                    <table class="manifest-table">
                        <tr><td>Config Hash</td><td><code>{config.get('config_hash', 'N/A')}</code></td></tr>
                        <tr><td>Period</td><td>{config.get('start_date')} to {config.get('end_date') or 'present'}</td></tr>
                        <tr><td>Initial Capital</td><td>{config.get('currency', 'EUR')} {config.get('initial_capital', 10000):,.0f}</td></tr>
                        <tr><td>Rebalance</td><td>{config.get('rebalance_frequency', 'monthly')}</td></tr>
                        <tr><td>Benchmark</td><td>{config.get('benchmark', 'None')}</td></tr>
                    </table>
                </div>
                <div class="manifest-section">
                    <h4>Cost Model</h4>
                    <table class="manifest-table">
                        <tr><td>Commission</td><td>{config.get('costs', {}).get('commission_pct', 0) * 100:.2f}%</td></tr>
                        <tr><td>Spread</td><td>{config.get('costs', {}).get('spread_bps', 0):.1f} bps</td></tr>
                        <tr><td>Slippage</td><td>{config.get('costs', {}).get('slippage_bps', 0):.1f} bps</td></tr>
                    </table>
                </div>
                <div class="manifest-section">
                    <h4>Strategy</h4>
                    <table class="manifest-table">
                        <tr><td>Name</td><td>{strategy.name}</td></tr>
                        <tr><td>Class</td><td><code>{strategy.class_name}</code></td></tr>
                        <tr><td>Assets</td><td>{', '.join(strategy.assets[:5])}{'...' if len(strategy.assets) > 5 else ''}</td></tr>
                    </table>
                </div>
                <div class="manifest-section">
                    <h4>Reproducibility</h4>
                    <table class="manifest-table">
                        <tr><td>Run ID</td><td><code>{self.run_id}</code></td></tr>
                        <tr><td>Git</td><td><code>{git_status}</code></td></tr>
                        <tr><td>Data Points</td><td>{self.data.data_points:,}</td></tr>
                        <tr><td>Timestamp</td><td>{self.timestamp[:19]}</td></tr>
                    </table>
                </div>
            </div>
        </div>
        """

    def manifest_css(self) -> str:
        """CSS styles for manifest block."""
        return """
        .manifest-block {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        .manifest-block h3 {
            font-size: 1rem;
            color: #475569;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .manifest-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }
        .manifest-section h4 {
            font-size: 0.875rem;
            color: #64748b;
            margin-bottom: 0.5rem;
        }
        .manifest-table {
            width: 100%;
            font-size: 0.8125rem;
        }
        .manifest-table td {
            padding: 0.25rem 0;
            border: none;
        }
        .manifest-table td:first-child {
            color: #64748b;
            width: 40%;
        }
        .manifest-table code {
            background: #e2e8f0;
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
            font-size: 0.75rem;
        }
        """


def generate_run_id(
    config: "RunConfig",
    strategy: "Strategy",
    timestamp: datetime
) -> str:
    """
    Generate a unique run ID.

    Format: {strategy_short}_{config_hash}_{timestamp}

    Args:
        config: Run configuration
        strategy: Strategy being tested
        timestamp: Run timestamp

    Returns:
        Unique run identifier
    """
    strategy_name = getattr(strategy, 'name', strategy.__class__.__name__)
    strategy_short = strategy_name.lower().replace(' ', '_')[:20]
    config_hash = config.config_hash
    ts = timestamp.strftime("%Y%m%d_%H%M%S")

    return f"{strategy_short}_{config_hash}_{ts}"


def compute_data_hash(data: "PriceData") -> str:
    """
    Compute a hash of the price data for reproducibility verification.

    Args:
        data: PriceData to hash

    Returns:
        16-character hex hash
    """
    # Create a deterministic representation
    content = data.prices.to_csv()
    hash_bytes = hashlib.sha256(content.encode()).digest()
    return hash_bytes[:8].hex()
