"""Pydantic models for request/response validation."""

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class StrategyParam(BaseModel):
    """Schema for a strategy parameter."""
    name: str
    type: str
    required: bool
    default: Optional[Any] = None
    choices: Optional[List[Any]] = None
    description: Optional[str] = None


class StrategyInfo(BaseModel):
    """Schema for strategy information."""
    name: str
    file_path: str
    file_name: str
    description: Optional[str] = None
    assets: List[str] = []
    rebalance_frequency: str = "monthly"


class StrategySchema(BaseModel):
    """Full schema for a strategy including parameters."""
    strategy_name: str
    description: Optional[str] = None
    assets: List[str] = []
    rebalance_frequency: str = "monthly"
    parameters: List[StrategyParam] = []


# ============================================================================
# Manual Data Provenance Schemas
# ============================================================================

class ManualProvenanceCreateRequest(BaseModel):
    """Request schema for creating manual data provenance entries."""
    file_path: str = Field(..., description="Path to local manual data file")
    dataset: str = Field(..., description="Dataset identifier")
    source: str = Field(..., description="Source name (e.g., SeekingAlpha)")
    quality_tag: str = Field(default="manual", description="official|proxy|community|manual")
    as_of_date: Optional[str] = Field(default=None, description="As-of date (YYYY-MM-DD)")
    import_method: str = Field(default="manual_upload", description="Import method label")
    license_tos_note: str = Field(default="", description="License / ToS note")
    source_url: Optional[str] = Field(default=None, description="Optional source URL")
    notes: Optional[str] = Field(default=None, description="Optional notes")
    entry_id: Optional[str] = Field(default=None, description="Optional explicit entry id")


class ManualProvenanceEntryResponse(BaseModel):
    """Single manual data provenance entry."""
    entry_id: str
    dataset: str
    file_path: str
    source: str
    quality_tag: str
    as_of_date: Optional[str] = None
    import_method: str
    license_tos_note: str
    source_url: Optional[str] = None
    imported_at: str
    checksum_sha256: Optional[str] = None
    file_size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    notes: Optional[str] = None


class ManualProvenanceListResponse(BaseModel):
    """List response for manual provenance entries."""
    total: int
    entries: List[ManualProvenanceEntryResponse]


class ManualProvenanceIssueResponse(BaseModel):
    """Single provenance verification issue."""
    entry_id: str
    status: str
    message: str


class ManualProvenanceVerifyResponse(BaseModel):
    """Verification response for manual provenance registry."""
    registry_path: str
    total_entries: int
    ok_entries: int
    issue_count: int
    issues: List[ManualProvenanceIssueResponse]


# ============================================================================
# Signals Schemas
# ============================================================================

class SignalPortfolioRequest(BaseModel):
    """Optional portfolio payload for live signal generation."""
    positions: Dict[str, float] = Field(default_factory=dict)
    cash: float = Field(default=0.0)
    last_rebalance: Optional[str] = Field(default=None, description="YYYY-MM-DD")


class MetaCandidateRequest(BaseModel):
    """One candidate strategy for meta decisioning."""
    strategy: str = Field(..., description="Path to candidate strategy file")
    params: Dict[str, Any] = Field(default_factory=dict, description="Optional candidate params")


class MetaDecisionRequest(BaseModel):
    """Optional meta-switch decision config for signals endpoint."""
    enabled: bool = Field(default=False)
    candidates: List[MetaCandidateRequest] = Field(default_factory=list)
    params_source: Literal["preset_first", "strategy_defaults", "manual_only"] = Field(default="preset_first")
    preset_params_file: Optional[str] = Field(default=None)
    scoring_mode: Literal["hybrid", "gate_only", "performance_only"] = Field(default="hybrid")
    performance_windows_days: List[int] = Field(default_factory=lambda: [63, 126, 252])
    performance_weights: List[float] = Field(default_factory=lambda: [0.5, 0.3, 0.2])
    confirm_points: int = Field(default=2, ge=1)
    switch_margin: float = Field(default=0.10, ge=0.0)
    decision_cadence: Literal["run_check_rebalance_switch", "immediate", "monthly_fixed"] = Field(
        default="run_check_rebalance_switch"
    )
    plan_mode: Literal["recommendation_only", "recommendation_with_portfolio_plan", "always_plan"] = Field(
        default="recommendation_with_portfolio_plan"
    )
    evidence_required: bool = Field(default=True)
    evidence_profile: Literal["defensiv", "ausgewogen", "aggressiv", "custom"] = Field(default="ausgewogen")
    evidence_compare_mode: Literal["vs_current"] = Field(default="vs_current")
    evidence_max_age_days: int = Field(default=30, ge=1)
    evidence_artifact_path: Optional[str] = Field(default=None)
    gate_fail_action: Literal["hold_current", "manual_override", "fallback_safe"] = Field(default="hold_current")
    custom_thresholds: Dict[str, float] = Field(default_factory=dict)
    regime_mode: Literal["none", "strategy_fragility"] = Field(default="strategy_fragility")
    regime_profile: Literal["defensiv", "ausgewogen", "aggressiv", "custom"] = Field(default="ausgewogen")
    alpha_tie_band: Optional[float] = Field(default=None, ge=0.0)
    stress_alpha_tolerance: Optional[float] = Field(default=None, ge=0.0)
    conditioned_min_windows: Optional[int] = Field(default=None, ge=1)


class SignalRequest(BaseModel):
    """Request schema for live signal generation."""
    strategy: str = Field(..., description="Path to strategy file")
    params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameter overrides")
    signal_date: Optional[str] = Field(default=None, description="Signal date YYYY-MM-DD")
    rebalance_frequency: Optional[str] = Field(default=None, description="Optional rebalance frequency override")
    portfolio: Optional[SignalPortfolioRequest] = Field(default=None, description="Current portfolio snapshot")
    drift_tolerance: float = Field(default=0.005, ge=0.0, description="Drift tolerance as absolute weight")
    skip_failed: bool = Field(default=True, description="Skip failed ticker downloads")
    exposure_policy: Optional[Dict[str, Any]] = Field(default=None, description="Optional 3x exposure controller config")
    meta_decision: Optional[MetaDecisionRequest] = Field(default=None, description="Optional meta-switch decision config")


class SignalResponse(BaseModel):
    """Response schema for live signal generation."""
    report: Dict[str, Any]


class MetaEvidenceRunRequest(BaseModel):
    """Request schema for explicit evidence artifact generation."""
    current_strategy: str = Field(..., description="Current strategy path")
    target_strategy: str = Field(..., description="Target strategy path")
    current_params: Dict[str, Any] = Field(default_factory=dict)
    target_params: Dict[str, Any] = Field(default_factory=dict)
    as_of: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    evidence_profile: Literal["defensiv", "ausgewogen", "aggressiv", "custom"] = Field(default="ausgewogen")
    evidence_compare_mode: Literal["vs_current"] = Field(default="vs_current")
    evidence_max_age_days: int = Field(default=30, ge=1)
    evidence_artifact_path: Optional[str] = Field(default=None)
    custom_thresholds: Dict[str, float] = Field(default_factory=dict)
    train_years: float = Field(default=5.0, gt=0.0)
    test_years: float = Field(default=1.0, gt=0.0)
    step_months: int = Field(default=12, ge=1)
    anchored: bool = Field(default=False)
    start_date: str = Field(default="2010-01-01")
    initial_capital: float = Field(default=10000.0)
    costs_pct: float = Field(default=0.001)
    skip_failed: bool = Field(default=True)
    metric_basis: Literal["gross", "net_realized", "net_liquidation"] = Field(default="gross")
    tuning_enabled: bool = Field(default=False, description="Enable 2-stage smart tuning for meta switch params")
    grid_confirm_points: List[int] = Field(default_factory=lambda: [1, 2, 3])
    grid_switch_margin: List[float] = Field(default_factory=lambda: [0.05, 0.10, 0.15])
    max_combinations: int = Field(default=120, ge=1)
    top_k: int = Field(default=10, ge=1)


class MetaEvidenceResponse(BaseModel):
    """Generic evidence artifact payload response."""
    artifact: Dict[str, Any]


class MetaPromotionRunRequest(BaseModel):
    """Request schema for meta-promotion governance reports."""
    strategies: Optional[List[str]] = Field(default=None, description="Strategy paths; None -> module default set")
    baseline: Optional[str] = Field(default=None, description="Incumbent baseline path; None -> default")
    start: str = Field(default="2016-01-01", description="Backtest start (YYYY-MM-DD)")
    end: Optional[str] = Field(default=None, description="Backtest end (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, gt=0.0)
    costs_pct: float = Field(default=0.001, ge=0.0)
    metric_basis: Literal["gross", "net_realized", "net_liquidation"] = Field(default="net_liquidation")
    tax_enabled: bool = Field(default=True)
    brokers: Optional[List[Literal["trade_republic", "maxblue"]]] = Field(default=None)
    align: Literal["intersection", "ffill"] = Field(default="ffill")
    skip_failed: bool = Field(default=True)
    validate_data: bool = Field(default=True, description="Pre/post-run validation")
    research_proxy_mode: Literal["live", "soxl_proxy"] = Field(default="live")
    tail_risk_gate_basis: Literal["daily", "rebalance"] = Field(default="daily")


class MetaPromotionResponse(BaseModel):
    """Promotion artifact payload response."""
    artifact: Dict[str, Any]


class MetaPromotionListResponse(BaseModel):
    """List of promotion artifact metadata for the artifact browser."""
    artifacts: List[Dict[str, Any]]


class MetaBootstrapRunRequest(BaseModel):
    """Request schema for neutral start strategy bootstrap decision."""
    strategy_a: str = Field(..., description="First strategy path")
    strategy_b: str = Field(..., description="Second strategy path")
    strategy_a_params: Dict[str, Any] = Field(default_factory=dict)
    strategy_b_params: Dict[str, Any] = Field(default_factory=dict)
    as_of: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    evidence_profile: Literal["defensiv", "ausgewogen", "aggressiv", "custom"] = Field(default="ausgewogen")
    evidence_compare_mode: Literal["vs_current"] = Field(default="vs_current")
    evidence_max_age_days: int = Field(default=30, ge=1)
    custom_thresholds: Dict[str, float] = Field(default_factory=dict)
    train_years: float = Field(default=5.0, gt=0.0)
    test_years: float = Field(default=1.0, gt=0.0)
    step_months: int = Field(default=12, ge=1)
    anchored: bool = Field(default=False)
    start_date: str = Field(default="2010-01-01")
    initial_capital: float = Field(default=10000.0)
    costs_pct: float = Field(default=0.001)
    skip_failed: bool = Field(default=True)
    metric_basis: Literal["gross", "net_realized", "net_liquidation"] = Field(default="gross")
    fallback_cagr_tie_band_pp: float = Field(default=1.0, ge=0.0)
    fallback_tie_breaker: Literal["maxdd", "sharpe"] = Field(default="maxdd")
    artifact_path: Optional[str] = Field(default=None)


class MetaBootstrapResponse(BaseModel):
    """Neutral bootstrap decision payload response."""
    artifact: Dict[str, Any]


class RunRequest(BaseModel):
    """Request schema for running a backtest."""
    strategy: str = Field(..., description="Path to strategy file")
    run_id: Optional[str] = Field(default=None, description="Client-side run identifier")
    start_date: str = Field(default="2010-01-01", description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, description="Initial capital")
    benchmark: str = Field(default="S&P 500", description="Benchmark name")
    costs_pct: float = Field(default=0.001, description="Transaction costs as decimal")
    rebalance_frequency: str = Field(
        default="monthly",
        description="Rebalance frequency"
    )

    # Tax settings
    tax_enabled: bool = Field(default=True, description="Enable German tax model")
    tax_exemption: float = Field(default=1000.0, description="Tax exemption amount (EUR)")
    metric_basis: str = Field(
        default="net_liquidation",
        description="Metric basis: gross, net_realized, or net_liquidation"
    )

    # Data settings
    drip_enabled: bool = Field(default=False, description="Enable dividend reinvestment")
    skip_failed: bool = Field(default=True, description="Skip failed ticker downloads")
    enable_validation: bool = Field(default=True, description="Enable validation checks")

    # Execution realism settings
    execution_lag_days: int = Field(default=0, description="Execution lag in trading days")
    max_volume_participation: Optional[float] = Field(default=None, description="Max share of daily volume per trade")
    min_daily_dollar_volume: float = Field(default=0.0, description="Skip trades below this daily notional volume")
    liquidity_on_missing_volume: str = Field(default="allow", description="allow or skip trades when volume data is missing")

    # Risk overlays
    exposure_policy: Optional[Dict[str, Any]] = Field(default=None, description="Optional 3x exposure controller config")
    max_position: Optional[float] = Field(default=None, description="Max target position weight per ticker")
    sector_caps: Dict[str, float] = Field(default_factory=dict, description="Sector cap mapping (sector -> max weight)")
    ticker_sectors: Dict[str, str] = Field(default_factory=dict, description="Ticker to sector mapping for sector caps")
    turnover_budget: Optional[float] = Field(default=None, description="Max one-way turnover per rebalance")
    drawdown_brake_threshold: Optional[float] = Field(default=None, description="Drawdown threshold to activate drawdown brake")
    drawdown_brake_cash_target: float = Field(default=1.0, description="Cash target while drawdown brake is active")
    drawdown_brake_release: Optional[float] = Field(default=None, description="Optional release drawdown level for brake hysteresis")

    # Strategy parameters
    params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameter overrides")


class MetricsResponse(BaseModel):
    """Response schema for metrics."""
    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: Optional[float] = None
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: Optional[float] = None
    win_rate_monthly: Optional[float] = None
    best_month: Optional[float] = None
    worst_month: Optional[float] = None
    num_trades: int = 0
    turnover_annual: Optional[float] = None
    total_costs: float = 0.0


class TaxSummaryResponse(BaseModel):
    """Response schema for tax summary."""
    total_tax_paid: float
    total_realized_gains: float
    total_realized_losses: float
    net_realized_gain: float
    exemption_used: float
    effective_tax_rate: float
    tax_drag_cagr_pp: float
    tax_drag_final_pct: float = 0.0
    final_value_gross: float
    final_value_net_realized: float
    final_value_net_liquidation: float
    # Liquidation tax fields
    tax_paid_liquidation: float = 0.0
    unrealized_gain_at_end: float = 0.0
    # CAGR values for comparison table
    cagr_gross: float = 0.0
    cagr_net_realized: float = 0.0
    cagr_net_liquidation: float = 0.0


class TradeResponse(BaseModel):
    """Response schema for a single trade."""
    date: str
    ticker: str
    action: str  # BUY or SELL
    shares: float
    price: float
    value: float
    costs: float
    tax_paid: Optional[float] = None


class TradingCostsResponse(BaseModel):
    """Response schema for trading costs breakdown."""
    total_costs: float
    costs_per_year: float
    costs_pct_of_final: float
    num_trades: int
    turnover_annual: float


class MonthlyReturnResponse(BaseModel):
    """Response schema for monthly returns."""
    year: int
    month: int
    return_pct: float


class BacktestResultResponse(BaseModel):
    """Response schema for backtest results."""
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    final_value_gross: Optional[float] = None
    final_value_net: Optional[float] = None
    years: float = 0.0
    benchmark_name: Optional[str] = None

    # Metrics
    metrics: MetricsResponse
    metrics_gross: Optional[MetricsResponse] = None
    metrics_net: Optional[MetricsResponse] = None

    # Tax
    tax_enabled: bool
    tax_summary: Optional[TaxSummaryResponse] = None

    # Trading costs
    trading_costs: Optional[TradingCostsResponse] = None

    # Trades (last 50)
    trades: List[TradeResponse] = []

    # Monthly returns
    monthly_returns: List[MonthlyReturnResponse] = []

    # Charts (Plotly JSON)
    equity_chart: Optional[Dict[str, Any]] = None
    drawdown_chart: Optional[Dict[str, Any]] = None
    monthly_returns_chart: Optional[Dict[str, Any]] = None

    # Data series (for custom charting)
    equity_curve: Optional[Dict[str, List]] = None
    benchmark_curve: Optional[Dict[str, List]] = None
    constraint_impact: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str
    detail: Optional[str] = None


# ============================================================================
# Compare Schemas
# ============================================================================

class StrategyConfig(BaseModel):
    """Configuration for a single strategy in a comparison."""
    strategy: str = Field(..., description="Path to strategy file")
    params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameter overrides")


class CompareRequest(BaseModel):
    """Request schema for comparing multiple strategies."""
    strategies: List[StrategyConfig] = Field(..., description="List of strategies to compare", min_length=1)
    run_id: Optional[str] = Field(default=None, description="Client-side run identifier")
    start_date: str = Field(default="2010-01-01", description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, description="Initial capital")
    benchmark: str = Field(default="S&P 500", description="Benchmark name")
    costs_pct: float = Field(default=0.001, description="Transaction costs as decimal")

    # Tax settings
    tax_enabled: bool = Field(default=True, description="Enable German tax model")
    tax_exemption: float = Field(default=1000.0, description="Tax exemption amount (EUR)")
    metric_basis: str = Field(
        default="net_liquidation",
        description="Metric basis: gross, net_realized, or net_liquidation"
    )

    # Data settings
    drip_enabled: bool = Field(default=False, description="Enable dividend reinvestment")
    skip_failed: bool = Field(default=True, description="Skip failed ticker downloads")
    enable_validation: bool = Field(default=True, description="Enable validation checks")

    # Execution realism settings
    execution_lag_days: int = Field(default=0, description="Execution lag in trading days")
    max_volume_participation: Optional[float] = Field(default=None, description="Max share of daily volume per trade")
    min_daily_dollar_volume: float = Field(default=0.0, description="Skip trades below this daily notional volume")
    liquidity_on_missing_volume: str = Field(default="allow", description="allow or skip trades when volume data is missing")

    # Risk overlays
    exposure_policy: Optional[Dict[str, Any]] = Field(default=None, description="Optional 3x exposure controller config")
    max_position: Optional[float] = Field(default=None, description="Max target position weight per ticker")
    sector_caps: Dict[str, float] = Field(default_factory=dict, description="Sector cap mapping (sector -> max weight)")
    ticker_sectors: Dict[str, str] = Field(default_factory=dict, description="Ticker to sector mapping for sector caps")
    turnover_budget: Optional[float] = Field(default=None, description="Max one-way turnover per rebalance")
    drawdown_brake_threshold: Optional[float] = Field(default=None, description="Drawdown threshold to activate drawdown brake")
    drawdown_brake_cash_target: float = Field(default=1.0, description="Cash target while drawdown brake is active")
    drawdown_brake_release: Optional[float] = Field(default=None, description="Optional release drawdown level for brake hysteresis")


class CompareRowResponse(BaseModel):
    """Single row in the comparison table."""
    strategy_name: str
    is_benchmark: bool = False

    # Values (depending on metric_basis)
    final_value: float
    cagr: float
    cagr_gross: Optional[float] = None
    excess_cagr: Optional[float] = None
    tax_drag_pp: Optional[float] = None

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate_monthly: Optional[float] = None

    # Costs & Tax
    total_tax: Optional[float] = None
    total_costs: float
    num_trades: int

    # Equity curve for chart
    equity_curve: Dict[str, List]


class CompareResultResponse(BaseModel):
    """Response schema for strategy comparison."""
    start_date: str
    end_date: str
    initial_capital: float
    years: float
    benchmark_name: Optional[str] = None
    metric_basis: str
    tax_enabled: bool

    # Comparison data
    rows: List[CompareRowResponse]

    # Combined benchmark curve (if available)
    benchmark_curve: Optional[Dict[str, List]] = None


# ============================================================================
# Sweep Schemas
# ============================================================================

class SweepRequest(BaseModel):
    """Request schema for sweep (robustness) analysis."""
    strategies: List[str] = Field(..., description="List of strategy file paths", min_length=1)
    run_id: Optional[str] = Field(default=None, description="Client-side run identifier")

    # Optional: Loaded parameters from optimization
    strategy_params: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Pre-optimized parameters per strategy class name"
    )

    # Windowing configuration
    mode: str = Field(default="rolling", description="Window mode: 'rolling' or 'end-fixed'")
    window_length: str = Field(default="10y", description="Window length (e.g., '5y', '10y')")
    from_date: Optional[str] = Field(default=None, description="Start date for window generation")
    to_date: Optional[str] = Field(default=None, description="End date for window starts")
    end_date: Optional[str] = Field(default=None, description="Fixed end date (for end-fixed mode)")
    start_grid: str = Field(default="monthly", description="Start date grid: 'weekly', 'monthly', 'yearly'")
    step: int = Field(default=1, description="Skip every N grid points", ge=1)

    # Backtest settings
    initial_capital: float = Field(default=10000.0, description="Initial capital")
    rebalance_frequency: str = Field(default="monthly", description="Rebalance frequency")
    costs_pct: float = Field(default=0.001, description="Transaction costs as decimal")
    benchmark: str = Field(default="SPY", description="Benchmark ticker")

    # Tax settings
    tax_enabled: bool = Field(default=True, description="Enable German tax model")
    tax_exemption: float = Field(default=1000.0, description="Tax exemption amount (EUR)")
    metric_basis: str = Field(
        default="net_liquidation",
        description="Metric basis: gross, net_realized, or net_liquidation"
    )

    # Data settings
    drip_enabled: bool = Field(default=False, description="Enable dividend reinvestment")
    skip_failed: bool = Field(default=True, description="Skip failed ticker downloads")

    # Execution realism settings
    execution_lag_days: int = Field(default=0, description="Execution lag in trading days")
    max_volume_participation: Optional[float] = Field(default=None, description="Max share of daily volume per trade")
    min_daily_dollar_volume: float = Field(default=0.0, description="Skip trades below this daily notional volume")
    liquidity_on_missing_volume: str = Field(default="allow", description="allow or skip trades when volume data is missing")

    # Risk overlays
    max_position: Optional[float] = Field(default=None, description="Max target position weight per ticker")
    sector_caps: Dict[str, float] = Field(default_factory=dict, description="Sector cap mapping (sector -> max weight)")
    ticker_sectors: Dict[str, str] = Field(default_factory=dict, description="Ticker to sector mapping for sector caps")
    turnover_budget: Optional[float] = Field(default=None, description="Max one-way turnover per rebalance")
    drawdown_brake_threshold: Optional[float] = Field(default=None, description="Drawdown threshold to activate drawdown brake")
    drawdown_brake_cash_target: float = Field(default=1.0, description="Cash target while drawdown brake is active")
    drawdown_brake_release: Optional[float] = Field(default=None, description="Optional release drawdown level for brake hysteresis")


class SweepSummaryRow(BaseModel):
    """Summary statistics for a single strategy across all windows."""
    strategy_name: str
    strategy_file: str
    rank: int
    pareto_front: bool = False

    # Window counts
    num_windows: int
    num_ok: int
    num_skipped: int

    # Return robustness (Optional to handle NaN/missing values)
    median_cagr: Optional[float] = None
    p10_cagr: Optional[float] = None
    p90_cagr: Optional[float] = None
    worst_cagr: Optional[float] = None
    best_cagr: Optional[float] = None
    prob_negative_cagr: Optional[float] = None

    # Risk metrics
    median_sharpe: Optional[float] = None
    p10_sharpe: Optional[float] = None
    median_sortino: Optional[float] = None
    median_maxdd: Optional[float] = None
    worst_maxdd: Optional[float] = None
    median_vol: Optional[float] = None
    median_calmar: Optional[float] = None

    # CAGR variants
    median_cagr_gross: Optional[float] = None
    median_cagr_net_realized: Optional[float] = None
    median_cagr_net_liquidation: Optional[float] = None
    p10_cagr_gross: Optional[float] = None
    p10_cagr_net_realized: Optional[float] = None
    p10_cagr_net_liquidation: Optional[float] = None

    # Benchmark comparison
    hit_rate_vs_benchmark: Optional[float] = None
    median_excess_cagr: Optional[float] = None
    p10_excess_cagr: Optional[float] = None
    prob_underperform_benchmark: Optional[float] = None


class SweepWindowResult(BaseModel):
    """Result for a single strategy in a single window."""
    strategy_name: str
    window_start: str
    window_end: str
    window_years: float
    status: str  # "ok", "skipped", "error"

    # Metrics
    cagr: float = 0.0
    cagr_gross: float = 0.0
    cagr_net_realized: float = 0.0
    cagr_net_liquidation: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0

    # Benchmark
    benchmark_cagr: Optional[float] = None
    excess_cagr: Optional[float] = None


class SweepResultResponse(BaseModel):
    """Response schema for sweep analysis."""
    # Configuration summary
    mode: str
    window_length: str
    start_grid: str
    metric_basis: str
    tax_enabled: bool
    benchmark_ticker: str

    # Window range
    num_windows: int
    first_window_start: Optional[str] = None
    last_window_end: Optional[str] = None

    # Strategy summaries (ranked)
    summaries: List[SweepSummaryRow]

    # Window results (for charts)
    window_results: List[SweepWindowResult]


class SweepProgressResponse(BaseModel):
    """Response schema for sweep progress."""
    run_id: str
    status: str
    run_count: int
    total_runs: int
    strategy_index: int
    strategy_total: int
    strategy_name: Optional[str] = None
    iteration_index: int
    iteration_total: int
    elapsed_seconds: float


class RunProgressResponse(BaseModel):
    """Response schema for run progress."""
    run_id: str
    status: str
    message: Optional[str] = None
    elapsed_seconds: float


class CompareProgressResponse(BaseModel):
    """Response schema for compare progress."""
    run_id: str
    status: str
    strategy_index: int
    strategy_total: int
    strategy_name: Optional[str] = None
    elapsed_seconds: float


class BatchOptimizeProgressResponse(BaseModel):
    """Response schema for batch optimization progress."""
    run_id: str
    status: str
    strategy_index: int
    strategy_total: int
    strategy_name: Optional[str] = None
    run_count: int
    total_runs: int
    elapsed_seconds: float


class OptimizeProgressResponse(BaseModel):
    """Response schema for optimization progress."""
    run_id: str
    status: str
    run_count: int
    total_runs: int
    elapsed_seconds: float
    message: Optional[str] = None


# ============================================================================
# Optimize Schemas
# ============================================================================

class ParamGridItem(BaseModel):
    """Single parameter with values to test."""
    name: str = Field(..., description="Parameter name")
    values: List[Any] = Field(..., description="List of values to test", min_length=1)


class OptimizeRequest(BaseModel):
    """Request schema for parameter optimization."""
    strategy: str = Field(..., description="Path to strategy file")
    run_id: Optional[str] = Field(default=None, description="Client-side run identifier")

    # Parameter grid
    param_grid: List[ParamGridItem] = Field(
        default_factory=list,
        description="Parameter grid to optimize"
    )
    rebalance_frequencies: List[str] = Field(
        default=["monthly"],
        description="List of rebalance frequencies to test"
    )

    # Optimization settings
    metric: str = Field(default="sharpe_ratio", description="Metric to optimize")
    minimize: bool = Field(default=False, description="Minimize metric instead of maximize")
    top_n: int = Field(default=10, description="Number of top results to return", ge=1, le=50)

    # Backtest settings
    start_date: str = Field(default="2010-01-01", description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, description="Initial capital")
    costs_pct: float = Field(default=0.001, description="Transaction costs as decimal")

    # Tax settings
    tax_enabled: bool = Field(default=True, description="Enable German tax model")
    tax_exemption: float = Field(default=1000.0, description="Tax exemption amount (EUR)")
    metric_basis: str = Field(
        default="net_liquidation",
        description="Metric basis: gross, net_realized, or net_liquidation"
    )

    # Data settings
    drip_enabled: bool = Field(default=False, description="Enable dividend reinvestment")
    skip_failed: bool = Field(default=True, description="Skip failed ticker downloads")
    enable_validation: bool = Field(default=True, description="Enable validation checks")

    # Execution realism settings
    execution_lag_days: int = Field(default=0, description="Execution lag in trading days")
    max_volume_participation: Optional[float] = Field(default=None, description="Max share of daily volume per trade")
    min_daily_dollar_volume: float = Field(default=0.0, description="Skip trades below this daily notional volume")
    liquidity_on_missing_volume: str = Field(default="allow", description="allow or skip trades when volume data is missing")

    # Risk overlays
    max_position: Optional[float] = Field(default=None, description="Max target position weight per ticker")
    sector_caps: Dict[str, float] = Field(default_factory=dict, description="Sector cap mapping (sector -> max weight)")
    ticker_sectors: Dict[str, str] = Field(default_factory=dict, description="Ticker to sector mapping for sector caps")
    turnover_budget: Optional[float] = Field(default=None, description="Max one-way turnover per rebalance")
    drawdown_brake_threshold: Optional[float] = Field(default=None, description="Drawdown threshold to activate drawdown brake")
    drawdown_brake_cash_target: float = Field(default=1.0, description="Cash target while drawdown brake is active")
    drawdown_brake_release: Optional[float] = Field(default=None, description="Optional release drawdown level for brake hysteresis")

    # Walk-Forward Optimization Settings
    walk_forward: bool = Field(default=False, description="Enable walk-forward optimization")
    walk_forward_nested: bool = Field(default=False, description="Enable nested walk-forward optimization")
    wf_train_years: Optional[float] = Field(default=5.0, description="Training window in years")
    wf_test_years: Optional[float] = Field(default=1.0, description="Test window in years")
    wf_step_months: Optional[int] = Field(default=12, description="Step size between windows in months")
    wf_anchored: bool = Field(default=False, description="Use anchored (vs rolling) training window")
    wf_inner_train_years: Optional[float] = Field(
        default=3.0,
        description="Nested mode: inner training window in years",
    )
    wf_inner_test_years: Optional[float] = Field(
        default=1.0,
        description="Nested mode: inner test window in years",
    )
    wf_inner_step_months: Optional[int] = Field(
        default=6,
        description="Nested mode: inner step size in months",
    )
    wf_inner_anchored: bool = Field(
        default=False,
        description="Nested mode: use anchored inner windows",
    )


class OptimizeResultRow(BaseModel):
    """Single optimization result row."""
    rank: int
    rebalance_frequency: str
    params: Dict[str, Any]

    # All metrics
    cagr: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    calmar_ratio: Optional[float] = None
    total_return: Optional[float] = None

    # The metric used for optimization
    metric_value: Optional[float] = None

    # Error if any
    error: Optional[str] = None


class WalkForwardWindowResponse(BaseModel):
    """Single walk-forward window result."""
    window_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]
    train_sharpe: Optional[float] = None
    test_sharpe: Optional[float] = None
    test_cagr: Optional[float] = None
    degradation: Optional[float] = None
    best_rebalance_frequency: Optional[str] = None
    inner_score_mean: Optional[float] = None
    inner_score_std: Optional[float] = None
    inner_windows: Optional[int] = None


class WalkForwardResponse(BaseModel):
    """Walk-forward optimization results."""
    strategy_name: str
    strategy_file: str
    metric: str
    start_date: str
    end_date: str

    # Configuration
    train_months: int
    test_months: int
    step_months: int
    anchored: bool
    mode: str = "standard"
    nested: bool = False
    inner_train_months: Optional[int] = None
    inner_test_months: Optional[int] = None
    inner_step_months: Optional[int] = None
    inner_anchored: Optional[bool] = None
    num_windows: int

    # Summary Metrics
    avg_is_sharpe: Optional[float] = None  # Average In-Sample Sharpe
    avg_oos_sharpe: Optional[float] = None  # Average Out-of-Sample Sharpe
    degradation_ratio: Optional[float] = None  # (IS - OOS) / IS
    overfitting_score: Optional[float] = None  # 0-1 score

    # Per-Window Results
    windows: List[WalkForwardWindowResponse]

    # Best Overall Parameters (based on OOS performance)
    best_params: Optional[Dict[str, Any]] = None
    best_oos_sharpe: Optional[float] = None


class OptimizeResultResponse(BaseModel):
    """Response schema for optimization results."""
    strategy_name: str
    strategy_file: str

    # Configuration
    metric: str
    minimize: bool
    start_date: str
    end_date: str
    metric_basis: str
    tax_enabled: bool

    # Results
    total_combinations: int
    successful_runs: int
    failed_runs: int

    # Top results (ranked)
    results: List[OptimizeResultRow]

    # Best parameters (convenience)
    best_params: Optional[Dict[str, Any]] = None
    best_rebalance_frequency: Optional[str] = None
    best_metric_value: Optional[float] = None

    # Walk-Forward Results (optional, only when walk_forward=true)
    walk_forward_result: Optional[WalkForwardResponse] = None


# ============================================================================
# Batch Optimize Schemas
# ============================================================================

class BatchOptimizeRequest(BaseModel):
    """Request schema for batch parameter optimization."""
    strategies: List[str] = Field(..., description="List of strategy file paths", min_length=1)
    run_id: Optional[str] = Field(default=None, description="Client-side run identifier")

    # Optimization settings
    metric: str = Field(default="sharpe_ratio", description="Metric to optimize")
    minimize: bool = Field(default=False, description="Minimize metric instead of maximize")
    rebalance_frequencies: List[str] = Field(
        default=["monthly", "quarterly"],
        description="List of rebalance frequencies to test"
    )

    # Backtest settings
    start_date: str = Field(default="2010-01-01", description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, description="Initial capital")
    costs_pct: float = Field(default=0.001, description="Transaction costs as decimal")

    # Tax settings
    tax_enabled: bool = Field(default=True, description="Enable German tax model")
    tax_exemption: float = Field(default=1000.0, description="Tax exemption amount (EUR)")
    metric_basis: str = Field(
        default="net_liquidation",
        description="Metric basis: gross, net_realized, or net_liquidation"
    )

    # Data settings
    drip_enabled: bool = Field(default=False, description="Enable dividend reinvestment")
    skip_failed: bool = Field(default=True, description="Skip failed ticker downloads")
    enable_validation: bool = Field(default=True, description="Enable validation checks")

    # Execution realism settings
    execution_lag_days: int = Field(default=0, description="Execution lag in trading days")
    max_volume_participation: Optional[float] = Field(default=None, description="Max share of daily volume per trade")
    min_daily_dollar_volume: float = Field(default=0.0, description="Skip trades below this daily notional volume")
    liquidity_on_missing_volume: str = Field(default="allow", description="allow or skip trades when volume data is missing")

    # Risk overlays
    max_position: Optional[float] = Field(default=None, description="Max target position weight per ticker")
    sector_caps: Dict[str, float] = Field(default_factory=dict, description="Sector cap mapping (sector -> max weight)")
    ticker_sectors: Dict[str, str] = Field(default_factory=dict, description="Ticker to sector mapping for sector caps")
    turnover_budget: Optional[float] = Field(default=None, description="Max one-way turnover per rebalance")
    drawdown_brake_threshold: Optional[float] = Field(default=None, description="Drawdown threshold to activate drawdown brake")
    drawdown_brake_cash_target: float = Field(default=1.0, description="Cash target while drawdown brake is active")
    drawdown_brake_release: Optional[float] = Field(default=None, description="Optional release drawdown level for brake hysteresis")

    # Walk-Forward settings
    walk_forward: bool = Field(default=False, description="Enable walk-forward analysis")
    walk_forward_nested: bool = Field(default=False, description="Enable nested walk-forward analysis")
    wf_train_years: float = Field(default=5.0, description="Walk-forward training window in years")
    wf_test_years: float = Field(default=1.0, description="Walk-forward test window in years")
    wf_step_months: int = Field(default=12, description="Walk-forward step size in months")
    wf_anchored: bool = Field(default=False, description="Use anchored (vs rolling) training windows")
    wf_inner_train_years: float = Field(default=3.0, description="Nested mode: inner training window in years")
    wf_inner_test_years: float = Field(default=1.0, description="Nested mode: inner test window in years")
    wf_inner_step_months: int = Field(default=6, description="Nested mode: inner step size in months")
    wf_inner_anchored: bool = Field(default=False, description="Nested mode: use anchored inner windows")


class WalkForwardSummaryResponse(BaseModel):
    """Walk-forward analysis summary for a single strategy."""
    num_windows: int = Field(..., description="Number of walk-forward windows")
    avg_is_sharpe: float = Field(..., description="Average in-sample Sharpe ratio")
    avg_oos_sharpe: float = Field(..., description="Average out-of-sample Sharpe ratio")
    degradation_ratio: float = Field(..., description="Performance degradation ratio")
    overfitting_score: float = Field(..., description="Overfitting score (0-1)")
    best_oos_sharpe: float = Field(..., description="Best out-of-sample Sharpe ratio")
    mode: str = Field(default="standard", description="standard|nested")
    nested: bool = Field(default=False, description="Whether nested walk-forward was used")
    inner_train_years: Optional[float] = Field(default=None)
    inner_test_years: Optional[float] = Field(default=None)
    inner_step_months: Optional[int] = Field(default=None)
    inner_anchored: Optional[bool] = Field(default=None)


class BatchOptimizeResultRow(BaseModel):
    """Single strategy optimization result in batch."""
    rank: int
    strategy_name: str
    strategy_file: str
    strategy_class: str
    best_params: Dict[str, Any]
    best_rebalance_frequency: str

    # Metrics
    metric_value: Optional[float] = None  # The optimized metric
    cagr: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    calmar_ratio: Optional[float] = None

    # Number of parameter combinations tested
    total_combinations: int = 0

    # Walk-Forward results (optional)
    walk_forward: Optional[WalkForwardSummaryResponse] = None

    # Error if any
    error: Optional[str] = None


class BatchOptimizeResponse(BaseModel):
    """Response schema for batch optimization results."""
    metric: str
    minimize: bool
    start_date: str
    end_date: str
    metric_basis: str
    tax_enabled: bool

    # Walk-Forward mode
    walk_forward_enabled: bool = False
    walk_forward_nested: Optional[bool] = None
    wf_train_years: Optional[float] = None
    wf_test_years: Optional[float] = None
    wf_step_months: Optional[int] = None
    wf_anchored: Optional[bool] = None
    wf_inner_train_years: Optional[float] = None
    wf_inner_test_years: Optional[float] = None
    wf_inner_step_months: Optional[int] = None
    wf_inner_anchored: Optional[bool] = None

    # Counts
    total_strategies: int
    successful_strategies: int
    failed_strategies: int

    # Results (sorted by metric)
    results: List[BatchOptimizeResultRow]

    # Export data: optimized params for all strategies
    export_params: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optimized parameters for all strategies, keyed by strategy class name"
    )
