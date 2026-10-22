"""Meta-promotion audit reports for strategy governance.

This module does not decide live orders. It creates reproducible artifacts for
quarterly strategy-promotion reviews: after-tax performance, rolling windows,
trade/cost load, and broker execution mapping status.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import pandas as pd

from backtest.backtester import BacktestConfig, BacktestResult, Backtester
from backtest.data import DataLoader, PriceData
from backtest.live.instrument_mapping import load_mapping
from backtest.metrics import MetricsCalculator
from backtest.strategy import Strategy
from strategies.levered_etf_momentum_sticky import LeveredETFMomentumSticky


DEFAULT_STRATEGIES: tuple[str, ...] = (
    "strategies/levered_etf_momentum_sticky.py",
    "strategies/sticky_levered_vol_targeted.py",
    "strategies/sticky_levered_cascade.py",
)

SOXL_PROXY_BASELINE = "builtin:sticky_levered_soxl_proxy"

SOXL_PROXY_STRATEGIES: tuple[str, ...] = (
    SOXL_PROXY_BASELINE,
    "strategies/sticky_levered_vol_targeted.py",
    "strategies/sticky_levered_cascade.py",
)

DEFAULT_BROKERS: tuple[str, ...] = ("trade_republic", "maxblue")

BROKER_PROXY_MAPS: dict[str, dict[str, str]] = {
    # Maxblue blocks the WisdomTree PHLX Semi 3x product; live execution uses
    # VanEck Semiconductor UCITS as the documented broker proxy.
    "maxblue": {"3SEM.L": "VVSM.DE"},
}

RESEARCH_PROXY_MAPS: dict[str, dict[str, str]] = {
    # research asset -> live asset. Used exclusively by
    # broker_mapping_status(): "if the strategy uses SOXL in
    # research, then 3SEM.L is the live equivalent (which may be
    # remapped further via the Maxblue proxy to VVSM.DE)".
    "soxl_proxy": {"SOXL": "3SEM.L"},
}

# Backtest substitutes: live asset -> research proxy with long history.
# Strictly separate from RESEARCH_PROXY_MAPS, because not every research
# proxy is also a sensible live substitute (e.g. QQQ is directly
# tradeable via L&S at TR, so it should NOT be flagged as a "proxy" in
# the broker audit, even though it is swapped for EXXT.DE in the backtest).
#
# Convention, two classes:
#   1) 1:1 substitute: same underlying index, same leverage,
#      only a different wrapper/venue. Backtest results directly comparable.
#   2) Approximate substitute: factor or region mismatch, documented.
#      Read backtest results as a proxy, not 1:1.
BACKTEST_SUBSTITUTE_MAPS: dict[str, dict[str, str]] = {
    "soxl_proxy": {
        # === 1:1 substitutes ===
        "3SEM.L":  "SOXL",   # WisdomTree 3x Semi (2024) -> Direxion 3x Semi (2010)
        "VVSM.DE": "SOXX",   # VanEck 1x Semi UCITS (2020) -> iShares 1x Semi (2001)
        "EXXT.DE": "QQQ",    # iShares Nasdaq-100 UCITS (2014) -> Invesco QQQ (1999)
        # === Approximate substitutes (factor/region mismatch) ===
        "QDVF.DE": "IVE",    # MSCI World Value Factor (2014) -> S&P 500 Value (2000)
        "QDVH.DE": "MTUM",   # MSCI World Momentum (2014) -> MSCI USA Momentum (2013)
        # === Deliberately NOT substituted ===
        # 3XFE.L (FTSE MIB 3x), 3XEE.L (FTSE 100 3x): no 3x EU proxy with
        # long history available. Strategies with these tickers still
        # have data gaps for these assets in soxl_proxy mode.
    },
}


def _backtest_substitute_map(mode: str) -> dict[str, str]:
    """Returns the live-asset -> research-proxy map for the backtest.

    Called by the strategy remap: `3SEM.L` in `strategy.candidates`
    is swapped for `SOXL` (in soxl_proxy mode), so that the strategy
    runs on the long history instead of the short live history.
    """
    return dict(BACKTEST_SUBSTITUTE_MAPS.get(mode, {}))


def _remap_strategy_candidates(strategy: Strategy, inverse_map: dict[str, str]) -> None:
    """Rewrites live-asset references in strategy.candidates/assets to
    research-proxy tickers. Uses simple list substitution; unsuitable
    for strategies whose asset universe does not flow through `candidates`.
    """
    if not inverse_map:
        return

    def swap(items):
        return [inverse_map.get(item, item) for item in items]

    if hasattr(strategy, "candidates"):
        strategy.candidates = swap(strategy.candidates)
    if hasattr(strategy, "assets"):
        # Preserve order, drop duplicates (e.g. if SOXL was already present)
        strategy.assets = list(dict.fromkeys(swap(strategy.assets)))


@dataclass(frozen=True)
class RollingWindow:
    start: str
    end: str
    cagr: float
    max_drawdown: float
    sharpe: float
    sortino: float
    calmar: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
        }


def load_strategy_from_path(path: str | Path) -> Strategy:
    """Load a Strategy instance without importing CLI to avoid circular imports."""
    import importlib.util
    import sys

    if str(path) == SOXL_PROXY_BASELINE:
        class StickyLeveredSOXLProxy(LeveredETFMomentumSticky):
            name = "[Baseline] Sticky/Core Levered (Backtest, SOXL-Proxy)"

            def __init__(self) -> None:
                super().__init__(candidates=["TQQQ", "UPRO", "SOXL"])

        return StickyLeveredSOXLProxy()

    strategy_path = Path(path)
    if not strategy_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")

    module_name = f"meta_promotion_{strategy_path.stem}_{hashlib.sha1(str(strategy_path).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if hasattr(module, "strategy") and isinstance(module.strategy, Strategy):
        return module.strategy

    strategy_class = None
    for obj in vars(module).values():
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
            strategy_class = obj
    if strategy_class is None:
        raise ValueError(f"No Strategy subclass found in {strategy_path}")
    return strategy_class()


TailRiskGateBasis = Literal["daily", "rebalance"]


def metric_curve(result: BacktestResult) -> pd.Series:
    """Return the curve matching the headline metric basis."""
    if result.tax_summary and result.tax_summary.tax_enabled:
        if result.tax_summary.metric_basis == "net_liquidation":
            return result.equity_curve
        if result.tax_summary.metric_basis == "net_realized":
            return result.equity_curve_net if result.equity_curve_net is not None else result.equity_curve
    return result.equity_curve


def daily_metric_curve(result: BacktestResult) -> pd.Series:
    """Return the daily MTM curve matching the headline metric basis."""
    daily_curve = result.equity_curve_daily
    if daily_curve is not None and len(daily_curve) > 0:
        return daily_curve
    return metric_curve(result)


def tail_risk_curve(result: BacktestResult, basis: TailRiskGateBasis) -> pd.Series:
    """Return the curve used for tail-risk gate calculations."""
    return daily_metric_curve(result) if basis == "daily" else metric_curve(result)


def _window_metrics(curve: pd.Series) -> RollingWindow:
    clean = curve.dropna()
    returns = MetricsCalculator.monthly_returns(clean)
    cagr = MetricsCalculator.cagr(clean)
    max_dd = MetricsCalculator.max_drawdown(clean)
    sharpe = MetricsCalculator.sharpe_ratio(returns) if len(returns) else 0.0
    sortino = MetricsCalculator.sortino_ratio(returns) if len(returns) else 0.0
    calmar = MetricsCalculator.calmar_ratio(cagr, max_dd)
    return RollingWindow(
        start=clean.index[0].strftime("%Y-%m-%d"),
        end=clean.index[-1].strftime("%Y-%m-%d"),
        cagr=float(cagr),
        max_drawdown=float(max_dd),
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(calmar),
    )


def rolling_windows(curve: pd.Series, years: int, *, step_months: int = 1) -> List[RollingWindow]:
    """Calculate rolling calendar-year windows.

    Backtest equity curves are stored at rebalance points. A monthly strategy
    has roughly 12 rows/year, a weekly strategy roughly 52 rows/year; calendar
    slicing keeps those mixed frequencies comparable.
    """
    clean = curve.dropna()
    if clean.empty:
        return []
    start = clean.index[0]
    end = clean.index[-1]
    if start + pd.DateOffset(years=years) > end:
        return []
    out: List[RollingWindow] = []
    current_start = start
    while current_start + pd.DateOffset(years=years) <= end:
        current_end = current_start + pd.DateOffset(years=years)
        window = clean[(clean.index >= current_start) & (clean.index <= current_end)]
        if len(window) >= 2:
            out.append(_window_metrics(window))
        current_start = current_start + pd.DateOffset(months=step_months)

    final_start = end - pd.DateOffset(years=years)
    if out and out[-1].end != end.strftime("%Y-%m-%d"):
        final_window = clean[(clean.index >= final_start) & (clean.index <= end)]
        if len(final_window) >= 2:
            out.append(_window_metrics(final_window))
    return out


def compare_rolling(candidate: List[RollingWindow], baseline: List[RollingWindow]) -> Dict[str, Any]:
    """Compare candidate windows to baseline windows by aligned order."""
    n = min(len(candidate), len(baseline))
    if n == 0:
        return {
            "windows": 0,
            "cagr_win_rate": None,
            "maxdd_win_rate": None,
            "worst_maxdd_delta_pp": None,
        }
    c = candidate[:n]
    b = baseline[:n]
    cagr_wins = sum(1 for cw, bw in zip(c, b) if cw.cagr > bw.cagr)
    dd_wins = sum(1 for cw, bw in zip(c, b) if cw.max_drawdown > bw.max_drawdown)
    worst_delta = min((cw.max_drawdown - bw.max_drawdown) * 100.0 for cw, bw in zip(c, b))
    return {
        "windows": n,
        "cagr_win_rate": cagr_wins / n,
        "maxdd_win_rate": dd_wins / n,
        "worst_maxdd_delta_pp": float(worst_delta),
    }


def _metric_value(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


def _tail_risk_summary(result: BacktestResult) -> Dict[str, Any]:
    daily_metrics = result.metrics_daily
    daily_curve = daily_metric_curve(result)
    return {
        "rebalance": {
            "max_drawdown": float(result.metrics.max_drawdown),
            "max_drawdown_duration": int(result.metrics.max_drawdown_duration),
            "sample_count": int(len(metric_curve(result))),
        },
        "daily": {
            "max_drawdown": _metric_value(daily_metrics.max_drawdown if daily_metrics else None),
            "max_drawdown_duration": (
                int(daily_metrics.max_drawdown_duration) if daily_metrics else None
            ),
            "underwater_days": (
                int(result.metrics.underwater_days_daily)
                if result.metrics.underwater_days_daily is not None
                else MetricsCalculator.underwater_days(daily_curve)
            ),
            "sample_count": int(len(daily_curve)),
        },
    }


def _max_drawdown_for_gate(metrics: Dict[str, Any], basis: TailRiskGateBasis) -> float:
    tail = metrics.get("tail_risk", {})
    basis_metrics = tail.get(basis, {})
    value = basis_metrics.get("max_drawdown")
    if value is None:
        return float(metrics["max_drawdown"])
    return float(value)


def summarize_result(result: BacktestResult) -> Dict[str, Any]:
    m = result.metrics
    t = result.tax_summary
    tail_risk = _tail_risk_summary(result)
    return {
        "strategy_name": result.strategy.name,
        "rebalance_frequency": result.config.rebalance_frequency,
        "metric_basis": result.headline_metric_basis,
        "final_value": float(result.equity_curve.iloc[-1]),
        "cagr": float(m.cagr),
        "max_drawdown": float(m.max_drawdown),
        "max_drawdown_rebalance": float(tail_risk["rebalance"]["max_drawdown"]),
        "max_drawdown_daily": _metric_value(tail_risk["daily"]["max_drawdown"]),
        "sharpe": float(m.sharpe_ratio),
        "sortino": float(m.sortino_ratio),
        "calmar": float(m.calmar_ratio),
        "trades": int(m.num_trades),
        "annual_turnover": float(m.turnover_annual),
        "total_costs": float(m.total_costs),
        "tax_total": float(t.total_tax_including_liquidation) if t else 0.0,
        "tax_drag_cagr_pp": float(t.tax_drag_for_metric_basis_pp) if t else 0.0,
        "tail_risk": tail_risk,
    }


def _normalize_pathish(value: str | Path) -> str:
    """Keep builtin strategy IDs stable while normalizing real file paths."""
    text = str(value)
    return text if text.startswith("builtin:") else str(Path(text))


def broker_mapping_status(
    strategy: Strategy,
    brokers: Iterable[str],
    *,
    research_proxy_mode: str = "live",
) -> Dict[str, Any]:
    """Audit if each strategy asset has a direct or explicit broker proxy mapping."""
    assets = [str(a).upper() for a in getattr(strategy, "assets", [])]
    proxy_map = {
        str(k).upper(): str(v).upper()
        for k, v in getattr(strategy, "execution_proxy_map", {}).items()
    }
    research_proxy_map = {
        str(k).upper(): str(v).upper()
        for k, v in RESEARCH_PROXY_MAPS.get(research_proxy_mode, {}).items()
    }
    out: Dict[str, Any] = {}
    for broker in brokers:
        mapping = load_mapping(broker)
        broker_proxy_map = {
            str(k).upper(): str(v).upper()
            for k, v in BROKER_PROXY_MAPS.get(str(broker), {}).items()
        }
        broker_proxy_map.update(proxy_map)
        rows = []
        ok = True
        for asset in assets:
            live_signal_asset = research_proxy_map.get(asset, asset)
            mapped_asset = broker_proxy_map.get(live_signal_asset, live_signal_asset)
            available = mapped_asset in mapping and bool(mapping[mapped_asset].isin)
            direct = asset == live_signal_asset == mapped_asset and available
            proxied = not direct and available
            if not available:
                ok = False
            rows.append(
                {
                    "signal_asset": asset,
                    "live_signal_asset": live_signal_asset,
                    "execution_asset": mapped_asset,
                    "status": "direct" if direct else "proxy" if proxied else "missing",
                }
            )
        out[broker] = {"ok": ok, "assets": rows}
    return out


def artifact_id(payload: Dict[str, Any]) -> str:
    body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(body).hexdigest()[:16]


def write_markdown_report(payload: Dict[str, Any], path: Path) -> None:
    gate_basis = payload["config"].get("tail_risk_gate_basis", "daily")
    lines = [
        "# Strategy Meta-Promotion Report",
        "",
        f"Generated: {payload['generated_at']}",
        f"Artifact ID: `{payload['artifact_id']}`",
        f"Date range: `{payload['config']['start']}` to `{payload['config']['end']}`",
        f"Metric basis: `{payload['config']['metric_basis']}`",
        f"Tail-risk gate basis: `{gate_basis}`",
        f"Research proxy mode: `{payload['config']['research_proxy_mode']}`",
        "",
        "> Hinweis: `soxl_proxy` ist der promotion-faehige Langhistorienmodus fuer Semi-/`3SEM.L`-Research. "
        "`live` nutzt echte Live-Ticker und ist bei kurzer `3SEM.L`-Historie nur ein Smoke-/Mapping-Check.",
        "",
        "> Tail-Risiko-Gates folgen Meta-Playbook v1.7: Daily-MTM ist Gate, Rebalance bleibt historische Referenz.",
        "",
        "## Summary",
        "",
        "| Strategy | Role | Frequency | Net CAGR | MaxDD rebalance | MaxDD daily | Gate MaxDD | Sharpe | Trades | Tax incl. liq |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["strategies"]:
        metrics = row["metrics"]
        gate_maxdd = _max_drawdown_for_gate(metrics, gate_basis)
        daily_maxdd = metrics.get("max_drawdown_daily")
        lines.append(
            "| {name} | {role} | {freq} | {cagr:.1%} | {dd_reb:.1%} | {dd_daily} | {dd_gate:.1%} | {sharpe:.2f} | {trades} | EUR {tax:,.0f} |".format(
                name=metrics["strategy_name"],
                role=row["role"],
                freq=metrics["rebalance_frequency"],
                cagr=metrics["cagr"],
                dd_reb=metrics["max_drawdown"],
                dd_daily="n/a" if daily_maxdd is None else f"{daily_maxdd:.1%}",
                dd_gate=gate_maxdd,
                sharpe=metrics["sharpe"],
                trades=metrics["trades"],
                tax=metrics["tax_total"],
            )
        )
    lines.extend(["", "## Promotion Diagnostics", ""])
    for row in payload["strategies"]:
        if row["role"] == "baseline":
            continue
        diag = row.get("vs_baseline", {})
        r3 = diag.get("rolling_3y", {})
        r5 = diag.get("rolling_5y", {})
        r3_rebalance = diag.get("rolling_3y_rebalance", {})
        r3_daily = diag.get("rolling_3y_daily", {})
        r5_rebalance = diag.get("rolling_5y_rebalance", {})
        r5_daily = diag.get("rolling_5y_daily", {})
        lines.extend(
            [
                f"### {row['metrics']['strategy_name']}",
                "",
                f"- Gate basis: `{diag.get('gate_basis', gate_basis)}`",
                f"- CAGR delta: `{diag.get('cagr_delta_pp', 0.0):+.2f}pp`",
                f"- MaxDD delta (gate): `{diag.get('maxdd_delta_pp', 0.0):+.2f}pp`",
                f"- MaxDD delta (rebalance ref): `{diag.get('maxdd_delta_rebalance_pp', 0.0):+.2f}pp`",
                f"- MaxDD delta (daily): `{diag.get('maxdd_delta_daily_pp', 0.0):+.2f}pp`",
                f"- Rolling-3Y CAGR win-rate: `{_fmt_pct_or_na(r3.get('cagr_win_rate'))}`",
                f"- Rolling-3Y MaxDD win-rate (gate): `{_fmt_pct_or_na(r3.get('maxdd_win_rate'))}`",
                f"- Rolling-3Y MaxDD win-rate (rebalance ref): `{_fmt_pct_or_na(r3_rebalance.get('maxdd_win_rate'))}`",
                f"- Rolling-3Y MaxDD win-rate (daily): `{_fmt_pct_or_na(r3_daily.get('maxdd_win_rate'))}`",
                f"- Worst Rolling-3Y MaxDD delta (gate): `{_fmt_pp_or_na(r3.get('worst_maxdd_delta_pp'))}`",
                f"- Worst Rolling-3Y MaxDD delta (rebalance ref): `{_fmt_pp_or_na(r3_rebalance.get('worst_maxdd_delta_pp'))}`",
                f"- Worst Rolling-3Y MaxDD delta (daily): `{_fmt_pp_or_na(r3_daily.get('worst_maxdd_delta_pp'))}`",
                f"- Rolling-5Y CAGR win-rate: `{_fmt_pct_or_na(r5.get('cagr_win_rate'))}`",
                f"- Rolling-5Y MaxDD win-rate (gate): `{_fmt_pct_or_na(r5.get('maxdd_win_rate'))}`",
                f"- Rolling-5Y MaxDD win-rate (rebalance ref): `{_fmt_pct_or_na(r5_rebalance.get('maxdd_win_rate'))}`",
                f"- Rolling-5Y MaxDD win-rate (daily): `{_fmt_pct_or_na(r5_daily.get('maxdd_win_rate'))}`",
                "",
            ]
        )
    lines.extend(["## Broker Mapping", ""])
    for row in payload["strategies"]:
        lines.append(f"### {row['metrics']['strategy_name']}")
        for broker, status in row["broker_mapping"].items():
            marker = "OK" if status["ok"] else "MISSING"
            missing = [a for a in status["assets"] if a["status"] == "missing"]
            suffix = "" if not missing else ": " + ", ".join(a["signal_asset"] for a in missing)
            lines.append(f"- `{broker}`: **{marker}**{suffix}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_pct_or_na(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _fmt_pp_or_na(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.2f}pp"


def run_meta_promotion_report(
    *,
    strategy_paths: Iterable[str | Path] = DEFAULT_STRATEGIES,
    baseline_path: str | Path = DEFAULT_STRATEGIES[0],
    start: str = "2016-01-01",
    end: Optional[str] = None,
    output_dir: str | Path = "results/meta_promotion",
    initial_capital: float = 10_000.0,
    costs_pct: float = 0.001,
    metric_basis: str = "net_liquidation",
    tax_enabled: bool = True,
    brokers: Iterable[str] = DEFAULT_BROKERS,
    skip_failed: bool = True,
    align: str = "ffill",
    validate: bool = True,
    research_proxy_mode: str = "live",
    equity_fund_map: Optional[Dict[str, bool]] = None,
    tail_risk_gate_basis: TailRiskGateBasis = "daily",
) -> Dict[str, Any]:
    if tail_risk_gate_basis not in {"daily", "rebalance"}:
        raise ValueError("tail_risk_gate_basis must be 'daily' or 'rebalance'")

    paths = [_normalize_pathish(p) for p in strategy_paths]
    baseline_path = _normalize_pathish(baseline_path)
    brokers_list = list(brokers)
    if baseline_path not in paths:
        paths.insert(0, baseline_path)

    strategies = [load_strategy_from_path(path) for path in paths]

    # Research-proxy auto-remap: in soxl_proxy mode we rewrite 3SEM.L -> SOXL
    # at the strategy level, so that live strategies (with a short 3SEM.L
    # history) run on the same long-history basis as the SOXL-proxy baseline.
    # Without this step, live 3SEM.L strategies would miss 14 years of
    # semiconductor edge (3SEM.L data only starts in 2024), which would
    # destroy comparability.
    substitute_map = _backtest_substitute_map(research_proxy_mode)
    if substitute_map:
        for strategy in strategies:
            _remap_strategy_candidates(strategy, substitute_map)

    all_assets = sorted({asset for strategy in strategies for asset in strategy.assets})
    data: PriceData = DataLoader.yahoo(
        tickers=all_assets,
        start=start,
        end=end,
        currency="EUR",
        align=align,
        skip_failed=skip_failed,
        load_dividends=tax_enabled,
        validate=validate,
    )

    results: list[BacktestResult] = []
    for strategy in strategies:
        config = BacktestConfig(
            initial_capital=initial_capital,
            costs_pct=costs_pct,
            rebalance_frequency=None,
            benchmark=None,
            tax_enabled=tax_enabled,
            metric_basis=metric_basis,  # type: ignore[arg-type]
            validate=validate,
            # Per-ticker Teilfreistellung: contains only individual-stock
            # tickers (->False). ETF strategy tickers are missing from the
            # map and retain the legacy 30% behavior via
            # _instrument_tax_flags -> safely applicable to all candidates.
            equity_fund_map=equity_fund_map,
        )
        results.append(Backtester(strategy, data, config).run())

    baseline_idx = paths.index(baseline_path)
    baseline_result = results[baseline_idx]
    baseline_curve_rebalance = metric_curve(baseline_result)
    baseline_curve_daily = daily_metric_curve(baseline_result)
    baseline_gate_curve = tail_risk_curve(baseline_result, tail_risk_gate_basis)
    baseline_3y_rebalance = rolling_windows(baseline_curve_rebalance, 3)
    baseline_5y_rebalance = rolling_windows(baseline_curve_rebalance, 5)
    baseline_3y_daily = rolling_windows(baseline_curve_daily, 3)
    baseline_5y_daily = rolling_windows(baseline_curve_daily, 5)
    baseline_3y_gate = rolling_windows(baseline_gate_curve, 3)
    baseline_5y_gate = rolling_windows(baseline_gate_curve, 5)
    baseline_metrics = summarize_result(baseline_result)

    rows = []
    for path, result in zip(paths, results):
        curve_rebalance = metric_curve(result)
        curve_daily = daily_metric_curve(result)
        curve_gate = tail_risk_curve(result, tail_risk_gate_basis)
        metrics = summarize_result(result)
        role = "baseline" if path == baseline_path else "candidate"
        rolling_3y_rebalance = rolling_windows(curve_rebalance, 3)
        rolling_5y_rebalance = rolling_windows(curve_rebalance, 5)
        rolling_3y_daily = rolling_windows(curve_daily, 3)
        rolling_5y_daily = rolling_windows(curve_daily, 5)
        rolling_3y_gate = rolling_windows(curve_gate, 3)
        rolling_5y_gate = rolling_windows(curve_gate, 5)
        row: Dict[str, Any] = {
            "path": path,
            "role": role,
            "metrics": metrics,
            "tail_risk_gate_basis": tail_risk_gate_basis,
            "rolling_3y": [w.to_dict() for w in rolling_3y_gate],
            "rolling_5y": [w.to_dict() for w in rolling_5y_gate],
            "rolling_3y_rebalance": [w.to_dict() for w in rolling_3y_rebalance],
            "rolling_5y_rebalance": [w.to_dict() for w in rolling_5y_rebalance],
            "rolling_3y_daily": [w.to_dict() for w in rolling_3y_daily],
            "rolling_5y_daily": [w.to_dict() for w in rolling_5y_daily],
            "broker_mapping": broker_mapping_status(
                result.strategy,
                brokers_list,
                research_proxy_mode=research_proxy_mode,
            ),
        }
        if role != "baseline":
            maxdd_gate = _max_drawdown_for_gate(metrics, tail_risk_gate_basis)
            baseline_maxdd_gate = _max_drawdown_for_gate(baseline_metrics, tail_risk_gate_basis)
            maxdd_rebalance = _max_drawdown_for_gate(metrics, "rebalance")
            baseline_maxdd_rebalance = _max_drawdown_for_gate(baseline_metrics, "rebalance")
            maxdd_daily = _max_drawdown_for_gate(metrics, "daily")
            baseline_maxdd_daily = _max_drawdown_for_gate(baseline_metrics, "daily")
            row["vs_baseline"] = {
                "baseline_name": baseline_metrics["strategy_name"],
                "gate_basis": tail_risk_gate_basis,
                "cagr_delta_pp": float((metrics["cagr"] - baseline_metrics["cagr"]) * 100.0),
                "maxdd_delta_pp": float((maxdd_gate - baseline_maxdd_gate) * 100.0),
                "maxdd_delta_rebalance_pp": float((maxdd_rebalance - baseline_maxdd_rebalance) * 100.0),
                "maxdd_delta_daily_pp": float((maxdd_daily - baseline_maxdd_daily) * 100.0),
                "sharpe_delta": float(metrics["sharpe"] - baseline_metrics["sharpe"]),
                "rolling_3y": compare_rolling(rolling_3y_gate, baseline_3y_gate),
                "rolling_5y": compare_rolling(rolling_5y_gate, baseline_5y_gate),
                "rolling_3y_rebalance": compare_rolling(rolling_3y_rebalance, baseline_3y_rebalance),
                "rolling_5y_rebalance": compare_rolling(rolling_5y_rebalance, baseline_5y_rebalance),
                "rolling_3y_daily": compare_rolling(rolling_3y_daily, baseline_3y_daily),
                "rolling_5y_daily": compare_rolling(rolling_5y_daily, baseline_5y_daily),
            }
        rows.append(row)

    generated_at = datetime.now(timezone.utc).isoformat()
    config_payload = {
        "start": start,
        "end": end,
        "initial_capital": initial_capital,
        "costs_pct": costs_pct,
        "metric_basis": metric_basis,
        "tax_enabled": tax_enabled,
        "strategy_paths": paths,
        "baseline_path": baseline_path,
        "brokers": brokers_list,
        "research_proxy_mode": research_proxy_mode,
        "tail_risk_gate_basis": tail_risk_gate_basis,
    }
    payload: Dict[str, Any] = {
        "generated_at": generated_at,
        "config": config_payload,
        "strategies": rows,
    }
    payload["artifact_id"] = artifact_id(payload)

    root = Path(output_dir)
    artifact_dir = root / generated_at[:10].replace("-", "") / payload["artifact_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifact_dir / "promotion_report.json"
    md_path = artifact_dir / "promotion_report.md"
    payload["paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(payload, md_path)
    return payload


DEFAULT_PROMOTION_DIR = "results/meta_promotion"


def load_promotion_artifact(path: str | Path) -> Dict[str, Any]:
    """Load a persisted promotion_report.json file."""
    artifact_path = Path(path).expanduser().resolve()
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def _sort_key_generated_at(path: Path) -> tuple[str, float]:
    """Sort key: primarily generated_at from the payload, secondarily mtime."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated = str(payload.get("generated_at") or "")
    except Exception:
        generated = ""
    return (generated, path.stat().st_mtime)


def find_latest_promotion_artifact(
    base_dir: str | Path = DEFAULT_PROMOTION_DIR,
) -> Optional[Dict[str, Any]]:
    """Return latest promotion artifact payload, if present.

    Ordered primarily by generated_at from the payload, secondarily by mtime —
    this keeps the audit trail stable if the file is touched later.
    """
    base = Path(base_dir).expanduser().resolve()
    if not base.exists():
        return None
    files = list(base.rglob("promotion_report.json"))
    if not files:
        return None
    files.sort(key=_sort_key_generated_at, reverse=True)
    return load_promotion_artifact(files[0])


def find_promotion_artifact_by_id(
    artifact_id: str,
    base_dir: str | Path = DEFAULT_PROMOTION_DIR,
) -> Optional[Dict[str, Any]]:
    """Find a promotion artifact by id under the base promotion directory.

    Artifacts live under <base_dir>/<YYYYMMDD>/<artifact_id>/promotion_report.json.
    """
    base = Path(base_dir).expanduser().resolve()
    if not base.exists():
        return None
    candidate = next(base.rglob(f"{artifact_id}/promotion_report.json"), None)
    if candidate is None:
        return None
    return load_promotion_artifact(candidate)


def list_promotion_artifacts(
    base_dir: str | Path = DEFAULT_PROMOTION_DIR,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return metadata index for the artifact-browser (no full payload).

    Sortierung: neueste zuerst (per generated_at, Fallback mtime).
    """
    base = Path(base_dir).expanduser().resolve()
    if not base.exists():
        return []
    files = list(base.rglob("promotion_report.json"))
    files.sort(key=_sort_key_generated_at, reverse=True)
    rows: List[Dict[str, Any]] = []
    for path in files[: max(0, int(limit))]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        strategies = payload.get("strategies") or []
        candidate_count = sum(1 for row in strategies if row.get("role") == "candidate")
        rows.append(
            {
                "artifact_id": payload.get("artifact_id"),
                "generated_at": payload.get("generated_at"),
                "baseline_path": (payload.get("config") or {}).get("baseline_path"),
                "research_proxy_mode": (payload.get("config") or {}).get("research_proxy_mode"),
                "candidate_count": candidate_count,
                "strategy_count": len(strategies),
                "json_path": str(path),
            }
        )
    return rows


def read_promotion_markdown_by_id(
    artifact_id: str,
    base_dir: str | Path = DEFAULT_PROMOTION_DIR,
) -> Optional[str]:
    """Read the promotion_report.md text for a given artifact id, if present."""
    base = Path(base_dir).expanduser().resolve()
    if not base.exists():
        return None
    candidate = next(base.rglob(f"{artifact_id}/promotion_report.md"), None)
    if candidate is None:
        return None
    return candidate.read_text(encoding="utf-8")
