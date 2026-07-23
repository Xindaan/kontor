"""Reusable risk overlays applied on top of strategy target allocations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from backtest.assets import resolve_asset
from backtest.strategy import Allocation


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Best-effort float conversion with optional fallback."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class DrawdownBrakeConfig:
    """Configuration for drawdown brake overlay."""

    threshold: float = 0.20
    cash_target: float = 1.0
    release_drawdown: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DrawdownBrakeConfig":
        threshold = _to_float(data.get("threshold"), 0.20)
        cash_target = _to_float(data.get("cash_target"), 1.0)
        release_drawdown = _to_float(data.get("release_drawdown"), None)
        return cls(
            threshold=0.20 if threshold is None else float(threshold),
            cash_target=1.0 if cash_target is None else float(cash_target),
            release_drawdown=release_drawdown,
        )

    def normalized(self) -> "DrawdownBrakeConfig":
        """Return a clamped copy suitable for runtime application."""
        threshold = min(max(float(self.threshold), 0.0), 1.0)
        cash_target = min(max(float(self.cash_target), 0.0), 1.0)
        release_drawdown = self.release_drawdown
        if release_drawdown is not None:
            release_drawdown = min(max(float(release_drawdown), 0.0), 1.0)
        return DrawdownBrakeConfig(
            threshold=threshold,
            cash_target=cash_target,
            release_drawdown=release_drawdown,
        )


@dataclass
class RiskOverlayConfig:
    """Configuration for reusable risk overlays."""

    max_position: Optional[float] = None
    sector_caps: Dict[str, float] = field(default_factory=dict)
    ticker_sectors: Dict[str, str] = field(default_factory=dict)
    turnover_budget: Optional[float] = None
    drawdown_brake: Optional[DrawdownBrakeConfig] = None

    @classmethod
    def from_raw(cls, raw: Optional[Mapping[str, Any]]) -> "RiskOverlayConfig":
        if raw is None:
            return cls()
        max_position = _to_float(raw.get("max_position"), None)
        turnover_budget = _to_float(raw.get("turnover_budget"), None)

        raw_sector_caps = raw.get("sector_caps") or {}
        sector_caps: Dict[str, float] = {}
        if isinstance(raw_sector_caps, Mapping):
            for key, value in raw_sector_caps.items():
                cap = _to_float(value, None)
                if cap is None:
                    continue
                sector_caps[str(key).strip().lower()] = float(cap)

        raw_ticker_sectors = raw.get("ticker_sectors") or {}
        ticker_sectors: Dict[str, str] = {}
        if isinstance(raw_ticker_sectors, Mapping):
            for ticker, sector in raw_ticker_sectors.items():
                ticker_sectors[str(ticker)] = str(sector).strip().lower()

        drawdown_brake = None
        raw_drawdown = raw.get("drawdown_brake")
        if isinstance(raw_drawdown, Mapping):
            drawdown_brake = DrawdownBrakeConfig.from_dict(raw_drawdown)

        return cls(
            max_position=max_position,
            sector_caps=sector_caps,
            ticker_sectors=ticker_sectors,
            turnover_budget=turnover_budget,
            drawdown_brake=drawdown_brake,
        )

    @property
    def enabled(self) -> bool:
        """True when at least one overlay is configured."""
        return any(
            [
                self.max_position is not None,
                bool(self.sector_caps),
                self.turnover_budget is not None,
                self.drawdown_brake is not None,
            ]
        )


class RiskOverlayEngine:
    """Applies deterministic, composable risk overlays to allocations."""

    def __init__(self, config: Optional[RiskOverlayConfig] = None):
        self.config = config or RiskOverlayConfig()
        self._peak_equity: Optional[float] = None
        self._drawdown_brake_active: bool = False
        self._stats: Dict[str, float] = {
            "apply_calls": 0.0,
            "changed_calls": 0.0,
            "max_position_bindings": 0.0,
            "sector_cap_bindings": 0.0,
            "turnover_budget_bindings": 0.0,
            "drawdown_brake_active_calls": 0.0,
            "drawdown_brake_activations": 0.0,
            "drawdown_brake_releases": 0.0,
            "drawdown_brake_scaled_calls": 0.0,
        }

    @classmethod
    def from_raw(cls, raw: Optional[Mapping[str, Any]]) -> "RiskOverlayEngine":
        return cls(RiskOverlayConfig.from_raw(raw))

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def apply(
        self,
        target: Allocation,
        current_weights: Optional[Mapping[str, float]] = None,
        current_equity: Optional[float] = None,
    ) -> Allocation:
        """
        Apply configured overlays in deterministic order:
        max position -> sector caps -> turnover budget -> drawdown brake.
        """
        if not self.enabled:
            return target

        self._stats["apply_calls"] += 1.0
        weights = self._sanitize_weights(target.weights)
        initial_weights = dict(weights)

        if self.config.max_position is not None:
            max_w = min(max(float(self.config.max_position), 0.0), 1.0)
            bound_count = sum(1 for weight in weights.values() if weight > max_w + 1e-12)
            if bound_count > 0:
                self._stats["max_position_bindings"] += float(bound_count)
            weights = self._apply_max_position(weights, max_w)

        if self.config.sector_caps:
            caps = {
                str(k).strip().lower(): min(max(float(v), 0.0), 1.0)
                for k, v in self.config.sector_caps.items()
            }
            sector_totals_before: Dict[str, float] = {}
            for ticker, weight in weights.items():
                sector = self._resolve_sector(ticker)
                sector_totals_before[sector] = sector_totals_before.get(sector, 0.0) + weight
            bound_sectors = sum(
                1 for sector, total_weight in sector_totals_before.items()
                if sector in caps and total_weight > caps[sector] + 1e-12
            )
            if bound_sectors > 0:
                self._stats["sector_cap_bindings"] += float(bound_sectors)
            weights = self._apply_sector_caps(weights)

        if self.config.turnover_budget is not None:
            budget = min(max(float(self.config.turnover_budget), 0.0), 1.0)
            target_sanitized = self._sanitize_weights(weights)
            current_sanitized = self._sanitize_weights(current_weights or {})
            all_tickers = set(target_sanitized) | set(current_sanitized)
            if all_tickers:
                one_way_turnover = 0.5 * sum(
                    abs(target_sanitized.get(t, 0.0) - current_sanitized.get(t, 0.0))
                    for t in all_tickers
                )
                if one_way_turnover > budget + 1e-12:
                    self._stats["turnover_budget_bindings"] += 1.0
            weights = self._apply_turnover_budget(
                target_weights=weights,
                current_weights=current_weights or {},
                turnover_budget=budget,
            )

        if self.config.drawdown_brake is not None:
            active_before = self._drawdown_brake_active
            risky_before = sum(weights.values())
            weights = self._apply_drawdown_brake(
                weights=weights,
                current_equity=current_equity,
                config=self.config.drawdown_brake.normalized(),
            )
            active_after = self._drawdown_brake_active
            risky_after = sum(weights.values())
            if active_after:
                self._stats["drawdown_brake_active_calls"] += 1.0
            if not active_before and active_after:
                self._stats["drawdown_brake_activations"] += 1.0
            if active_before and not active_after:
                self._stats["drawdown_brake_releases"] += 1.0
            if active_after and risky_after < risky_before - 1e-12:
                self._stats["drawdown_brake_scaled_calls"] += 1.0

        if self._weights_different(initial_weights, weights):
            self._stats["changed_calls"] += 1.0

        return Allocation(weights)

    @staticmethod
    def _weights_different(
        left: Mapping[str, float],
        right: Mapping[str, float],
        tol: float = 1e-10,
    ) -> bool:
        all_tickers = set(left) | set(right)
        for ticker in all_tickers:
            if abs(float(left.get(ticker, 0.0)) - float(right.get(ticker, 0.0))) > tol:
                return True
        return False

    def diagnostics(self) -> Dict[str, float]:
        """Return overlay diagnostics suitable for reporting and APIs."""
        result = dict(self._stats)
        result["drawdown_brake_currently_active"] = 1.0 if self._drawdown_brake_active else 0.0
        result["peak_equity"] = float(self._peak_equity) if self._peak_equity is not None else 0.0
        return result

    @staticmethod
    def _sanitize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
        """Drop invalid weights, clip negatives, and ensure sum <= 1."""
        cleaned: Dict[str, float] = {}
        for ticker, weight in weights.items():
            w = _to_float(weight, None)
            if w is None or not np.isfinite(w):
                continue
            if w <= 0:
                continue
            cleaned[str(ticker)] = float(w)
        total = sum(cleaned.values())
        if total > 1.0 and total > 0.0:
            scale = 1.0 / total
            cleaned = {ticker: weight * scale for ticker, weight in cleaned.items()}
        return cleaned

    def _apply_max_position(self, weights: Mapping[str, float], max_position: float) -> Dict[str, float]:
        max_w = min(max(max_position, 0.0), 1.0)
        return {ticker: min(weight, max_w) for ticker, weight in weights.items() if min(weight, max_w) > 0}

    def _apply_sector_caps(self, weights: Mapping[str, float]) -> Dict[str, float]:
        if not weights:
            return {}

        caps = {str(k).strip().lower(): min(max(float(v), 0.0), 1.0) for k, v in self.config.sector_caps.items()}
        if not caps:
            return dict(weights)

        sector_totals: Dict[str, float] = {}
        ticker_sector: Dict[str, str] = {}
        for ticker, weight in weights.items():
            sector = self._resolve_sector(ticker)
            ticker_sector[ticker] = sector
            sector_totals[sector] = sector_totals.get(sector, 0.0) + weight

        adjusted = dict(weights)
        for sector, total_weight in sector_totals.items():
            cap = caps.get(sector)
            if cap is None or total_weight <= cap or total_weight <= 0:
                continue
            scale = cap / total_weight
            for ticker, ticker_weight in list(adjusted.items()):
                if ticker_sector.get(ticker) == sector:
                    adjusted[ticker] = ticker_weight * scale

        return self._sanitize_weights(adjusted)

    def _resolve_sector(self, ticker: str) -> str:
        mapped = self.config.ticker_sectors.get(ticker)
        if mapped:
            return mapped.strip().lower()
        try:
            asset = resolve_asset(ticker)
            return str(asset.asset_class).strip().lower()
        except Exception:
            return "unknown"

    def _apply_turnover_budget(
        self,
        target_weights: Mapping[str, float],
        current_weights: Mapping[str, float],
        turnover_budget: float,
    ) -> Dict[str, float]:
        budget = min(max(turnover_budget, 0.0), 1.0)
        if budget <= 0.0:
            return self._sanitize_weights(current_weights)

        target = self._sanitize_weights(target_weights)
        current = self._sanitize_weights(current_weights)
        all_tickers = set(target) | set(current)
        if not all_tickers:
            return {}

        one_way_turnover = 0.5 * sum(abs(target.get(t, 0.0) - current.get(t, 0.0)) for t in all_tickers)
        if one_way_turnover <= budget or one_way_turnover <= 1e-12:
            return target

        scale = budget / one_way_turnover
        blended = {
            ticker: current.get(ticker, 0.0) + (target.get(ticker, 0.0) - current.get(ticker, 0.0)) * scale
            for ticker in all_tickers
        }
        return self._sanitize_weights(blended)

    def _apply_drawdown_brake(
        self,
        weights: Mapping[str, float],
        current_equity: Optional[float],
        config: DrawdownBrakeConfig,
    ) -> Dict[str, float]:
        if current_equity is None:
            return dict(weights)

        equity = float(current_equity)
        if not np.isfinite(equity):
            return dict(weights)

        if self._peak_equity is None:
            self._peak_equity = equity
        else:
            self._peak_equity = max(self._peak_equity, equity)

        peak = self._peak_equity if self._peak_equity and self._peak_equity > 0 else 0.0
        if peak <= 0.0:
            return dict(weights)

        drawdown = max(0.0, (peak - equity) / peak)

        if self._drawdown_brake_active and config.release_drawdown is not None:
            if drawdown <= config.release_drawdown:
                self._drawdown_brake_active = False

        if drawdown >= config.threshold:
            self._drawdown_brake_active = True

        if not self._drawdown_brake_active:
            return dict(weights)

        risk_budget = 1.0 - config.cash_target
        if risk_budget <= 0.0:
            return {}

        damped = {ticker: weight * risk_budget for ticker, weight in weights.items()}
        return self._sanitize_weights(damped)
