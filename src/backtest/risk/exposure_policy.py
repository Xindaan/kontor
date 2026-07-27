"""Optional exposure policy applied between strategy signal and execution.

The policy is deliberately separate from strategy classes: strategies keep
choosing the alpha asset, while this layer may de-lever execution when a
predefined 3x shock guard is active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.assets import get_yahoo_ticker
from backtest.strategy import Allocation


DEFAULT_PROXY_MAPS: Dict[str, Dict[str, str]] = {
    "trade_republic": {
        "A3GL7E": "A0F5UF",
        "A1VBKR": "A0YEDG",
        "A4ANZ5": "A2QC5J",
    },
    "tr": {
        "A3GL7E": "A0F5UF",
        "A1VBKR": "A0YEDG",
        "A4ANZ5": "A2QC5J",
    },
    "maxblue": {
        "A3GL7E": "A0F5UF",
        "A1VBKR": "A0YEDG",
        "A4ANZ5": "A2QC5J",
        "A3GWDS": "A142NY",
        "A3GWD0": "A142NX",
    },
    "us": {
        "TQQQ": "QQQ",
        "UPRO": "SPY",
        "SOXL": "SOXX",
        "TECL": "XLK",
        "FAS": "XLF",
        "DUSL": "XLI",
        "ERX": "XLE",
        "CURE": "XLV",
        "UYM": "XLB",
        "UTSL": "XLU",
        "DRN": "XLRE",
    },
}

DEFAULT_CORE_ASSETS: Dict[str, str] = {
    "trade_republic": "A0YEDG",
    "tr": "A0YEDG",
    "maxblue": "A0YEDG",
    "us": "SPY",
}


@dataclass(frozen=True)
class ExposurePolicyConfig:
    """Configuration for the optional exposure controller."""

    enabled: bool = False
    mode: str = "full_until_guard"
    profile: str = "trade_republic"
    proxy_map: Dict[str, str] = field(default_factory=dict)
    core_asset: Optional[str] = None
    level1_ret_5d_floor: float = -0.12
    level1_drawdown_21d_floor: float = -0.25
    level2_ret_21d_3x_floor: float = -0.25
    level2_proxy_ret_21d_floor: float = 0.0
    release_ret_5d_floor: float = -0.05
    release_proxy_ret_21d_floor: float = 0.0
    release_confirmation_periods: int = 2

    @classmethod
    def from_raw(cls, raw: Optional[Mapping[str, Any]]) -> "ExposurePolicyConfig":
        if not raw:
            return cls()
        enabled = bool(raw.get("enabled", False))
        profile = str(raw.get("profile", "trade_republic")).strip().lower()
        mode = str(raw.get("mode", "full_until_guard")).strip().lower()

        raw_map = dict(DEFAULT_PROXY_MAPS.get(profile, {}))
        if isinstance(raw.get("proxy_map"), Mapping):
            raw_map.update({str(k): str(v) for k, v in raw.get("proxy_map", {}).items()})
        proxy_map = {
            get_yahoo_ticker(str(k)): get_yahoo_ticker(str(v))
            for k, v in raw_map.items()
            if str(k).strip() and str(v).strip()
        }

        core_asset = raw.get("core_asset", raw.get("safe_asset", DEFAULT_CORE_ASSETS.get(profile)))
        resolved_core = get_yahoo_ticker(str(core_asset)) if core_asset else None

        return cls(
            enabled=enabled,
            mode=mode,
            profile=profile,
            proxy_map=proxy_map,
            core_asset=resolved_core,
            level1_ret_5d_floor=_float(raw.get("level1_ret_5d_floor"), -0.12),
            level1_drawdown_21d_floor=_float(raw.get("level1_drawdown_21d_floor"), -0.25),
            level2_ret_21d_3x_floor=_float(raw.get("level2_ret_21d_3x_floor"), -0.25),
            level2_proxy_ret_21d_floor=_float(raw.get("level2_proxy_ret_21d_floor"), 0.0),
            release_ret_5d_floor=_float(raw.get("release_ret_5d_floor"), -0.05),
            release_proxy_ret_21d_floor=_float(raw.get("release_proxy_ret_21d_floor"), 0.0),
            release_confirmation_periods=max(1, int(_float(raw.get("release_confirmation_periods"), 2))),
        )

    @property
    def required_assets(self) -> List[str]:
        if not self.enabled:
            return []
        assets = set(self.proxy_map.keys()) | set(self.proxy_map.values())
        if self.core_asset:
            assets.add(self.core_asset)
        return sorted(assets)


@dataclass(frozen=True)
class ExposurePolicyDecision:
    """Result of applying the policy to one allocation."""

    allocation: Allocation
    exposure_state: str
    raw_strategy_target: Dict[str, float]
    policy_adjusted_target: Dict[str, float]
    raw_signal_asset: Optional[str]
    execution_asset: Optional[str]
    fallback_reason: Optional[str]
    guard_checks: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "exposure_state": self.exposure_state,
            "raw_strategy_target": dict(self.raw_strategy_target),
            "policy_adjusted_target": dict(self.policy_adjusted_target),
            "raw_signal_asset": self.raw_signal_asset,
            "execution_asset": self.execution_asset,
            "fallback_reason": self.fallback_reason,
            "guard_checks": list(self.guard_checks),
        }


class ExposurePolicyEngine:
    """Stateful 3x exposure controller with causal shock/release guards."""

    def __init__(self, config: Optional[ExposurePolicyConfig] = None):
        self.config = config or ExposurePolicyConfig()
        self._active_state_by_asset: Dict[str, str] = {}
        self._release_streak_by_asset: Dict[str, int] = {}
        self._stats: Dict[str, float] = {
            "apply_calls": 0.0,
            "changed_calls": 0.0,
            "normal_calls": 0.0,
            "deleveraged_1x_calls": 0.0,
            "safe_calls": 0.0,
            "blocked_calls": 0.0,
            "guard_activations": 0.0,
            "release_confirmations": 0.0,
        }

    @classmethod
    def from_raw(cls, raw: Optional[Mapping[str, Any]]) -> "ExposurePolicyEngine":
        return cls(ExposurePolicyConfig.from_raw(raw))

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def apply(self, target: Allocation, historical_prices: pd.DataFrame) -> ExposurePolicyDecision:
        if not self.enabled:
            return ExposurePolicyDecision(
                allocation=target,
                exposure_state="disabled",
                raw_strategy_target=dict(target.weights),
                policy_adjusted_target=dict(target.weights),
                raw_signal_asset=_largest_weight_asset(target.weights),
                execution_asset=_largest_weight_asset(target.weights),
                fallback_reason=None,
                guard_checks=[],
            )
        if self.config.mode != "full_until_guard":
            raise ValueError(f"Unsupported exposure policy mode: {self.config.mode}")

        self._stats["apply_calls"] += 1.0
        raw_weights = _normalize_tickers(target.weights)
        adjusted: Dict[str, float] = {}
        guard_checks: List[Dict[str, Any]] = []
        states: List[str] = []
        fallback_reasons: List[str] = []

        for ticker, weight in raw_weights.items():
            proxy = self.config.proxy_map.get(ticker)
            if proxy is None:
                adjusted[ticker] = adjusted.get(ticker, 0.0) + weight
                states.append("normal")
                continue

            state, execution_asset, reason, checks = self._choose_execution_asset(
                raw_asset=ticker,
                proxy_asset=proxy,
                historical_prices=historical_prices,
            )
            guard_checks.extend(checks)
            states.append(state)
            if reason:
                fallback_reasons.append(reason)
            adjusted[execution_asset] = adjusted.get(execution_asset, 0.0) + weight

        adjusted_allocation = Allocation(_sanitize_weights(adjusted))
        exposure_state = _dominant_state(states)
        self._stats[f"{exposure_state}_calls"] = self._stats.get(f"{exposure_state}_calls", 0.0) + 1.0

        raw_asset = _largest_weight_asset(raw_weights)
        execution_asset = _largest_weight_asset(adjusted_allocation.weights)
        if _weights_different(raw_weights, adjusted_allocation.weights):
            self._stats["changed_calls"] += 1.0

        return ExposurePolicyDecision(
            allocation=adjusted_allocation,
            exposure_state=exposure_state,
            raw_strategy_target=raw_weights,
            policy_adjusted_target=dict(adjusted_allocation.weights),
            raw_signal_asset=raw_asset,
            execution_asset=execution_asset,
            fallback_reason="; ".join(fallback_reasons) if fallback_reasons else None,
            guard_checks=guard_checks,
        )

    def diagnostics(self) -> Dict[str, Any]:
        result: Dict[str, Any] = dict(self._stats)
        result["enabled"] = bool(self.enabled)
        result["mode"] = self.config.mode
        result["profile"] = self.config.profile
        result["proxy_map"] = dict(self.config.proxy_map)
        result["core_asset"] = self.config.core_asset
        result["active_states"] = dict(self._active_state_by_asset)
        return result

    def _choose_execution_asset(
        self,
        raw_asset: str,
        proxy_asset: str,
        historical_prices: pd.DataFrame,
    ) -> Tuple[str, str, Optional[str], List[Dict[str, Any]]]:
        checks: List[Dict[str, Any]] = []
        raw_ret_5d = _return(historical_prices, raw_asset, 5)
        raw_ret_21d = _return(historical_prices, raw_asset, 21)
        raw_dd_21d = _drawdown(historical_prices, raw_asset, 21)
        proxy_ret_21d = _return(historical_prices, proxy_asset, 21)

        level1 = (
            _lte(raw_ret_5d, self.config.level1_ret_5d_floor)
            or _lte(raw_dd_21d, self.config.level1_drawdown_21d_floor)
        )
        level2 = bool(
            level1
            and (
                _lt(proxy_ret_21d, self.config.level2_proxy_ret_21d_floor)
                or _lte(raw_ret_21d, self.config.level2_ret_21d_3x_floor)
            )
        )

        checks.extend(
            [
                _check("ret_5d_3x", raw_ret_5d, "<=", self.config.level1_ret_5d_floor, _lte(raw_ret_5d, self.config.level1_ret_5d_floor)),
                _check("drawdown_21d_3x", raw_dd_21d, "<=", self.config.level1_drawdown_21d_floor, _lte(raw_dd_21d, self.config.level1_drawdown_21d_floor)),
                _check("ret_21d_3x", raw_ret_21d, "<=", self.config.level2_ret_21d_3x_floor, _lte(raw_ret_21d, self.config.level2_ret_21d_3x_floor)),
                _check("ret_21d_proxy", proxy_ret_21d, "<", self.config.level2_proxy_ret_21d_floor, _lt(proxy_ret_21d, self.config.level2_proxy_ret_21d_floor)),
            ]
        )

        previous_state = self._active_state_by_asset.get(raw_asset, "normal")
        release_ok = (
            _gt(raw_ret_5d, self.config.release_ret_5d_floor)
            and _gte(proxy_ret_21d, self.config.release_proxy_ret_21d_floor)
        )

        if level2:
            self._release_streak_by_asset[raw_asset] = 0
            return self._activate(raw_asset, "safe", self.config.core_asset, "Level-2 shock: 3x and proxy/theme both weak", checks, historical_prices)
        if level1:
            self._release_streak_by_asset[raw_asset] = 0
            return self._activate(raw_asset, "deleveraged_1x", proxy_asset, "Level-1 shock: 3x de-levered to 1x proxy", checks, historical_prices)

        if previous_state in {"deleveraged_1x", "safe"}:
            if release_ok:
                self._release_streak_by_asset[raw_asset] = self._release_streak_by_asset.get(raw_asset, 0) + 1
                self._stats["release_confirmations"] += 1.0
            else:
                self._release_streak_by_asset[raw_asset] = 0

            checks.append(
                {
                    "key": "release_confirmation",
                    "label": "Release Confirmation",
                    "status": "pass" if self._release_streak_by_asset.get(raw_asset, 0) >= self.config.release_confirmation_periods else "fail",
                    "value": self._release_streak_by_asset.get(raw_asset, 0),
                    "threshold": self.config.release_confirmation_periods,
                    "detail": f"Need {self.config.release_confirmation_periods} confirmed signal day(s) before returning to 3x",
                }
            )
            if self._release_streak_by_asset.get(raw_asset, 0) < self.config.release_confirmation_periods:
                if previous_state == "safe":
                    return self._activate(raw_asset, "safe", self.config.core_asset, "Waiting for shock-release confirmation", checks, historical_prices, count_activation=False)
                return self._activate(raw_asset, "deleveraged_1x", proxy_asset, "Waiting for shock-release confirmation", checks, historical_prices, count_activation=False)

        self._active_state_by_asset[raw_asset] = "normal"
        self._release_streak_by_asset[raw_asset] = 0
        return "normal", raw_asset, None, checks

    def _activate(
        self,
        raw_asset: str,
        state: str,
        execution_asset: Optional[str],
        reason: str,
        checks: List[Dict[str, Any]],
        historical_prices: pd.DataFrame,
        count_activation: bool = True,
    ) -> Tuple[str, str, str, List[Dict[str, Any]]]:
        if not execution_asset:
            self._active_state_by_asset[raw_asset] = "blocked"
            return "blocked", raw_asset, f"{reason}; no fallback asset configured", checks
        if historical_prices is None or execution_asset not in historical_prices.columns:
            self._active_state_by_asset[raw_asset] = "blocked"
            return "blocked", raw_asset, f"{reason}; fallback asset {execution_asset} has no loaded price data", checks
        if count_activation and self._active_state_by_asset.get(raw_asset) != state:
            self._stats["guard_activations"] += 1.0
        self._active_state_by_asset[raw_asset] = state
        return state, execution_asset, reason, checks


def required_assets_from_raw(raw: Optional[Mapping[str, Any]]) -> List[str]:
    """Return tickers that need to be loaded when this policy is enabled."""
    return ExposurePolicyConfig.from_raw(raw).required_assets


def _float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _normalize_tickers(weights: Mapping[str, float]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for ticker, weight in weights.items():
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(w) or w <= 0:
            continue
        mapped = get_yahoo_ticker(str(ticker))
        result[mapped] = result.get(mapped, 0.0) + w
    return _sanitize_weights(result)


def _sanitize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    cleaned = {str(t): float(w) for t, w in weights.items() if np.isfinite(float(w)) and float(w) > 0}
    total = sum(cleaned.values())
    if total > 1.0 and total > 0:
        cleaned = {ticker: weight / total for ticker, weight in cleaned.items()}
    return cleaned


def _largest_weight_asset(weights: Mapping[str, float]) -> Optional[str]:
    if not weights:
        return None
    return max(weights.items(), key=lambda item: item[1])[0]


def _dominant_state(states: List[str]) -> str:
    order = ["safe", "deleveraged_1x", "blocked", "normal"]
    for state in order:
        if state in states:
            return state
    return "normal"


def _series(data: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    if data is None or data.empty or ticker not in data.columns:
        return None
    clean = pd.to_numeric(data[ticker], errors="coerce").dropna()
    if clean.empty:
        return None
    return clean


def _return(data: pd.DataFrame, ticker: str, days: int) -> Optional[float]:
    values = _series(data, ticker)
    if values is None or len(values) <= days:
        return None
    start = float(values.iloc[-days - 1])
    end = float(values.iloc[-1])
    if start <= 0:
        return None
    return end / start - 1.0


def _drawdown(data: pd.DataFrame, ticker: str, days: int) -> Optional[float]:
    values = _series(data, ticker)
    if values is None or len(values) < days:
        return None
    window = values.tail(days)
    peak = float(window.max())
    end = float(window.iloc[-1])
    if peak <= 0:
        return None
    return end / peak - 1.0


def _lte(value: Optional[float], threshold: float) -> bool:
    return value is not None and value <= threshold


def _lt(value: Optional[float], threshold: float) -> bool:
    return value is not None and value < threshold


def _gt(value: Optional[float], threshold: float) -> bool:
    return value is not None and value > threshold


def _gte(value: Optional[float], threshold: float) -> bool:
    return value is not None and value >= threshold


def _check(key: str, value: Optional[float], op: str, threshold: float, triggered: bool) -> Dict[str, Any]:
    return {
        "key": key,
        "label": key.replace("_", " "),
        "status": "triggered" if triggered else "pass",
        "value": value,
        "operator": op,
        "threshold": threshold,
        "detail": "insufficient_history" if value is None else f"{value:+.2%} {op} {threshold:+.2%}",
    }


def _weights_different(left: Mapping[str, float], right: Mapping[str, float], tol: float = 1e-10) -> bool:
    all_tickers = set(left) | set(right)
    return any(abs(float(left.get(ticker, 0.0)) - float(right.get(ticker, 0.0))) > tol for ticker in all_tickers)
