"""Meta allocator that combines multiple strategies into one allocation."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from backtest.strategy import Allocation, Strategy


DEFAULT_MEMBER_STRATEGIES = [
    "classic_60_40",
    "dual_momentum",
    "inverse_vol_risk_parity",
]


def _unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _parse_member_identifiers(value: Optional[str]) -> List[str]:
    if value is None:
        return DEFAULT_MEMBER_STRATEGIES.copy()
    identifiers = [part.strip() for part in str(value).split(",") if part.strip()]
    if not identifiers:
        return DEFAULT_MEMBER_STRATEGIES.copy()
    return _unique_preserve_order(identifiers)


class MetaAllocatorStrategy(Strategy):
    """
    Strategy-of-strategies allocator.

    Combines child strategy allocations using either:
    - ``equal_weight``: equal sleeve weights across members
    - ``risk_budgeted``: inverse-volatility sleeve weights based on recent
      realized vol of each member's current allocation
    """

    name = "Meta Allocator"

    def __init__(
        self,
        members_csv: str = "classic_60_40,dual_momentum,inverse_vol_risk_parity",
        allocator: Literal[
            "equal_weight", "risk_budgeted", "equal_risk_contribution"
        ] = "equal_weight",
        lookback_days: int = 63,
        rebalance_frequency: str = "monthly",
    ) -> None:
        member_identifiers = _parse_member_identifiers(members_csv)
        if not member_identifiers:
            raise ValueError("MetaAllocator requires at least one member strategy.")

        allocator_name = str(allocator).strip().lower()
        if allocator_name not in {
            "equal_weight",
            "risk_budgeted",
            "equal_risk_contribution",
        }:
            raise ValueError(
                "allocator must be 'equal_weight', 'risk_budgeted' "
                "or 'equal_risk_contribution'"
            )

        self._member_identifiers = member_identifiers
        self._allocator = allocator_name
        self._lookback_days = max(10, int(lookback_days))
        self.rebalance_frequency = rebalance_frequency

        self._members = self._load_members(member_identifiers)
        if not self._members:
            raise ValueError("No member strategies could be loaded.")

        assets: List[str] = []
        for member in self._members:
            assets.extend(list(getattr(member, "assets", []) or []))
        self.assets = _unique_preserve_order(assets)

        self.params = {
            "members_csv": ",".join(member_identifiers),
            "allocator": self._allocator,
            "lookback_days": self._lookback_days,
            "rebalance_frequency": self.rebalance_frequency,
        }

    @classmethod
    def get_param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "allocator": [
                "equal_weight",
                "risk_budgeted",
                "equal_risk_contribution",
            ],
            "lookback_days": [42, 63, 126],
        }

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        if data is None or data.empty:
            return Allocation({})

        member_allocations: List[Dict[str, float]] = []
        member_risks: List[float] = []
        available_columns = set(data.columns)

        for member in self._members:
            try:
                raw_allocation = member.signal(date, data)
            except Exception:
                raw_allocation = Allocation({})

            weights = self._sanitize_allocation(raw_allocation, available_columns)
            member_allocations.append(weights)
            member_risks.append(self._estimate_member_risk(weights, data))

        sleeve_weights = self._build_sleeve_weights(member_risks)

        combined: Dict[str, float] = {}
        for sleeve_weight, member_weights in zip(sleeve_weights, member_allocations):
            if sleeve_weight <= 0.0:
                continue
            for ticker, weight in member_weights.items():
                combined[ticker] = combined.get(ticker, 0.0) + sleeve_weight * weight

        total = sum(combined.values())
        if total > 1.0 and total > 0.0:
            scale = 1.0 / total
            combined = {ticker: weight * scale for ticker, weight in combined.items()}

        return Allocation(combined)

    def _build_sleeve_weights(self, member_risks: List[float]) -> List[float]:
        n = len(member_risks)
        if n == 0:
            return []
        if self._allocator == "equal_weight":
            return [1.0 / n] * n

        risk = np.asarray(member_risks, dtype=float)
        valid = np.isfinite(risk) & (risk > 0.0)
        if not valid.any():
            return [1.0 / n] * n

        fallback = float(np.median(risk[valid]))
        if not np.isfinite(fallback) or fallback <= 0.0:
            fallback = 1.0
        risk[~valid] = fallback

        inv = 1.0 / np.maximum(risk, 1e-8)
        inv_sum = float(inv.sum())
        if not np.isfinite(inv_sum) or inv_sum <= 0.0:
            return [1.0 / n] * n
        # Phase E4: in the allocator path, `equal_risk_contribution` is
        # reduced to inverse-vol equivalence, because no cross-member
        # covariance is available here. True ERC with a correlation
        # matrix lives in `portfolio.risk_parity.erc_weights` and is
        # used in `strategies/ml_forecast_tilt.py`
        # (Codex R4.15: Phase-E ERC deliberately kept to the tilt strategy only).
        return (inv / inv_sum).tolist()

    def _estimate_member_risk(self, weights: Dict[str, float], data: pd.DataFrame) -> float:
        if not weights:
            return float("nan")

        tickers = [ticker for ticker in weights.keys() if ticker in data.columns]
        if not tickers:
            return float("nan")

        prices = data[tickers].dropna(how="all")
        returns = prices.pct_change(fill_method=None).dropna(how="all")
        if len(returns) < 10:
            return float("nan")
        window = returns.iloc[-self._lookback_days:] if len(returns) > self._lookback_days else returns
        if len(window) < 10:
            return float("nan")

        weight_vector = pd.Series(weights, dtype=float)
        portfolio_returns = window.fillna(0.0).mul(weight_vector, axis=1).sum(axis=1)
        vol = float(portfolio_returns.std(ddof=0) * np.sqrt(252.0))
        if not np.isfinite(vol) or vol <= 0.0:
            return float("nan")
        return vol

    @staticmethod
    def _sanitize_allocation(allocation: Allocation, available_columns: set[str]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if not isinstance(allocation, Allocation):
            return result
        for ticker, weight in allocation.items():
            if ticker not in available_columns:
                continue
            try:
                value = float(weight)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value) or value <= 0.0:
                continue
            result[ticker] = value

        total = sum(result.values())
        if total > 1.0 and total > 0.0:
            scale = 1.0 / total
            result = {ticker: weight * scale for ticker, weight in result.items()}
        return result

    def _load_members(self, identifiers: List[str]) -> List[Strategy]:
        members: List[Strategy] = []
        for identifier in identifiers:
            strategy = self._load_member(identifier)
            if isinstance(strategy, MetaAllocatorStrategy):
                raise ValueError("Nested MetaAllocatorStrategy members are not supported.")
            members.append(strategy)
        return members

    def _load_member(self, identifier: str) -> Strategy:
        strategy_path = self._resolve_strategy_path(identifier)
        module_name = f"meta_member_{strategy_path.stem}_{abs(hash(str(strategy_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, strategy_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load strategy module: {strategy_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        strategy_instance: Optional[Strategy] = None
        strategy_class: Optional[type[Strategy]] = None

        if hasattr(module, "strategy"):
            candidate = getattr(module, "strategy")
            if isinstance(candidate, Strategy):
                strategy_instance = candidate
                strategy_class = type(candidate)

        if strategy_class is None:
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, Strategy) and attr is not Strategy:
                    strategy_class = attr
                    break

        if strategy_instance is None:
            if strategy_class is None:
                raise ValueError(f"No Strategy subclass found in {strategy_path}")
            try:
                strategy_instance = strategy_class()
            except Exception as exc:
                raise ValueError(
                    f"Could not instantiate member strategy from {strategy_path}: {exc}"
                ) from exc

        return strategy_instance

    def _resolve_strategy_path(self, identifier: str) -> Path:
        raw = str(identifier).strip()
        if not raw:
            raise ValueError("Empty member strategy identifier.")

        repo_root = Path(__file__).resolve().parents[2]
        strategies_dir = repo_root / "strategies"
        archive_dir = strategies_dir / "_archive"
        candidates: List[Path] = []

        raw_path = Path(raw).expanduser()
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(Path.cwd() / raw_path)
            candidates.append(strategies_dir / raw_path)
            if raw_path.suffix != ".py":
                candidates.append(strategies_dir / f"{raw_path}.py")
                candidates.append(Path.cwd() / f"{raw_path}.py")
                if archive_dir.exists():
                    candidates.extend(sorted(archive_dir.glob(f"**/{raw_path.name}.py")))
            elif archive_dir.exists():
                candidates.extend(sorted(archive_dir.glob(f"**/{raw_path.name}")))

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                if candidate.resolve().name == "meta_allocator.py":
                    raise ValueError(
                        "Meta allocator cannot include itself as member strategy."
                    )
                return candidate.resolve()

        raise FileNotFoundError(
            f"Member strategy '{identifier}' could not be resolved to a strategy file."
        )
