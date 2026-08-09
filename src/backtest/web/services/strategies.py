"""Strategy discovery and schema generation service."""

import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, get_type_hints, get_origin, get_args, Literal, Union

from backtest.strategy import Strategy
from backtest.web.config import DEFAULT_STRATEGIES_DIR


@dataclass
class ParamInfo:
    """Information about a strategy parameter."""
    name: str
    type: str
    default: Any
    required: bool
    description: Optional[str] = None
    choices: Optional[List[Any]] = None


@dataclass
class StrategyInfo:
    """Information about a strategy."""
    name: str
    file_path: str
    class_name: str
    description: Optional[str]
    assets: List[str]
    params: List[ParamInfo]
    rebalance_frequency: str


def list_strategies(strategies_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """List all available strategies in the strategies directory.

    Returns a list of basic strategy info (name, file_path, description).
    """
    if strategies_dir is None:
        strategies_dir = DEFAULT_STRATEGIES_DIR

    hidden_files = {
        # Root dependency for the visible LBH challenger preset, not a curated UI choice.
        "levered_momentum_crash_guard.py",
        # Phase B/C/D demo strategies: without an external-features provider
        # they degrade to trivial baselines. They remain runnable via CLI
        # and tested, but are not in the curated frontend dropdown.
        "ml_forecast_tilt.py",
        "sentiment_tilt.py",
        "analyst_momentum_filter.py",
    }

    strategies_dir = Path(strategies_dir)
    if not strategies_dir.exists():
        return []

    strategies = []
    for path in sorted(strategies_dir.glob("*.py")):
        # Skip private files
        if path.name.startswith("_"):
            continue

        try:
            if path.name in hidden_files:
                continue
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"PIT universe CSV not found:.*",
                    category=UserWarning,
                )
                info = get_strategy_info(path)
            strategies.append({
                "name": info.name,
                "file_path": str(path.resolve()),  # Use absolute path
                "file_name": path.name,
                "description": info.description,
                "assets": info.assets,
                "rebalance_frequency": info.rebalance_frequency,
            })
        except Exception as e:
            # Skip files that can't be loaded
            print(f"Warning: Could not load strategy from {path}: {e}")
            continue

    strategies.sort(key=lambda item: _strategy_sort_key(item["file_name"], item["name"]))
    return strategies


def _strategy_sort_key(file_name: str, name: str) -> tuple[int, str]:
    """Sort UI dropdown by governance tier: Production -> Pilot -> Research -> Benchmark -> Experimental.

    The tier prefix in strategy.name (e.g. ``[Production]``, ``[Pilot]``,
    ``[Research]``, ``[Benchmark]``, ``[Experimental]``) gives users the
    status at a glance. Order:

    - Tier 1 ``[Production]``: the framework's default reference strategies.
    - Tier 2 ``[Pilot]``: candidates with a serious promotion prospect.
    - Tier 3 ``[Research]``: legitimate alternatives, but not a default.
    - Tier 4 ``[Benchmark]``: references (Buy & Hold).
    - Tier 5 ``[Experimental]``: only with the greatest caution.
    """

    preferred_order = {
        # Tier 1: Production Defaults
        "levered_etf_momentum_sticky.py": 2,
        # Tier 2: Pilot / Promotion Candidates
        "sticky_levered_vol_targeted.py": 13,
        "sticky_levered_cascade.py": 14,
        # Tier 3: Research Variants
        "sticky_levered_vol_targeted_sector_aware.py": 20,
        "levered_momentum_crash_guard.py": 22,
        "levered_etf_momentum_sticky_adaptive_v2.py": 23,
        "levered_momentum_crash_guard_lbh_challenger.py": 24,
        # Rebalance-friction research variants.
        "sticky_levered_tax_aware.py": 25,
        "sticky_levered_entry_staged.py": 26,
        # AI infrastructure basket (thematic research candidate).
        "ai_infra_basket.py": 27,
        # Regime vol gate: a cyclical vol-timing bet that failed out-of-sample
        # robustness checks -> stays Research, do not promote.
        "sticky_levered_vol_targeted_pct120.py": 28,
        # Tier 4: Benchmarks / classic reference strategies
        "buy_and_hold.py": 80,
        "3x_bh.py": 81,
        "classic_60_40.py": 82,
        "all_weather.py": 83,
        "dual_momentum.py": 84,
        "trend_following_sma.py": 85,
        "volatility_targeting.py": 86,
        "inverse_vol_risk_parity.py": 87,
        "momentum_topn_pit_sp500.py": 88,
        "sector_rotation_momentum.py": 89,
        # Tier 5: Experimental
        "levered_5x_momentum_guard.py": 95,
    }
    return (preferred_order.get(file_name, 100), name.lower())


def get_strategy_info(path: Path) -> StrategyInfo:
    """Get detailed information about a strategy including its parameters."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    # Ensure strategies package imports resolve (for strategies/_momentum_utils.py, etc.)
    repo_root = path.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Load the module
    module_name = f"strategy_module_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"PIT universe CSV not found:.*",
            category=UserWarning,
        )
        spec.loader.exec_module(module)

    # Find the Strategy subclass
    strategy_class = None
    strategy_instance = None

    for name, obj in vars(module).items():
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
            strategy_class = obj
        elif isinstance(obj, Strategy) and name == "strategy":
            strategy_instance = obj

    if strategy_class is None and strategy_instance is None:
        raise ValueError(f"No Strategy found in {path}")

    # Get parameter info from class
    params = _extract_params(strategy_class) if strategy_class else []

    # Get instance info
    if strategy_instance:
        instance = strategy_instance
    elif strategy_class:
        # Try to instantiate with no args to get defaults
        try:
            instance = strategy_class()
        except TypeError:
            # Requires arguments, create a minimal instance for info
            instance = None
    else:
        instance = None

    # Build info
    name = instance.name if instance else (strategy_class.__name__ if strategy_class else path.stem)
    description = strategy_class.__doc__ if strategy_class else None
    assets = instance.assets if instance else []
    rebalance_freq = instance.rebalance_frequency if instance else "monthly"

    return StrategyInfo(
        name=name,
        file_path=str(path),
        class_name=strategy_class.__name__ if strategy_class else "",
        description=description.strip() if description else None,
        assets=assets,
        params=params,
        rebalance_frequency=rebalance_freq,
    )


def _extract_params(strategy_class: Type[Strategy]) -> List[ParamInfo]:
    """Extract parameter information from a Strategy class."""
    params = []

    if strategy_class is None:
        return params

    # Get __init__ signature
    try:
        sig = inspect.signature(strategy_class.__init__)
    except (ValueError, TypeError):
        return params

    # Try to get type hints
    try:
        hints = get_type_hints(strategy_class.__init__)
    except Exception:
        hints = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "args", "kwargs"):
            continue

        # Get type info
        type_hint = hints.get(param_name, param.annotation)
        type_str, choices = _parse_type_hint(type_hint)

        # Determine if required
        has_default = param.default is not inspect.Parameter.empty
        default_value = param.default if has_default else None

        params.append(ParamInfo(
            name=param_name,
            type=type_str,
            default=default_value,
            required=not has_default,
            choices=choices,
        ))

    return params


def _parse_type_hint(hint) -> tuple[str, Optional[List[Any]]]:
    """Parse a type hint to get a string representation and any choices."""
    if hint is inspect.Parameter.empty:
        return "any", None

    origin = get_origin(hint)
    args = get_args(hint)

    # Handle Literal types (for choices)
    if origin is Literal:
        return "choice", list(args)

    # Handle List types
    if origin is list:
        if args:
            inner_type, _ = _parse_type_hint(args[0])
            return f"list[{inner_type}]", None
        return "list", None

    # Handle Optional types
    if origin is type(None) or (hasattr(hint, "__origin__") and str(hint).startswith("typing.Optional")):
        if args:
            inner_type, choices = _parse_type_hint(args[0])
            return inner_type, choices
        return "any", None

    # Handle Union types (for Optional)
    if hasattr(origin, "__name__") and origin.__name__ == "Union":
        # Filter out NoneType
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _parse_type_hint(non_none_args[0])
        return "any", None

    # Basic types
    type_map = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "list",
        dict: "object",
    }

    if hint in type_map:
        return type_map[hint], None

    # Try to get name
    if hasattr(hint, "__name__"):
        return hint.__name__.lower(), None

    return "any", None


def get_param_schema(path: Path) -> Dict[str, Any]:
    """Generate a JSON Schema-like structure for strategy parameters.

    This is used by the frontend to dynamically generate form fields.
    """
    info = get_strategy_info(path)

    schema = {
        "strategy_name": info.name,
        "description": info.description,
        "assets": info.assets,
        "rebalance_frequency": info.rebalance_frequency,
        "parameters": [],
    }

    for param in info.params:
        param_schema = {
            "name": param.name,
            "type": param.type,
            "required": param.required,
            "default": param.default,
        }

        if param.choices:
            param_schema["choices"] = param.choices

        if param.description:
            param_schema["description"] = param.description

        schema["parameters"].append(param_schema)

    return schema


def load_strategy_instance(
    path: Path,
    params: Optional[Dict[str, Any]] = None
) -> Strategy:
    """Load and instantiate a strategy with optional parameter overrides.

    This is a thin wrapper around the CLI's load_strategy_from_file
    that also handles parameter overrides.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    # Load the module
    module_name = f"strategy_module_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Find strategy class and instance
    strategy_class = None
    strategy_instance = None

    for name, obj in vars(module).items():
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
            strategy_class = obj
        elif isinstance(obj, Strategy) and name == "strategy":
            strategy_instance = obj

    if strategy_class is None and strategy_instance is None:
        raise ValueError(f"No Strategy found in {path}")

    # Instantiate with parameters
    if params:
        if strategy_class is None:
            raise ValueError("Cannot override parameters - no strategy class found")
        coerced_params = coerce_params_to_signature(strategy_class, params)
        return strategy_class(**coerced_params)
    elif strategy_instance:
        return strategy_instance
    else:
        return strategy_class()


def _normalize_bool(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return value


def _coerce_value_to_type(value: Any, target_type: Any) -> Any:
    if value is None or target_type is Any:
        return value

    origin = get_origin(target_type)
    args = get_args(target_type)

    if origin is Union:
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return _coerce_value_to_type(value, non_none[0])
        return value

    if origin is list and args:
        if isinstance(value, list):
            return [_coerce_value_to_type(item, args[0]) for item in value]
        return value

    if origin is Literal:
        return value

    if target_type is bool:
        return bool(_normalize_bool(value))
    if target_type is int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value
    if target_type is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if target_type is str:
        try:
            return str(value)
        except Exception:
            return value

    return value


def coerce_params_to_signature(strategy_class: Type[Strategy], params: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce parameter values to the strategy __init__ signature types."""
    try:
        hints = get_type_hints(strategy_class.__init__)
    except Exception:
        hints = {}

    coerced: Dict[str, Any] = {}
    for key, value in params.items():
        target_type = hints.get(key, Any)
        coerced[key] = _coerce_value_to_type(value, target_type)

    return coerced
