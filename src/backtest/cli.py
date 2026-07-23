"""
Command Line Interface for the backtest framework.

Uses argparse for maximum compatibility across Python versions.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

from backtest.presets import get_preset_profile, preset_profile_names


def load_strategy_from_file(path: str):
    """Dynamically load a strategy from a Python file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    spec = importlib.util.spec_from_file_location("strategy_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_module"] = module
    spec.loader.exec_module(module)

    from backtest.strategy import Strategy

    # First, check for a pre-instantiated 'strategy' variable
    if hasattr(module, 'strategy'):
        obj = getattr(module, 'strategy')
        if isinstance(obj, Strategy):
            return obj

    # Otherwise, find and instantiate the first Strategy subclass
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, Strategy)
            and obj is not Strategy
        ):
            return obj()

    raise ValueError(f"No Strategy subclass found in {path}")


def _load_cost_profile(path: str | None):
    """
    Load optional per-ticker cost profile from JSON.

    Expected schema (all keys optional):
      {
        "default": {...},
        "asset_classes": {"class": {...}},
        "ticker_asset_class": {"AAPL": "class"},
        "tickers": {"AAPL": {...}}
      }
    """
    if not path:
        return None

    profile_path = Path(path)
    if not profile_path.exists():
        print(f"Error: Cost profile file not found: {profile_path}")
        sys.exit(1)

    try:
        with open(profile_path, "r") as f:
            profile = json.load(f)
    except Exception as e:
        print(f"Error loading cost profile JSON: {e}")
        sys.exit(1)

    if not isinstance(profile, dict):
        print("Error: Cost profile JSON must be an object/dictionary")
        sys.exit(1)

    return profile


def _resolve_execution_lag(args) -> int:
    """Resolve execution lag from explicit days and optional T+1 shortcut."""
    lag_days = int(getattr(args, "execution_lag_days", 0) or 0)
    if getattr(args, "t_plus_one", False):
        lag_days = max(lag_days, 1)
    if lag_days < 0:
        print("Error: --execution-lag-days must be >= 0")
        sys.exit(1)
    return lag_days


def _apply_preset_profile_defaults(args) -> None:
    """
    Apply preset execution/risk defaults if requested.

    Presets are applied as defaults only: fields that differ from the parser
    defaults are treated as user overrides and are left untouched.
    """
    profile_name = getattr(args, "preset_profile", None)
    if not profile_name:
        return

    settings = get_preset_profile(profile_name)
    if settings is None:
        print(
            f"Error: Unknown preset profile '{profile_name}'. "
            f"Available: {', '.join(preset_profile_names())}"
        )
        sys.exit(1)

    # Parser/default baselines used by execution/risk flags.
    defaults = {
        "execution_lag_days": 0,
        "max_volume_participation": None,
        "min_daily_dollar_volume": 0.0,
        "liquidity_on_missing_volume": "allow",
        "max_position": None,
        "turnover_budget": None,
        "drawdown_brake_threshold": None,
        "drawdown_brake_cash_target": 1.0,
        "drawdown_brake_release": None,
    }

    for key, default_value in defaults.items():
        if not hasattr(args, key):
            continue
        current_value = getattr(args, key)
        if current_value == default_value:
            setattr(args, key, settings.get(key, current_value))


def _resolve_liquidity_settings(args):
    """Resolve liquidity guard settings from CLI args."""
    max_volume_participation = getattr(args, "max_volume_participation", None)
    min_daily_dollar_volume = float(getattr(args, "min_daily_dollar_volume", 0.0) or 0.0)
    liquidity_on_missing_volume = getattr(args, "liquidity_on_missing_volume", "allow")
    return max_volume_participation, min_daily_dollar_volume, liquidity_on_missing_volume


def _needs_volume_data(max_volume_participation, min_daily_dollar_volume) -> bool:
    """True if current settings require loading volume series."""
    return (
        (max_volume_participation is not None and max_volume_participation > 0)
        or float(min_daily_dollar_volume) > 0.0
    )


def _load_json_mapping(path: str | None, label: str) -> dict:
    """Load a JSON object from file and validate mapping shape."""
    if not path:
        return {}

    json_path = Path(path)
    if not json_path.exists():
        print(f"Error: {label} file not found: {json_path}")
        sys.exit(1)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {label} JSON: {e}")
        sys.exit(1)

    if not isinstance(data, dict):
        print(f"Error: {label} JSON must be an object/dictionary")
        sys.exit(1)

    return data


def _parse_sector_caps(entries) -> dict:
    """Parse repeated --sector-cap entries in format sector=cap."""
    caps = {}
    if not entries:
        return caps

    for entry in entries:
        raw = str(entry).strip()
        if not raw:
            continue
        if "=" in raw:
            sector, cap_raw = raw.split("=", 1)
        elif ":" in raw:
            sector, cap_raw = raw.split(":", 1)
        else:
            print(f"Error: Invalid --sector-cap '{entry}'. Use sector=cap (e.g., equity=0.6)")
            sys.exit(1)

        sector_name = sector.strip().lower()
        if not sector_name:
            print(f"Error: Invalid --sector-cap '{entry}'. Sector must not be empty.")
            sys.exit(1)
        try:
            caps[sector_name] = float(cap_raw.strip())
        except ValueError:
            print(f"Error: Invalid --sector-cap '{entry}'. Cap must be numeric.")
            sys.exit(1)

    return caps


def _resolve_risk_overlay(args):
    """Build risk overlay config from CLI args."""
    max_position = getattr(args, "max_position", None)
    turnover_budget = getattr(args, "turnover_budget", None)
    sector_caps = _parse_sector_caps(getattr(args, "sector_caps", None))
    ticker_sectors_raw = _load_json_mapping(getattr(args, "sector_map_file", None), "Sector map")
    ticker_sectors = {
        str(ticker): str(sector).strip().lower()
        for ticker, sector in ticker_sectors_raw.items()
        if str(sector).strip()
    }

    drawdown_threshold = getattr(args, "drawdown_brake_threshold", None)
    drawdown_cash_target = getattr(args, "drawdown_brake_cash_target", 1.0)
    drawdown_release = getattr(args, "drawdown_brake_release", None)

    has_drawdown = drawdown_threshold is not None
    if has_drawdown:
        drawdown_brake = {
            "threshold": float(drawdown_threshold),
            "cash_target": float(drawdown_cash_target),
        }
        if drawdown_release is not None:
            drawdown_brake["release_drawdown"] = float(drawdown_release)
    else:
        drawdown_brake = None

    has_overlay = any(
        [
            max_position is not None,
            turnover_budget is not None,
            bool(sector_caps),
            bool(ticker_sectors),
            drawdown_brake is not None,
        ]
    )
    if not has_overlay:
        return None

    overlay = {}
    if max_position is not None:
        overlay["max_position"] = float(max_position)
    if turnover_budget is not None:
        overlay["turnover_budget"] = float(turnover_budget)
    if sector_caps:
        overlay["sector_caps"] = sector_caps
    if ticker_sectors:
        overlay["ticker_sectors"] = ticker_sectors
    if drawdown_brake is not None:
        overlay["drawdown_brake"] = drawdown_brake

    return overlay


def _resolve_exposure_policy(args):
    """Build optional exposure policy config from CLI args."""
    policy_file = getattr(args, "exposure_policy_file", None)
    if policy_file:
        policy = _load_json_mapping(policy_file, "Exposure policy")
    elif not getattr(args, "exposure_policy_enable", False):
        return None
    else:
        policy = {
            "enabled": True,
            "profile": getattr(args, "exposure_policy_profile", None) or "trade_republic",
            "mode": "full_until_guard",
        }

    policy = dict(policy)
    if getattr(args, "exposure_policy_enable", False):
        policy["enabled"] = True
    if getattr(args, "exposure_policy_profile", None):
        policy["profile"] = getattr(args, "exposure_policy_profile")
    if getattr(args, "exposure_policy_core_asset", None):
        policy["core_asset"] = getattr(args, "exposure_policy_core_asset")

    threshold_fields = {
        "level1_ret_5d_floor": "exposure_level1_ret_5d_floor",
        "level1_drawdown_21d_floor": "exposure_level1_drawdown_21d_floor",
        "level2_ret_21d_3x_floor": "exposure_level2_ret_21d_3x_floor",
        "level2_proxy_ret_21d_floor": "exposure_level2_proxy_ret_21d_floor",
        "release_ret_5d_floor": "exposure_release_ret_5d_floor",
        "release_confirmation_periods": "exposure_release_confirmation_periods",
    }
    for policy_key, arg_key in threshold_fields.items():
        value = getattr(args, arg_key, None)
        if value is not None:
            policy[policy_key] = value
    return policy


def _build_external_features_config(args):
    """Translate CLI flags into an ExternalFeaturesConfig.

    Default is fully off — backward compatible. ``--external-features-enable``
    activates the pipeline; ``--external-features-dataset`` selects the
    snapshot namespace; ``--external-features-provenance-mode`` controls
    strict/warn/off enforcement.
    """
    from backtest.external_features.config import ExternalFeaturesConfig

    enabled = bool(getattr(args, "external_features_enable", False))
    dataset = getattr(args, "external_features_dataset", None)
    mode = getattr(args, "external_features_provenance_mode", None) or "warn"
    root = getattr(args, "external_features_root", None) or "data/external_features"
    registry = (
        getattr(args, "external_features_registry", None)
        or "data/manual/provenance.json"
    )
    if not enabled and not dataset:
        return ExternalFeaturesConfig()
    return ExternalFeaturesConfig(
        enabled=enabled,
        dataset=dataset,
        root=root,
        provenance_mode=mode,
        registry_path=registry,
    )


def _build_external_features_loader(args):
    """Convenience: ExternalFeaturesConfig -> Optional[loader]."""
    from backtest.external_features.config import build_loader_from_config

    return build_loader_from_config(_build_external_features_config(args))


def _add_external_features_arguments(parser) -> None:
    """Add the --external-features-* flag trio to a subparser."""
    parser.add_argument(
        "--external-features-enable",
        action="store_true",
        default=False,
        help="Enable the external features pipeline (analyst/news/ML).",
    )
    parser.add_argument(
        "--external-features-dataset",
        default=None,
        help="External features dataset identifier (e.g., mock_analyst).",
    )
    parser.add_argument(
        "--external-features-provenance-mode",
        choices=["off", "warn", "strict"],
        default="warn",
        help="Provenance enforcement mode for external features (default: warn).",
    )
    parser.add_argument(
        "--external-features-root",
        default=None,
        help="Root directory for external features (default: data/external_features).",
    )
    parser.add_argument(
        "--external-features-registry",
        default=None,
        help="Optional provenance registry path (default: data/manual/provenance.json).",
    )
    # Phase C news flags (runtime). Engine choice + intraday cutoff are
    # PULL-TIME concerns and live on `features pull` instead.
    parser.add_argument(
        "--news-dataset",
        default=None,
        help=(
            "External features dataset id that carries news_sentiment_score "
            "(used by SignalGenerator and run_meta_decision)."
        ),
    )
    parser.add_argument(
        "--news-score-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the news_sentiment_score component in the meta-decision "
            "live_score. Defaults to 0.0 (off). w_analyst + w_news must be < 1."
        ),
    )
    # Phase D ml flags (runtime). Bundle dir / stacking-only live on
    # `features pull` because they are pull-time concerns.
    parser.add_argument(
        "--ml-dataset",
        default=None,
        help=(
            "External features dataset id that carries ml_forecast_score "
            "(used by SignalGenerator and run_meta_decision)."
        ),
    )
    parser.add_argument(
        "--ml-score-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the ml_forecast_score component in the meta-decision "
            "live_score. Defaults to 0.0 (off). w_analyst + w_news + w_ml "
            "must be < 1."
        ),
    )
    # Phase E1: Cross-Product-Konsens-Gate.
    parser.add_argument(
        "--cross-product-require",
        action="store_true",
        default=False,
        help=(
            "Phase E1: enforce cross-product consensus across "
            "analyst/news/ml as a pre-switch gate (Codex R3.7). "
            "Default off."
        ),
    )
    parser.add_argument(
        "--cross-product-threshold",
        type=float,
        default=None,
        help=(
            "Phase E1: override the profile-default cross-product "
            "consensus threshold (defensiv=0.7, ausgewogen=0.5, "
            "aggressiv=0.3)."
        ),
    )


def _resolve_news_runtime_args(args):
    dataset = getattr(args, "news_dataset", None)
    weight = float(getattr(args, "news_score_weight", 0.0) or 0.0)
    return dataset, weight


def _resolve_cross_product_args(args):
    """Phase E1 (T-0306): Cross-Product Gate-Flags."""

    require = bool(getattr(args, "cross_product_require", False))
    threshold = getattr(args, "cross_product_threshold", None)
    return require, threshold


def _resolve_ml_runtime_args(args):
    """Phase D analogue of :func:`_resolve_news_runtime_args` (T-0228)."""

    dataset = getattr(args, "ml_dataset", None)
    weight = float(getattr(args, "ml_score_weight", 0.0) or 0.0)
    return dataset, weight


def _extend_assets_for_exposure_policy(assets, exposure_policy):
    """Add policy proxy/core assets to a list of tickers."""
    if not exposure_policy:
        return list(assets)
    from backtest.risk.exposure_policy import required_assets_from_raw

    ordered = list(assets)
    seen = set(ordered)
    for ticker in required_assets_from_raw(exposure_policy):
        if ticker not in seen:
            ordered.append(ticker)
            seen.add(ticker)
    return ordered


def cmd_run(args):
    """Run a backtest for a single strategy."""
    from backtest.backtester import Backtester, BacktestConfig
    from backtest.data import DataLoader
    from backtest.reporter import Reporter
    from backtest.metadata import (
        build_run_metadata,
        build_strategy_info,
        build_data_info,
    )

    print(f"\nLoading strategy from {args.strategy}...")

    # Load strategy with optional parameter overrides
    strategy_path = Path(args.strategy)
    if not strategy_path.exists():
        print(f"Error: Strategy file not found: {strategy_path}")
        sys.exit(1)

    # Import strategy module
    spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_module"] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error loading strategy module: {e}")
        sys.exit(1)

    # Find strategy class and instance
    from backtest.strategy import Strategy as BaseStrategy
    strategy_class = None
    strategy_instance = None

    for name, obj in vars(module).items():
        # Find Strategy subclass (but not the base class itself)
        if (isinstance(obj, type)
            and issubclass(obj, BaseStrategy)
            and obj is not BaseStrategy
            and hasattr(obj, 'signal')):
            strategy_class = obj
        # Find strategy instance
        elif (hasattr(obj, 'signal')
              and hasattr(obj, 'assets')
              and isinstance(obj, BaseStrategy)):
            if name == 'strategy' or strategy_instance is None:
                strategy_instance = obj

    if strategy_class is None and strategy_instance is None:
        print("Error: No strategy found in file")
        sys.exit(1)

    # Parse parameter overrides
    param_overrides = {}
    if args.params:
        for param_str in args.params:
            if "=" not in param_str:
                print(f"Error: Invalid param format '{param_str}'. Use: --param name=value")
                sys.exit(1)
            name, value_str = param_str.split("=", 1)
            value_str = value_str.strip()
            # Try to parse as Python literal (list, dict, etc.)
            try:
                import ast
                param_overrides[name] = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                # Try to parse as number
                try:
                    if "." in value_str:
                        param_overrides[name] = float(value_str)
                    else:
                        param_overrides[name] = int(value_str)
                except ValueError:
                    param_overrides[name] = value_str

    # Instantiate strategy with parameters
    try:
        if param_overrides:
            if strategy_class is None:
                print("Error: Cannot override parameters - no strategy class found")
                sys.exit(1)
            strategy = strategy_class(**param_overrides)
            print(f"Strategy: {strategy.name}")
            print(f"Parameters: {param_overrides}")
        elif strategy_instance:
            strategy = strategy_instance
            print(f"Strategy: {strategy.name}")
        else:
            strategy = strategy_class()
            print(f"Strategy: {strategy.name}")
    except Exception as e:
        print(f"Error instantiating strategy: {e}")
        sys.exit(1)

    print(f"Assets: {', '.join(strategy.assets)}\n")

    print("Loading price data...")

    # Determine metric basis
    tax_enabled = not args.no_tax
    if args.liquidate_at_end:
        metric_basis = "net_liquidation"
    elif args.metric_basis:
        metric_basis = args.metric_basis
    else:
        # Default: net_liquidation if tax enabled, gross otherwise
        metric_basis = "net_liquidation" if tax_enabled else "gross"
    execution_lag_days = _resolve_execution_lag(args)
    max_volume_participation, min_daily_dollar_volume, liquidity_on_missing_volume = _resolve_liquidity_settings(args)
    load_volumes = _needs_volume_data(max_volume_participation, min_daily_dollar_volume)
    exposure_policy = _resolve_exposure_policy(args)

    try:
        assets = _extend_assets_for_exposure_policy(strategy.assets.copy(), exposure_policy)
        benchmark_setting = args.benchmark
        bench_ticker = None
        if args.benchmark_ticker:
            bench_ticker = args.benchmark_ticker
            benchmark_setting = args.benchmark_ticker
        elif args.benchmark:
            from backtest.assets import get_benchmark_ticker
            bench_ticker = get_benchmark_ticker(args.benchmark)
        if bench_ticker and bench_ticker not in assets:
            assets.append(bench_ticker)

        drip_enabled = getattr(args, 'drip', False)
        validate = not getattr(args, 'no_validate', False)
        data = DataLoader.yahoo(
            tickers=assets,
            start=args.start,
            end=args.end,
            currency="EUR",
            align=getattr(args, 'align', 'ffill'),
            skip_failed=getattr(args, 'skip_failed', True),
            load_dividends=drip_enabled or tax_enabled,
            load_volumes=load_volumes,
            validate=validate,
        )
        print(f"Loaded {len(data.prices)} days of data\n")
        if data.dividends is not None:
            div_count = (data.dividends > 0).sum().sum()
            print(f"  Loaded {div_count} dividend events")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    print("Running backtest...")
    cost_profile = _load_cost_profile(getattr(args, "cost_profile_file", None))
    risk_overlay = _resolve_risk_overlay(args)

    config = BacktestConfig(
        initial_capital=args.capital,
        costs_pct=args.costs,
        cost_profile=cost_profile,
        execution_lag_days=execution_lag_days,
        max_volume_participation=max_volume_participation,
        min_daily_dollar_volume=min_daily_dollar_volume,
        liquidity_on_missing_volume=liquidity_on_missing_volume,
        exposure_policy=exposure_policy,
        risk_overlay=risk_overlay,
        benchmark=benchmark_setting,
        rebalance_frequency=args.rebalance_frequency,
        tax_enabled=tax_enabled,
        tax_exemption_amount=args.tax_exemption,
        cash_rate=args.cash_rate,
        metric_basis=metric_basis,
        allow_universe_lookahead=getattr(args, 'allow_universe_lookahead', False),
        validate=not getattr(args, 'no_validate', False),
        drip_enabled=drip_enabled,
        external_features_loader=_build_external_features_loader(args),
    )

    backtester = Backtester(strategy, data, config)
    result = backtester.run()

    print(result.summary())

    # Build metadata for the report
    strategy_info = build_strategy_info(strategy, args.strategy)
    data_info = build_data_info(data, requested_assets=assets)
    metadata = build_run_metadata(
        config=config,
        data_info=data_info,
        strategy_info=strategy_info,
        mode="single",
        cli_rebalance_frequency=args.rebalance_frequency,
        cli_start=args.start,
        cli_end=args.end,
    )

    if args.output:
        reporter = Reporter(result, metadata=metadata)
        if args.format.lower() == "json":
            reporter.to_json(args.output)
        else:
            reporter.to_html(args.output)
        print(f"\nReport saved to: {args.output}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = strategy.name.lower().replace(' ', '_').replace('/', '_')
        filename = f"reports/{safe_name}_{timestamp}.html"
        Path("reports").mkdir(exist_ok=True)
        Reporter(result, metadata=metadata).to_html(filename)
        print(f"\nReport saved to: {filename}")


def cmd_compare(args):
    """Compare multiple strategies."""
    from backtest.backtester import BacktestConfig
    from backtest.comparator import Comparator
    from backtest.data import DataLoader, PriceData
    from backtest.metadata import (
        build_run_metadata,
        build_strategy_info,
        build_data_info,
    )

    print("\nLoading strategies...")

    strategies = []
    strategy_files = {}  # strategy.name -> file path

    for path in args.strategies:
        try:
            strategy = load_strategy_from_file(path)
            strategies.append(strategy)
            strategy_files[strategy.name] = path
            print(f"  + {strategy.name}")
        except Exception as e:
            print(f"  - {path}: {e}")

    if not strategies:
        print("No valid strategies loaded")
        sys.exit(1)

    # Load PriceData per strategy and collect date ranges
    print("\nLoading price data per strategy...")
    strategy_data: dict = {}  # strategy.name -> PriceData
    date_info: list = []  # [(strategy_name, start_date, end_date), ...]
    validate = not getattr(args, 'no_validate', False)
    execution_lag_days = _resolve_execution_lag(args)
    max_volume_participation, min_daily_dollar_volume, liquidity_on_missing_volume = _resolve_liquidity_settings(args)
    load_volumes = _needs_volume_data(max_volume_participation, min_daily_dollar_volume)
    exposure_policy = _resolve_exposure_policy(args)

    for strategy in strategies:
        try:
            data = DataLoader.yahoo(
                tickers=_extend_assets_for_exposure_policy(strategy.assets, exposure_policy),
                start=args.start,
                end=args.end,
                currency="EUR",
                align=getattr(args, 'align', 'ffill'),
                skip_failed=getattr(args, 'skip_failed', True),
                validate=validate,
            )
            strategy_data[strategy.name] = data
            date_info.append((strategy.name, data.start_date, data.end_date))
            print(f"  {strategy.name}: {data.start_date.strftime('%Y-%m-%d')} -> {data.end_date.strftime('%Y-%m-%d')} ({len(data.prices)} days)")
        except Exception as e:
            print(f"  Error loading data for {strategy.name}: {e}")
            sys.exit(1)

    # Validate date alignment across all strategies
    start_dates = set(d[1].date() for d in date_info)
    end_dates = set(d[2].date() for d in date_info)

    dates_aligned = len(start_dates) == 1 and len(end_dates) == 1

    if not dates_aligned:
        print("\n" + "=" * 60)
        print("WARNING: Date ranges differ across strategies!")
        print("=" * 60)
        print("\nStrategy date ranges:")
        for name, start, end in date_info:
            print(f"  {name}:")
            print(f"    Start: {start.strftime('%Y-%m-%d')}")
            print(f"    End:   {end.strftime('%Y-%m-%d')}")

        # Calculate intersection
        common_start = max(d[1] for d in date_info)
        common_end = min(d[2] for d in date_info)

        if common_start >= common_end:
            print("\nERROR: No overlapping date range found. Cannot compare strategies.")
            sys.exit(1)

        print(f"\nCommon date range: {common_start.strftime('%Y-%m-%d')} -> {common_end.strftime('%Y-%m-%d')}")

        if not getattr(args, "allow_misaligned", False):
            print("\nTo compare with trimmed intersection, use --allow-misaligned")
            print("Aborting comparison.")
            sys.exit(1)

        print("\n--allow-misaligned set: trimming all data to common range...")

    # Collect all assets and load unified data for comparison
    all_assets = set()
    for strategy in strategies:
        all_assets.update(strategy.assets)
    all_assets.update(_extend_assets_for_exposure_policy([], exposure_policy))

    # Add benchmark ticker
    from backtest.assets import get_benchmark_ticker
    if args.benchmark_ticker:
        bench_ticker = args.benchmark_ticker
    else:
        bench_ticker = get_benchmark_ticker(args.benchmark)
    if bench_ticker not in all_assets:
        all_assets.add(bench_ticker)

    # Determine final date range (intersection if misaligned)
    if dates_aligned:
        final_start = args.start
        final_end = args.end
    else:
        final_start = max(d[1] for d in date_info).strftime('%Y-%m-%d')
        final_end = min(d[2] for d in date_info).strftime('%Y-%m-%d')

    print(f"\nLoading unified price data for {len(all_assets)} assets...")
    drip_enabled = getattr(args, 'drip', False)
    tax_enabled = not args.no_tax
    try:
        data = DataLoader.yahoo(
            tickers=list(all_assets),
            start=final_start,
            end=final_end,
            currency="EUR",
            align=getattr(args, 'align', 'ffill'),
            skip_failed=getattr(args, 'skip_failed', True),
            load_dividends=drip_enabled or tax_enabled,
            load_volumes=load_volumes,
            validate=validate,
        )
        print(f"Loaded {len(data.prices)} days of data")
        print(f"Date range: {data.start_date.strftime('%Y-%m-%d')} -> {data.end_date.strftime('%Y-%m-%d')}\n")
        if data.dividends is not None:
            div_count = (data.dividends > 0).sum().sum()
            print(f"  Loaded {div_count} dividend events")
    except Exception as e:
        print(f"Error loading unified data: {e}")
        sys.exit(1)

    print("Running backtests...")
    cost_profile = _load_cost_profile(getattr(args, "cost_profile_file", None))
    risk_overlay = _resolve_risk_overlay(args)

    # Determine metric basis
    if args.liquidate_at_end:
        metric_basis = "net_liquidation"
    elif args.metric_basis:
        metric_basis = args.metric_basis
    else:
        # Default: net_liquidation if tax enabled, gross otherwise
        metric_basis = "net_liquidation" if tax_enabled else "gross"

    drip_enabled = getattr(args, 'drip', False)
    config = BacktestConfig(
        initial_capital=args.capital,
        costs_pct=args.costs,
        cost_profile=cost_profile,
        execution_lag_days=execution_lag_days,
        max_volume_participation=max_volume_participation,
        min_daily_dollar_volume=min_daily_dollar_volume,
        liquidity_on_missing_volume=liquidity_on_missing_volume,
        exposure_policy=exposure_policy,
        risk_overlay=risk_overlay,
        rebalance_frequency=args.rebalance_frequency,
        benchmark=args.benchmark,
        tax_enabled=tax_enabled,
        tax_exemption_amount=args.tax_exemption,
        cash_rate=args.cash_rate,
        metric_basis=metric_basis,
        allow_universe_lookahead=getattr(args, 'allow_universe_lookahead', False),
        validate=not getattr(args, 'no_validate', False),
        drip_enabled=drip_enabled,
        external_features_loader=_build_external_features_loader(args),
    )
    comparator = Comparator(strategies, data, config, benchmark_ticker=bench_ticker)
    result = comparator.run()

    print(result.summary())

    # Build metadata for the comparison report
    compare_strategy_infos = [
        build_strategy_info(s, strategy_files.get(s.name))
        for s in strategies
    ]
    data_info = build_data_info(data, requested_assets=list(all_assets))
    metadata = build_run_metadata(
        config=config,
        data_info=data_info,
        mode="compare",
        compare_strategies=compare_strategy_infos,
        cli_rebalance_frequency=args.rebalance_frequency,
        cli_start=final_start,
        cli_end=final_end,
    )

    # Attach metadata to result for HTML generation
    result.metadata = metadata

    if args.output:
        result.to_html(args.output)
        output = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/comparison_{timestamp}.html"
        Path("reports").mkdir(exist_ok=True)
        result.to_html(filename)
        output = filename

    print(f"\nReport saved to: {output}")


def cmd_meta_promotion(args):
    """Create a governance artifact for strategy-promotion reviews."""
    from backtest.meta_promotion import (
        DEFAULT_BROKERS,
        DEFAULT_STRATEGIES,
        SOXL_PROXY_BASELINE,
        SOXL_PROXY_STRATEGIES,
        run_meta_promotion_report,
    )

    strategy_paths = args.strategies or list(DEFAULT_STRATEGIES)
    baseline = args.baseline
    research_proxy_mode = "live"
    if args.soxl_proxy:
        research_proxy_mode = "soxl_proxy"
        if not args.strategies:
            strategy_paths = list(SOXL_PROXY_STRATEGIES)
        if baseline == "strategies/levered_etf_momentum_sticky.py":
            baseline = SOXL_PROXY_BASELINE

    brokers = args.brokers or list(DEFAULT_BROKERS)
    payload = run_meta_promotion_report(
        strategy_paths=strategy_paths,
        baseline_path=baseline,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        initial_capital=args.capital,
        costs_pct=args.costs,
        metric_basis=args.metric_basis,
        tax_enabled=not args.no_tax,
        brokers=brokers,
        skip_failed=args.skip_failed,
        align=args.align,
        validate=not args.no_validate,
        research_proxy_mode=research_proxy_mode,
        tail_risk_gate_basis=args.tail_risk_gate_basis,
    )

    print("\nMeta-promotion report created")
    print(f"  Artifact ID: {payload['artifact_id']}")
    print(f"  JSON:        {payload['paths']['json']}")
    print(f"  Markdown:    {payload['paths']['markdown']}")
    print("\nStrategies:")
    for row in payload["strategies"]:
        metrics = row["metrics"]
        daily_maxdd = metrics.get("max_drawdown_daily")
        daily_maxdd = metrics["max_drawdown"] if daily_maxdd is None else daily_maxdd
        print(
            f"  - {metrics['strategy_name']} | role={row['role']} | "
            f"freq={metrics['rebalance_frequency']} | "
            f"CAGR={metrics['cagr']:.1%} | "
            f"MaxDD(reb)={metrics['max_drawdown']:.1%} | "
            f"MaxDD(daily)={daily_maxdd:.1%}"
        )


def cmd_metrics(args):
    """Display metrics for a strategy."""
    from backtest.backtester import Backtester, BacktestConfig
    from backtest.data import DataLoader

    strategy = load_strategy_from_file(args.strategy)

    data = DataLoader.yahoo(
        tickers=strategy.assets,
        start=args.start,
        end=args.end,
        currency="EUR",
        align="ffill",  # Default to ffill for metrics
    )

    config = BacktestConfig()
    result = Backtester(strategy, data, config).run()

    m = result.metrics
    print(f"\n{strategy.name} Metrics")
    print("=" * 40)
    print(f"{'CAGR':<25} {m.cagr:>12.2%}")
    print(f"{'Volatility':<25} {m.volatility:>12.2%}")
    print(f"{'Sharpe Ratio':<25} {m.sharpe_ratio:>12.2f}")
    print(f"{'Sortino Ratio':<25} {m.sortino_ratio:>12.2f}")
    print(f"{'Max Drawdown':<25} {m.max_drawdown:>12.2%}")
    print(f"{'Calmar Ratio':<25} {m.calmar_ratio:>12.2f}")
    print(f"{'Win Rate (Monthly)':<25} {m.win_rate_monthly:>12.2%}")
    print(f"{'Total Return':<25} {m.total_return:>12.2%}")
    print(f"{'Number of Trades':<25} {m.num_trades:>12}")
    print(f"{'Total Costs':<25} {'€':>1}{m.total_costs:>11,.2f}")
    print()


def cmd_assets(args):
    """List available assets."""
    from backtest.assets import list_assets
    print("\nAvailable Assets:\n")
    print(list_assets())
    print()


def cmd_new(args):
    """Generate a new strategy template."""
    name = args.name
    template = f'''"""
{name} Strategy

Description of your strategy here.
"""

from datetime import date

import pandas as pd

from backtest.strategy import Strategy, Allocation


class {name.replace(" ", "")}(Strategy):
    """
    {name} strategy implementation.

    Add a detailed description of how the strategy works.
    """

    name = "{name}"

    def __init__(self):
        """Initialize the strategy."""
        self.params = {{}}
        self.assets = ["SPY", "BND"]  # Define required assets

    def signal(self, date: date, data: pd.DataFrame) -> Allocation:
        """
        Generate allocation signal.

        Args:
            date: Current date
            data: Historical price data

        Returns:
            Target allocation
        """
        # Implement your strategy logic here
        return Allocation({{"SPY": 0.6, "BND": 0.4}})
'''

    filename = f"strategies/{name.lower().replace(' ', '_')}.py"
    Path("strategies").mkdir(exist_ok=True)

    if Path(filename).exists():
        print(f"Error: File already exists: {filename}")
        sys.exit(1)

    Path(filename).write_text(template)
    print(f"Created strategy template: {filename}")


def cmd_data_download(args):
    """Download and cache price data."""
    from backtest.data import DataLoader

    print(f"\nDownloading data for {len(args.tickers)} tickers...")

    try:
        data = DataLoader.yahoo(
            tickers=args.tickers,
            start=args.start,
            end=args.end,
            cache=True
        )
        print(f"Downloaded {len(data.prices)} days of data")
        print(f"Date range: {data.start_date.strftime('%Y-%m-%d')} to {data.end_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_data_list(args):
    """List cached data files."""
    from backtest.data import DataLoader

    files = DataLoader.list_cache()
    if not files:
        print("No cached data files")
        return

    print(f"\nCached data files ({len(files)}):")
    for f in sorted(files):
        print(f"  {f}")


def cmd_data_clear(args):
    """Clear all cached data."""
    from backtest.data import DataLoader

    count = DataLoader.clear_cache()
    print(f"Cleared {count} cached files")


def cmd_data_provenance_add(args):
    """Register manual data file provenance metadata."""
    from backtest.provenance import ManualDataProvenanceRegistry

    source = args.source
    import_method = args.import_method
    license_note = args.license_note

    if args.seekingalpha:
        if not source:
            source = "SeekingAlpha"
        if import_method == "manual_upload":
            import_method = "manual_csv_export"
        if not license_note:
            license_note = (
                "Manual export from personal SeekingAlpha access; "
                "use only in compliance with SeekingAlpha ToS."
            )

    registry = ManualDataProvenanceRegistry(path=args.registry)

    try:
        entry = registry.register_entry(
            file_path=args.file_path,
            dataset=args.dataset,
            source=source,
            quality_tag=args.quality_tag,
            as_of_date=args.as_of_date,
            import_method=import_method,
            license_tos_note=license_note or "",
            source_url=args.source_url,
            notes=args.notes,
            entry_id=args.entry_id,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(entry.to_dict(), indent=2))
        return

    print("\nManual data provenance registered:")
    print(f"  Entry ID:     {entry.entry_id}")
    print(f"  Dataset:      {entry.dataset}")
    print(f"  File:         {entry.file_path}")
    print(f"  Source:       {entry.source}")
    print(f"  Quality tag:  {entry.quality_tag}")
    if entry.as_of_date:
        print(f"  As-of date:   {entry.as_of_date}")
    print(f"  Import method:{entry.import_method}")
    if entry.license_tos_note:
        print(f"  License/ToS:  {entry.license_tos_note}")
    if entry.source_url:
        print(f"  Source URL:   {entry.source_url}")
    print(f"  Imported at:  {entry.imported_at}")
    print(f"  SHA256:       {entry.checksum_sha256}")
    print(f"  File size:    {entry.file_size_bytes} bytes")
    if entry.row_count is not None and entry.column_count is not None:
        print(f"  Shape:        {entry.row_count} rows x {entry.column_count} cols")
    if entry.notes:
        print(f"  Notes:        {entry.notes}")


def cmd_data_provenance_list(args):
    """List registered manual data provenance entries."""
    from backtest.provenance import ManualDataProvenanceRegistry

    registry = ManualDataProvenanceRegistry(path=args.registry)
    try:
        entries = registry.list_entries(dataset=args.dataset, source=args.source)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    rows = [entry.to_dict() for entry in entries]
    if args.format == "json":
        print(json.dumps({"total": len(rows), "entries": rows}, indent=2))
        return

    if not rows:
        print("No manual provenance entries found.")
        return

    print(f"\nManual provenance entries ({len(rows)}):")
    print("  " + "-" * 108)
    print(
        f"  {'Entry ID':<30} {'Dataset':<16} {'Source':<14} "
        f"{'Tag':<10} {'As-of':<10} {'Imported At':<20}"
    )
    print("  " + "-" * 108)
    for row in rows:
        print(
            f"  {row['entry_id'][:30]:<30} "
            f"{str(row['dataset'])[:16]:<16} "
            f"{str(row['source'])[:14]:<14} "
            f"{str(row['quality_tag'])[:10]:<10} "
            f"{str(row['as_of_date'] or '-')[:10]:<10} "
            f"{str(row['imported_at'])[:20]:<20}"
        )


def cmd_data_provenance_show(args):
    """Show a single manual data provenance entry."""
    from backtest.provenance import ManualDataProvenanceRegistry

    registry = ManualDataProvenanceRegistry(path=args.registry)
    try:
        entry = registry.get_entry(args.entry_id)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if entry is None:
        print(f"Error: entry not found: {args.entry_id}")
        sys.exit(1)

    payload = entry.to_dict()
    if args.format == "json":
        print(json.dumps(payload, indent=2))
        return

    print("\nManual provenance entry:")
    for key in [
        "entry_id",
        "dataset",
        "file_path",
        "source",
        "quality_tag",
        "as_of_date",
        "import_method",
        "license_tos_note",
        "source_url",
        "imported_at",
        "checksum_sha256",
        "file_size_bytes",
        "row_count",
        "column_count",
        "notes",
    ]:
        print(f"  {key}: {payload.get(key)}")


def cmd_data_provenance_verify(args):
    """Verify registered manual provenance entries against files/checksums."""
    from backtest.provenance import ManualDataProvenanceRegistry

    registry = ManualDataProvenanceRegistry(path=args.registry)
    try:
        result = registry.verify_entries(check_hash=not args.skip_hash)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(result, indent=2))
        return

    print("\nManual provenance verification:")
    print(f"  Registry:     {result['registry_path']}")
    print(f"  Total entries:{result['total_entries']}")
    print(f"  OK entries:   {result['ok_entries']}")
    print(f"  Issues:       {result['issue_count']}")
    if result["issues"]:
        print("\nIssues:")
        for issue in result["issues"]:
            print(f"  - {issue['entry_id']}: {issue['status']} ({issue['message']})")


def cmd_features_pull(args):
    """Pull a snapshot from the configured external features adapter."""
    from datetime import date

    from backtest.external_features import (
        SNAPSHOT_DIR,
        datasets_allowing_empty_tickers,
        get_adapter,
    )
    from backtest.provenance import ManualDataProvenanceRegistry

    dataset = args.dataset
    try:
        adapter = get_adapter(dataset)
    except KeyError as e:
        print(f"Error: {e}")
        sys.exit(1)

    tickers_raw = getattr(args, "tickers", None)
    if tickers_raw:
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    else:
        tickers = []
    if not tickers and dataset not in set(datasets_allowing_empty_tickers()):
        print(
            f"Error: dataset '{dataset}' requires --tickers AAA,BBB."
        )
        sys.exit(2)

    try:
        as_of = date.fromisoformat(args.as_of)
    except ValueError:
        print(f"Error: invalid --as-of date '{args.as_of}' (expected YYYY-MM-DD)")
        sys.exit(2)

    # Phase C: optional news-engine + intraday-cutoff. We swap the
    # adapter for a freshly-constructed instance carrying the desired
    # engine + cutoff so the registered prototype stays unchanged.
    engine_name = getattr(args, "news_engine", None)
    cutoff = getattr(args, "news_intraday_cutoff", None)
    adapter_to_use = adapter
    if engine_name or cutoff:
        adapter_to_use = _build_news_adapter_override(adapter, engine_name, cutoff)

    # Phase D: pull-time ML options (Codex D5/D20). Only apply when the
    # adapter advertises the with_options() factory.
    ml_bundle = getattr(args, "ml_model_bundle", None)
    ml_stacking_only = bool(getattr(args, "ml_stacking_only", False))
    if ml_bundle or ml_stacking_only:
        ml_kwargs = {}
        if ml_bundle:
            ml_kwargs["bundle_override"] = Path(ml_bundle).expanduser()
        if ml_stacking_only:
            ml_kwargs["stacking_only"] = True
        try:
            adapter_to_use = adapter_to_use.with_options(**ml_kwargs)
        except TypeError:
            # Adapter does not accept the requested options; leave it alone.
            pass

    root = args.root if getattr(args, "root", None) else SNAPSHOT_DIR
    registry = ManualDataProvenanceRegistry(path=getattr(args, "registry", None))
    pull_kwargs = dict(
        registry=registry,
        root=root,
        force=getattr(args, "force", False),
    )
    if cutoff:
        pull_kwargs["cutoff_ts_utc"] = cutoff
    path = adapter_to_use.pull_snapshot(tickers, as_of, **pull_kwargs)

    if args.format == "json":
        print(json.dumps({"snapshot_path": str(path), "dataset": dataset}, indent=2))
        return
    print(f"Snapshot written: {path}")


def cmd_ml_train(args):
    """Phase D `backtest ml train` (T-0227).

    Resolves the training universe (Codex D4) — hard-fails when neither
    --tickers nor --universe-source is set — and hands off to the
    walk-forward trainer that writes one ``manifest.json`` per outer
    window plus stage pickles + imputer state + zscore stats.
    """

    from datetime import date as _date

    from backtest.data import DataLoader
    from backtest.external_features.ml.config import MLTrainingConfig
    from backtest.external_features.ml.training import run_walk_forward_training
    from backtest.provenance import ManualDataProvenanceRegistry

    if args.ml_command != "train":
        print("Error: only 'backtest ml train' is supported.")
        sys.exit(2)

    try:
        start = _date.fromisoformat(args.start)
        end = _date.fromisoformat(args.end)
    except ValueError as exc:
        print(f"Error: invalid date — {exc}")
        sys.exit(2)
    if end <= start:
        print("Error: --end must be after --start")
        sys.exit(2)

    tickers_raw = (args.tickers or "").strip()
    universe_source = args.universe_source
    if not tickers_raw and not universe_source:
        print(
            "Error: ML training requires --tickers AAA,BBB ODER "
            "--universe-source PATH (Codex D4 — Survivorship-Schutz)."
        )
        sys.exit(2)

    tickers: tuple[str, ...] = ()
    if tickers_raw:
        tickers = tuple(
            t.strip().upper() for t in tickers_raw.split(",") if t.strip()
        )
    elif universe_source:
        universe_path = Path(universe_source).expanduser()
        if not universe_path.exists():
            print(f"Error: --universe-source not found: {universe_path}")
            sys.exit(2)
        rows: list[str] = []
        for raw in universe_path.read_text(encoding="utf-8").splitlines():
            value = raw.strip().upper()
            if value and not value.startswith("#"):
                rows.append(value.split(",")[0].strip())
        tickers = tuple(dict.fromkeys(rows))

    try:
        horizons = tuple(int(h) for h in str(args.horizons).split(",") if h.strip())
    except ValueError:
        print("Error: --horizons must be comma-separated integers (e.g. 21,63,252)")
        sys.exit(2)
    families = tuple(
        f.strip().lower() for f in str(args.models).split(",") if f.strip()
    )

    config = MLTrainingConfig(
        horizons=horizons,
        model_families=families,
        inner_train_years=float(args.inner_train_years),
        inner_test_months=int(args.inner_test_months),
        grid_size=int(args.grid_size),
        seed=int(args.seed),
        tickers=tickers,
        universe_source=universe_source,
    )

    print(
        f"Loading prices for {len(tickers)} tickers between "
        f"{start.isoformat()} and {end.isoformat()}..."
    )
    price_data = DataLoader.yahoo(
        tickers=list(tickers),
        start=start.isoformat(),
        end=end.isoformat(),
        currency="EUR",
        align="ffill",
        skip_failed=True,
    )
    # DataLoader.yahoo returns PriceData; .prices is the actual
    # DataFrame.
    prices = (
        getattr(price_data, "prices", price_data)
        if price_data is not None
        else None
    )
    if prices is None or prices.empty:
        print("Error: no price data available for training universe.")
        sys.exit(1)

    registry = ManualDataProvenanceRegistry()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_walk_forward_training(
        prices=prices,
        config=config,
        output_dir=output_dir,
        registry=registry,
    )
    print(
        f"Training done: {len(result.manifest_paths)} bundle manifests "
        f"written under {output_dir}."
    )
    for manifest_path in result.manifest_paths:
        print(f"  - {manifest_path}")


def _build_execution_plan_adapter(broker: str):
    """Phase E2 (T-0367..T-0369): instantiates the matching
    ExecutionPlanAdapter. NOT a real broker submit (Codex R2.2)."""

    from backtest.live.alpaca_paper_adapter import AlpacaPaperAdapter
    from backtest.live.dry_run_adapter import DryRunAdapter
    from backtest.live.ibkr_basket_csv_adapter import IBKRBasketCsvAdapter
    from backtest.live.maxblue_brief import MaxblueBriefAdapter
    from backtest.live.trade_republic_brief import TradeRepublicBriefAdapter

    if broker == "dry_run":
        return DryRunAdapter()
    if broker == "ibkr_basket_csv":
        return IBKRBasketCsvAdapter()
    if broker == "alpaca_paper_preview":
        return AlpacaPaperAdapter()
    if broker == "trade_republic_brief":
        return TradeRepublicBriefAdapter()
    if broker == "maxblue_brief":
        return MaxblueBriefAdapter()
    raise ValueError(f"unknown broker '{broker}'")


def _orders_from_signal_report(
    report: dict,
    *,
    run_id: str,
    strategy_hash: str,
    portfolio_snapshot_hash: str,
    broker_label: str,
    signals_as_of_iso: str,
):
    """Converts SignalReport.actionable_orders -> order list."""

    from backtest.live.orders import Order, stable_order_plan_id

    orders_in = []
    # Preferred source: the `orders` block from the SignalReport
    # (contains only actionable orders with shares_delta != None/0).
    raw_orders = report.get("orders") or []
    for entry in raw_orders:
        ticker = entry.get("ticker")
        action = entry.get("action")
        shares_delta = entry.get("shares_delta")
        if not ticker or action not in {"BUY", "SELL"}:
            continue
        if shares_delta is None or float(shares_delta) == 0:
            continue
        orders_in.append(
            {
                "ticker": ticker,
                "action": action,
                "shares_delta": float(shares_delta),
                "value_delta": float(entry.get("value_delta") or 0),
            }
        )
    # Fallback 1: legacy `signals`-Block.
    if not orders_in:
        for s in report.get("signals") or []:
            if s.get("shares_delta") is None or float(s.get("shares_delta", 0)) == 0:
                continue
            orders_in.append(
                {
                    "ticker": s.get("ticker"),
                    "action": s.get("action"),
                    "shares_delta": float(s.get("shares_delta") or 0),
                    "value_delta": float(s.get("value_delta") or 0),
                }
            )
    # Fallback 2: `summary.orders.actionable_list` (selten).
    if not orders_in:
        summary = report.get("summary") or {}
        orders_section = summary.get("orders") or {}
        for entry in orders_section.get("actionable_list", []) or []:
            orders_in.append(entry)
    out = []
    for raw in orders_in:
        ticker = str(raw.get("ticker") or "").upper()
        action = str(raw.get("action") or "").upper()
        if not ticker or action not in {"BUY", "SELL"}:
            continue
        target_shares = float(raw.get("shares_delta") or 0)
        target_value = float(raw.get("value_delta") or 0)
        spi = stable_order_plan_id(
            run_id=run_id,
            strategy_hash=strategy_hash,
            portfolio_snapshot_hash=portfolio_snapshot_hash,
            broker_label=broker_label,
            ticker=ticker,
            action=action,
            target_shares=target_shares,
        )
        out.append(
            Order(
                ticker=ticker,
                action=action,
                target_shares=target_shares,
                target_value=target_value,
                broker_label=broker_label,
                stable_order_plan_id=spi,
                run_id=run_id,
                strategy_hash=strategy_hash,
                portfolio_snapshot_hash=portfolio_snapshot_hash,
                signals_as_of_iso=signals_as_of_iso,
            )
        )
    return out


def cmd_live_plan(args):
    """Phase E2 (T-0372..T-0376)."""

    import hashlib

    from backtest.live.order_plan_log import OrderPlanLog
    from backtest.live.orders import compute_run_id
    from backtest.live.signal_report_io import (
        canonical_signal_report_hash,
        load_signal_report,
    )

    report = load_signal_report(args.signals_report)
    price_warnings = report.get("price_warnings") or []
    if price_warnings and not getattr(args, "allow_price_warnings", False):
        print(
            "Error: SignalReport contains price warnings; refusing to emit "
            "an order plan. Fix the portfolio quote/ticker data or rerun "
            "with --allow-price-warnings after manual verification."
        )
        for warning in price_warnings:
            print(f"  - {warning}")
        sys.exit(2)

    signal_report_hash = canonical_signal_report_hash(report)
    signals_as_of_iso = str(
        report.get("signals_as_of_iso")
        or report.get("as_of")
        or ""
    )

    # Portfolio-Hash-Fallback (Codex R3.4).
    portfolio_snapshot_hash = None
    if args.portfolio:
        portfolio_text = Path(args.portfolio).expanduser().read_text(encoding="utf-8")
        portfolio_snapshot_hash = hashlib.sha256(
            portfolio_text.encode("utf-8")
        ).hexdigest()[:16]
    else:
        portfolio_snapshot_hash = report.get("portfolio_snapshot_hash")
    if not portfolio_snapshot_hash:
        print(
            "Error: portfolio_snapshot_hash is required for live plan "
            "idempotency. Provide --portfolio PATH or store "
            "portfolio_snapshot_hash in the SignalReport JSON."
        )
        sys.exit(2)

    strategy_path = str(report.get("strategy_path") or report.get("strategy_name") or "")
    strategy_hash = hashlib.sha256(strategy_path.encode("utf-8")).hexdigest()[:16]

    run_id = compute_run_id(
        signal_report_hash=signal_report_hash,
        broker_label=args.broker,
        portfolio_snapshot_hash=portfolio_snapshot_hash,
        new_run_token=args.new_run or "",
    )

    orders = _orders_from_signal_report(
        report,
        run_id=run_id,
        strategy_hash=strategy_hash,
        portfolio_snapshot_hash=portfolio_snapshot_hash,
        broker_label=args.broker,
        signals_as_of_iso=signals_as_of_iso,
    )

    adapter = _build_execution_plan_adapter(args.broker)
    log = OrderPlanLog(args.log_path)
    receipts = adapter.emit_order_plan(orders, log=log)

    print(
        f"Live plan emitted: {len(receipts)} receipt(s); broker="
        f"{args.broker}; run_id={run_id}; plan_only=True (Phase E)."
    )
    for r in receipts:
        print(f"  - {r.stable_order_plan_id} {r.ticker} {r.action} -> {r.status}")


def cmd_live_status(args):
    """Phase E2 (T-0375)."""

    from backtest.live.order_plan_log import OrderPlanLog

    log = OrderPlanLog(args.log_path)
    rows = log.read_all()
    since = args.since
    if since:
        rows = [r for r in rows if r.get("emitted_at_iso", "").startswith(since[:10]) or r.get("emitted_at_iso", "") >= since]
    print(f"OrderPlanLog: {len(rows)} entries.")
    for r in rows:
        print(
            f"  {r.get('emitted_at_iso')} {r.get('broker_label')} "
            f"{r.get('ticker')} {r.get('action')} -> {r.get('status')}"
        )


def cmd_live_reconcile(args):
    """Phase E2 (T-0372). Phase E has NO submit; the reconcile path is
    just a simple consistency check of the portfolio JSON file."""

    portfolio_path = Path(args.portfolio).expanduser()
    if not portfolio_path.exists():
        print(f"Error: portfolio JSON not found: {portfolio_path}")
        sys.exit(2)
    text = portfolio_path.read_text(encoding="utf-8")
    payload = json.loads(text)
    positions = payload.get("positions") or {}
    cash = float(payload.get("cash") or 0.0)
    currency = str(payload.get("currency") or "EUR")
    print(
        f"Reconcile (broker={args.broker}, plan_only=True): "
        f"{len(positions)} positions, cash={cash:.2f} {currency}."
    )


def _parse_position_update(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        raise ValueError(f"Invalid position update '{raw}'. Use TICKER=SHARES.")
    ticker, shares_raw = raw.split("=", 1)
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError(f"Invalid position update '{raw}': empty ticker.")
    try:
        shares = float(shares_raw.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid shares for {ticker}: {shares_raw!r}") from exc
    if shares < 0:
        raise ValueError(f"Invalid shares for {ticker}: must be >= 0.")
    return ticker, shares


def _target_shares_from_signal_report(path: str) -> dict[str, float]:
    report = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    updates: dict[str, float] = {}
    for row in report.get("orders") or []:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        target = row.get("target_shares")
        if target is None and row.get("current_shares") is not None and row.get("shares_delta") is not None:
            target = float(row.get("current_shares") or 0.0) + float(row.get("shares_delta") or 0.0)
        if target is None:
            continue
        updates[ticker] = max(0.0, float(target))
    return updates


def _write_portfolio_updates(
    portfolio_path: Path,
    updates: dict[str, float],
    *,
    stand: str,
) -> None:
    payload = json.loads(portfolio_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Portfolio JSON must be an object")

    if isinstance(payload.get("positionen"), list):
        seen: set[str] = set()
        for row in payload["positionen"]:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("price_ticker") or row.get("ticker") or "").strip().upper()
            if ticker in updates:
                row["shares"] = updates[ticker]
                seen.add(ticker)
        for ticker, shares in updates.items():
            if ticker not in seen:
                payload["positionen"].append(
                    {
                        "name": ticker,
                        "price_ticker": ticker,
                        "waehrung": "EUR",
                        "shares": shares,
                        "rolle": "manuell_ergaenzt",
                        "ticker_verify": True,
                    }
                )
    elif isinstance(payload.get("positions"), dict):
        for ticker, shares in updates.items():
            existing = payload["positions"].get(ticker)
            if isinstance(existing, dict):
                existing["shares"] = shares
                existing.setdefault("price_ticker", ticker)
            else:
                payload["positions"][ticker] = shares
    else:
        payload["positionen"] = [
            {
                "name": ticker,
                "price_ticker": ticker,
                "waehrung": "EUR",
                "shares": shares,
                "rolle": "manuell_ergaenzt",
                "ticker_verify": True,
            }
            for ticker, shares in sorted(updates.items())
        ]

    payload["stand"] = stand
    portfolio_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def cmd_live_update_portfolio(args):
    """Update manual broker portfolio JSON after a real manual execution."""

    from datetime import date

    portfolio_path = Path(args.portfolio).expanduser()
    if not portfolio_path.exists():
        print(f"Error: portfolio JSON not found: {portfolio_path}")
        sys.exit(2)

    updates: dict[str, float] = {}
    if args.signals_report:
        try:
            updates.update(_target_shares_from_signal_report(args.signals_report))
        except Exception as e:
            print(f"Error reading signals report: {e}")
            sys.exit(2)

    try:
        for raw in args.position or []:
            ticker, shares = _parse_position_update(raw)
            updates[ticker] = shares
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(2)

    if not updates:
        print("Error: provide --position TICKER=SHARES or --signals-report PATH.")
        sys.exit(2)

    stand = args.stand or date.today().isoformat()
    try:
        _write_portfolio_updates(portfolio_path, updates, stand=stand)
    except Exception as e:
        print(f"Error updating portfolio JSON: {e}")
        sys.exit(2)

    print(f"Portfolio updated: {portfolio_path}")
    print(f"Stand: {stand}")
    for ticker, shares in sorted(updates.items()):
        print(f"  {ticker}: {shares:g} shares")


def _build_news_adapter_override(adapter, engine_name, cutoff):
    """Phase C: re-instantiate the news adapter with pull-time options."""

    from backtest.external_features.sentiment import get_sentiment_engine

    engine = get_sentiment_engine(engine_name) if engine_name else None
    klass = type(adapter)
    try:
        return klass(engine=engine, cutoff_override=cutoff)
    except TypeError:
        # Adapters without an engine kwarg (e.g. analyst adapters) fall
        # back to a no-op override: use the original.
        return adapter


def cmd_features_list(args):
    """List external feature snapshot files plus their registry entries."""
    from backtest.external_features import SNAPSHOT_DIR, iter_snapshot_files
    from backtest.provenance import ManualDataProvenanceRegistry

    root = args.root if getattr(args, "root", None) else SNAPSHOT_DIR
    paths = list(iter_snapshot_files(getattr(args, "dataset", None), root=root))
    registry = ManualDataProvenanceRegistry(path=getattr(args, "registry", None))
    entries = registry.list_entries(dataset=getattr(args, "dataset", None))
    entry_paths = {Path(entry.file_path).resolve(): entry for entry in entries}

    if args.format == "json":
        records = []
        for path in paths:
            resolved = path.resolve()
            entry = entry_paths.get(resolved)
            records.append(
                {
                    "snapshot_path": str(path),
                    "registered": entry is not None,
                    "entry_id": entry.entry_id if entry else None,
                    "source": entry.source if entry else None,
                }
            )
        print(json.dumps({"snapshots": records, "root": str(root)}, indent=2))
        return
    print(f"Snapshots under {root}:")
    if not paths:
        print("  (none)")
        return
    for path in paths:
        resolved = path.resolve()
        entry = entry_paths.get(resolved)
        marker = entry.entry_id if entry else "(no provenance)"
        print(f"  {path}  -- {marker}")


def cmd_features_verify(args):
    """Verify external feature snapshot schema and provenance hashes."""
    from backtest.external_features import (
        REQUIRED_COLUMNS,
        SNAPSHOT_DIR,
        iter_snapshot_files,
        read_snapshot_csv,
    )
    from backtest.external_features.adapters.base import sha256_of
    from backtest.provenance import ManualDataProvenanceRegistry

    root = args.root if getattr(args, "root", None) else SNAPSHOT_DIR
    registry = ManualDataProvenanceRegistry(path=getattr(args, "registry", None))
    entries = registry.list_entries(dataset=getattr(args, "dataset", None))
    entry_paths = {Path(entry.file_path).resolve(): entry for entry in entries}

    issues = []
    ok = 0
    paths = list(iter_snapshot_files(getattr(args, "dataset", None), root=root))
    for path in paths:
        resolved = path.resolve()
        try:
            df = read_snapshot_csv(path)
        except Exception as exc:
            issues.append({"path": str(path), "status": "unreadable", "message": str(exc)})
            continue
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            issues.append(
                {"path": str(path), "status": "schema_missing", "message": ", ".join(missing)}
            )
            continue
        entry = entry_paths.get(resolved)
        if entry is None:
            issues.append({"path": str(path), "status": "no_provenance", "message": "no registry entry"})
            continue
        current = sha256_of(path)
        if entry.checksum_sha256 and entry.checksum_sha256 != current:
            issues.append(
                {"path": str(path), "status": "checksum_mismatch", "message": "hash drift"}
            )
            continue
        ok += 1

    payload = {
        "root": str(root),
        "total": len(paths),
        "ok": ok,
        "issue_count": len(issues),
        "issues": issues,
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2))
        return
    print(f"\nExternal features verification:")
    print(f"  Root:         {root}")
    print(f"  Snapshots:    {len(paths)}")
    print(f"  OK:           {ok}")
    print(f"  Issues:       {len(issues)}")
    for issue in issues:
        print(f"  - {issue['path']}: {issue['status']} ({issue['message']})")


def _parse_param_assignments(param_items, flag_name: str = "--param"):
    """Parse repeated NAME=VALUE arguments into a dict."""
    parsed = {}
    if not param_items:
        return parsed
    for param_str in param_items:
        if "=" not in param_str:
            raise ValueError(f"Invalid {flag_name} format '{param_str}'. Use: {flag_name} name=value")
        name, value_str = param_str.split("=", 1)
        name = name.strip()
        value_str = value_str.strip()
        try:
            import ast
            parsed[name] = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            try:
                if "." in value_str:
                    parsed[name] = float(value_str)
                else:
                    parsed[name] = int(value_str)
            except ValueError:
                parsed[name] = value_str
    return parsed


def cmd_signals(args):
    """Generate live trading signals for a strategy."""
    from datetime import date
    from backtest.signals import (
        SignalGenerator,
        Portfolio,
        format_signal_report,
    )
    from backtest.meta_decision import run_meta_decision

    print(f"\nLoading strategy from {args.strategy}...")

    # Load and configure strategy with parameters
    strategy_path = Path(args.strategy)
    if not strategy_path.exists():
        print(f"Error: Strategy file not found: {strategy_path}")
        sys.exit(1)

    # Import strategy module
    spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_module"] = module
    spec.loader.exec_module(module)

    from backtest.strategy import Strategy

    # Find strategy class
    strategy_class = None
    strategy_instance = None

    # First, look for a pre-instantiated 'strategy' variable
    if hasattr(module, 'strategy'):
        obj = getattr(module, 'strategy')
        if isinstance(obj, Strategy):
            strategy_instance = obj
            strategy_class = type(obj)

    # If no instance, look for a Strategy class
    if strategy_class is None:
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                strategy_class = obj
                break

    if strategy_class is None:
        print("Error: No Strategy class found in file")
        sys.exit(1)

    try:
        param_overrides = _parse_param_assignments(args.params, flag_name="--param")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Instantiate strategy with parameters
    try:
        if param_overrides:
            # Create new instance with overridden parameters
            strategy = strategy_class(**param_overrides)
            print(f"Strategy: {strategy.display_name}")
            print(f"Parameters: {param_overrides}")
        elif strategy_instance:
            strategy = strategy_instance
            print(f"Strategy: {strategy.display_name}")
        else:
            strategy = strategy_class()
            print(f"Strategy: {strategy.display_name}")
    except Exception as e:
        print(f"Error instantiating strategy: {e}")
        sys.exit(1)

    # Override rebalance frequency if specified
    if args.rebalance_frequency:
        strategy.rebalance_frequency = args.rebalance_frequency
        print(f"Rebalance frequency: {args.rebalance_frequency}")

    # Load portfolio if specified
    portfolio = None
    if args.portfolio:
        portfolio_path = Path(args.portfolio)
        if not portfolio_path.exists():
            print(f"Error: Portfolio file not found: {portfolio_path}")
            sys.exit(1)
        try:
            portfolio = Portfolio.from_json(str(portfolio_path))
            print(f"Portfolio loaded: {len(portfolio.positions)} positions, {portfolio.cash:.2f} EUR cash")
        except Exception as e:
            print(f"Error loading portfolio: {e}")
            sys.exit(1)

    # Parse date
    as_of = None
    if args.date:
        try:
            as_of = date.fromisoformat(args.date)
        except ValueError:
            print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD.")
            sys.exit(1)

    print("\nGenerating signals...")

    try:
        exposure_policy = _resolve_exposure_policy(args)
        # Phase B: a single provider shared between SignalGenerator and
        # run_meta_decision so they see the same analyst snapshot.
        external_provider = _build_external_features_loader(args)
        analyst_dataset = getattr(args, "external_features_dataset", None) or None
        news_dataset_cli, _news_weight = _resolve_news_runtime_args(args)
        ml_dataset_cli, _ml_weight = _resolve_ml_runtime_args(args)
        _cp_require, _cp_threshold = _resolve_cross_product_args(args)
        generator = SignalGenerator(
            strategy,
            portfolio,
            skip_failed=getattr(args, 'skip_failed', True),
            drift_tolerance=getattr(args, 'drift_tolerance', 0.005),
            exposure_policy=exposure_policy,
            external_features_provider=external_provider,
            analyst_dataset=analyst_dataset,
            news_dataset=news_dataset_cli,
            ml_dataset=ml_dataset_cli,
        )
        report = generator.generate(as_of=as_of)
    except Exception as e:
        print(f"Error generating signals: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if getattr(args, "meta_enable", False):
        meta_candidates = []
        for path in getattr(args, "meta_candidates", []) or []:
            meta_candidates.append({"strategy": str(Path(path).expanduser().resolve()), "params": {}})
        meta_file = getattr(args, "meta_candidates_file", None)
        if meta_file:
            meta_file_path = Path(meta_file)
            if not meta_file_path.exists():
                print(f"Error: Meta candidates file not found: {meta_file_path}")
                sys.exit(1)
            try:
                payload = json.loads(meta_file_path.read_text())
            except Exception as e:
                print(f"Error: Invalid meta candidates file JSON: {e}")
                sys.exit(1)
            if not isinstance(payload, list):
                print("Error: Meta candidates file must contain a JSON list")
                sys.exit(1)
            for row in payload:
                if isinstance(row, str):
                    meta_candidates.append(
                        {"strategy": str(Path(row).expanduser().resolve()), "params": {}}
                    )
                elif isinstance(row, dict) and row.get("strategy"):
                    meta_candidates.append(
                        {
                            "strategy": str(Path(row["strategy"]).expanduser().resolve()),
                            "params": row.get("params", {}) or {},
                        }
                    )

        try:
            meta_result = run_meta_decision(
                as_of=as_of or report.as_of,
                current_strategy=str(strategy_path.resolve()),
                current_params=param_overrides,
                candidates=meta_candidates,
                portfolio=portfolio,
                skip_failed=getattr(args, "skip_failed", True),
                drift_tolerance=getattr(args, "drift_tolerance", 0.005),
                params_source=getattr(args, "meta_params_source", "preset_first"),
                preset_params_file=getattr(args, "meta_preset_file", None),
                scoring_mode=getattr(args, "meta_scoring", "hybrid"),
                confirm_points=getattr(args, "meta_confirm_points", 2),
                switch_margin=getattr(args, "meta_switch_margin", 0.10),
                decision_cadence=getattr(args, "meta_decision_cadence", "run_check_rebalance_switch"),
                plan_mode=getattr(args, "meta_plan_mode", "recommendation_with_portfolio_plan"),
                evidence_required=getattr(args, "meta_evidence_required", True),
                evidence_profile=getattr(args, "meta_evidence_profile", "ausgewogen"),
                evidence_compare_mode="vs_current",
                evidence_max_age_days=getattr(args, "meta_evidence_max_age_days", 30),
                evidence_artifact_path=getattr(args, "meta_evidence_artifact_path", None),
                gate_fail_action=getattr(args, "meta_gate_fail_action", "hold_current"),
                regime_mode=getattr(args, "meta_regime_mode", "strategy_fragility"),
                regime_profile=getattr(args, "meta_regime_profile", "ausgewogen"),
                alpha_tie_band=getattr(args, "meta_alpha_tie_band", None),
                stress_alpha_tolerance=getattr(args, "meta_stress_alpha_tolerance", None),
                conditioned_min_windows=getattr(args, "meta_conditioned_min_windows", None),
                external_features_provider=external_provider,
                news_score_weight=_news_weight,
                news_datasets=(news_dataset_cli,) if news_dataset_cli else (),
                ml_score_weight=_ml_weight,
                ml_datasets=(ml_dataset_cli,) if ml_dataset_cli else (),
                cross_product_require=_cp_require,
                cross_product_threshold=_cp_threshold,
            )
            report.meta_decision = meta_result
        except Exception as e:
            print(f"Error in meta decisioning: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Output
    if args.output:
        output_path = Path(args.output)
        if args.format == "json" or output_path.suffix == ".json":
            output_path.write_text(report.to_json())
            print(f"\nSignals saved to: {output_path}")
        else:
            # CSV format
            import csv
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "ticker", "action", "current_weight", "target_weight",
                    "weight_change", "momentum_score", "momentum_rank", "reason",
                    "order_action", "current_shares", "target_shares", "shares_delta",
                    "current_value", "target_value", "value_delta",
                    "drift_bps", "drift_in_tolerance",
                ])
                writer.writeheader()
                for signal in report.signals:
                    writer.writerow(signal.to_dict())
            print(f"\nSignals saved to: {output_path}")
            if report.meta_decision:
                meta_output = output_path.with_suffix(output_path.suffix + ".meta.json")
                meta_output.write_text(json.dumps(report.meta_decision, indent=2))
                print(f"Meta decision saved to: {meta_output}")
    else:
        # Print to console
        print()
        print(format_signal_report(report))


def cmd_meta_evidence(args):
    """Run historical OOS evidence analysis for a strategy switch pair."""
    from backtest.meta_evidence import run_meta_evidence_analysis
    import itertools
    from datetime import datetime, timezone

    try:
        current_params = _parse_param_assignments(
            getattr(args, "current_params", []), flag_name="--current-param"
        )
        target_params = _parse_param_assignments(
            getattr(args, "target_params", []), flag_name="--target-param"
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    custom_thresholds = {}
    try:
        for item in getattr(args, "custom_thresholds", []) or []:
            if "=" not in item:
                raise ValueError(f"Invalid --custom-threshold '{item}'. Use key=value")
            key, value = item.split("=", 1)
            custom_thresholds[key.strip()] = float(value.strip())
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        artifact = run_meta_evidence_analysis(
            current_strategy=args.current_strategy,
            target_strategy=args.target_strategy,
            current_params=current_params,
            target_params=target_params,
            as_of=getattr(args, "date", None),
            evidence_profile=args.evidence_profile,
            evidence_compare_mode="vs_current",
            evidence_max_age_days=args.evidence_max_age_days,
            evidence_artifact_path=getattr(args, "evidence_artifact_path", None),
            custom_thresholds=custom_thresholds or None,
            train_years=args.train_years,
            test_years=args.test_years,
            step_months=args.step_months,
            anchored=args.anchored,
            start_date=args.start,
            initial_capital=args.capital,
            costs_pct=args.costs,
            skip_failed=getattr(args, "skip_failed", True),
            metric_basis=args.metric_basis,
            save_artifact=True,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if getattr(args, "tuning_enabled", False):
        try:
            confirm_grid = sorted(
                {int(x.strip()) for x in str(args.grid_confirm_points).split(",") if x.strip()}
            )
            margin_grid = sorted(
                {float(x.strip()) for x in str(args.grid_switch_margin).split(",") if x.strip()}
            )
        except ValueError as e:
            print(f"Error: invalid tuning grid: {e}")
            sys.exit(1)

        combos = list(itertools.product(confirm_grid, margin_grid))
        total_combinations = len(combos)
        capped = combos[: max(1, int(args.max_combinations))]

        edge_pp = float(artifact.get("summary", {}).get("oos_cagr_edge_pp", 0.0) or 0.0)
        degradation_pct = float(artifact.get("summary", {}).get("oos_degradation_pct", 0.0) or 0.0)
        dd_delta_pp = float(artifact.get("summary", {}).get("oos_dd_delta_pp", 0.0) or 0.0)

        stage1 = []
        for confirm_points, switch_margin in capped:
            score = edge_pp - 0.35 * (confirm_points - 1) - 2.0 * switch_margin
            stage1.append(
                {
                    "confirm_points": int(confirm_points),
                    "switch_margin": float(switch_margin),
                    "stage1_score": float(score),
                }
            )
        stage1.sort(key=lambda row: row["stage1_score"], reverse=True)
        top_k = max(1, int(args.top_k))
        stage2 = []
        for row in stage1[:top_k]:
            stage2_score = (
                row["stage1_score"]
                - 0.02 * max(0.0, degradation_pct - 25.0)
                - 0.05 * max(0.0, dd_delta_pp - 5.0)
            )
            stage2.append({**row, "stage2_score": float(stage2_score)})
        stage2.sort(key=lambda row: row["stage2_score"], reverse=True)
        best = stage2[0] if stage2 else None

        artifact["tuning"] = {
            "enabled": True,
            "mode": "2-stage-smart",
            "total_combinations": total_combinations,
            "capped_combinations": len(capped),
            "max_combinations": int(args.max_combinations),
            "top_k": top_k,
            "stage1_top": stage1[:top_k],
            "stage2_results": stage2,
            "best_setup": best,
        }
        if best is not None:
            defaults_path = Path("results/meta_allocator_defaults.json").resolve()
            defaults_path.parent.mkdir(parents=True, exist_ok=True)
            defaults_payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_strategy": args.current_strategy,
                "target_strategy": args.target_strategy,
                "evidence_profile": args.evidence_profile,
                "evidence_max_age_days": args.evidence_max_age_days,
                "confirm_points": best["confirm_points"],
                "switch_margin": best["switch_margin"],
                "tuning_source_artifact_id": artifact.get("artifact_id"),
            }
            defaults_path.write_text(json.dumps(defaults_payload, indent=2))
            artifact["defaults_path"] = str(defaults_path)

    print("\nMeta Evidence")
    print("=" * 60)
    print(f"Current strategy: {artifact['current_strategy']['path']}")
    print(f"Target strategy:  {artifact['target_strategy']['path']}")
    print(f"As of:            {artifact['as_of']}")
    print(f"Profile:          {artifact['evidence_profile']}")
    print(f"Windows:          {artifact['summary']['num_windows']}")
    print(f"OOS CAGR edge:    {artifact['summary']['oos_cagr_edge_pp']:+.2f}pp")
    print(f"OOS hit-rate:     {artifact['summary']['oos_hit_rate']:.1%}")
    print(f"OOS degradation:  {artifact['summary']['oos_degradation_pct']:.1f}%")
    print(f"OOS dd delta:     {artifact['summary']['oos_dd_delta_pp']:+.2f}pp")
    print(f"Gate pass:        {artifact['gates']['pass']}")
    if artifact["gates"]["reasons"]:
        print("Reasons:")
        for reason in artifact["gates"]["reasons"]:
            print(f"  - {reason}")
    if artifact.get("tuning", {}).get("best_setup"):
        best = artifact["tuning"]["best_setup"]
        print(
            "Best setup:      "
            f"confirm_points={best['confirm_points']}, switch_margin={best['switch_margin']}, "
            f"score={best['stage2_score']:.3f}"
        )
    print(f"Artifact:         {artifact['artifact_path']}")

    if getattr(args, "output", None):
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2))
        print(f"Exported:         {output_path}")


def cmd_meta_bootstrap(args):
    """Run neutral bootstrap start decision for two strategies."""
    from backtest.meta_bootstrap import run_meta_bootstrap_decision

    try:
        params_a = _parse_param_assignments(
            getattr(args, "strategy_a_params", []), flag_name="--a-param"
        )
        params_b = _parse_param_assignments(
            getattr(args, "strategy_b_params", []), flag_name="--b-param"
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    custom_thresholds = {}
    try:
        for item in getattr(args, "custom_thresholds", []) or []:
            if "=" not in item:
                raise ValueError(f"Invalid --custom-threshold '{item}'. Use key=value")
            key, value = item.split("=", 1)
            custom_thresholds[key.strip()] = float(value.strip())
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        artifact = run_meta_bootstrap_decision(
            strategy_a=args.strategy_a,
            strategy_b=args.strategy_b,
            strategy_a_params=params_a,
            strategy_b_params=params_b,
            as_of=getattr(args, "date", None),
            evidence_profile=args.evidence_profile,
            evidence_compare_mode="vs_current",
            evidence_max_age_days=args.evidence_max_age_days,
            custom_thresholds=custom_thresholds or None,
            train_years=args.train_years,
            test_years=args.test_years,
            step_months=args.step_months,
            anchored=args.anchored,
            start_date=args.start,
            initial_capital=args.capital,
            costs_pct=args.costs,
            skip_failed=getattr(args, "skip_failed", True),
            metric_basis=args.metric_basis,
            fallback_cagr_tie_band_pp=args.fallback_cagr_tie_band_pp,
            fallback_tie_breaker=args.fallback_tie_breaker,
            artifact_path=getattr(args, "artifact_path", None),
            save_artifact=True,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    decision = artifact.get("decision", {})
    a_to_b = artifact.get("evidence", {}).get("a_to_b", {})
    b_to_a = artifact.get("evidence", {}).get("b_to_a", {})
    fallback = artifact.get("fallback") or {}

    print("\nMeta Bootstrap Start Decision")
    print("=" * 60)
    print(f"Strategy A:       {artifact['strategy_a']['path']}")
    print(f"Strategy B:       {artifact['strategy_b']['path']}")
    print(f"As of:            {artifact['as_of']}")
    print(f"Evidence profile: {artifact['evidence_profile']}")
    print(f"A->B pass:        {a_to_b.get('pass')} | edge={a_to_b.get('oos_cagr_edge_pp', 0.0):+.2f}pp | hit={a_to_b.get('oos_hit_rate', 0.0):.1%}")
    print(f"B->A pass:        {b_to_a.get('pass')} | edge={b_to_a.get('oos_cagr_edge_pp', 0.0):+.2f}pp | hit={b_to_a.get('oos_hit_rate', 0.0):.1%}")
    if fallback:
        cagr_edge = float(fallback.get("cagr_edge_b_minus_a_pp", 0.0))
        print(f"Fallback edge:    B-A CAGR {cagr_edge:+.2f}pp | tie-band={float(fallback.get('tie_band_pp', 0.0)):.2f}pp")
    print(f"Decision rule:    {decision.get('decision_rule')}")
    print(f"Recommended:      {decision.get('recommended_start_strategy')}")
    reasons = decision.get("reasons") or []
    if reasons:
        print("Reasons:")
        for reason in reasons:
            print(f"  - {reason}")
    print(f"Artifact:         {artifact.get('artifact_path')}")

    if getattr(args, "output", None):
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2))
        print(f"Exported:         {output_path}")


def cmd_sweep(args):
    """Run sweep analysis over multiple time windows."""
    from backtest.sweep import (
        SweepConfig,
        resolve_strategy_paths,
        run_sweep,
        run_sweep_with_strategies,
        save_sweep_results,
        render_sweep_html,
    )
    from backtest.batch_optimize import load_strategy_from_file
    import copy
    import json

    # Resolve strategy paths (supports globs)
    try:
        strategy_paths = resolve_strategy_paths(args.strategies)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Determine metric basis
    tax_enabled = not args.no_tax
    if args.liquidate_at_window_end:
        metric_basis = "net_liquidation"
    elif args.metric_basis:
        metric_basis = args.metric_basis
    else:
        # Default: net_liquidation if tax enabled, gross otherwise
        metric_basis = "net_liquidation" if tax_enabled else "gross"

    # Build sweep config
    execution_lag_days = _resolve_execution_lag(args)
    max_volume_participation, min_daily_dollar_volume, liquidity_on_missing_volume = _resolve_liquidity_settings(args)
    cost_profile = None if args.no_costs else _load_cost_profile(getattr(args, "cost_profile_file", None))
    risk_overlay = _resolve_risk_overlay(args)
    config = SweepConfig(
        mode=args.mode,
        window_length=args.window,
        end_date=args.end,
        from_date=args.start_from,
        to_date=args.start_to,
        start_grid=args.start_grid,
        step=args.step,
        warmup_days=args.warmup_days,
        initial_capital=args.capital,
        rebalance_frequency=args.rebalance_frequency,
        costs_enabled=not args.no_costs,
        costs_pct=args.cost_bps / 10000 if args.cost_bps else 0.001,
        cost_profile=cost_profile,
        execution_lag_days=execution_lag_days,
        max_volume_participation=max_volume_participation,
        min_daily_dollar_volume=min_daily_dollar_volume,
        liquidity_on_missing_volume=liquidity_on_missing_volume,
        risk_overlay=risk_overlay,
        tax_enabled=tax_enabled,
        tax_rate=args.tax_rate,
        tax_exemption=args.allowance,
        benchmark_ticker=args.benchmark_ticker,
        metric_basis=metric_basis,
        allow_universe_lookahead=getattr(args, 'allow_universe_lookahead', False),
        jobs=args.jobs,
        fail_fast=args.fail_fast,
        align=getattr(args, 'align', 'ffill'),
        skip_failed=getattr(args, 'skip_failed', True),
        validate=not getattr(args, 'no_validate', False),
        drip_enabled=getattr(args, 'drip', False),
        external_features=_build_external_features_config(args),
    )

    # Run sweep
    try:
        if args.params_file:
            with open(args.params_file, "r") as f:
                strategy_params = json.load(f)

            strategies = []
            strategy_files = {}

            for path in strategy_paths:
                instance, cls, _ = load_strategy_from_file(str(path))
                params_to_apply = {}
                rebal_freq = instance.rebalance_frequency if instance else "monthly"

                if strategy_params and cls.__name__ in strategy_params:
                    opt_params = copy.deepcopy(strategy_params[cls.__name__])
                    rebal_freq = opt_params.pop("rebalance_frequency", rebal_freq)
                    params_to_apply = {k: v for k, v in opt_params.items() if not k.startswith("_")}

                    if params_to_apply:
                        try:
                            instance = cls(**params_to_apply)
                        except TypeError:
                            instance = cls()
                            for k, v in params_to_apply.items():
                                setattr(instance, k, v)

                if instance is None:
                    instance = cls()

                instance.rebalance_frequency = rebal_freq
                strategies.append(instance)
                strategy_files[instance.name] = str(path)

            result = run_sweep_with_strategies(
                strategies=strategies,
                strategy_files=strategy_files,
                config=config,
                progress=True,
            )
        else:
            result = run_sweep(strategy_paths, config, progress=True)
    except Exception as e:
        print(f"Error during sweep: {e}")
        if args.fail_fast:
            raise
        sys.exit(1)

    # Determine output directory
    if args.out:
        output_dir = Path(args.out)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/sweep_{timestamp}")

    # Save results
    save_sweep_results(result, output_dir)

    # Generate HTML report
    html_path = output_dir / "report.html"
    render_sweep_html(result, html_path)
    print(f"  - report.html")


def _determine_default_metric_basis(no_tax: bool, liquidate_at_end: bool) -> str:
    """Determine metric basis for optimize and walk-forward modes."""
    if liquidate_at_end:
        return "net_liquidation"
    return "net_liquidation" if not no_tax else "gross"


def _select_metrics_for_optimization(result):
    """
    Return the metric set aligned with BacktestConfig.metric_basis.

    result.metrics is already the primary headline metric set for the configured basis.
    Fallbacks are only for defensive robustness.
    """
    if result.metrics is not None:
        return result.metrics
    if result.metrics_net is not None:
        return result.metrics_net
    return result.metrics_gross


def _build_walk_forward_windows(
    dates,
    train_days: int,
    test_days: int,
    step_days: int,
    anchored: bool,
):
    """Build walk-forward window boundaries from a date index."""
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train_days, test_days, and step_days must be positive")

    windows = []
    total_days = len(dates)
    if total_days < train_days + test_days:
        return windows

    if anchored:
        train_start_idx = 0
        test_end_idx = train_days + test_days
        while test_end_idx <= total_days:
            train_end_idx = test_end_idx - test_days
            windows.append(
                {
                    "train_start": dates[train_start_idx],
                    "train_end": dates[train_end_idx - 1],
                    "test_start": dates[train_end_idx],
                    "test_end": dates[test_end_idx - 1],
                }
            )
            test_end_idx += step_days
    else:
        start_idx = 0
        while start_idx + train_days + test_days <= total_days:
            train_end_idx = start_idx + train_days
            test_end_idx = train_end_idx + test_days
            windows.append(
                {
                    "train_start": dates[start_idx],
                    "train_end": dates[train_end_idx - 1],
                    "test_start": dates[train_end_idx],
                    "test_end": dates[test_end_idx - 1],
                }
            )
            start_idx += step_days

    return windows


def _instantiate_strategy_for_params(strategy_class, strategy_instance, params):
    """Create a strategy instance from parameter overrides."""
    import copy
    import inspect

    try:
        if strategy_instance:
            init_sig = inspect.signature(strategy_class.__init__)
            init_params = [p for p in init_sig.parameters.keys() if p != "self"]
            kwargs = {}
            for p in init_params:
                if p in params:
                    kwargs[p] = params[p]
                elif hasattr(strategy_instance, p):
                    kwargs[p] = getattr(strategy_instance, p)
            return strategy_class(**kwargs)
        return strategy_class(**params)
    except Exception:
        if strategy_instance is None:
            raise
        strategy = copy.deepcopy(strategy_instance)
        for k, v in params.items():
            setattr(strategy, k, v)
        if hasattr(strategy, "_rebuild_assets"):
            strategy._rebuild_assets()
        return strategy


def _run_metric_backtest(
    args,
    strategy_class,
    strategy_instance,
    eval_data,
    rebalance_frequency: str,
    params: dict,
    metric_basis: str,
    cost_profile=None,
    risk_overlay=None,
):
    """Run one backtest and return objective metric value and metric object."""
    import pandas as pd

    from backtest.backtester import Backtester, BacktestConfig

    strategy = _instantiate_strategy_for_params(strategy_class, strategy_instance, params)
    config = BacktestConfig(
        initial_capital=args.capital,
        costs_pct=args.costs,
        cost_profile=cost_profile,
        execution_lag_days=_resolve_execution_lag(args),
        max_volume_participation=getattr(args, "max_volume_participation", None),
        min_daily_dollar_volume=float(getattr(args, "min_daily_dollar_volume", 0.0) or 0.0),
        liquidity_on_missing_volume=getattr(args, "liquidity_on_missing_volume", "allow"),
        risk_overlay=risk_overlay,
        rebalance_frequency=rebalance_frequency,
        tax_enabled=not args.no_tax,
        cash_rate=getattr(args, "cash_rate", 0.0),
        metric_basis=metric_basis,
        validate=not getattr(args, "no_validate", False),
        drip_enabled=getattr(args, "drip", False),
        external_features_loader=_build_external_features_loader(args),
    )
    backtester = Backtester(strategy, eval_data, config)
    result = backtester.run()
    metrics = _select_metrics_for_optimization(result)
    metric_value = getattr(metrics, args.metric, float("nan"))
    if pd.isna(metric_value):
        raise ValueError(f"Metric '{args.metric}' returned NaN")
    return float(metric_value), metrics


def _compute_degradation_pct(train_metric: float, test_metric: float, minimize: bool) -> float:
    """Compute degradation where positive values always mean OOS got worse."""
    if train_metric == 0:
        return 0.0
    if minimize:
        return (test_metric - train_metric) / abs(train_metric) * 100.0
    return (train_metric - test_metric) / abs(train_metric) * 100.0


def _calculate_parameter_drift(window_results, param_names):
    """Return average parameter drift and per-parameter drift ratio."""
    import numpy as np

    if len(window_results) < 2:
        return 0.0, {}

    tracked_keys = ["best_rebalance"] + [f"best_{name}" for name in param_names]
    transitions = len(window_results) - 1
    drift_by_key = {}

    for key in tracked_keys:
        changes = 0
        for idx in range(1, len(window_results)):
            if window_results[idx].get(key) != window_results[idx - 1].get(key):
                changes += 1
        drift_by_key[key] = changes / transitions if transitions > 0 else 0.0

    avg_drift = float(np.mean(list(drift_by_key.values()))) if drift_by_key else 0.0
    return avg_drift, drift_by_key


def _run_walk_forward_optimization(
    args,
    data,
    strategy_class,
    strategy_instance,
    param_grid,
    rebalance_frequencies,
    cost_profile=None,
    risk_overlay=None,
):
    """Run walk-forward optimization (single-layer or nested)."""
    import itertools
    from collections import Counter

    import numpy as np
    import pandas as pd

    nested_mode = bool(getattr(args, "walk_forward_nested", False))
    outer_train_days = int(args.train_years * 252)
    outer_test_days = int(args.test_years * 252)
    outer_step_days = int(args.step_months * 21)  # ~21 trading days/month

    if outer_train_days <= 0 or outer_test_days <= 0 or outer_step_days <= 0:
        print("Error: --train-years, --test-years, and --step-months must be positive")
        sys.exit(1)

    dates = data.prices.index
    total_days = len(dates)
    min_required = outer_train_days + outer_test_days
    if total_days < min_required:
        print(f"Error: Not enough data. Need {min_required} days, have {total_days}")
        sys.exit(1)

    outer_windows = _build_walk_forward_windows(
        dates=dates,
        train_days=outer_train_days,
        test_days=outer_test_days,
        step_days=outer_step_days,
        anchored=args.anchored,
    )
    if not outer_windows:
        print("Error: Could not generate any walk-forward windows")
        sys.exit(1)

    inner_train_days = inner_test_days = inner_step_days = 0
    if nested_mode:
        inner_train_days = int(args.inner_train_years * 252)
        inner_test_days = int(args.inner_test_years * 252)
        inner_step_days = int(args.inner_step_months * 21)
        if inner_train_days <= 0 or inner_test_days <= 0 or inner_step_days <= 0:
            print(
                "Error: --inner-train-years, --inner-test-years, and --inner-step-months "
                "must be positive in nested mode"
            )
            sys.exit(1)
        if outer_train_days < inner_train_days + inner_test_days:
            print(
                "Error: Outer training window too short for nested walk-forward.\n"
                f"Need at least {inner_train_days + inner_test_days} days "
                f"({args.inner_train_years}y + {args.inner_test_years}y), "
                f"but outer train has {outer_train_days} days ({args.train_years}y)."
            )
            sys.exit(1)

    print("\nWalk-Forward Optimization")
    print(f"  Mode: {'nested' if nested_mode else 'standard'}")
    print(f"  Train: {args.train_years} years, Test: {args.test_years} years")
    print(f"  Step: {args.step_months} months, Windows: {len(outer_windows)}")
    print(f"  Window type: {'anchored' if args.anchored else 'rolling'}")
    if nested_mode:
        print(
            f"  Inner train/test/step: {args.inner_train_years}y / "
            f"{args.inner_test_years}y / {args.inner_step_months}m"
        )
    print()

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values)) if param_values else [()]
    total_candidates = len(param_combinations) * len(rebalance_frequencies)
    print(f"Parameter combinations per window: {total_candidates}")
    print("=" * 70)

    metric_basis = _determine_default_metric_basis(args.no_tax, args.liquidate_at_end)
    window_results = []

    for outer_idx, outer_window in enumerate(outer_windows):
        print(f"\n[Window {outer_idx + 1}/{len(outer_windows)}]")
        print(
            f"  Train: {outer_window['train_start'].strftime('%Y-%m-%d')} -> "
            f"{outer_window['train_end'].strftime('%Y-%m-%d')}"
        )
        print(
            f"  Test:  {outer_window['test_start'].strftime('%Y-%m-%d')} -> "
            f"{outer_window['test_end'].strftime('%Y-%m-%d')}"
        )

        outer_train_data = data.filter_dates(outer_window["train_start"], outer_window["train_end"])
        outer_test_data = data.filter_dates(outer_window["test_start"], outer_window["test_end"])

        best_params = None
        best_rebalance = None
        inner_score_mean = float("nan")
        inner_score_std = float("nan")
        inner_window_count = 0

        if nested_mode:
            inner_windows = _build_walk_forward_windows(
                dates=outer_train_data.prices.index,
                train_days=inner_train_days,
                test_days=inner_test_days,
                step_days=inner_step_days,
                anchored=getattr(args, "inner_anchored", False),
            )
            inner_window_count = len(inner_windows)
            if not inner_windows:
                print("  WARN: No valid inner windows (insufficient warmup for this outer window)")
                continue

            candidate_scores = []
            for rebalance_frequency in rebalance_frequencies:
                for param_combo in param_combinations:
                    params = dict(zip(param_names, param_combo))
                    inner_scores = []
                    for inner_window in inner_windows:
                        inner_test_data = outer_train_data.filter_dates(
                            inner_window["test_start"],
                            inner_window["test_end"],
                        )
                        try:
                            metric_value, _ = _run_metric_backtest(
                                args=args,
                                strategy_class=strategy_class,
                                strategy_instance=strategy_instance,
                                eval_data=inner_test_data,
                                rebalance_frequency=rebalance_frequency,
                                params=params,
                                metric_basis=metric_basis,
                                cost_profile=cost_profile,
                                risk_overlay=risk_overlay,
                            )
                            inner_scores.append(metric_value)
                        except Exception:
                            continue

                    if inner_scores:
                        score_mean = float(np.mean(inner_scores))
                        score_std = float(np.std(inner_scores))
                        candidate_scores.append((rebalance_frequency, params, score_mean, score_std))

            if not candidate_scores:
                print("  WARN: No valid parameter set in nested inner optimization")
                continue

            if args.minimize:
                candidate_scores.sort(key=lambda x: (x[2], x[3]))
            else:
                candidate_scores.sort(key=lambda x: (-x[2], x[3]))
            best_rebalance, best_params, inner_score_mean, inner_score_std = candidate_scores[0]
            print(
                f"  Nested inner score ({args.metric}): "
                f"mean={inner_score_mean:.4f}, std={inner_score_std:.4f}, windows={inner_window_count}"
            )
        else:
            best_metric = float("inf") if args.minimize else float("-inf")
            for rebalance_frequency in rebalance_frequencies:
                for param_combo in param_combinations:
                    params = dict(zip(param_names, param_combo))
                    try:
                        metric_value, _ = _run_metric_backtest(
                            args=args,
                            strategy_class=strategy_class,
                            strategy_instance=strategy_instance,
                            eval_data=outer_train_data,
                            rebalance_frequency=rebalance_frequency,
                            params=params,
                            metric_basis=metric_basis,
                            cost_profile=cost_profile,
                            risk_overlay=risk_overlay,
                        )
                    except Exception:
                        continue
                    if args.minimize:
                        is_better = metric_value < best_metric
                    else:
                        is_better = metric_value > best_metric
                    if is_better:
                        best_metric = metric_value
                        best_params = params.copy()
                        best_rebalance = rebalance_frequency

        if best_params is None or best_rebalance is None:
            print("  WARN: No valid results in optimization phase")
            continue

        try:
            train_metric, train_metrics = _run_metric_backtest(
                args=args,
                strategy_class=strategy_class,
                strategy_instance=strategy_instance,
                eval_data=outer_train_data,
                rebalance_frequency=best_rebalance,
                params=best_params,
                metric_basis=metric_basis,
                cost_profile=cost_profile,
                risk_overlay=risk_overlay,
            )
            test_metric, test_metrics = _run_metric_backtest(
                args=args,
                strategy_class=strategy_class,
                strategy_instance=strategy_instance,
                eval_data=outer_test_data,
                rebalance_frequency=best_rebalance,
                params=best_params,
                metric_basis=metric_basis,
                cost_profile=cost_profile,
                risk_overlay=risk_overlay,
            )
        except Exception as e:
            print(f"  WARN: Evaluation failed for selected parameters: {e}")
            continue

        degradation = _compute_degradation_pct(train_metric, test_metric, args.minimize)
        params_str = ", ".join(f"{k}={v}" for k, v in best_params.items()) or "<default>"
        print(f"  Best: {best_rebalance}, {params_str}")
        print(f"  Train {args.metric}: {train_metric:.4f}")
        print(f"  Test  {args.metric}: {test_metric:.4f} (degradation: {degradation:+.1f}%)")

        row = {
            "window": outer_idx + 1,
            "train_start": outer_window["train_start"].strftime("%Y-%m-%d"),
            "train_end": outer_window["train_end"].strftime("%Y-%m-%d"),
            "test_start": outer_window["test_start"].strftime("%Y-%m-%d"),
            "test_end": outer_window["test_end"].strftime("%Y-%m-%d"),
            "best_rebalance": best_rebalance,
            f"train_{args.metric}": train_metric,
            f"test_{args.metric}": test_metric,
            "degradation_pct": degradation,
            "test_sharpe": getattr(test_metrics, "sharpe_ratio", float("nan")),
            "test_cagr": getattr(test_metrics, "cagr", float("nan")),
            "test_max_drawdown": getattr(test_metrics, "max_drawdown", float("nan")),
        }
        row.update({f"best_{k}": v for k, v in best_params.items()})
        if nested_mode:
            row["inner_score_mean"] = inner_score_mean
            row["inner_score_std"] = inner_score_std
            row["inner_windows"] = inner_window_count
        window_results.append(row)

    if not window_results:
        print("\nNo valid window results")
        return

    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    train_metrics = [r[f"train_{args.metric}"] for r in window_results]
    test_metrics = [r[f"test_{args.metric}"] for r in window_results]
    degradations = [r["degradation_pct"] for r in window_results]

    print(f"\nIn-Sample (Train) {args.metric}:")
    print(f"  Mean: {np.mean(train_metrics):.4f}")
    print(f"  Std:  {np.std(train_metrics):.4f}")

    print(f"\nOut-of-Sample (Test) {args.metric}:")
    print(f"  Mean: {np.mean(test_metrics):.4f}")
    print(f"  Std:  {np.std(test_metrics):.4f}")

    print("\nDegradation (positive means worse OOS):")
    print(f"  Mean: {np.mean(degradations):+.1f}%")
    print(f"  Std:  {np.std(degradations):.1f}%")

    avg_degradation = float(np.mean(degradations))
    if avg_degradation > 50:
        print("\nOVERFITTING RISK: HIGH (average degradation > 50%)")
    elif avg_degradation > 25:
        print("\nOVERFITTING RISK: MODERATE (average degradation 25-50%)")
    else:
        print("\nOVERFITTING RISK: LOW (average degradation < 25%)")

    rebal_counts = Counter(r["best_rebalance"] for r in window_results)
    stable_rebalance, stable_rebalance_count = rebal_counts.most_common(1)[0]
    print(
        "\nParameter Stability:"
        f"\n  Most stable rebalance: {stable_rebalance} "
        f"({stable_rebalance_count}/{len(window_results)} windows)"
    )
    for param in param_names:
        key = f"best_{param}"
        value_counts = Counter(r[key] for r in window_results if key in r)
        if not value_counts:
            continue
        most_common_value, most_common_count = value_counts.most_common(1)[0]
        print(
            f"  {param}: most common={most_common_value} "
            f"({most_common_count}/{len(window_results)} windows)"
        )

    avg_drift, drift_by_key = _calculate_parameter_drift(window_results, param_names)
    print(f"  Parameter drift score: {avg_drift:.2f} (0=stable, 1=changes every window)")
    for key, drift_ratio in drift_by_key.items():
        if key == "best_rebalance":
            label = "rebalance_frequency"
        else:
            label = key.replace("best_", "")
        print(f"    drift[{label}]={drift_ratio:.2f}")

    if nested_mode:
        inner_means = [r["inner_score_mean"] for r in window_results if "inner_score_mean" in r]
        inner_stds = [r["inner_score_std"] for r in window_results if "inner_score_std" in r]
        if inner_means:
            print(
                f"\nNested score ({args.metric}) across windows:"
                f"\n  Mean(inner mean): {np.mean(inner_means):.4f}"
                f"\n  Mean(inner std):  {np.mean(inner_stds):.4f}"
            )

    robustness_pass = avg_degradation <= 25 and avg_drift <= 0.60
    print(f"\nRobustness gate: {'PASS' if robustness_pass else 'FAIL'}")

    if args.output:
        df = pd.DataFrame(window_results)
        df["robustness_pass"] = robustness_pass
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


def cmd_optimize(args):
    """Run parameter optimization for a strategy."""
    import itertools

    from backtest.data import DataLoader
    from backtest.backtester import Backtester, BacktestConfig

    # Load strategy file to get the strategy class
    strategy_path = Path(args.strategy)
    if not strategy_path.exists():
        print(f"Error: Strategy file not found: {strategy_path}")
        sys.exit(1)

    # Import strategy module
    spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_module"] = module
    spec.loader.exec_module(module)

    from backtest.strategy import Strategy

    # Find strategy instance first (preferred), then class
    strategy_class = None
    strategy_instance = None

    # First, look for a pre-instantiated 'strategy' variable
    if hasattr(module, 'strategy'):
        obj = getattr(module, 'strategy')
        if isinstance(obj, Strategy):
            strategy_instance = obj
            strategy_class = type(obj)

    # If no instance, look for a Strategy class
    if strategy_class is None:
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
                strategy_class = obj
                break
            elif isinstance(obj, Strategy) and strategy_instance is None:
                strategy_instance = obj
                strategy_class = type(obj)

    if strategy_class is None:
        print("Error: No Strategy class found in file")
        sys.exit(1)

    # Parse parameter grid
    param_grid = {}

    # Parse --param arguments
    if args.params:
        for param_str in args.params:
            if "=" not in param_str:
                print(f"Error: Invalid param format '{param_str}'. Use: --param name=val1,val2,...")
                sys.exit(1)
            name, values_str = param_str.split("=", 1)
            values = []
            for v in values_str.split(","):
                v = v.strip()
                # Try to parse as number
                try:
                    if "." in v:
                        values.append(float(v))
                    else:
                        values.append(int(v))
                except ValueError:
                    values.append(v)
            param_grid[name] = values

    # Parse rebalance frequencies
    rebalance_frequencies = [f.strip() for f in args.rebalance_frequencies.split(",")]
    cost_profile = _load_cost_profile(getattr(args, "cost_profile_file", None))
    execution_lag_days = _resolve_execution_lag(args)
    max_volume_participation, min_daily_dollar_volume, liquidity_on_missing_volume = _resolve_liquidity_settings(args)
    load_volumes = _needs_volume_data(max_volume_participation, min_daily_dollar_volume)
    risk_overlay = _resolve_risk_overlay(args)

    # Get base assets from strategy instance
    if strategy_instance:
        base_assets = strategy_instance.assets.copy()
    else:
        # Try to instantiate with minimal args to get assets
        try:
            temp_strategy = strategy_class.__new__(strategy_class)
            if hasattr(temp_strategy, 'assets'):
                base_assets = temp_strategy.assets
            else:
                base_assets = []
        except Exception:
            base_assets = []

    if not base_assets:
        print("Error: Could not determine strategy assets")
        sys.exit(1)

    print(f"Loading strategy class: {strategy_class.__name__}")
    print(f"Assets: {len(base_assets)} tickers")
    print(f"Parameters to optimize: {param_grid if param_grid else 'None'}")
    print(f"Rebalance frequencies: {rebalance_frequencies}")
    print(f"Optimizing for: {args.metric} ({'minimize' if args.minimize else 'maximize'})")
    print()

    # Load data
    print("Loading price data...")
    drip_enabled = getattr(args, 'drip', False)
    tax_enabled = not args.no_tax
    validate = not getattr(args, 'no_validate', False)
    try:
        data = DataLoader.yahoo(
            tickers=base_assets,
            start=args.start,
            end=args.end,
            currency="EUR",
            align=getattr(args, 'align', 'ffill'),
            skip_failed=getattr(args, 'skip_failed', True),
            load_dividends=drip_enabled or tax_enabled,
            load_volumes=load_volumes,
            validate=validate,
        )
        print(f"Loaded {len(data.prices)} days of data")
        print(f"Date range: {data.prices.index[0].strftime('%Y-%m-%d')} → {data.prices.index[-1].strftime('%Y-%m-%d')}")
        if data.dividends is not None:
            div_count = (data.dividends > 0).sum().sum()
            print(f"  Loaded {div_count} dividend events")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Walk-Forward mode (standard or nested)
    if args.walk_forward or getattr(args, "walk_forward_nested", False):
        _run_walk_forward_optimization(
            args, data, strategy_class, strategy_instance,
            param_grid, rebalance_frequencies, cost_profile=cost_profile, risk_overlay=risk_overlay
        )
        return

    # Single-window mode: Generate all combinations
    all_results = []

    # Build combinations including rebalance_frequency
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    if param_values:
        param_combinations = list(itertools.product(*param_values))
    else:
        param_combinations = [()]  # No params to vary

    total_runs = len(param_combinations) * len(rebalance_frequencies)
    print(f"\nRunning {total_runs} backtests...")
    print("-" * 60)

    metric_basis = _determine_default_metric_basis(args.no_tax, args.liquidate_at_end)
    run_count = 0
    for rebal_freq in rebalance_frequencies:
        for param_combo in param_combinations:
            run_count += 1
            params = dict(zip(param_names, param_combo))

            # Create strategy with these parameters
            try:
                import inspect
                if strategy_instance:
                    # Get __init__ parameters to know which attributes to copy
                    init_sig = inspect.signature(strategy_class.__init__)
                    init_params = [p for p in init_sig.parameters.keys() if p != 'self']

                    # Build kwargs from instance attributes + overrides
                    kwargs = {}
                    for p in init_params:
                        if p in params:
                            kwargs[p] = params[p]
                        elif hasattr(strategy_instance, p):
                            kwargs[p] = getattr(strategy_instance, p)

                    strategy = strategy_class(**kwargs)
                else:
                    strategy = strategy_class(**params)
            except Exception as e:
                # Fallback: copy instance and override attributes
                import copy
                if strategy_instance:
                    strategy = copy.deepcopy(strategy_instance)
                    for k, v in params.items():
                        setattr(strategy, k, v)
                    # Rebuild assets if needed
                    if hasattr(strategy, '_rebuild_assets'):
                        strategy._rebuild_assets()
                else:
                    print(f"  [{run_count:3d}/{total_runs}] ERROR creating strategy: {e}")
                    continue

            # Configure backtest
            config = BacktestConfig(
                initial_capital=args.capital,
                costs_pct=args.costs,
                cost_profile=cost_profile,
                execution_lag_days=execution_lag_days,
                max_volume_participation=max_volume_participation,
                min_daily_dollar_volume=min_daily_dollar_volume,
                liquidity_on_missing_volume=liquidity_on_missing_volume,
                risk_overlay=risk_overlay,
                rebalance_frequency=rebal_freq,
                tax_enabled=not args.no_tax,
                cash_rate=getattr(args, "cash_rate", 0.0),
                metric_basis=metric_basis,
                validate=not getattr(args, 'no_validate', False),
                drip_enabled=drip_enabled,
                external_features_loader=_build_external_features_loader(args),
            )

            # Run backtest
            try:
                backtester = Backtester(strategy, data, config)
                result = backtester.run()
                m = _select_metrics_for_optimization(result)

                metric_value = getattr(m, args.metric, None)
                if metric_value is None:
                    metric_value = float('nan')

                result_entry = {
                    'rebalance_frequency': rebal_freq,
                    **params,
                    'sharpe_ratio': m.sharpe_ratio,
                    'sortino_ratio': m.sortino_ratio,
                    'cagr': m.cagr,
                    'volatility': m.volatility,
                    'max_drawdown': m.max_drawdown,
                    'calmar_ratio': m.calmar_ratio,
                    'total_return': m.total_return,
                    '_metric_value': metric_value,
                }
                all_results.append(result_entry)

                # Progress indicator
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                if param_str:
                    param_str = f", {param_str}"
                print(f"  [{run_count:3d}/{total_runs}] {rebal_freq}{param_str} → {args.metric}={metric_value:.4f}")

            except Exception as e:
                print(f"  [{run_count:3d}/{total_runs}] ERROR: {e}")
                all_results.append({
                    'rebalance_frequency': rebal_freq,
                    **params,
                    'error': str(e),
                    '_metric_value': float('nan'),
                })

    # Sort results
    reverse = not args.minimize
    valid_results = [r for r in all_results if not (isinstance(r.get('_metric_value'), float) and r['_metric_value'] != r['_metric_value'])]
    sorted_results = sorted(valid_results, key=lambda x: x['_metric_value'], reverse=reverse)

    # Print top results
    print()
    print("=" * 60)
    print(f"TOP {min(args.top, len(sorted_results))} RESULTS (by {args.metric})")
    print("=" * 60)

    for i, r in enumerate(sorted_results[:args.top], 1):
        params_str = ", ".join(f"{k}={v}" for k, v in r.items()
                               if k not in ['_metric_value', 'sharpe_ratio', 'sortino_ratio', 'cagr',
                                          'volatility', 'max_drawdown', 'calmar_ratio', 'total_return', 'error'])
        print(f"{i:2d}. {params_str}")
        print(f"    Sharpe={r.get('sharpe_ratio', 0):.3f}  CAGR={r.get('cagr', 0)*100:.1f}%  "
              f"MaxDD={r.get('max_drawdown', 0)*100:.1f}%  Sortino={r.get('sortino_ratio', 0):.3f}")
        print()

    # Best result
    if sorted_results:
        best = sorted_results[0]
        print("-" * 60)
        print("BEST PARAMETERS:")
        for k, v in best.items():
            if k not in ['_metric_value', 'sharpe_ratio', 'sortino_ratio', 'cagr',
                        'volatility', 'max_drawdown', 'calmar_ratio', 'total_return', 'error']:
                print(f"  {k}: {v}")

    # Save to CSV if requested
    if args.output:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df = df.drop(columns=['_metric_value'], errors='ignore')
        df = df.sort_values(args.metric, ascending=args.minimize)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


def cmd_batch_optimize(args):
    """Run batch optimization across all strategies."""
    from pathlib import Path
    from datetime import datetime
    from backtest.batch_optimize import (
        run_batch_optimization,
        save_batch_results,
    )
    from backtest.sweep import resolve_strategy_paths

    # Resolve strategy paths (supports globs)
    try:
        strategy_paths = resolve_strategy_paths(args.strategies)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    walk_forward = getattr(args, 'walk_forward', False)
    walk_forward_nested = getattr(args, 'walk_forward_nested', False)
    train_years = getattr(args, 'train_years', 5.0)
    test_years = getattr(args, 'test_years', 1.0)
    step_months = getattr(args, 'step_months', 12)
    anchored = getattr(args, 'anchored', False)
    inner_train_years = getattr(args, 'inner_train_years', 3.0)
    inner_test_years = getattr(args, 'inner_test_years', 1.0)
    inner_step_months = getattr(args, 'inner_step_months', 6)
    inner_anchored = getattr(args, 'inner_anchored', False)

    print(f"\nBatch Optimization")
    print("=" * 70)
    print(f"Strategies: {len(strategy_paths)}")
    print(f"Metric: {args.metric} ({'minimize' if args.minimize else 'maximize'})")
    if walk_forward:
        mode = "anchored" if anchored else "rolling"
        wf_mode = "nested " if walk_forward_nested else ""
        print(
            f"Mode: {wf_mode}Walk-Forward Analysis "
            f"(train={train_years}y, test={test_years}y, step={step_months}m, {mode})"
        )
        if walk_forward_nested:
            inner_mode = "anchored" if inner_anchored else "rolling"
            print(
                "  Nested inner windows: "
                f"train={inner_train_years}y, test={inner_test_years}y, "
                f"step={inner_step_months}m, {inner_mode}"
            )
    else:
        print(f"Mode: Grid Search")
    print(f"Rebalance frequencies: {args.rebalance_frequencies}")

    # Parse rebalance frequencies
    rebalance_frequencies = [f.strip() for f in args.rebalance_frequencies.split(",")]
    cost_profile = _load_cost_profile(getattr(args, "cost_profile_file", None))
    execution_lag_days = _resolve_execution_lag(args)
    max_volume_participation, min_daily_dollar_volume, liquidity_on_missing_volume = _resolve_liquidity_settings(args)
    risk_overlay = _resolve_risk_overlay(args)

    # Determine metric basis
    tax_enabled = not args.no_tax
    metric_basis = _determine_default_metric_basis(args.no_tax, args.liquidate_at_end)

    # Run batch optimization
    try:
        result = run_batch_optimization(
            strategy_files=strategy_paths,
            start=args.start,
            end=args.end,
            metric=args.metric,
            minimize=args.minimize,
            rebalance_frequencies=rebalance_frequencies,
            initial_capital=args.capital,
            costs_pct=args.costs,
            cost_profile=cost_profile,
            execution_lag_days=execution_lag_days,
            max_volume_participation=max_volume_participation,
            min_daily_dollar_volume=min_daily_dollar_volume,
            liquidity_on_missing_volume=liquidity_on_missing_volume,
            risk_overlay=risk_overlay,
            tax_enabled=tax_enabled,
            metric_basis=metric_basis,
            progress=True,
            fail_fast=args.fail_fast,
            align=getattr(args, 'align', 'ffill'),
            skip_failed=getattr(args, 'skip_failed', True),
            validate=not getattr(args, 'no_validate', False),
            drip_enabled=getattr(args, 'drip', False),
            walk_forward=walk_forward,
            walk_forward_nested=walk_forward_nested,
            train_years=train_years,
            test_years=test_years,
            step_months=step_months,
            anchored=anchored,
            inner_train_years=inner_train_years,
            inner_test_years=inner_test_years,
            inner_step_months=inner_step_months,
            inner_anchored=inner_anchored,
            external_features=_build_external_features_config(args),
        )
    except Exception as e:
        print(f"\nError during optimization: {e}")
        if args.fail_fast:
            raise
        sys.exit(1)

    # Print summary
    print(result.summary())

    # Determine output directory
    if args.out:
        output_dir = Path(args.out)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/batch_optimize_{timestamp}")

    # Save results
    print(f"\nSaving results to {output_dir}/")
    save_batch_results(result, output_dir)

    print("\nDone!")


def cmd_web(args):
    """Start the web frontend server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: Web dependencies not installed.")
        print("Install with: poetry install --with web")
        print("Or: pip install fastapi uvicorn python-multipart")
        sys.exit(1)

    from backtest.web.app import create_app

    print(f"\nStarting Kontor Web UI...")
    print(f"Server running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(
        "backtest.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def _add_risk_overlay_args(parser):
    """Add reusable risk overlay CLI flags to a parser."""
    parser.add_argument(
        "--max-position",
        type=float,
        default=None,
        help="Maximum target weight per position (0.25 = max 25%% per ticker)",
    )
    parser.add_argument(
        "--sector-cap",
        dest="sector_caps",
        action="append",
        default=[],
        help="Sector cap as sector=weight (repeatable, e.g., --sector-cap equity=0.6)",
    )
    parser.add_argument(
        "--sector-map-file",
        default=None,
        help="Path to JSON file mapping ticker->sector used by --sector-cap",
    )
    parser.add_argument(
        "--turnover-budget",
        type=float,
        default=None,
        help="Max one-way turnover per rebalance (0.2 = 20%%)",
    )
    parser.add_argument(
        "--drawdown-brake-threshold",
        type=float,
        default=None,
        help="Activate drawdown brake at this drawdown level (0.15 = 15%%)",
    )
    parser.add_argument(
        "--drawdown-brake-cash-target",
        type=float,
        default=1.0,
        help="Cash target while drawdown brake is active (1.0 = fully in cash)",
    )
    parser.add_argument(
        "--drawdown-brake-release",
        type=float,
        default=None,
        help="Optional drawdown level to release brake (hysteresis), e.g. 0.08",
    )


def _add_exposure_policy_args(parser):
    """Add optional 3x exposure controller CLI flags."""
    parser.add_argument(
        "--exposure-policy-enable",
        action="store_true",
        default=False,
        help="Enable optional 3x exposure policy (default: disabled)",
    )
    parser.add_argument(
        "--exposure-policy-file",
        default=None,
        help="Path to exposure policy JSON config",
    )
    parser.add_argument(
        "--exposure-policy-profile",
        choices=["trade_republic", "tr", "maxblue", "us"],
        default=None,
        help="Default proxy/core mapping profile for exposure policy",
    )
    parser.add_argument(
        "--exposure-policy-core-asset",
        default=None,
        help="Core/safe fallback asset used by level-2 exposure policy",
    )
    parser.add_argument("--exposure-level1-ret-5d-floor", type=float, default=None)
    parser.add_argument("--exposure-level1-drawdown-21d-floor", type=float, default=None)
    parser.add_argument("--exposure-level2-ret-21d-3x-floor", type=float, default=None)
    parser.add_argument("--exposure-level2-proxy-ret-21d-floor", type=float, default=None)
    parser.add_argument("--exposure-release-ret-5d-floor", type=float, default=None)
    parser.add_argument("--exposure-release-confirmation-periods", type=int, default=None)


def _add_preset_profile_arg(parser):
    """Add reusable execution/risk preset profile flag to a parser."""
    parser.add_argument(
        "--preset-profile",
        choices=preset_profile_names(),
        default=None,
        help=(
            "Apply execution/risk defaults profile "
            "(research, realistic, defensive). "
            "Explicit non-default flags still take precedence."
        ),
    )


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="backtest",
        description="A modular Python framework for systematic backtesting of investment strategies."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    def add_skip_failed_flags(target_parser, skip_help: str) -> None:
        """Add skip-failed toggles with framework default = True."""
        target_parser.add_argument(
            "--skip-failed",
            dest="skip_failed",
            action="store_true",
            help=f"{skip_help} (default: enabled)"
        )
        target_parser.add_argument(
            "--no-skip-failed",
            dest="skip_failed",
            action="store_false",
            help="Fail fast when any ticker download fails"
        )
        target_parser.set_defaults(skip_failed=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run a backtest for a single strategy")
    run_parser.add_argument("strategy", help="Path to strategy Python file")
    run_parser.add_argument("-s", "--start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    run_parser.add_argument("-e", "--end", default=None, help="End date (YYYY-MM-DD)")
    run_parser.add_argument("-c", "--capital", type=float, default=10000.0, help="Initial capital")
    run_parser.add_argument("-b", "--benchmark", default="S&P 500", help="Benchmark name (default: S&P 500)")
    run_parser.add_argument("--costs", type=float, default=0.001, help="Transaction costs (e.g., 0.001 = 0.1%%)")
    run_parser.add_argument(
        "--cost-profile-file",
        default=None,
        help="Path to JSON file with per-ticker/asset-class execution cost overrides"
    )
    run_parser.add_argument(
        "--execution-lag-days",
        type=int,
        default=0,
        help="Execution lag in trading days (0 = same-day, 1 = T+1)"
    )
    run_parser.add_argument(
        "--t-plus-one",
        action="store_true",
        default=False,
        help="Shortcut for --execution-lag-days 1"
    )
    run_parser.add_argument(
        "--max-volume-participation",
        type=float,
        default=None,
        help="Max fraction of daily share volume per trade (e.g., 0.1 = 10%%)"
    )
    run_parser.add_argument(
        "--min-daily-dollar-volume",
        type=float,
        default=0.0,
        help="Skip trades when ticker daily notional volume is below this threshold"
    )
    run_parser.add_argument(
        "--liquidity-on-missing-volume",
        choices=["allow", "skip"],
        default="allow",
        help="Behavior when volume data is missing: allow trade or skip trade"
    )
    _add_preset_profile_arg(run_parser)
    _add_risk_overlay_args(run_parser)
    _add_exposure_policy_args(run_parser)
    run_parser.add_argument("-o", "--output", default=None, help="Output file path")
    run_parser.add_argument(
        "--rebalance-frequency",
        "--frequency",
        dest="rebalance_frequency",
        choices=["daily", "weekly", "monthly", "quarterly", "yearly"],
        default=None,
        help=(
            "Override strategy rebalance frequency "
            "(daily|weekly|monthly|quarterly|yearly). Defaults to the strategy file."
        ),
    )
    run_parser.add_argument("-f", "--format", default="html", choices=["html", "json"], help="Output format")
    run_parser.add_argument(
        "--no-tax",
        action="store_true",
        default=False,
        help="Disable German tax model (Abgeltungssteuer)",
    )
    run_parser.add_argument(
        "--tax-exemption",
        type=float,
        default=1000.0,
        help="Freistellungsauftrag in EUR (default: 1000 single, use 2000 for joint)",
    )
    run_parser.add_argument(
        "--cash-rate",
        type=float,
        default=0.0,
        help="Annual interest rate applied to uninvested cash (e.g., 0.02 = 2%%)",
    )
    run_parser.add_argument(
        "--metric-basis",
        dest="metric_basis",
        choices=["gross", "net_realized", "net_liquidation"],
        default=None,
        help="Metric basis: gross (no tax), net_realized (realized tax only), "
             "net_liquidation (virtual liquidation with tax on unrealized). "
             "Default: net_liquidation if tax enabled, gross otherwise."
    )
    run_parser.add_argument(
        "--liquidate-at-end",
        action="store_true",
        default=False,
        help="Alias for --metric-basis net_liquidation (virtual liquidation for metrics)"
    )
    run_parser.add_argument(
        "--benchmark-ticker",
        default=None,
        help="Override benchmark ticker (default: derived from --benchmark name)"
    )
    run_parser.add_argument(
        "--allow-universe-lookahead",
        action="store_true",
        default=False,
        help="Allow strategies to use non-point-in-time universe data for historical backtests. "
             "WARNING: This may introduce survivorship/selection bias!"
    )
    run_parser.add_argument(
        "--align",
        choices=["intersection", "ffill"],
        default="ffill",
        help="Data alignment mode: 'intersection' drops rows with any NaN (shortest common range), "
             "'ffill' forward-fills gaps to keep longer history (default: ffill)"
    )
    run_parser.add_argument(
        "-p", "--param",
        action="append",
        dest="params",
        metavar="NAME=VALUE",
        help="Override strategy parameter (can be used multiple times). "
             "Supports Python literals: --param top_n=5 --param assets=\"['SPY','QQQ']\""
    )
    add_skip_failed_flags(
        run_parser,
        "Skip tickers that fail to download (useful for PIT strategies with delisted stocks)"
    )
    run_parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Disable pre/post-run validation checks (suppresses warnings about missing assets, low coverage, etc.)"
    )
    run_parser.add_argument(
        "--drip",
        action="store_true",
        default=False,
        help="Enable Dividend Reinvestment Plan (DRIP): automatically reinvest dividends into the paying stock"
    )
    _add_external_features_arguments(run_parser)
    run_parser.set_defaults(func=cmd_run)

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple strategies")
    compare_parser.add_argument("strategies", nargs="+", help="Paths to strategy Python files")
    compare_parser.add_argument("-s", "--start", default="2010-01-01", help="Start date")
    compare_parser.add_argument("-e", "--end", default=None, help="End date")
    compare_parser.add_argument("-c", "--capital", type=float, default=10000.0, help="Initial capital")
    compare_parser.add_argument(
        "--rebalance-frequency",
        "--frequency",
        dest="rebalance_frequency",
        choices=["daily", "weekly", "monthly", "quarterly", "yearly"],
        default=None,
        help=(
            "Override strategy rebalance frequency "
            "(daily|weekly|monthly|quarterly|yearly). Defaults to each strategy file."
        ),
    )
    compare_parser.add_argument("-o", "--output", default=None, help="Output file path")
    compare_parser.add_argument(
        "--allow-misaligned",
        action="store_true",
        default=False,
        help="Allow comparison when date ranges differ (uses intersection)",
    )
    compare_parser.add_argument(
        "--no-tax",
        action="store_true",
        default=False,
        help="Disable German tax model (Abgeltungssteuer)",
    )
    compare_parser.add_argument(
        "--tax-exemption",
        type=float,
        default=1000.0,
        help="Freistellungsauftrag in EUR (default: 1000 single, use 2000 for joint)",
    )
    compare_parser.add_argument(
        "--cash-rate",
        type=float,
        default=0.0,
        help="Annual interest rate applied to uninvested cash (e.g., 0.02 = 2%%)",
    )
    compare_parser.add_argument(
        "--metric-basis",
        dest="metric_basis",
        choices=["gross", "net_realized", "net_liquidation"],
        default=None,
        help="Metric basis: gross (no tax), net_realized (realized tax only), "
             "net_liquidation (virtual liquidation with tax on unrealized). "
             "Default: net_liquidation if tax enabled, gross otherwise."
    )
    compare_parser.add_argument(
        "--liquidate-at-end",
        action="store_true",
        default=False,
        help="Alias for --metric-basis net_liquidation"
    )
    compare_parser.add_argument(
        "-b", "--benchmark",
        default="S&P 500",
        help="Benchmark name (default: S&P 500)"
    )
    compare_parser.add_argument(
        "--benchmark-ticker",
        default=None,
        help="Override benchmark ticker (default: derived from --benchmark)"
    )
    compare_parser.add_argument(
        "--costs",
        type=float,
        default=0.001,
        help="Transaction costs (e.g., 0.001 = 0.1%%)"
    )
    compare_parser.add_argument(
        "--cost-profile-file",
        default=None,
        help="Path to JSON file with per-ticker/asset-class execution cost overrides"
    )
    compare_parser.add_argument(
        "--execution-lag-days",
        type=int,
        default=0,
        help="Execution lag in trading days (0 = same-day, 1 = T+1)"
    )
    compare_parser.add_argument(
        "--t-plus-one",
        action="store_true",
        default=False,
        help="Shortcut for --execution-lag-days 1"
    )
    compare_parser.add_argument(
        "--max-volume-participation",
        type=float,
        default=None,
        help="Max fraction of daily share volume per trade (e.g., 0.1 = 10%%)"
    )
    compare_parser.add_argument(
        "--min-daily-dollar-volume",
        type=float,
        default=0.0,
        help="Skip trades when ticker daily notional volume is below this threshold"
    )
    compare_parser.add_argument(
        "--liquidity-on-missing-volume",
        choices=["allow", "skip"],
        default="allow",
        help="Behavior when volume data is missing: allow trade or skip trade"
    )
    _add_preset_profile_arg(compare_parser)
    _add_risk_overlay_args(compare_parser)
    _add_exposure_policy_args(compare_parser)
    compare_parser.add_argument(
        "--allow-universe-lookahead",
        action="store_true",
        default=False,
        help="Allow strategies to use non-point-in-time universe data for historical backtests. "
             "WARNING: This may introduce survivorship/selection bias!"
    )
    compare_parser.add_argument(
        "--align",
        choices=["intersection", "ffill"],
        default="ffill",
        help="Data alignment mode: 'intersection' or 'ffill' (default: ffill)"
    )
    add_skip_failed_flags(
        compare_parser,
        "Skip tickers that fail to download (delisted, etc.)"
    )
    compare_parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Disable pre/post-run validation checks"
    )
    compare_parser.add_argument(
        "--drip",
        action="store_true",
        default=False,
        help="Enable Dividend Reinvestment Plan (DRIP)"
    )
    _add_external_features_arguments(compare_parser)
    compare_parser.set_defaults(func=cmd_compare)

    # meta-promotion command
    meta_promotion_parser = subparsers.add_parser(
        "meta-promotion",
        help="Create a governance artifact for strategy-promotion reviews",
    )
    meta_promotion_parser.add_argument(
        "strategies",
        nargs="*",
        help="Strategy files to compare. Defaults to Sticky/Core, VolTarget, Cascade.",
    )
    meta_promotion_parser.add_argument(
        "--baseline",
        default="strategies/levered_etf_momentum_sticky.py",
        help="Baseline/incumbent strategy file (default: Sticky/Core Levered).",
    )
    meta_promotion_parser.add_argument(
        "--soxl-proxy",
        action="store_true",
        default=False,
        help=(
            "Use SOXL as long-history proxy for 3SEM.L/Semi research. "
            "Recommended for promotion-grade reviews."
        ),
    )
    meta_promotion_parser.add_argument("-s", "--start", default="2016-01-01", help="Start date")
    meta_promotion_parser.add_argument("-e", "--end", default=None, help="End date")
    meta_promotion_parser.add_argument("-c", "--capital", type=float, default=10000.0, help="Initial capital")
    meta_promotion_parser.add_argument(
        "--costs",
        type=float,
        default=0.001,
        help="Transaction costs (e.g., 0.001 = 0.1%%)",
    )
    meta_promotion_parser.add_argument(
        "--metric-basis",
        choices=["gross", "net_realized", "net_liquidation"],
        default="net_liquidation",
        help="Promotion metric basis (default: net_liquidation).",
    )
    meta_promotion_parser.add_argument(
        "--tail-risk-gate-basis",
        choices=["daily", "rebalance"],
        default="daily",
        help="Tail-risk gate curve basis; daily is the Meta-Playbook v1.7 default.",
    )
    meta_promotion_parser.add_argument(
        "--no-tax",
        action="store_true",
        default=False,
        help="Disable German tax model for the promotion artifact.",
    )
    meta_promotion_parser.add_argument(
        "--broker",
        action="append",
        dest="brokers",
        choices=["trade_republic", "maxblue"],
        help="Broker mapping profile to audit (can be repeated). Defaults to both.",
    )
    meta_promotion_parser.add_argument(
        "--output-dir",
        default="results/meta_promotion",
        help="Output root for JSON/Markdown artifacts.",
    )
    meta_promotion_parser.add_argument(
        "--align",
        choices=["intersection", "ffill"],
        default="ffill",
        help="Data alignment mode (default: ffill).",
    )
    add_skip_failed_flags(meta_promotion_parser, "Skip tickers that fail to download")
    meta_promotion_parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Disable pre/post-run validation checks.",
    )
    meta_promotion_parser.set_defaults(func=cmd_meta_promotion)

    # sweep command
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run sweep analysis over multiple time windows for robustness testing"
    )
    sweep_parser.add_argument(
        "strategies",
        nargs="+",
        help="Strategy files or glob patterns (e.g., strategies/[!_]*.py)"
    )

    # Windowing options
    sweep_parser.add_argument(
        "--mode",
        choices=["rolling", "end-fixed"],
        default="rolling",
        help="Window mode: rolling (fixed size) or end-fixed (all end at same date)"
    )
    sweep_parser.add_argument(
        "--window",
        default="10y",
        help="Window length for rolling mode (e.g., 3y, 5y, 10y, 15y)"
    )
    sweep_parser.add_argument(
        "--end",
        default=None,
        help="End date for windows (default: last available data date)"
    )
    sweep_parser.add_argument(
        "--from",
        dest="start_from",
        default=None,
        help="Minimum start date for windows"
    )
    sweep_parser.add_argument(
        "--to",
        dest="start_to",
        default=None,
        help="Maximum start date for windows"
    )
    sweep_parser.add_argument(
        "--start-grid",
        choices=["weekly", "monthly", "yearly"],
        default="monthly",
        help="Frequency of start date candidates"
    )
    sweep_parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Skip every N grid points (default: 1 = every point)"
    )
    sweep_parser.add_argument(
        "--warmup-days",
        type=int,
        default=260,
        help="Days of data before window start for indicator calculation"
    )

    # Backtest parameters
    sweep_parser.add_argument(
        "-c", "--capital",
        type=float,
        default=10000.0,
        help="Initial capital"
    )
    sweep_parser.add_argument(
        "--rebalance-frequency",
        dest="rebalance_frequency",
        choices=["daily", "weekly", "monthly", "quarterly", "yearly"],
        default="monthly",
        help="Rebalance frequency"
    )

    # Costs
    sweep_parser.add_argument(
        "--no-costs",
        action="store_true",
        default=False,
        help="Disable transaction costs"
    )
    sweep_parser.add_argument(
        "--cost-bps",
        type=float,
        default=None,
        help="Transaction costs in basis points (e.g., 10 = 0.1%%)"
    )
    sweep_parser.add_argument(
        "--cost-profile-file",
        default=None,
        help="Path to JSON file with per-ticker/asset-class execution cost overrides"
    )
    sweep_parser.add_argument(
        "--execution-lag-days",
        type=int,
        default=0,
        help="Execution lag in trading days (0 = same-day, 1 = T+1)"
    )
    sweep_parser.add_argument(
        "--t-plus-one",
        action="store_true",
        default=False,
        help="Shortcut for --execution-lag-days 1"
    )
    sweep_parser.add_argument(
        "--max-volume-participation",
        type=float,
        default=None,
        help="Max fraction of daily share volume per trade (e.g., 0.1 = 10%%)"
    )
    sweep_parser.add_argument(
        "--min-daily-dollar-volume",
        type=float,
        default=0.0,
        help="Skip trades when ticker daily notional volume is below this threshold"
    )
    sweep_parser.add_argument(
        "--liquidity-on-missing-volume",
        choices=["allow", "skip"],
        default="allow",
        help="Behavior when volume data is missing: allow trade or skip trade"
    )
    _add_preset_profile_arg(sweep_parser)
    _add_risk_overlay_args(sweep_parser)

    # Taxes
    sweep_parser.add_argument(
        "--no-tax",
        action="store_true",
        default=False,
        help="Disable German tax model"
    )
    sweep_parser.add_argument(
        "--tax-rate",
        type=float,
        default=0.26375,
        help="Tax rate (default: 26.375%% = Abgeltungssteuer + Soli)"
    )
    sweep_parser.add_argument(
        "--allowance",
        type=float,
        default=1000.0,
        help="Freistellungsauftrag in EUR (1000 single, 2000 joint)"
    )
    sweep_parser.add_argument(
        "--cash-rate",
        type=float,
        default=0.0,
        help="Annual interest rate applied to uninvested cash (e.g., 0.02 = 2%%)"
    )
    sweep_parser.add_argument(
        "--metric-basis",
        "--terminal-valuation",
        dest="metric_basis",
        choices=["gross", "net_realized", "net_liquidation"],
        default=None,
        help="Metric basis: gross (no tax), net_realized (realized tax only), "
             "net_liquidation (virtual liquidation with tax on unrealized). "
             "Default: net_liquidation if tax enabled, gross otherwise."
    )
    sweep_parser.add_argument(
        "--liquidate-at-window-end",
        "--liquidate-at-end",
        action="store_true",
        default=False,
        help="Alias for --metric-basis net_liquidation"
    )

    # Benchmark
    sweep_parser.add_argument(
        "--benchmark-ticker",
        default="SPY",
        help="Benchmark ticker symbol (default: SPY = S&P 500)"
    )

    # Execution
    sweep_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)"
    )
    sweep_parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop on first error instead of skipping"
    )
    sweep_parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: results/sweep_<timestamp>)"
    )
    sweep_parser.add_argument(
        "--allow-universe-lookahead",
        action="store_true",
        default=False,
        help="Allow strategies to use non-point-in-time universe data for historical backtests. "
             "WARNING: This may introduce survivorship/selection bias!"
    )
    sweep_parser.add_argument(
        "--align",
        choices=["intersection", "ffill"],
        default="ffill",
        help="Data alignment mode: 'intersection' or 'ffill' (default: ffill)"
    )
    add_skip_failed_flags(
        sweep_parser,
        "Skip tickers that fail to download (delisted, etc.)"
    )
    sweep_parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Disable pre/post-run validation checks"
    )
    sweep_parser.add_argument(
        "--drip",
        action="store_true",
        default=False,
        help="Enable Dividend Reinvestment Plan (DRIP)"
    )
    sweep_parser.add_argument(
        "--params-file",
        default=None,
        help="JSON file with optimized parameters (from batch-optimize)"
    )
    _add_external_features_arguments(sweep_parser)
    sweep_parser.set_defaults(func=cmd_sweep)

    # metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Display metrics for a strategy")
    metrics_parser.add_argument("strategy", help="Path to strategy Python file")
    metrics_parser.add_argument("-s", "--start", default="2010-01-01", help="Start date")
    metrics_parser.add_argument("-e", "--end", default=None, help="End date")
    metrics_parser.set_defaults(func=cmd_metrics)

    # signals command
    signals_parser = subparsers.add_parser(
        "signals",
        help="Generate live trading signals",
        description="Generate BUY/SELL/HOLD signals for portfolio management."
    )
    signals_parser.add_argument("strategy", help="Path to strategy Python file")
    signals_parser.add_argument(
        "-p", "--param",
        action="append",
        dest="params",
        metavar="NAME=VALUE",
        help="Strategy parameter override (can be used multiple times). "
             "Example: --param top_n=10 --param lookback_days=252"
    )
    signals_parser.add_argument(
        "--rebalance-frequency",
        dest="rebalance_frequency",
        choices=["daily", "weekly", "monthly", "quarterly", "yearly"],
        default=None,
        help="Override strategy's rebalance frequency"
    )
    signals_parser.add_argument(
        "--portfolio",
        default=None,
        help="Path to portfolio JSON file (for comparing current vs target)"
    )
    signals_parser.add_argument(
        "-d", "--date",
        default=None,
        help="Date to generate signals for (default: today). Format: YYYY-MM-DD"
    )
    signals_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (JSON or CSV based on extension)"
    )
    signals_parser.add_argument(
        "-f", "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)"
    )
    add_skip_failed_flags(
        signals_parser,
        "Skip tickers that fail to download (delisted, etc.)"
    )
    signals_parser.add_argument(
        "--drift-tolerance",
        type=float,
        default=0.005,
        help="Weight drift tolerance for reconciliation (default: 0.005 = 0.5%%)",
    )
    _add_exposure_policy_args(signals_parser)
    signals_parser.add_argument(
        "--meta-enable",
        action="store_true",
        default=False,
        help="Enable meta decisioning with evidence-gated strategy switching",
    )
    signals_parser.add_argument(
        "--meta-candidate",
        dest="meta_candidates",
        action="append",
        default=[],
        help="Candidate strategy file path (repeatable)",
    )
    signals_parser.add_argument(
        "--meta-candidates-file",
        default=None,
        help="JSON file with candidate list [{\"strategy\": ..., \"params\": {...}}]",
    )
    signals_parser.add_argument(
        "--meta-params-source",
        choices=["preset_first", "strategy_defaults", "manual_only"],
        default="preset_first",
        help="Candidate parameter source for meta decisioning",
    )
    signals_parser.add_argument(
        "--meta-preset-file",
        default=None,
        help="Optional optimized-params JSON to seed candidate params",
    )
    signals_parser.add_argument(
        "--meta-scoring",
        choices=["hybrid", "gate_only", "performance_only"],
        default="hybrid",
        help="Meta candidate scoring mode",
    )
    signals_parser.add_argument(
        "--meta-confirm-points",
        type=int,
        default=2,
        help="Consecutive confirmation points required before switch (default: 2)",
    )
    signals_parser.add_argument(
        "--meta-switch-margin",
        type=float,
        default=0.10,
        help="Minimum score edge required for switch (default: 0.10)",
    )
    signals_parser.add_argument(
        "--meta-decision-cadence",
        choices=["run_check_rebalance_switch", "immediate", "monthly_fixed"],
        default="run_check_rebalance_switch",
        help="Decision cadence gate (default: run_check_rebalance_switch)",
    )
    signals_parser.add_argument(
        "--meta-plan-mode",
        choices=["recommendation_only", "recommendation_with_portfolio_plan", "always_plan"],
        default="recommendation_with_portfolio_plan",
        help="Switch plan generation mode",
    )
    signals_parser.add_argument(
        "--meta-evidence-required",
        dest="meta_evidence_required",
        action="store_true",
        help="Require valid historical evidence gate for switch (default: enabled)",
    )
    signals_parser.add_argument(
        "--no-meta-evidence-required",
        dest="meta_evidence_required",
        action="store_false",
        help="Allow switch without evidence gate",
    )
    signals_parser.set_defaults(meta_evidence_required=True)
    signals_parser.add_argument(
        "--meta-evidence-profile",
        choices=["defensiv", "ausgewogen", "aggressiv", "custom"],
        default="ausgewogen",
        help="Evidence profile for hard switch gate",
    )
    signals_parser.add_argument(
        "--meta-evidence-max-age-days",
        type=int,
        default=30,
        help="Max age of evidence artifact in days (default: 30)",
    )
    signals_parser.add_argument(
        "--meta-evidence-artifact-path",
        default=None,
        help="Optional explicit evidence artifact JSON path",
    )
    signals_parser.add_argument(
        "--meta-gate-fail-action",
        choices=["hold_current", "manual_override", "fallback_safe"],
        default="hold_current",
        help="Action when evidence gate fails (default: hold_current)",
    )
    signals_parser.add_argument(
        "--meta-regime-mode",
        choices=["none", "strategy_fragility"],
        default="strategy_fragility",
        help="Enable generic strategy fragility buckets for regime-aware switching",
    )
    signals_parser.add_argument(
        "--meta-regime-profile",
        choices=["defensiv", "ausgewogen", "aggressiv", "custom"],
        default="ausgewogen",
        help="Fragility bucket profile (default: ausgewogen)",
    )
    signals_parser.add_argument(
        "--meta-alpha-tie-band",
        type=float,
        default=None,
        help="Allowed performance gap for fragile-state challenger switches",
    )
    signals_parser.add_argument(
        "--meta-stress-alpha-tolerance",
        type=float,
        default=None,
        help="Allowed performance gap for stressed-state challenger switches",
    )
    signals_parser.add_argument(
        "--meta-conditioned-min-windows",
        type=int,
        default=None,
        help="Override minimum conditioned OOS windows for fragility-driven evidence",
    )
    _add_external_features_arguments(signals_parser)
    signals_parser.set_defaults(func=cmd_signals)

    # meta-evidence command
    meta_evidence_parser = subparsers.add_parser(
        "meta-evidence",
        help="Generate historical OOS evidence artifact for strategy switching",
    )
    meta_evidence_parser.add_argument(
        "--current-strategy",
        required=True,
        help="Current strategy file path",
    )
    meta_evidence_parser.add_argument(
        "--target-strategy",
        required=True,
        help="Target strategy file path",
    )
    meta_evidence_parser.add_argument(
        "--current-param",
        dest="current_params",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Current strategy parameter override (repeatable)",
    )
    meta_evidence_parser.add_argument(
        "--target-param",
        dest="target_params",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Target strategy parameter override (repeatable)",
    )
    meta_evidence_parser.add_argument(
        "--profile",
        dest="evidence_profile",
        choices=["defensiv", "ausgewogen", "aggressiv", "custom"],
        default="ausgewogen",
        help="Evidence gate profile",
    )
    meta_evidence_parser.add_argument(
        "--custom-threshold",
        dest="custom_thresholds",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Custom threshold override, only applied with --profile custom",
    )
    meta_evidence_parser.add_argument(
        "--evidence-max-age-days",
        type=int,
        default=30,
        help="Staleness gate in days (default: 30)",
    )
    meta_evidence_parser.add_argument(
        "--evidence-artifact-path",
        default=None,
        help="Optional explicit output artifact path",
    )
    meta_evidence_parser.add_argument(
        "-d",
        "--date",
        default=None,
        help="As-of date (YYYY-MM-DD, default: today)",
    )
    meta_evidence_parser.add_argument("-s", "--start", default="2010-01-01", help="Data start date")
    meta_evidence_parser.add_argument("--train-years", type=float, default=5.0, help="Train window in years")
    meta_evidence_parser.add_argument("--test-years", type=float, default=1.0, help="Test window in years")
    meta_evidence_parser.add_argument("--step-months", type=int, default=12, help="Step size in months")
    meta_evidence_parser.add_argument("--anchored", action="store_true", default=False, help="Use anchored windows")
    meta_evidence_parser.add_argument(
        "-c",
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital per window backtest",
    )
    meta_evidence_parser.add_argument("--costs", type=float, default=0.001, help="Transaction costs")
    meta_evidence_parser.add_argument(
        "--metric-basis",
        choices=["gross", "net_realized", "net_liquidation"],
        default="gross",
        help="Metric basis for evidence backtests",
    )
    meta_evidence_parser.add_argument(
        "--tuning-enabled",
        action="store_true",
        default=False,
        help="Enable 2-stage smart tuning over confirm_points/switch_margin",
    )
    meta_evidence_parser.add_argument(
        "--grid-confirm-points",
        default="1,2,3",
        help="Comma-separated confirm_points grid (default: 1,2,3)",
    )
    meta_evidence_parser.add_argument(
        "--grid-switch-margin",
        default="0.05,0.10,0.15",
        help="Comma-separated switch_margin grid (default: 0.05,0.10,0.15)",
    )
    meta_evidence_parser.add_argument(
        "--max-combinations",
        type=int,
        default=120,
        help="Maximum combinations to evaluate (default: 120)",
    )
    meta_evidence_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K candidates to advance to stage 2 (default: 10)",
    )
    add_skip_failed_flags(
        meta_evidence_parser,
        "Skip tickers that fail to download (delisted, etc.)",
    )
    meta_evidence_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional export file path for full artifact JSON",
    )
    meta_evidence_parser.set_defaults(func=cmd_meta_evidence)

    # meta-bootstrap command
    meta_bootstrap_parser = subparsers.add_parser(
        "meta-bootstrap",
        help="Select neutral start strategy from two candidates using bilateral evidence + fallback",
    )
    meta_bootstrap_parser.add_argument(
        "--strategy-a",
        required=True,
        help="First strategy file path",
    )
    meta_bootstrap_parser.add_argument(
        "--strategy-b",
        required=True,
        help="Second strategy file path",
    )
    meta_bootstrap_parser.add_argument(
        "--a-param",
        dest="strategy_a_params",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Strategy A parameter override (repeatable)",
    )
    meta_bootstrap_parser.add_argument(
        "--b-param",
        dest="strategy_b_params",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Strategy B parameter override (repeatable)",
    )
    meta_bootstrap_parser.add_argument(
        "--profile",
        dest="evidence_profile",
        choices=["defensiv", "ausgewogen", "aggressiv", "custom"],
        default="ausgewogen",
        help="Evidence gate profile",
    )
    meta_bootstrap_parser.add_argument(
        "--custom-threshold",
        dest="custom_thresholds",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Custom threshold override, only applied with --profile custom",
    )
    meta_bootstrap_parser.add_argument(
        "--evidence-max-age-days",
        type=int,
        default=30,
        help="Staleness gate in days (default: 30)",
    )
    meta_bootstrap_parser.add_argument(
        "-d",
        "--date",
        default=None,
        help="As-of date (YYYY-MM-DD, default: today)",
    )
    meta_bootstrap_parser.add_argument("-s", "--start", default="2010-01-01", help="Data start date")
    meta_bootstrap_parser.add_argument("--train-years", type=float, default=5.0, help="Train window in years")
    meta_bootstrap_parser.add_argument("--test-years", type=float, default=1.0, help="Test window in years")
    meta_bootstrap_parser.add_argument("--step-months", type=int, default=12, help="Step size in months")
    meta_bootstrap_parser.add_argument("--anchored", action="store_true", default=False, help="Use anchored windows")
    meta_bootstrap_parser.add_argument(
        "-c",
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital per backtest",
    )
    meta_bootstrap_parser.add_argument("--costs", type=float, default=0.001, help="Transaction costs")
    meta_bootstrap_parser.add_argument(
        "--metric-basis",
        choices=["gross", "net_realized", "net_liquidation"],
        default="gross",
        help="Metric basis for evidence and fallback backtests",
    )
    meta_bootstrap_parser.add_argument(
        "--fallback-cagr-tie-band-pp",
        type=float,
        default=1.0,
        help="If |CAGR edge| is below this, use tie-breaker (default: 1.0pp)",
    )
    meta_bootstrap_parser.add_argument(
        "--fallback-tie-breaker",
        choices=["maxdd", "sharpe"],
        default="maxdd",
        help="Fallback tie-breaker when CAGR edge is inside tie band (default: maxdd)",
    )
    meta_bootstrap_parser.add_argument(
        "--artifact-path",
        default=None,
        help="Optional explicit output artifact path",
    )
    add_skip_failed_flags(
        meta_bootstrap_parser,
        "Skip tickers that fail to download (delisted, etc.)",
    )
    meta_bootstrap_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional export file path for full artifact JSON",
    )
    meta_bootstrap_parser.set_defaults(func=cmd_meta_bootstrap)

    # assets command
    assets_parser = subparsers.add_parser("assets", help="List available assets")
    assets_parser.set_defaults(func=cmd_assets)

    # new command
    new_parser = subparsers.add_parser("new", help="Generate a new strategy template")
    new_parser.add_argument("name", help="Name for the new strategy")
    new_parser.set_defaults(func=cmd_new)

    # data subcommands
    data_parser = subparsers.add_parser("data", help="Data management commands")
    data_subparsers = data_parser.add_subparsers(dest="data_command", help="Data commands")

    # data download
    download_parser = data_subparsers.add_parser("download", help="Download and cache price data")
    download_parser.add_argument("tickers", nargs="+", help="Ticker symbols to download")
    download_parser.add_argument("-s", "--start", default="2000-01-01", help="Start date")
    download_parser.add_argument("-e", "--end", default=None, help="End date")
    download_parser.set_defaults(func=cmd_data_download)

    # data list
    list_parser = data_subparsers.add_parser("list", help="List cached data files")
    list_parser.set_defaults(func=cmd_data_list)

    # data clear
    clear_parser = data_subparsers.add_parser("clear", help="Clear all cached data")
    clear_parser.set_defaults(func=cmd_data_clear)

    # data provenance subcommands
    provenance_parser = data_subparsers.add_parser(
        "provenance",
        help="Manage manual data provenance metadata"
    )
    provenance_subparsers = provenance_parser.add_subparsers(
        dest="provenance_command",
        help="Provenance commands"
    )

    # data provenance add
    provenance_add_parser = provenance_subparsers.add_parser(
        "add",
        help="Register provenance for a manual data file",
    )
    provenance_add_parser.add_argument("file_path", help="Path to manual data file")
    provenance_add_parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset identifier (e.g., fundamentals_sp500, sentiment_export)"
    )
    provenance_add_parser.add_argument(
        "--source",
        default="",
        help="Source name (e.g., SeekingAlpha, SEC, FRED)"
    )
    provenance_add_parser.add_argument(
        "--quality-tag",
        choices=["official", "proxy", "community", "manual"],
        default="manual",
        help="Data quality tag"
    )
    provenance_add_parser.add_argument(
        "--as-of-date",
        default=None,
        help="As-of date for the data snapshot (YYYY-MM-DD)"
    )
    provenance_add_parser.add_argument(
        "--import-method",
        default="manual_upload",
        help="Import method description (default: manual_upload)"
    )
    provenance_add_parser.add_argument(
        "--license-note",
        default="",
        help="License/ToS note"
    )
    provenance_add_parser.add_argument(
        "--source-url",
        default=None,
        help="Optional source URL"
    )
    provenance_add_parser.add_argument(
        "--notes",
        default=None,
        help="Optional notes"
    )
    provenance_add_parser.add_argument(
        "--entry-id",
        default=None,
        help="Optional explicit entry id"
    )
    provenance_add_parser.add_argument(
        "--seekingalpha",
        action="store_true",
        default=False,
        help="Apply SeekingAlpha defaults (source/import-method/license note)"
    )
    provenance_add_parser.add_argument(
        "--registry",
        default=None,
        help="Optional custom provenance registry JSON path"
    )
    provenance_add_parser.add_argument(
        "-f", "--format",
        choices=["table", "json"],
        default="table",
        help="Output format"
    )
    provenance_add_parser.set_defaults(func=cmd_data_provenance_add)

    # data provenance list
    provenance_list_parser = provenance_subparsers.add_parser(
        "list",
        help="List manual data provenance entries",
    )
    provenance_list_parser.add_argument(
        "--dataset",
        default=None,
        help="Filter by dataset"
    )
    provenance_list_parser.add_argument(
        "--source",
        default=None,
        help="Filter by source"
    )
    provenance_list_parser.add_argument(
        "--registry",
        default=None,
        help="Optional custom provenance registry JSON path"
    )
    provenance_list_parser.add_argument(
        "-f", "--format",
        choices=["table", "json"],
        default="table",
        help="Output format"
    )
    provenance_list_parser.set_defaults(func=cmd_data_provenance_list)

    # data provenance show
    provenance_show_parser = provenance_subparsers.add_parser(
        "show",
        help="Show a manual data provenance entry",
    )
    provenance_show_parser.add_argument("entry_id", help="Provenance entry id")
    provenance_show_parser.add_argument(
        "--registry",
        default=None,
        help="Optional custom provenance registry JSON path"
    )
    provenance_show_parser.add_argument(
        "-f", "--format",
        choices=["table", "json"],
        default="table",
        help="Output format"
    )
    provenance_show_parser.set_defaults(func=cmd_data_provenance_show)

    # data provenance verify
    provenance_verify_parser = provenance_subparsers.add_parser(
        "verify",
        help="Verify manual data provenance files/checksums",
    )
    provenance_verify_parser.add_argument(
        "--skip-hash",
        action="store_true",
        default=False,
        help="Skip checksum verification and only check file existence"
    )
    provenance_verify_parser.add_argument(
        "--registry",
        default=None,
        help="Optional custom provenance registry JSON path"
    )
    provenance_verify_parser.add_argument(
        "-f", "--format",
        choices=["table", "json"],
        default="table",
        help="Output format"
    )
    provenance_verify_parser.set_defaults(func=cmd_data_provenance_verify)

    # features subcommands (external features pipeline)
    features_parser = subparsers.add_parser(
        "features",
        help="Manage external feature snapshots (analyst/news/ML)",
    )
    features_subparsers = features_parser.add_subparsers(
        dest="features_command", help="Features commands"
    )

    features_pull_parser = features_subparsers.add_parser(
        "pull",
        help="Pull a snapshot from a registered adapter",
    )
    features_pull_parser.add_argument(
        "--dataset", required=True, help="Dataset identifier (e.g., mock_analyst)"
    )
    features_pull_parser.add_argument(
        "--as-of", required=True, help="Snapshot as-of date (YYYY-MM-DD)"
    )
    features_pull_parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated ticker list (required except for mock_analyst)",
    )
    features_pull_parser.add_argument(
        "--root", default=None, help="Snapshot root (default: data/external_features/snapshots)"
    )
    features_pull_parser.add_argument(
        "--registry", default=None, help="Optional custom provenance registry JSON path"
    )
    features_pull_parser.add_argument(
        "--force", action="store_true", default=False, help="Re-fetch even if cache exists"
    )
    # Phase C: pull-time sentiment engine and intraday-cutoff.
    features_pull_parser.add_argument(
        "--news-engine",
        choices=["mock", "vader", "finbert"],
        default=None,
        help="Sentiment engine for news adapters (default depends on adapter).",
    )
    features_pull_parser.add_argument(
        "--news-intraday-cutoff",
        default=None,
        help="HH:MM UTC cutoff for headlines aggregated into the snapshot.",
    )
    # Phase D: pull-time ML bundle override + stacking-only mode.
    features_pull_parser.add_argument(
        "--ml-model-bundle",
        default=None,
        help=(
            "Override path to a specific ML bundle directory (manifest.json + "
            "stage pickles). Default selects the latest with available_from <= as_of."
        ),
    )
    features_pull_parser.add_argument(
        "--ml-stacking-only",
        action="store_true",
        default=False,
        help=(
            "Run only the Stage-3 stacking head on pre-cached Stage 1/2 OOF "
            "outputs (skips Stage-1 cross-sectional + Stage-2 residual fits)."
        ),
    )
    features_pull_parser.add_argument(
        "-f", "--format", choices=["table", "json"], default="table", help="Output format"
    )
    features_pull_parser.set_defaults(func=cmd_features_pull)

    features_list_parser = features_subparsers.add_parser(
        "list",
        help="List external feature snapshot files and registry status",
    )
    features_list_parser.add_argument(
        "--dataset", default=None, help="Filter by dataset"
    )
    features_list_parser.add_argument(
        "--root", default=None, help="Snapshot root (default: data/external_features/snapshots)"
    )
    features_list_parser.add_argument(
        "--registry", default=None, help="Optional custom provenance registry JSON path"
    )
    features_list_parser.add_argument(
        "-f", "--format", choices=["table", "json"], default="table", help="Output format"
    )
    features_list_parser.set_defaults(func=cmd_features_list)

    features_verify_parser = features_subparsers.add_parser(
        "verify",
        help="Verify external feature snapshot schema and provenance hashes",
    )
    features_verify_parser.add_argument(
        "--dataset", default=None, help="Filter by dataset"
    )
    features_verify_parser.add_argument(
        "--root", default=None, help="Snapshot root (default: data/external_features/snapshots)"
    )
    features_verify_parser.add_argument(
        "--registry", default=None, help="Optional custom provenance registry JSON path"
    )
    features_verify_parser.add_argument(
        "-f", "--format", choices=["table", "json"], default="table", help="Output format"
    )
    features_verify_parser.set_defaults(func=cmd_features_verify)

    # Phase D: `ml train` subcommand (T-0227).
    ml_parser = subparsers.add_parser(
        "ml",
        help="ML forecast training and bundle management",
    )
    ml_subparsers = ml_parser.add_subparsers(
        dest="ml_command", help="ML commands"
    )
    ml_train_parser = ml_subparsers.add_parser(
        "train",
        help="Train an ML forecast bundle (walk-forward, label-purge OOF).",
    )
    ml_train_parser.add_argument(
        "--start", required=True, help="Training start date YYYY-MM-DD"
    )
    ml_train_parser.add_argument(
        "--end", required=True, help="Training end date YYYY-MM-DD"
    )
    ml_train_parser.add_argument(
        "--horizons",
        default="21,63,252",
        help="Comma-separated forward-return horizons in trading days (default 21,63,252).",
    )
    ml_train_parser.add_argument(
        "--models",
        default="lightgbm",
        help="Comma-separated model families to train (lightgbm,xgboost).",
    )
    ml_train_parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated tickers — required unless --universe-source is set (Codex D4).",
    )
    ml_train_parser.add_argument(
        "--universe-source",
        default=None,
        help="Path to PIT-Universe CSV (one ticker per row). Preferred over --tickers.",
    )
    ml_train_parser.add_argument(
        "--output-dir",
        default="data/external_features/ml/models",
        help="Bundle output root (default: data/external_features/ml/models).",
    )
    ml_train_parser.add_argument(
        "--inner-train-years",
        type=int,
        default=3,
        help="Nested inner-CV train horizon in years (default 3).",
    )
    ml_train_parser.add_argument(
        "--inner-test-months",
        type=int,
        default=6,
        help="Nested inner-CV test horizon in months (default 6).",
    )
    ml_train_parser.add_argument(
        "--grid-size",
        type=int,
        default=4,
        help="Hyperparameter grid size sampled per inner fold (default 4).",
    )
    ml_train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed pinned in the manifest (default 42).",
    )
    ml_train_parser.set_defaults(func=cmd_ml_train)

    # Phase E2: `backtest live` Subcommand (T-0372).
    live_parser = subparsers.add_parser(
        "live",
        help=(
            "Phase E2: execution-plan generation (paper/dry/brief only; "
            "no real broker submission)."
        ),
    )
    live_subparsers = live_parser.add_subparsers(
        dest="live_command", help="Live commands"
    )

    live_plan_parser = live_subparsers.add_parser(
        "plan",
        help="Emit an order plan (paper/dry/brief) from a SignalReport.",
    )
    live_plan_parser.add_argument(
        "--signals-report",
        required=True,
        help=(
            "Path to a SignalReport JSON (Codex R2.13: Phase E does not "
            "support strategy-fallback)."
        ),
    )
    live_plan_parser.add_argument(
        "--broker",
        required=True,
        choices=[
            "dry_run",
            "ibkr_basket_csv",
            "alpaca_paper_preview",
            "trade_republic_brief",
            "maxblue_brief",
        ],
        help="Execution-plan adapter (all Phase E adapters are plan_only).",
    )
    live_plan_parser.add_argument(
        "--portfolio",
        default=None,
        help=(
            "Optional path to Portfolio JSON. Used to derive "
            "portfolio_snapshot_hash (Codex R3.4)."
        ),
    )
    live_plan_parser.add_argument(
        "--position-drift-tol",
        type=float,
        default=0.005,
        help="Tolerance (fraction) for position-reconciliation drift.",
    )
    live_plan_parser.add_argument(
        "--new-run",
        default="",
        help=(
            "Optional token to force a new run_id (default reuses the "
            "deterministic hash of SignalReport + broker + portfolio)."
        ),
    )
    live_plan_parser.add_argument(
        "--log-path",
        default="results/live_orders/order_plan_log.jsonl",
        help="OrderPlanLog NDJSON path (Codex R3.5).",
    )
    live_plan_parser.add_argument(
        "--allow-price-warnings",
        action="store_true",
        help=(
            "Allow order-plan emission even when the SignalReport contains "
            "price_warnings. Use only after manually verifying the quotes."
        ),
    )
    live_plan_parser.set_defaults(func=cmd_live_plan)

    live_status_parser = live_subparsers.add_parser(
        "status",
        help="Read the OrderPlanLog and print recent entries.",
    )
    live_status_parser.add_argument(
        "--log-path",
        default="results/live_orders/order_plan_log.jsonl",
    )
    live_status_parser.add_argument(
        "--since",
        default=None,
        help="ISO-Date filter (YYYY-MM-DD).",
    )
    live_status_parser.set_defaults(func=cmd_live_status)

    live_reconcile_parser = live_subparsers.add_parser(
        "reconcile",
        help="Reconcile a Portfolio JSON against the broker adapter only.",
    )
    live_reconcile_parser.add_argument(
        "--broker",
        required=True,
        choices=[
            "dry_run",
            "ibkr_basket_csv",
            "alpaca_paper_preview",
            "trade_republic_brief",
            "maxblue_brief",
        ],
    )
    live_reconcile_parser.add_argument(
        "--portfolio",
        required=True,
    )
    live_reconcile_parser.set_defaults(func=cmd_live_reconcile)

    live_update_portfolio_parser = live_subparsers.add_parser(
        "update-portfolio",
        help="Write executed manual order results back to a Portfolio JSON.",
    )
    live_update_portfolio_parser.add_argument(
        "--portfolio",
        required=True,
        help="Path to the manual broker portfolio JSON.",
    )
    live_update_portfolio_parser.add_argument(
        "--position",
        action="append",
        metavar="TICKER=SHARES",
        help=(
            "Executed final share count for one ticker. Can be passed "
            "multiple times, e.g. --position SPY=10.5."
        ),
    )
    live_update_portfolio_parser.add_argument(
        "--signals-report",
        default=None,
        help=(
            "Optional SignalReport JSON; applies each order row's "
            "target_shares. Use only after executing that exact plan."
        ),
    )
    live_update_portfolio_parser.add_argument(
        "--stand",
        default=None,
        help="Portfolio stand date to write (YYYY-MM-DD, default today).",
    )
    live_update_portfolio_parser.set_defaults(func=cmd_live_update_portfolio)

    # optimize command
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Optimize strategy parameters",
        description="Find optimal parameters for a strategy by testing all combinations."
    )
    optimize_parser.add_argument("strategy", help="Path to strategy Python file")
    optimize_parser.add_argument(
        "-p", "--param",
        action="append",
        dest="params",
        metavar="NAME=VAL1,VAL2,...",
        help="Parameter to optimize (can be used multiple times). "
             "Example: --param lookback_days=126,189,252 --param top_n=5,10"
    )
    optimize_parser.add_argument(
        "--rebalance-frequency",
        dest="rebalance_frequencies",
        default="monthly",
        help="Comma-separated rebalance frequencies to test. "
             "Example: monthly,quarterly,yearly (default: monthly)"
    )
    optimize_parser.add_argument(
        "-m", "--metric",
        default="sharpe_ratio",
        choices=["sharpe_ratio", "sortino_ratio", "cagr", "calmar_ratio", "max_drawdown"],
        help="Metric to optimize (default: sharpe_ratio)"
    )
    optimize_parser.add_argument(
        "--minimize",
        action="store_true",
        default=False,
        help="Minimize metric instead of maximize (e.g., for max_drawdown)"
    )
    optimize_parser.add_argument("-s", "--start", default="2010-01-01", help="Start date")
    optimize_parser.add_argument("-e", "--end", default=None, help="End date")
    optimize_parser.add_argument("-c", "--capital", type=float, default=10000.0, help="Initial capital")
    optimize_parser.add_argument("--costs", type=float, default=0.001, help="Transaction costs")
    optimize_parser.add_argument(
        "--cost-profile-file",
        default=None,
        help="Path to JSON file with per-ticker/asset-class execution cost overrides"
    )
    optimize_parser.add_argument(
        "--execution-lag-days",
        type=int,
        default=0,
        help="Execution lag in trading days (0 = same-day, 1 = T+1)"
    )
    optimize_parser.add_argument(
        "--t-plus-one",
        action="store_true",
        default=False,
        help="Shortcut for --execution-lag-days 1"
    )
    optimize_parser.add_argument(
        "--max-volume-participation",
        type=float,
        default=None,
        help="Max fraction of daily share volume per trade (e.g., 0.1 = 10%%)"
    )
    optimize_parser.add_argument(
        "--min-daily-dollar-volume",
        type=float,
        default=0.0,
        help="Skip trades when ticker daily notional volume is below this threshold"
    )
    optimize_parser.add_argument(
        "--liquidity-on-missing-volume",
        choices=["allow", "skip"],
        default="allow",
        help="Behavior when volume data is missing: allow trade or skip trade"
    )
    _add_preset_profile_arg(optimize_parser)
    _add_risk_overlay_args(optimize_parser)
    optimize_parser.add_argument(
        "--no-tax",
        action="store_true",
        default=False,
        help="Disable German tax model"
    )
    optimize_parser.add_argument(
        "--liquidate-at-end",
        action="store_true",
        default=False,
        help="Use net liquidation value for metrics (virtual sell at end)"
    )
    optimize_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show top N results (default: 10)"
    )
    optimize_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file for results (CSV)"
    )
    # Walk-Forward options
    optimize_parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=False,
        help="Enable walk-forward optimization (train/test split)"
    )
    optimize_parser.add_argument(
        "--walk-forward-nested",
        action="store_true",
        default=False,
        help="Enable nested walk-forward (inner optimization inside each outer train window)"
    )
    optimize_parser.add_argument(
        "--train-years",
        type=float,
        default=5.0,
        help="Training window length in years (default: 5)"
    )
    optimize_parser.add_argument(
        "--test-years",
        type=float,
        default=1.0,
        help="Test window length in years (default: 1)"
    )
    optimize_parser.add_argument(
        "--anchored",
        action="store_true",
        default=False,
        help="Use anchored (expanding) training window instead of rolling"
    )
    optimize_parser.add_argument(
        "--step-months",
        type=int,
        default=12,
        help="Step size between windows in months (default: 12)"
    )
    optimize_parser.add_argument(
        "--inner-train-years",
        type=float,
        default=3.0,
        help="Inner training window length in years for nested walk-forward (default: 3)"
    )
    optimize_parser.add_argument(
        "--inner-test-years",
        type=float,
        default=1.0,
        help="Inner test window length in years for nested walk-forward (default: 1)"
    )
    optimize_parser.add_argument(
        "--inner-step-months",
        type=int,
        default=6,
        help="Inner step size in months for nested walk-forward (default: 6)"
    )
    optimize_parser.add_argument(
        "--inner-anchored",
        action="store_true",
        default=False,
        help="Use anchored inner windows in nested walk-forward"
    )
    optimize_parser.add_argument(
        "--allow-universe-lookahead",
        action="store_true",
        default=False,
        help="Allow strategies to use non-point-in-time universe data for historical backtests. "
             "WARNING: This may introduce survivorship/selection bias!"
    )
    optimize_parser.add_argument(
        "--align",
        choices=["intersection", "ffill"],
        default="ffill",
        help="Data alignment mode: 'intersection' or 'ffill' (default: ffill)"
    )
    add_skip_failed_flags(
        optimize_parser,
        "Skip tickers that fail to download (delisted, etc.)"
    )
    optimize_parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Disable pre/post-run validation checks"
    )
    optimize_parser.add_argument(
        "--drip",
        action="store_true",
        default=False,
        help="Enable Dividend Reinvestment Plan (DRIP)"
    )
    _add_external_features_arguments(optimize_parser)
    optimize_parser.set_defaults(func=cmd_optimize)

    # batch-optimize command
    batch_opt_parser = subparsers.add_parser(
        "batch-optimize",
        help="Optimize parameters for all strategies at once",
        description="Find optimal parameters for each strategy and rank them."
    )
    batch_opt_parser.add_argument(
        "strategies",
        nargs="+",
        help="Strategy files or glob patterns (e.g., strategies/[!_]*.py)"
    )
    batch_opt_parser.add_argument(
        "-m", "--metric",
        default="sharpe_ratio",
        choices=["sharpe_ratio", "sortino_ratio", "cagr", "calmar_ratio", "max_drawdown"],
        help="Metric to optimize (default: sharpe_ratio)"
    )
    batch_opt_parser.add_argument(
        "--minimize",
        action="store_true",
        default=False,
        help="Minimize metric instead of maximize"
    )
    batch_opt_parser.add_argument(
        "--rebalance-frequency",
        dest="rebalance_frequencies",
        default="monthly,quarterly",
        help="Comma-separated rebalance frequencies to test (default: monthly,quarterly)"
    )
    batch_opt_parser.add_argument("-s", "--start", default="2010-01-01", help="Start date")
    batch_opt_parser.add_argument("-e", "--end", default=None, help="End date")
    batch_opt_parser.add_argument("-c", "--capital", type=float, default=10000.0, help="Initial capital")
    batch_opt_parser.add_argument("--costs", type=float, default=0.001, help="Transaction costs")
    batch_opt_parser.add_argument(
        "--cost-profile-file",
        default=None,
        help="Path to JSON file with per-ticker/asset-class execution cost overrides"
    )
    batch_opt_parser.add_argument(
        "--execution-lag-days",
        type=int,
        default=0,
        help="Execution lag in trading days (0 = same-day, 1 = T+1)"
    )
    batch_opt_parser.add_argument(
        "--t-plus-one",
        action="store_true",
        default=False,
        help="Shortcut for --execution-lag-days 1"
    )
    batch_opt_parser.add_argument(
        "--max-volume-participation",
        type=float,
        default=None,
        help="Max fraction of daily share volume per trade (e.g., 0.1 = 10%%)"
    )
    batch_opt_parser.add_argument(
        "--min-daily-dollar-volume",
        type=float,
        default=0.0,
        help="Skip trades when ticker daily notional volume is below this threshold"
    )
    batch_opt_parser.add_argument(
        "--liquidity-on-missing-volume",
        choices=["allow", "skip"],
        default="allow",
        help="Behavior when volume data is missing: allow trade or skip trade"
    )
    _add_preset_profile_arg(batch_opt_parser)
    _add_risk_overlay_args(batch_opt_parser)
    batch_opt_parser.add_argument(
        "--no-tax",
        action="store_true",
        default=False,
        help="Disable German tax model"
    )
    batch_opt_parser.add_argument(
        "--liquidate-at-end",
        action="store_true",
        default=False,
        help="Use net liquidation value for metrics"
    )
    batch_opt_parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: results/batch_optimize_<timestamp>)"
    )
    batch_opt_parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop on first error"
    )
    batch_opt_parser.add_argument(
        "--align",
        choices=["intersection", "ffill"],
        default="ffill",
        help="Data alignment mode: 'intersection' or 'ffill' (default: ffill)"
    )
    add_skip_failed_flags(
        batch_opt_parser,
        "Skip tickers that fail to download (delisted, etc.)"
    )
    batch_opt_parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Disable pre/post-run validation checks"
    )
    batch_opt_parser.add_argument(
        "--drip",
        action="store_true",
        default=False,
        help="Enable Dividend Reinvestment Plan (DRIP)"
    )
    # Walk-Forward options
    batch_opt_parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=False,
        help="Use walk-forward analysis for out-of-sample validation"
    )
    batch_opt_parser.add_argument(
        "--walk-forward-nested",
        action="store_true",
        default=False,
        help="Use nested walk-forward (inner optimization inside each outer train window)"
    )
    batch_opt_parser.add_argument(
        "--train-years",
        type=float,
        default=5.0,
        help="Walk-forward training window in years (default: 5)"
    )
    batch_opt_parser.add_argument(
        "--test-years",
        type=float,
        default=1.0,
        help="Walk-forward test window in years (default: 1)"
    )
    batch_opt_parser.add_argument(
        "--step-months",
        type=int,
        default=12,
        help="Walk-forward step size in months (default: 12)"
    )
    batch_opt_parser.add_argument(
        "--anchored",
        action="store_true",
        default=False,
        help="Use anchored (expanding) training window in walk-forward mode"
    )
    batch_opt_parser.add_argument(
        "--inner-train-years",
        type=float,
        default=3.0,
        help="Inner training window length in years for nested walk-forward (default: 3)"
    )
    batch_opt_parser.add_argument(
        "--inner-test-years",
        type=float,
        default=1.0,
        help="Inner test window length in years for nested walk-forward (default: 1)"
    )
    batch_opt_parser.add_argument(
        "--inner-step-months",
        type=int,
        default=6,
        help="Inner step size in months for nested walk-forward (default: 6)"
    )
    batch_opt_parser.add_argument(
        "--inner-anchored",
        action="store_true",
        default=False,
        help="Use anchored inner windows in nested walk-forward"
    )
    _add_external_features_arguments(batch_opt_parser)
    batch_opt_parser.set_defaults(func=cmd_batch_optimize)

    # web command
    web_parser = subparsers.add_parser(
        "web",
        help="Start the web frontend server"
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    web_parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development"
    )
    web_parser.set_defaults(func=cmd_web)

    return parser


def app():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "data" and args.data_command is None:
        parser.parse_args(["data", "--help"])
        sys.exit(0)
    if (
        args.command == "data"
        and args.data_command == "provenance"
        and getattr(args, "provenance_command", None) is None
    ):
        parser.parse_args(["data", "provenance", "--help"])
        sys.exit(0)

    _apply_preset_profile_defaults(args)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    app()
