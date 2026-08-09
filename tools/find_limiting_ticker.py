#!/usr/bin/env python3
"""
Find the ticker(s) that limit the start date of a strategy's backtest.

When backtesting, the DataLoader aligns all price series by dropping rows
with any NaN (implicit inner join). This script identifies which ticker(s)
have the latest start date and thus limit the common date range.

Usage (run from project root with poetry):
    poetry run python tools/find_limiting_ticker.py --strategy strategies/momentum_topn_universe.py
    poetry run python tools/find_limiting_ticker.py --strategy strategies/dual_momentum.py --start 2000-01-01
    poetry run python tools/find_limiting_ticker.py --tickers SPY EFA BND URTH

Or if you have activated the virtual environment:
    python tools/find_limiting_ticker.py --strategy strategies/momentum_topn_universe.py
"""

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
    import yfinance as yf
except ImportError as e:
    print(f"Error: {e}")
    print("\nThis script requires pandas and yfinance.")
    print("Run it using poetry from the project root:")
    print("  poetry run python tools/find_limiting_ticker.py --strategy <strategy.py>")
    sys.exit(1)


def load_strategy_from_file(path: str):
    """Load a strategy from a Python file and return its assets list."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    # Add src to path for imports
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    spec = importlib.util.spec_from_file_location("strategy_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy_module"] = module
    spec.loader.exec_module(module)

    # Import Strategy base class
    from backtest.strategy import Strategy

    # First check for pre-instantiated strategy
    if hasattr(module, 'strategy'):
        obj = getattr(module, 'strategy')
        if isinstance(obj, Strategy):
            return obj

    # Otherwise find and instantiate first Strategy subclass
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, Strategy)
            and obj is not Strategy
        ):
            return obj()

    raise ValueError(f"No Strategy subclass found in {path}")


def download_ticker_data(ticker: str, start: str) -> tuple[str, pd.Timestamp | None, str | None]:
    """
    Download data for a single ticker and return its first valid date.

    Returns:
        (ticker, first_valid_date, error_message)
    """
    try:
        data = yf.download(
            ticker,
            start=start,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data.empty:
            return (ticker, None, "No data returned")

        # Handle both single and multi-column DataFrames
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"][ticker] if ticker in data["Close"].columns else data["Close"].iloc[:, 0]
        elif "Close" in data.columns:
            close = data["Close"]
        else:
            return (ticker, None, "No 'Close' column found")

        first_valid = close.first_valid_index()
        if first_valid is None:
            return (ticker, None, "All Close values are NaN")

        return (ticker, first_valid, None)

    except Exception as e:
        return (ticker, None, str(e))


def find_limiting_tickers(
    tickers: list[str],
    start: str = "1900-01-01",
    verbose: bool = True
) -> dict:
    """
    Find which ticker(s) limit the common start date.

    Returns dict with:
        - limiting_date: The latest start date (limits the backtest)
        - culprits: List of tickers with this limiting date
        - all_tickers: Dict of ticker -> start_date (sorted by date)
        - failed: List of tickers that failed to download
    """
    results = {}
    failed = []

    if verbose:
        print(f"\nDownloading data for {len(tickers)} tickers...")
        print("-" * 50)

    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"  [{i:3d}/{len(tickers)}] {ticker:<10}", end=" ")

        ticker_name, first_date, error = download_ticker_data(ticker, start)

        if error:
            failed.append((ticker, error))
            if verbose:
                print(f"FAILED: {error}")
        else:
            results[ticker] = first_date
            if verbose:
                print(f"-> {first_date.strftime('%Y-%m-%d')}")

    if not results:
        return {
            "limiting_date": None,
            "culprits": [],
            "all_tickers": {},
            "failed": failed,
        }

    # Find the limiting (latest) start date
    limiting_date = max(results.values())
    culprits = [t for t, d in results.items() if d == limiting_date]

    # Sort by date (latest first)
    sorted_tickers = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    return {
        "limiting_date": limiting_date,
        "culprits": culprits,
        "all_tickers": sorted_tickers,
        "failed": failed,
    }


def find_combined_limiting_ticker(
    tickers: list[str],
    start: str = "1900-01-01",
    verbose: bool = True,
    include_fx: bool = True,
) -> dict:
    """
    Download all tickers combined and find the ACTUAL limiting start date
    after dropna() alignment (simulates DataLoader behavior).

    This catches cases where gaps in data (not just late starts) limit the range.

    Args:
        tickers: List of ticker symbols
        start: Start date for download
        verbose: Print progress
        include_fx: Also check EUR/USD FX rates (important for EUR backtests!)

    Returns dict with:
        - combined_start: The actual start date after dropna()
        - combined_end: The actual end date
        - days_available: Number of trading days
        - limiting_tickers: Tickers that have NaN on the first available date
        - individual_starts: Dict of ticker -> individual start date
        - fx_start: Start date of EUR/USD FX rates (if include_fx=True)
    """
    if verbose:
        print(f"\nDownloading combined data for {len(tickers)} tickers...")

    try:
        # Download all at once
        data = yf.download(
            tickers,
            start=start,
            auto_adjust=True,
            progress=verbose,
            threads=True,
        )

        if data.empty:
            return {"error": "No data returned for any ticker"}

        # Extract Close prices
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            # Single ticker case
            close = data[["Close"]]
            close.columns = [tickers[0]]

        # Get individual first valid dates
        individual_starts = {}
        for ticker in close.columns:
            first_valid = close[ticker].first_valid_index()
            if first_valid is not None:
                individual_starts[ticker] = first_valid

        # Apply dropna() like DataLoader does (Step 1: align price tickers)
        aligned = close.dropna()

        if aligned.empty:
            return {
                "error": "No overlapping data after dropna()",
                "individual_starts": individual_starts,
            }

        combined_start = aligned.index[0]
        combined_end = aligned.index[-1]

        # Check FX rates - this is critical for EUR backtests!
        fx_start = None
        fx_limiting = False
        if include_fx:
            if verbose:
                print("Downloading EUR/USD FX rates (EURUSD=X)...")
            try:
                fx_data = yf.download(
                    "EURUSD=X",
                    start=start,
                    auto_adjust=True,
                    progress=False,
                )
                if not fx_data.empty:
                    if isinstance(fx_data.columns, pd.MultiIndex):
                        fx_close = fx_data["Close"]["EURUSD=X"]
                    else:
                        fx_close = fx_data["Close"]
                    fx_start = fx_close.first_valid_index()

                    # The DataLoader does: fx_rates.reindex(prices_df.index, method="ffill")
                    # This means dates BEFORE fx_start will be NaN after reindex!
                    # When in_eur() divides prices by fx_rates, NaN propagates.
                    if fx_start is not None and fx_start > combined_start:
                        fx_limiting = True
                        # The ACTUAL start is limited by FX data
                        combined_start = fx_start
                        individual_starts["EURUSD=X (FX)"] = fx_start

            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not download FX rates: {e}")

        # Recalculate aligned data with FX constraint
        if fx_limiting:
            aligned = aligned[aligned.index >= fx_start]

        # Find which tickers had NaN just before the combined start
        # These are the "culprits" that forced the late start
        limiting_tickers = []
        for ticker in close.columns:
            # Check if this ticker has NaN at dates before combined_start
            pre_start_data = close.loc[close.index < combined_start, ticker]
            if pre_start_data.isna().any() or pre_start_data.empty:
                # Check if the ticker's individual start is close to combined start
                if ticker in individual_starts:
                    ind_start = individual_starts[ticker]
                    # If individual start is within 30 days of combined start, it's a culprit
                    if abs((ind_start - combined_start).days) <= 30:
                        limiting_tickers.append(ticker)
                    # Or if it has the latest individual start
                    elif ind_start == max(individual_starts.values()):
                        limiting_tickers.append(ticker)

        # Add FX as limiting ticker if it's the bottleneck
        if fx_limiting:
            limiting_tickers = ["EURUSD=X (FX rates)"] + limiting_tickers

        # If no obvious culprits found, find tickers with most NaN before combined_start
        if not limiting_tickers:
            nan_counts = {}
            for ticker in close.columns:
                pre_data = close.loc[close.index < combined_start, ticker]
                nan_counts[ticker] = pre_data.isna().sum()
            if nan_counts:
                max_nans = max(nan_counts.values())
                limiting_tickers = [t for t, c in nan_counts.items() if c == max_nans]

        return {
            "combined_start": combined_start,
            "combined_end": combined_end,
            "days_available": len(aligned),
            "limiting_tickers": limiting_tickers,
            "individual_starts": individual_starts,
            "raw_days": len(close),
            "days_lost": len(close) - len(aligned),
            "fx_start": fx_start,
            "fx_limiting": fx_limiting,
        }

    except Exception as e:
        return {"error": str(e)}


def print_results(result: dict, combined_result: dict = None, strategy_name: str = None):
    """Pretty print the analysis results."""
    print("\n" + "=" * 70)
    if strategy_name:
        print(f"LIMITING TICKER ANALYSIS: {strategy_name}")
    else:
        print("LIMITING TICKER ANALYSIS")
    print("=" * 70)

    # Combined analysis results (most important!)
    if combined_result and "combined_start" in combined_result:
        print("\n" + "█" * 70)
        print("  ACTUAL BACKTEST DATE RANGE (after dropna + FX alignment)")
        print("█" * 70)
        print(f"\n  Combined Start Date:  {combined_result['combined_start'].strftime('%Y-%m-%d')}")
        print(f"  Combined End Date:    {combined_result['combined_end'].strftime('%Y-%m-%d')}")
        print(f"  Trading Days:         {combined_result['days_available']}")
        print(f"  Days Lost to Gaps:    {combined_result['days_lost']}")

        # FX rate info
        if combined_result.get("fx_start"):
            fx_start = combined_result["fx_start"]
            print(f"\n  EUR/USD FX Start:     {fx_start.strftime('%Y-%m-%d')}", end="")
            if combined_result.get("fx_limiting"):
                print("  ⚠️  <-- FX RATES ARE LIMITING!")
            else:
                print("  (not limiting)")

        if combined_result.get("limiting_tickers"):
            print(f"\n  LIMITING TICKER(S):   {', '.join(combined_result['limiting_tickers'])}")
        print()

    elif combined_result and "error" in combined_result:
        print(f"\n  Combined analysis error: {combined_result['error']}")

    # Individual ticker analysis
    if result["limiting_date"] is None:
        print("\nERROR: No valid ticker data found!")
        if result["failed"]:
            print("\nFailed downloads:")
            for ticker, error in result["failed"]:
                print(f"  {ticker}: {error}")
        return

    limiting_date = result["limiting_date"]
    culprits = result["culprits"]

    print("-" * 70)
    print("INDIVIDUAL TICKER START DATES")
    print("-" * 70)
    print(f"\nLatest individual start: {limiting_date.strftime('%Y-%m-%d')}")
    print(f"Ticker(s) with latest:   {', '.join(culprits)}")

    # DataLoader behavior note
    print("\n" + "-" * 70)
    print("NOTE: The DataLoader uses dropna() to align all series.")
    print("      Gaps in ANY ticker's data will push the start date forward.")
    print("      The 'Combined Start Date' above shows the ACTUAL backtest start.")
    print("-" * 70)

    # Full list sorted by date (latest first)
    print("\nFull ticker list (sorted by start date, latest first):")
    print("-" * 70)
    print(f"{'Ticker':<12} {'Start Date':<12} {'Days Behind':<12} Note")
    print("-" * 70)

    combined_culprits = combined_result.get("limiting_tickers", []) if combined_result else []

    for ticker, date in result["all_tickers"].items():
        days_behind = (limiting_date - date).days
        notes = []
        if ticker in culprits:
            notes.append("LATEST-START")
        if ticker in combined_culprits:
            notes.append("LIMITING")
        note = " | ".join(notes)
        print(f"{ticker:<12} {date.strftime('%Y-%m-%d'):<12} {days_behind:>8} days  {note}")

    # Failed downloads
    if result["failed"]:
        print("\n" + "-" * 70)
        print("FAILED DOWNLOADS:")
        print("-" * 70)
        for ticker, error in result["failed"]:
            print(f"  {ticker}: {error}")

    # Summary statistics
    print("\n" + "-" * 70)
    print("SUMMARY:")
    dates = list(result["all_tickers"].values())
    earliest = min(dates)
    print(f"  Earliest individual start:  {earliest.strftime('%Y-%m-%d')}")
    print(f"  Latest individual start:    {limiting_date.strftime('%Y-%m-%d')}")
    if combined_result and "combined_start" in combined_result:
        actual_start = combined_result["combined_start"]
        print(f"  ACTUAL combined start:      {actual_start.strftime('%Y-%m-%d')}  <-- USE THIS")
        print(f"  Data range lost:            {(actual_start - earliest).days} days ({(actual_start - earliest).days / 365.25:.1f} years)")
    print(f"  Tickers analyzed:           {len(result['all_tickers'])}")
    print(f"  Tickers failed:             {len(result['failed'])}")
    print("=" * 70)


def use_actual_dataloader(tickers: list[str], start: str, verbose: bool = True) -> dict:
    """
    Use the actual DataLoader to get the REAL date range.
    This is the most accurate way to determine what dates the backtest will use.
    """
    try:
        from backtest.data import DataLoader

        if verbose:
            print("\nUsing actual DataLoader (most accurate)...")

        data = DataLoader.yahoo(
            tickers=tickers,
            start=start,
            currency="EUR",
            cache=False,  # Don't use cache for fresh data
        )

        loaded_tickers = list(data.prices.columns)
        missing_tickers = [t for t in tickers if t not in loaded_tickers]

        # Find per-ticker first valid dates in the loaded data
        ticker_starts = {}
        for ticker in loaded_tickers:
            first_valid = data.prices[ticker].first_valid_index()
            if first_valid:
                ticker_starts[ticker] = first_valid

        # Sort by date (latest first) to find the limiting ticker
        sorted_starts = dict(sorted(ticker_starts.items(), key=lambda x: x[1], reverse=True))
        limiting_ticker = list(sorted_starts.keys())[0] if sorted_starts else None
        limiting_date = sorted_starts.get(limiting_ticker) if limiting_ticker else None

        return {
            "start": data.start_date,
            "end": data.end_date,
            "days": len(data.prices),
            "tickers_loaded": loaded_tickers,
            "tickers_requested": tickers,
            "missing_tickers": missing_tickers,
            "ticker_starts": sorted_starts,
            "limiting_ticker": limiting_ticker,
            "limiting_date": limiting_date,
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(
        description="Find ticker(s) that limit the start date of a strategy's backtest."
    )
    parser.add_argument(
        "--strategy", "-s",
        help="Path to strategy Python file"
    )
    parser.add_argument(
        "--tickers", "-t",
        nargs="+",
        help="List of tickers to analyze (alternative to --strategy)"
    )
    parser.add_argument(
        "--start",
        default="1900-01-01",
        help="Start date for data download (default: 1900-01-01)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress download progress output"
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip combined download analysis (faster but less accurate)"
    )
    parser.add_argument(
        "--use-dataloader",
        action="store_true",
        help="Also use actual DataLoader for verification (most accurate)"
    )

    args = parser.parse_args()

    if not args.strategy and not args.tickers:
        parser.error("Either --strategy or --tickers must be provided")

    # Get ticker list
    strategy_name = None
    if args.strategy:
        print(f"Loading strategy from {args.strategy}...")
        try:
            strategy = load_strategy_from_file(args.strategy)
            tickers = strategy.assets
            strategy_name = strategy.name
            print(f"Strategy: {strategy_name}")
            print(f"Assets ({len(tickers)}): {', '.join(tickers)}")
        except Exception as e:
            print(f"Error loading strategy: {e}")
            sys.exit(1)
    else:
        tickers = args.tickers
        print(f"Analyzing {len(tickers)} tickers: {', '.join(tickers)}")

    # Run individual ticker analysis
    result = find_limiting_tickers(
        tickers=tickers,
        start=args.start,
        verbose=not args.quiet
    )

    # Run combined analysis (simulates actual DataLoader behavior)
    combined_result = None
    if not args.skip_combined:
        combined_result = find_combined_limiting_ticker(
            tickers=tickers,
            start=args.start,
            verbose=not args.quiet
        )

    # Print results
    print_results(result, combined_result, strategy_name)

    # Optionally use actual DataLoader for verification
    if args.use_dataloader:
        dl_result = use_actual_dataloader(tickers, args.start, verbose=not args.quiet)
        print("\n" + "█" * 70)
        print("  ACTUAL DATALOADER RESULT (EUR conversion applied)")
        print("█" * 70)
        if "error" in dl_result:
            print(f"\n  Error: {dl_result['error']}")
            if "traceback" in dl_result:
                print(f"\n  {dl_result['traceback']}")
        else:
            print(f"\n  DataLoader Start:  {dl_result['start'].strftime('%Y-%m-%d')}")
            print(f"  DataLoader End:    {dl_result['end'].strftime('%Y-%m-%d')}")
            print(f"  Trading Days:      {dl_result['days']}")
            print(f"  Tickers Requested: {len(dl_result['tickers_requested'])}")
            print(f"  Tickers Loaded:    {len(dl_result['tickers_loaded'])}")

            # Show missing tickers (CRITICAL!)
            if dl_result.get("missing_tickers"):
                print(f"\n  ⚠️  MISSING TICKERS: {', '.join(dl_result['missing_tickers'])}")
                print("      These tickers failed to load and may cause date range issues!")

            # Show the limiting ticker from DataLoader's perspective
            if dl_result.get("limiting_ticker"):
                lt = dl_result["limiting_ticker"]
                ld = dl_result["limiting_date"]
                print(f"\n  LIMITING TICKER:   {lt}")
                print(f"  Limiting Date:     {ld.strftime('%Y-%m-%d') if ld else 'N/A'}")

            # Show ALL tickers sorted by start date
            if dl_result.get("ticker_starts"):
                print("\n  ALL tickers (sorted by start date, latest first):")
                print("  " + "-" * 50)
                limiting_date = dl_result.get("limiting_date")
                count_at_limiting = 0
                for ticker, date in dl_result["ticker_starts"].items():
                    is_limiting = (date == limiting_date) if limiting_date else False
                    if is_limiting:
                        count_at_limiting += 1
                    marker = " <-- LIMITING" if is_limiting else ""
                    print(f"    {ticker:<12} {date.strftime('%Y-%m-%d')}{marker}")

                if count_at_limiting > 1:
                    print(f"\n  ⚠️  {count_at_limiting} tickers share the limiting date!")
                    print(f"      Removing just one won't help - you need to remove ALL of them.")

        print("█" * 70)


if __name__ == "__main__":
    main()
