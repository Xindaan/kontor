#!/usr/bin/env python3
"""
CLI for historical index constituents.

Commands:
  constituents list          - List supported indices
  constituents download      - Download constituent data
  constituents query         - Query members as of a date
  constituents health        - Generate health report
  constituents export        - Export to CSV/JSON
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.constituents import (
    IndexRegistry,
    DataQuality,
    HealthReport,
    ConstituentSnapshot,
)


def cmd_list(args):
    """List all supported indices."""
    summary = IndexRegistry.get_summary()

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("\nSupported Indices")
    print("=" * 70)
    print(f"{'ID':<20} {'Name':<25} {'Region':<8} {'Quality':<20}")
    print("-" * 70)

    for idx in summary["indices"]:
        print(f"{idx['id']:<20} {idx['name']:<25} {idx['region']:<8} {idx['quality']:<20}")

    print("-" * 70)
    print(f"Total: {summary['total_indices']} indices")
    print()

    # By region
    print("By Region:")
    for region, count in summary["by_region"].items():
        print(f"  {region}: {count}")

    # By quality
    print("\nBy Quality:")
    for quality, count in summary["by_quality"].items():
        print(f"  {quality}: {count}")


def cmd_download(args):
    """Download constituent data."""
    import logging
    from backtest.constituents.nport import NPortProvider, ETF_INDEX_MAP

    # Enable logging for verbose mode
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    indices = args.index if args.index else []

    if args.all:
        indices = IndexRegistry.list_indices()

    if not indices:
        print("Error: Specify indices or use --all")
        print("\nExamples:")
        print("  constituents download SP500              # Download S&P 500 (auto source)")
        print("  constituents download SP500 --source nport    # Use N-PORT ETF filings")
        print("  constituents download SPY --source nport      # Download SPY ETF holdings")
        print("  constituents download --all              # Download all indices")
        sys.exit(1)

    source = getattr(args, 'source', 'auto')
    print(f"Downloading data for {len(indices)} {'index' if len(indices) == 1 else 'indices'} (source: {source})...")

    for index_id in indices:
        try:
            print(f"\n[{index_id}] ", end="", flush=True)

            provider = None

            # Check if it's an ETF ticker for N-PORT
            if source == "nport" or (source == "auto" and index_id.upper() in ETF_INDEX_MAP):
                etf_ticker = index_id.upper()
                if etf_ticker in ETF_INDEX_MAP:
                    print(f"(N-PORT: {etf_ticker}) ", end="", flush=True)
                    provider = NPortProvider(etf_ticker)
                elif source == "nport":
                    # Try to use as ETF ticker anyway
                    print(f"(N-PORT: {etf_ticker}) ", end="", flush=True)
                    provider = NPortProvider(etf_ticker)

            # Check if it's an index ID that maps to N-PORT
            if provider is None and source == "nport":
                # Map index IDs to ETF tickers
                index_to_etf = {
                    "SP500": "SPY",
                    "NDX": "QQQ",
                    "R1000": "IWB",
                    "R2000": "IWM",
                    "EAFE": "EFA",
                    "SX5E": "FEZ",
                }
                etf = index_to_etf.get(index_id.upper())
                if etf:
                    print(f"(N-PORT via {etf}) ", end="", flush=True)
                    provider = NPortProvider(etf)
                else:
                    print(f"SKIPPED (no N-PORT source for {index_id})")
                    continue

            # Use registry for auto/wikipedia
            if provider is None:
                provider = IndexRegistry.get_provider(index_id)

            provider.download(force=args.force)

            # Show result
            dates = provider.available_dates()
            if dates:
                print(f"OK ({len(dates)} snapshots, {min(dates)} to {max(dates)})")
            else:
                print("OK (no data yet - may need --force or check source)")
        except Exception as e:
            print(f"FAILED: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print("\nDone.")


def cmd_query(args):
    """Query members as of a date."""
    try:
        as_of = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        print(f"Invalid date format: {args.date}. Use YYYY-MM-DD.")
        sys.exit(1)

    try:
        provider = IndexRegistry.get_provider(args.index)
        snapshot = provider.snapshot(as_of)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.json:
        print(json.dumps(snapshot.to_dict(), indent=2, default=str))
        return

    print(f"\n{snapshot.index_id} as of {snapshot.as_of}")
    print(f"Quality: {snapshot.quality.value}")
    print(f"Source: {snapshot.source}")
    print(f"Members: {snapshot.size}")
    print("-" * 50)

    if args.tickers:
        # Show only tickers
        tickers = snapshot.tickers
        if tickers:
            print(", ".join(tickers))
        else:
            print("(no tickers available)")
    else:
        # Show all members
        for i, member in enumerate(snapshot.members[:args.limit], 1):
            id_info = f"{member.id_type.value}: {member.primary_id}"
            name = f" ({member.name})" if member.name else ""
            print(f"  {i:3}. {id_info}{name}")

        if len(snapshot.members) > args.limit:
            print(f"  ... and {len(snapshot.members) - args.limit} more")


def cmd_health(args):
    """Generate health report."""
    # Determine which indices to check
    if args.all:
        indices = IndexRegistry.list_indices()
    else:
        indices = [args.index] if args.index else []

    if not indices:
        print("Error: Specify an index or use --all")
        sys.exit(1)

    reports = []
    for index_id in indices:
        try:
            provider = IndexRegistry.get_provider(index_id)
            report = HealthReport.generate(provider)
            reports.append(report)
        except Exception as e:
            print(f"[{index_id}] Error: {e}")
            continue

    if not reports:
        print("No reports generated.")
        sys.exit(1)

    # Output
    if args.json:
        if len(reports) == 1:
            print(reports[0].to_json())
        else:
            print(json.dumps([r.to_dict() for r in reports], indent=2, default=str))
    elif args.summary or len(reports) > 1:
        _print_health_summary(reports)
    else:
        reports[0].print_report()


def _print_health_summary(reports):
    """Print compact summary table for multiple indices."""
    print("\n" + "=" * 100)
    print("Health Summary - All Indices")
    print("=" * 100)
    print(f"{'Index':<12} {'Quality':<20} {'Coverage':<25} {'Snapshots':>10} {'Avg Size':>10} {'Status':<10}")
    print("-" * 100)

    for r in sorted(reports, key=lambda x: x.index_id):
        coverage = ""
        if r.coverage_start and r.coverage_end:
            coverage = f"{r.coverage_start} → {r.coverage_end}"

        status = "OK" if r.is_healthy else "ISSUES"
        if r.has_warnings:
            status = f"{sum(1 for i in r.issues if i.severity == 'warning')}W"
        if r.has_errors:
            status = f"{sum(1 for i in r.issues if i.severity == 'error')}E"

        print(f"{r.index_id:<12} {r.quality.value:<20} {coverage:<25} {r.total_snapshots:>10} {r.avg_member_count:>10.0f} {status:<10}")

    print("-" * 100)
    print(f"Total: {len(reports)} indices")

    # Show issues summary
    all_errors = [(r.index_id, i) for r in reports for i in r.issues if i.severity == "error"]
    all_warnings = [(r.index_id, i) for r in reports for i in r.issues if i.severity == "warning"]

    if all_errors:
        print(f"\nErrors ({len(all_errors)}):")
        for idx, issue in all_errors[:5]:
            print(f"  [{idx}] {issue.message}")
        if len(all_errors) > 5:
            print(f"  ... and {len(all_errors) - 5} more")

    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for idx, issue in all_warnings[:5]:
            print(f"  [{idx}] {issue.message}")
        if len(all_warnings) > 5:
            print(f"  ... and {len(all_warnings) - 5} more")

    print()


def cmd_export(args):
    """Export constituent data to file."""
    try:
        provider = IndexRegistry.get_provider(args.index)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    available = provider.available_dates()
    if not available:
        print("No data available for export.")
        sys.exit(1)

    # Filter dates
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        available = [d for d in available if d >= start]
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
        available = [d for d in available if d <= end]

    print(f"Exporting {len(available)} snapshots...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "csv":
        _export_csv(provider, available, output_path)
    else:
        _export_json(provider, available, output_path)

    print(f"Exported to {output_path}")


def _export_csv(provider, dates, output_path):
    """Export to CSV (long format)."""
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["as_of", "ticker", "primary_id", "id_type", "name"])

        for d in dates:
            snapshot = provider.snapshot(d)
            for member in snapshot.members:
                writer.writerow([
                    d.isoformat(),
                    member.ticker or "",
                    member.primary_id,
                    member.id_type.value,
                    member.name or "",
                ])


def _export_json(provider, dates, output_path):
    """Export to JSON."""
    data = {
        "index_id": provider.index_id,
        "quality": provider.quality.value,
        "exported_at": date.today().isoformat(),
        "snapshots": [],
    }

    for d in dates:
        snapshot = provider.snapshot(d)
        data["snapshots"].append(snapshot.to_dict())

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def cmd_generate(args):
    """Generate monthly PIT snapshots for strategies."""
    import logging
    import pandas as pd

    # Enable logging for verbose mode
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    elif args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    # Parse dates
    try:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
    except ValueError:
        print(f"Invalid start date: {args.start}")
        sys.exit(1)

    end = None
    if args.end:
        try:
            end = datetime.strptime(args.end, "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid end date: {args.end}")
            sys.exit(1)

    source = getattr(args, 'source', 'wikipedia')
    index_upper = args.index.upper()

    # Handle N-PORT source
    if source == "nport":
        from backtest.constituents.nport import NPortProvider, ETF_INDEX_MAP

        # Map index to ETF or use directly
        index_to_etf = {
            "SP500": "SPY",
            "NDX": "QQQ",
            "NASDAQ100": "QQQ",
            "R1000": "IWB",
            "R2000": "IWM",
            "EAFE": "EFA",
        }

        etf_ticker = index_to_etf.get(index_upper, index_upper)
        if etf_ticker not in ETF_INDEX_MAP and index_upper not in ETF_INDEX_MAP:
            print(f"Unsupported index/ETF for N-PORT: {args.index}")
            print(f"Supported: {', '.join(sorted(ETF_INDEX_MAP.keys()))}")
            sys.exit(1)

        if etf_ticker in ETF_INDEX_MAP:
            index_name = ETF_INDEX_MAP[etf_ticker]["index"]
        else:
            index_name = index_upper

        print(f"Generating {index_name} monthly snapshots from N-PORT ({etf_ticker})...")
        default_output = f"data/universes/{index_name.lower()}_nport_constituents.csv"
        output_path = Path(args.output or default_output)

        print(f"  Start date: {start}")
        print(f"  End date: {end or 'today'}")
        print(f"  Output: {output_path}")
        if args.force:
            print("  Force refresh: enabled (clearing cache)")
        print()

        try:
            print("Loading N-PORT filings from SEC EDGAR...")
            provider = NPortProvider(etf_ticker)

            # Get CIK for debugging
            cik = provider._parser.get_cik_for_etf(etf_ticker)
            print(f"  ETF: {etf_ticker}, CIK: {cik}")

            # Find filings first to show progress
            print("  Searching for N-PORT filings...")
            filings = provider._parser.find_nport_filings(cik, start_date=start)
            print(f"  Found {len(filings)} N-PORT filings from SEC")

            if not filings:
                print("\n  No filings found. This could be because:")
                print("  - The ETF is a Unit Investment Trust (UITs don't file N-PORT)")
                print("  - The CIK mapping is incorrect")
                print("  - Network/API issues")
                print(f"\n  Check manually: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=N-PORT")
                sys.exit(1)

            # Show first few filings
            if args.verbose:
                print("  Recent filings:")
                for f in filings[-5:]:
                    print(f"    {f.report_date}: {f.accession_number} -> {f.primary_doc}")

            # Now do the full download and parse
            print("  Downloading and parsing holdings...")

            # First try without force
            provider.download(force=args.force, start_date=start)
            snapshots = provider.available_dates()

            # If we found filings but no snapshots, force re-download
            if not snapshots and filings and not args.force:
                print("  Cache appears empty, forcing re-download...")
                provider.download(force=True, start_date=start)
                snapshots = provider.available_dates()

            print(f"  Parsed {len(snapshots)} monthly snapshots with holdings")

            if not snapshots:
                print("\nWarning: Filings found but no holdings parsed!")
                print("This might mean the XML parsing failed or all filings are for other funds.")
                print("Try with --debug for more details.")
                sys.exit(1)

            # Export to CSV
            print("\nExporting to CSV...")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            rows = []
            for snap_date in sorted(snapshots):
                if snap_date < start:
                    continue
                if end and snap_date > end:
                    continue

                snapshot = provider.snapshot(snap_date)
                for member in snapshot.members:
                    ticker = member.ticker
                    if ticker:
                        rows.append({"as_of": snap_date.isoformat(), "ticker": ticker})

            if not rows:
                print("\nWarning: No data to export!")
                sys.exit(1)

            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)

            months = df["as_of"].nunique()
            avg_tickers = len(df) / months if months > 0 else 0

            print(f"\nSuccess! Exported {len(df)} rows to {output_path}")
            print(f"  Months: {months}")
            print(f"  Avg tickers/month: {avg_tickers:.0f}")
            print(f"  Date range: {df['as_of'].min()} to {df['as_of'].max()}")

        except Exception as e:
            print(f"\nError: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)

        return

    # Wikipedia source (default)
    from backtest.constituents.wikipedia import (
        WikipediaSP500Provider,
        WikipediaNasdaq100Provider,
    )

    # Get provider
    if index_upper in ("SP500", "SPX", "SPY"):
        print("Generating S&P 500 monthly snapshots from Wikipedia...")
        provider = WikipediaSP500Provider()
        default_output = "data/universes/sp500_constituents.csv"
    elif index_upper in ("NDX", "NASDAQ100", "QQQ"):
        print("Generating Nasdaq-100 monthly snapshots from Wikipedia...")
        provider = WikipediaNasdaq100Provider()
        default_output = "data/universes/nasdaq100_constituents.csv"
    else:
        print(f"Unsupported index for Wikipedia source: {args.index}")
        print("Supported: SP500, NDX")
        print("\nTip: Use --source nport for ETF-based data")
        sys.exit(1)

    output_path = Path(args.output or default_output)

    print(f"  Start date: {start}")
    print(f"  End date: {end or 'today'}")
    print(f"  Output: {output_path}")
    if args.force:
        print("  Force refresh: enabled (clearing cache)")
    print()

    # Load data - use force flag to clear cache if requested
    print("Loading data from Wikipedia...")
    try:
        if args.force:
            provider.download(force=True)
        else:
            provider._ensure_loaded()
            if not provider._current_members:
                provider._current_members, provider._current_date = provider._load_current_members()

        changes = provider.changes
        print(f"  Found {len(changes)} historical changes")

        current = provider._current_members
        current_date = provider._current_date
        print(f"  Current constituents: {len(current)} (as of {current_date})")

        if not current:
            print("\nError: No current constituents loaded!")
            print("This could be due to:")
            print("  - Network connectivity issues")
            print("  - Wikipedia API rate limiting")
            print("  - Changes to Wikipedia table structure")
            print("\nTry again with --force to clear cache and retry.")
            sys.exit(1)

    except Exception as e:
        print(f"\nError loading Wikipedia data: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try again with --force to clear cache")
        print("  3. Try again with --debug for more details")
        sys.exit(1)

    # Generate and export
    print("\nGenerating monthly snapshots...")
    try:
        rows = provider.export_to_csv(output_path, start, end)

        if rows == 0:
            print("\nWarning: No data exported!")
            print("This could mean no snapshots were generated.")
            sys.exit(1)

        print(f"\nSuccess! Exported {rows} rows to {output_path}")

        # Show summary
        df = pd.read_csv(output_path)
        months = df["as_of"].nunique()
        avg_tickers = len(df) / months if months > 0 else 0
        print(f"  Months: {months}")
        print(f"  Avg tickers/month: {avg_tickers:.0f}")

        # Show date range
        if not df.empty:
            print(f"  Date range: {df['as_of'].min()} to {df['as_of'].max()}")

    except Exception as e:
        print(f"\nError generating snapshots: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Historical Index Constituents CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List supported indices")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download constituent data")
    dl_parser.add_argument("index", nargs="*", help="Index IDs (SP500, NDX) or ETF tickers (SPY, QQQ)")
    dl_parser.add_argument("--all", action="store_true", help="Download all indices")
    dl_parser.add_argument("--force", action="store_true", help="Force redownload")
    dl_parser.add_argument(
        "--source",
        choices=["nport", "wikipedia", "auto"],
        default="auto",
        help="Data source: nport (SEC filings), wikipedia, or auto (default)"
    )
    dl_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query members as of date")
    query_parser.add_argument("index", help="Index ID (e.g., SP500, NDX)")
    query_parser.add_argument("date", help="Date in YYYY-MM-DD format")
    query_parser.add_argument("--json", action="store_true", help="Output as JSON")
    query_parser.add_argument("--tickers", action="store_true", help="Show only tickers")
    query_parser.add_argument("--limit", type=int, default=50, help="Max members to show")

    # Health command
    health_parser = subparsers.add_parser("health", help="Generate health report")
    health_parser.add_argument("index", nargs="?", help="Index ID (optional with --all)")
    health_parser.add_argument("--all", action="store_true", help="Check all indices")
    health_parser.add_argument("--summary", action="store_true", help="Show compact summary")
    health_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export to file")
    export_parser.add_argument("index", help="Index ID")
    export_parser.add_argument("output", help="Output file path")
    export_parser.add_argument("--format", choices=["csv", "json"], default="csv")
    export_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    export_parser.add_argument("--end", help="End date (YYYY-MM-DD)")

    # Generate command - for PIT strategy CSV files
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate monthly PIT snapshots for strategies",
        description="Generate monthly constituent snapshots for use with PIT strategies."
    )
    gen_parser.add_argument("index", help="Index: SP500, NDX, or ETF ticker (SPY, QQQ)")
    gen_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    gen_parser.add_argument("--end", help="End date (default: today)")
    gen_parser.add_argument(
        "--output", "-o",
        help="Output CSV path (default: data/universes/<index>_constituents.csv)"
    )
    gen_parser.add_argument(
        "--source",
        choices=["wikipedia", "nport"],
        default="wikipedia",
        help="Data source: wikipedia (changelog-based) or nport (SEC ETF filings)"
    )
    gen_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force refresh (clear cache and re-fetch)"
    )
    gen_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    gen_parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug output with full stack traces"
    )

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "health":
        cmd_health(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
