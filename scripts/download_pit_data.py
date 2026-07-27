#!/usr/bin/env python3
"""
Download and convert Point-in-Time (PIT) index constituent data.

This script downloads historical index constituent data from free sources
and converts it to the format required by PIT strategies.

Sources:
- S&P 500: https://github.com/fja05680/sp500 (since 1996)
- NASDAQ 100: https://github.com/jmccarrell/n100tickers (since 2015)
- Russell 2000: Not freely available with PIT data (needs WRDS access)

Output format (one row per ticker per date):
    as_of,ticker
    2010-01-01,AAPL
    2010-01-01,MSFT
    ...

Usage:
    python scripts/download_pit_data.py              # Download all
    python scripts/download_pit_data.py --index sp500
    python scripts/download_pit_data.py --index nasdaq100
    python scripts/download_pit_data.py --index sp500 --monthly

Requirements:
    pip install requests pandas
    pip install nasdaq-100-ticker-history  # For NASDAQ 100
"""

import argparse
import io
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


# =============================================================================
# Data Sources
# =============================================================================

SOURCES = {
    "sp500": {
        "name": "S&P 500",
        "url": "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes(11-16-2025).csv",
        "format": "wide",  # date,tickers (comma-separated)
        "output": "sp500_constituents.csv",
        "start_year": 1996,
    },
    "nasdaq100": {
        "name": "NASDAQ 100",
        "format": "n100tickers",  # Use nasdaq-100-ticker-history package
        "output": "nasdaq100_constituents.csv",
        "start_year": 2015,
    },
}


# =============================================================================
# Download Functions
# =============================================================================

def download_file(url: str, max_retries: int = 3) -> str:
    """Download file content with retries."""
    import time

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for attempt in range(max_retries):
        try:
            print(f"  Downloading from {url[:60]}...")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def generate_nasdaq100_pit(monthly_only: bool = False) -> pd.DataFrame:
    """
    Generate NASDAQ 100 PIT data using nasdaq-100-ticker-history package.

    Requires: pip install nasdaq-100-ticker-history
    """
    try:
        from nasdaq_100_ticker_history import tickers_as_of
    except ImportError:
        print("  ERROR: nasdaq-100-ticker-history not installed!")
        print("  Install with: pip install nasdaq-100-ticker-history")
        sys.exit(1)

    print("  Generating NASDAQ 100 PIT data from n100tickers package...")

    # Generate monthly snapshots from 2015 to now
    start_date = date(2015, 1, 1)
    end_date = date.today()

    records = []
    current = start_date

    while current <= end_date:
        try:
            tickers = tickers_as_of(current.year, current.month, current.day)
            for ticker in sorted(tickers):
                records.append({'as_of': current, 'ticker': ticker})

            if monthly_only:
                # Move to first of next month
                if current.month == 12:
                    current = date(current.year + 1, 1, 1)
                else:
                    current = date(current.year, current.month + 1, 1)
            else:
                # Weekly snapshots (every Monday)
                current += timedelta(days=7)
                # Adjust to Monday
                while current.weekday() != 0:
                    current += timedelta(days=1)
        except Exception as e:
            print(f"  Warning: Could not get tickers for {current}: {e}")
            current += timedelta(days=7)

    df = pd.DataFrame(records)
    print(f"  Generated {len(df)} ticker-date records")
    print(f"  Date range: {df['as_of'].min()} to {df['as_of'].max()}")

    return df


def convert_wide_to_long(content: str, monthly_only: bool = False) -> pd.DataFrame:
    """
    Convert wide format (date,tickers) to long format (as_of,ticker).

    Input format:
        date,tickers
        2010-01-04,"AAPL,MSFT,GOOGL,..."

    Output format:
        as_of,ticker
        2010-01-04,AAPL
        2010-01-04,MSFT
        ...
    """
    # Read the wide format
    df = pd.read_csv(io.StringIO(content))

    # Identify columns
    date_col = df.columns[0]  # Usually 'date'
    ticker_col = df.columns[1]  # Usually 'tickers'

    print(f"  Found {len(df)} dates in source data")

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col])

    # If monthly only, keep only first trading day of each month
    if monthly_only:
        df['year_month'] = df[date_col].dt.to_period('M')
        df = df.groupby('year_month').first().reset_index(drop=True)
        print(f"  Reduced to {len(df)} monthly snapshots")

    # Explode tickers into separate rows
    records = []
    for _, row in df.iterrows():
        date = row[date_col].date()
        tickers_str = row[ticker_col]

        # Handle different formats
        if isinstance(tickers_str, str):
            tickers = [t.strip() for t in tickers_str.split(',') if t.strip()]
        else:
            continue

        for ticker in tickers:
            records.append({'as_of': date, 'ticker': ticker})

    result = pd.DataFrame(records)
    print(f"  Generated {len(result)} ticker-date records")

    return result


# =============================================================================
# Main Functions
# =============================================================================

def download_index(index_id: str, output_dir: Path, monthly_only: bool = False) -> Path:
    """Download and convert a single index."""
    if index_id not in SOURCES:
        print(f"Error: Unknown index '{index_id}'")
        print(f"Available indices: {', '.join(SOURCES.keys())}")
        sys.exit(1)

    source = SOURCES[index_id]
    print(f"\n{'='*60}")
    print(f"Downloading {source['name']} PIT data")
    print(f"{'='*60}")

    # Handle different formats
    if source['format'] == 'n100tickers':
        # Use nasdaq-100-ticker-history package
        df = generate_nasdaq100_pit(monthly_only=monthly_only)
    elif source['format'] == 'wide':
        # Download and convert wide format
        content = download_file(source['url'])
        df = convert_wide_to_long(content, monthly_only=monthly_only)
    else:
        # Already in long format
        content = download_file(source['url'])
        df = pd.read_csv(io.StringIO(content))

    # Ensure correct column names
    if 'as_of' not in df.columns:
        # Try to rename first column
        df.columns = ['as_of', 'ticker'] + list(df.columns[2:])

    # Sort by date and ticker
    df = df.sort_values(['as_of', 'ticker']).reset_index(drop=True)

    # Save
    output_path = output_dir / source['output']
    df.to_csv(output_path, index=False)

    # Stats
    date_range = f"{df['as_of'].min()} to {df['as_of'].max()}"
    n_dates = df['as_of'].nunique()
    n_tickers = df['ticker'].nunique()

    print(f"\n  Saved: {output_path}")
    print(f"  Date range: {date_range}")
    print(f"  Unique dates: {n_dates}")
    print(f"  Unique tickers: {n_tickers}")
    print(f"  Total records: {len(df)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download Point-in-Time index constituent data"
    )
    parser.add_argument(
        "--index", "-i",
        choices=list(SOURCES.keys()) + ["all"],
        default="all",
        help="Which index to download (default: all)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/universes"),
        help="Output directory (default: data/universes)"
    )
    parser.add_argument(
        "--monthly", "-m",
        action="store_true",
        help="Keep only monthly snapshots (smaller file, faster loading)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available indices and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("Available indices:")
        for idx, info in SOURCES.items():
            print(f"  {idx}: {info['name']}")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download requested indices
    if args.index == "all":
        indices = list(SOURCES.keys())
    else:
        indices = [args.index]

    for index_id in indices:
        download_index(index_id, args.output_dir, monthly_only=args.monthly)

    print(f"\n{'='*60}")
    print("Done! PIT data ready for backtesting.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
