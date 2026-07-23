"""
AI infrastructure basket: segment classification + PIT universe builder.

Provides the static value-chain classification
(``data/universes/ai_infra_segments.csv``) and builds from it a
point-in-time, survivorship-complete universe by intersecting the historical
NDX/SP500 memberships (which include delisted/acquired names up to their
removal date) with the segment tickers.

Important (honesty): the segment map is current-GICS-vintage — the only
look-ahead assumption of the U-A basket (a company's industry is stable over
time and far less hindsight-laden than "is this a winner"). Foreign report
names without NDX/SP500 membership are a documented coverage gap.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backtest.universe import CsvPITUniverseProvider

# Semiconductor complex (overlap with the existing 3x semi core) -> deliberately capped.
SEMI_CAP_GROUP = "semi"
DIVERSIFIER_CAP_GROUP = "diversifier"

DEFAULT_SEGMENTS_CSV = "data/universes/ai_infra_segments.csv"
DEFAULT_AI_INFRA_PIT_CSV = "data/universes/ai_infra_pit.csv"
DEFAULT_AI_INFRA_UB_CSV = "data/universes/ai_infra_ub.csv"
DEFAULT_INDEX_CSVS = (
    "data/universes/sp500_constituents.csv",
    "data/universes/nasdaq100_constituents.csv",
)

# U-B (foil, NOT promotable): the literal report names (US-tradeable, incl.
# ADRs TSM/ASML/ARM) + delisted AI peers for survivorship control. Static
# membership over the entire history = look-ahead by construction; serves
# ONLY to measure the hindsight premium vs U-A. Foreign-only names
# (900001.KS/900002.KS/ASM.AS/BESI.AS/ENR.DE) are omitted -> documented
# coverage gap.
# Ticker-recycling contamination (Yahoo audit 2026-06-01, scripts/
# ai_infra_data_coverage.py): these symbols historically belonged to a
# delisted AI-infra company, but Yahoo serves them with the prices of an
# ENTIRELY DIFFERENT present-day company -> would feed in false series.
# Therefore excluded from the tradeable universe (NOT from the documented
# membership thesis). Examples: COR = Cencora (formerly CoreSite, acq 2021);
# EMC = a different company (Dell/EMC acq 2016); CAVM/OCLR = recycled.
CONTAMINATED_TICKERS = {"COR", "EMC", "CAVM", "OCLR"}

UB_REPORT_TICKERS = [
    # Report core (US-listed / ADR)
    "NVDA", "AMD", "INTC", "ARM", "QCOM", "AVGO", "MRVL", "MU",
    "WDC", "STX", "PSTG", "NTAP", "TSM", "ASML", "AMAT", "LRCX", "KLAC",
    "AMKR", "ANET", "CSCO", "VRT", "ETN", "GEV", "MSFT", "ORCL", "CRM",
    "NOW", "EQIX", "DLR",
    # Delisted/acquired AI peers (survivorship control)
    "XLNX", "ALTR", "MLNX", "INPHI", "CY", "CAVM", "FNSR", "OCLR", "ACIA",
    "CONE", "COR", "QTS", "DFT", "SNDK", "EMC",
]


def _read_segments_df(path: str | Path = DEFAULT_SEGMENTS_CSV) -> pd.DataFrame:
    """Read the segment CSV (comment lines starting with '#' are skipped)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Segment-Map nicht gefunden: {p}")
    df = pd.read_csv(p, comment="#")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"] != ""]
    return df


def load_segment_map(
    path: str | Path = DEFAULT_SEGMENTS_CSV,
    *,
    ticker_col: str = "ticker",
    segment_col: str = "segment",
) -> Dict[str, str]:
    """ticker -> segment label."""
    df = _read_segments_df(path)
    return dict(zip(df[ticker_col], df[segment_col]))


def load_cap_groups(path: str | Path = DEFAULT_SEGMENTS_CSV) -> Dict[str, str]:
    """ticker -> cap_group ('semi' | 'diversifier')."""
    df = _read_segments_df(path)
    return dict(zip(df["ticker"], df["cap_group"]))


def segment_tickers(
    path: str | Path = DEFAULT_SEGMENTS_CSV,
    *,
    cap_group: Optional[str] = None,
) -> List[str]:
    """All tickers of the segment map, optionally filtered to a cap_group."""
    df = _read_segments_df(path)
    if cap_group is not None:
        df = df[df["cap_group"] == cap_group]
    return sorted(df["ticker"].unique().tolist())


def equity_fund_map(
    path: str | Path = DEFAULT_SEGMENTS_CSV,
    *,
    is_equity_fund: bool = False,
) -> Dict[str, bool]:
    """ticker -> is_equity_fund for all basket tickers.

    Default False: individual stocks get NO Teilfreistellung (full
    26.375%). Injected into ``BacktestConfig.equity_fund_map``; ETF
    benchmarks are missing from the map and therefore retain the legacy 30% behavior.
    """
    return {t: is_equity_fund for t in segment_tickers(path)}


def build_ai_infra_pit_universe(
    *,
    segments_csv: str | Path = DEFAULT_SEGMENTS_CSV,
    index_csvs=DEFAULT_INDEX_CSVS,
    out_csv: str | Path = DEFAULT_AI_INFRA_PIT_CSV,
    cap_group: Optional[str] = None,
    exclude: Optional[set] = None,
    freq: str = "MS",
) -> pd.DataFrame:
    """Build a PIT, survivorship-complete AI-infra universe (U-A).

    For every first-of-month date within the coverage range of the index
    CSVs: universe = (SP500 members ∪ NDX members as of that date, each
    most-recent-≤-date via CsvPITUniverseProvider) ∩ segment-map tickers.
    Since the index CSVs include delisted/acquired names up to their
    removal date, MEMBERSHIP is survivorship-free.

    ``exclude`` (default ``CONTAMINATED_TICKERS``) removes recycled symbols
    that would otherwise feed in false price series. NOTE: truly-delisted
    names (XLNX/ALTR/JNPR/SNDK ...) deliberately remain in the CSV — Yahoo
    provides no data for them and drops them during the backtest, which
    makes the data survivorship gap VISIBLE/auditable (see the coverage
    manifest).

    Writes ``out_csv`` in (as_of, ticker) format (directly readable by
    CsvPITUniverseProvider) and returns the DataFrame.
    """
    exclude = CONTAMINATED_TICKERS if exclude is None else set(exclude)
    seg_set = set(segment_tickers(segments_csv, cap_group=cap_group)) - exclude

    providers = [CsvPITUniverseProvider(path=str(p)) for p in index_csvs]
    earliest = min(prov.earliest_date for prov in providers)
    latest = max(prov.latest_date for prov in providers)

    grid = pd.date_range(start=earliest, end=latest, freq=freq)
    rows: List[dict] = []
    for ts in grid:
        d = ts.date()
        members: set[str] = set()
        for prov in providers:
            members.update(prov.snapshot(d).tickers)
        selected = sorted(members & seg_set)
        for ticker in selected:
            rows.append({"as_of": d.isoformat(), "ticker": ticker})

    out = pd.DataFrame(rows, columns=["as_of", "ticker"])
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def build_ai_infra_ub_universe(
    *,
    tickers: Optional[List[str]] = None,
    index_csvs=DEFAULT_INDEX_CSVS,
    out_csv: str | Path = DEFAULT_AI_INFRA_UB_CSV,
    freq: str = "MS",
) -> pd.DataFrame:
    """Build the U-B foil universe: STATIC report-names membership.

    Writes the same fixed ticker list for every first-of-month date (same
    date grid as U-A, for comparability). Look-ahead by construction;
    price availability governs entry/exit of names (e.g. ARM from its
    2023 IPO, XLNX until its 2022 delisting). Measures the hindsight
    premium against U-A.
    """
    tickers = sorted(set(tickers or UB_REPORT_TICKERS))
    providers = [CsvPITUniverseProvider(path=str(p)) for p in index_csvs]
    earliest = min(prov.earliest_date for prov in providers)
    latest = max(prov.latest_date for prov in providers)

    grid = pd.date_range(start=earliest, end=latest, freq=freq)
    rows: List[dict] = []
    for ts in grid:
        d = ts.date().isoformat()
        for ticker in tickers:
            rows.append({"as_of": d, "ticker": ticker})

    out = pd.DataFrame(rows, columns=["as_of", "ticker"])
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def segment_cap_report(
    allocations: pd.DataFrame,
    segment_map: Dict[str, str],
    caps: Dict[str, float],
    *,
    tol: float = 1e-9,
) -> pd.DataFrame:
    """Rows (rebalance dates) on which a segment exceeds its cap.

    ``caps`` maps segment label -> max weight. Uses
    ``MetricsCalculator.segment_exposure``. Empty DataFrame == the cap held.
    """
    from backtest.metrics import MetricsCalculator

    exposure = MetricsCalculator.segment_exposure(allocations, segment_map)
    if exposure.empty:
        return exposure

    breaches = pd.DataFrame(index=exposure.index)
    any_breach = pd.Series(False, index=exposure.index)
    for seg, cap in caps.items():
        if seg in exposure.columns:
            over = exposure[seg] > (cap + tol)
            breaches[seg] = exposure[seg].where(over)
            any_breach = any_breach | over
    return breaches[any_breach]
