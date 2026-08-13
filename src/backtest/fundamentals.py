"""
Fundamentals data loader for point-in-time (PIT) factor data.

Expected input: CSV files in data/fundamentals/*.csv
Each file represents a ticker (file name as ticker) or contains a "ticker" column.
The data must include a date column and the following factors:
- P/E
- EV/EBITDA
- P/B
- ROE
- Gross Margin
- Debt/Equity
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence
import warnings

import pandas as pd

REQUIRED_FACTORS = (
    "pe",
    "ev_ebitda",
    "pb",
    "roe",
    "gross_margin",
    "debt_to_equity",
)

COLUMN_ALIASES = {
    "p/e": "pe",
    "pe": "pe",
    "pe_ratio": "pe",
    "price_to_earnings": "pe",
    "ev/ebitda": "ev_ebitda",
    "ev_ebitda": "ev_ebitda",
    "ev_to_ebitda": "ev_ebitda",
    "enterprise_value_to_ebitda": "ev_ebitda",
    "p/b": "pb",
    "pb": "pb",
    "price_to_book": "pb",
    "roe": "roe",
    "return_on_equity": "roe",
    "gross_margin": "gross_margin",
    "grossmargin": "gross_margin",
    "gross_profit_margin": "gross_margin",
    "debt/equity": "debt_to_equity",
    "debt_to_equity": "debt_to_equity",
    "debt_equity": "debt_to_equity",
    "published_at": "release_date",
    "filing_date": "release_date",
    "available_at": "release_date",
    "publication_date": "release_date",
}

DATE_COLUMNS = ("date", "as_of", "period_end", "report_date")
RELEASE_DATE_COLUMNS = (
    "release_date",
    "published_at",
    "filing_date",
    "available_at",
    "publication_date",
)


@dataclass
class FundamentalsSnapshot:
    """Point-in-time fundamentals snapshot for a set of tickers."""

    as_of: date
    data: pd.DataFrame


class FundamentalsLoader:
    """Load point-in-time fundamentals from CSV files."""

    def __init__(
        self,
        root: str | Path = "data/fundamentals",
        date_columns: Iterable[str] = DATE_COLUMNS,
        provenance_mode: str = "off",
        provenance_verify_hash: bool = True,
        provenance_registry_path: Optional[str | Path] = None,
        provenance_dataset: Optional[str] = None,
        provenance_source: Optional[str] = None,
        provenance_quality_tags: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.date_columns = tuple(date_columns)
        self._data: Dict[str, pd.DataFrame] = {}
        self.provenance_mode = str(provenance_mode).strip().lower() or "off"
        if self.provenance_mode not in {"off", "warn", "strict"}:
            raise ValueError(
                "provenance_mode must be one of: off, warn, strict"
            )
        self.provenance_verify_hash = bool(provenance_verify_hash)
        self.provenance_registry_path = Path(provenance_registry_path) if provenance_registry_path else None
        self.provenance_dataset = str(provenance_dataset).strip() if provenance_dataset else None
        self.provenance_source = str(provenance_source).strip() if provenance_source else None
        self.provenance_quality_tags = (
            {str(tag).strip().lower() for tag in provenance_quality_tags if str(tag).strip()}
            if provenance_quality_tags
            else None
        )
        self.provenance_issues: list[str] = []

    @property
    def tickers(self) -> Iterable[str]:
        """Return tickers with loaded fundamentals."""
        return self._data.keys()

    def load(self) -> None:
        """Load all fundamentals CSVs from the root directory."""
        if not self.root.exists():
            raise FileNotFoundError(f"Fundamentals directory not found: {self.root}")

        csv_paths = sorted(self.root.glob("*.csv"))
        self._validate_provenance(csv_paths)

        for path in csv_paths:
            df = pd.read_csv(path)
            df = self._normalize_columns(df)
            ticker_col = "ticker" if "ticker" in df.columns else None

            if ticker_col:
                for ticker, group in df.groupby(ticker_col):
                    self._data[str(ticker).upper()] = self._prepare_ticker_df(group)
            else:
                ticker = path.stem.upper()
                self._data[ticker] = self._prepare_ticker_df(df)

    def snapshot(
        self,
        as_of: date,
        tickers: Optional[Iterable[str]] = None,
    ) -> FundamentalsSnapshot:
        """Return point-in-time fundamentals for the requested tickers."""
        as_of_ts = pd.Timestamp(as_of)
        tickers = [t.upper() for t in tickers] if tickers is not None else list(self._data)
        rows = {}

        for ticker in tickers:
            df = self._data.get(ticker)
            if df is None or df.empty:
                continue
            if "release_date" in df.columns:
                eligible = df.loc[df["release_date"] <= as_of_ts]
            else:
                eligible = df.loc[df.index <= as_of_ts]
            if eligible.empty:
                continue
            rows[ticker] = eligible.iloc[-1]

        result = pd.DataFrame.from_dict(rows, orient="index")
        if not result.empty:
            result = result.loc[:, list(REQUIRED_FACTORS)]
        return FundamentalsSnapshot(as_of=as_of, data=result)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        normalized.columns = [str(col).strip().lower() for col in normalized.columns]
        rename_map = {}
        for col in normalized.columns:
            if col in COLUMN_ALIASES:
                rename_map[col] = COLUMN_ALIASES[col]
        if rename_map:
            normalized = normalized.rename(columns=rename_map)
        return normalized

    def _prepare_ticker_df(self, df: pd.DataFrame) -> pd.DataFrame:
        date_col = self._detect_date_column(df)
        prepared = df.copy()
        prepared[date_col] = pd.to_datetime(prepared[date_col])
        release_col = self._detect_release_date_column(prepared)
        if release_col is None:
            prepared["release_date"] = prepared[date_col]
        else:
            prepared["release_date"] = pd.to_datetime(prepared[release_col], errors="coerce")
            prepared["release_date"] = prepared["release_date"].fillna(prepared[date_col])
        prepared = prepared.sort_values(["release_date", date_col])
        prepared = prepared.set_index(date_col)

        for column in REQUIRED_FACTORS:
            if column not in prepared.columns:
                prepared[column] = pd.NA

        prepared = prepared.loc[:, [*REQUIRED_FACTORS, "release_date"]]
        numeric_columns = list(REQUIRED_FACTORS)
        prepared[numeric_columns] = prepared[numeric_columns].apply(pd.to_numeric, errors="coerce")
        prepared["release_date"] = pd.to_datetime(prepared["release_date"], errors="coerce")
        return prepared

    def _detect_date_column(self, df: pd.DataFrame) -> str:
        for col in self.date_columns:
            if col in df.columns:
                return col
        raise ValueError(
            "Fundamentals CSV is missing a date column. "
            f"Expected one of: {', '.join(self.date_columns)}"
        )

    def _detect_release_date_column(self, df: pd.DataFrame) -> Optional[str]:
        for col in RELEASE_DATE_COLUMNS:
            if col in df.columns:
                return col
        return None

    def _validate_provenance(self, csv_paths: list[Path]) -> None:
        """Validate manual fundamentals files against optional provenance registry."""
        self.provenance_issues = []
        if self.provenance_mode == "off" or not csv_paths:
            return

        from backtest.provenance import ManualDataProvenanceRegistry

        registry = ManualDataProvenanceRegistry(path=self.provenance_registry_path)
        entries = registry.list_entries(
            dataset=self.provenance_dataset,
            source=self.provenance_source,
        )
        if self.provenance_quality_tags:
            entries = [
                entry
                for entry in entries
                if entry.quality_tag.lower() in self.provenance_quality_tags
            ]

        by_path: Dict[Path, list] = {}
        for entry in entries:
            resolved = self._resolve_registered_path(entry.file_path)
            by_path.setdefault(resolved, []).append(entry)

        for csv_path in csv_paths:
            resolved_csv = csv_path.resolve()
            matching = by_path.get(resolved_csv, [])
            if not matching:
                self.provenance_issues.append(
                    f"Missing provenance entry for fundamentals file: {csv_path}"
                )
                continue

            if self.provenance_verify_hash:
                current_hash = self._sha256(csv_path)
                if not any(
                    entry.checksum_sha256 and entry.checksum_sha256 == current_hash
                    for entry in matching
                ):
                    self.provenance_issues.append(
                        f"Checksum mismatch or missing checksum for: {csv_path}"
                    )

        if not self.provenance_issues:
            return

        message = (
            "Manual fundamentals provenance validation found issues:\n- "
            + "\n- ".join(self.provenance_issues)
        )
        if self.provenance_mode == "strict":
            raise ValueError(message)
        warnings.warn(message, UserWarning)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _resolve_registered_path(path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate)
        return candidate.resolve()
