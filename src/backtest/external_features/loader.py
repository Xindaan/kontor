"""Point-in-time loader for external feature snapshots.

Reads CSV snapshots produced by adapters, enforces PIT cutoff and an
optional provenance check. Snapshot format is defined in schema.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import warnings

import pandas as pd

from backtest.external_features.schema import (
    REQUIRED_COLUMNS,
    iter_snapshot_files,
    read_snapshot_csv,
)


PROVENANCE_MODES = ("off", "warn", "strict")


@dataclass
class ExternalFeatureSnapshot:
    """Point-in-time external feature snapshot for a dataset."""

    as_of: date
    dataset: str
    data: pd.DataFrame


class ExternalFeaturesLoader:
    """Load external feature snapshots from CSV files under a root.

    A loader is bound to a single dataset. Multi-dataset aggregation is
    out of scope for Phase A.
    """

    def __init__(
        self,
        root: str | Path = "data/external_features/snapshots",
        dataset: Optional[str] = None,
        provenance_mode: str = "off",
        provenance_verify_hash: bool = True,
        provenance_registry_path: Optional[str | Path] = None,
    ) -> None:
        self.root = Path(root)
        self.dataset = dataset
        self.provenance_mode = str(provenance_mode).strip().lower() or "off"
        if self.provenance_mode not in PROVENANCE_MODES:
            raise ValueError(
                "provenance_mode must be one of: " + ", ".join(PROVENANCE_MODES)
            )
        self.provenance_verify_hash = bool(provenance_verify_hash)
        self.provenance_registry_path = (
            Path(provenance_registry_path) if provenance_registry_path else None
        )
        self.provenance_issues: List[str] = []
        self._frame: Optional[pd.DataFrame] = None
        self._loaded_paths: List[Path] = []

    def load(self, enforce_expiry: bool = False) -> None:
        """Read all snapshot CSVs for the configured dataset.

        Phase C T-0103d: ``enforce_expiry`` controls live-vs-research
        semantics. When ``True`` (live path / ``cmd_signals``), snapshots
        whose provenance ``fresh_until`` lies in the past are skipped
        entirely. When ``False`` (default; backtests / research), all
        snapshots are loaded regardless of freshness.
        """

        if self.dataset is None:
            raise ValueError("ExternalFeaturesLoader requires a dataset")

        paths = list(iter_snapshot_files(self.dataset, root=self.root))
        self._loaded_paths = paths
        self._validate_provenance(paths)

        expired_paths: set[Path] = set()
        if enforce_expiry:
            expired_paths = self._expired_snapshot_paths(paths)

        frames: List[pd.DataFrame] = []
        for path in paths:
            if path.resolve() in expired_paths:
                continue
            df = read_snapshot_csv(path)
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(
                    f"snapshot {path} is missing required column(s): "
                    + ", ".join(missing)
                )
            frames.append(df)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
        else:
            combined = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
        self._frame = combined

    def history(
        self,
        feature_name: str,
        start: date,
        end: date,
        tickers: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """Return all snapshot rows for ``feature_name`` in ``[start, end]``.

        Phase C T-0103d / Codex C22. Used by :mod:`backtest.meta_regime`
        to compute trailing percentiles. The frame is sorted by
        ``release_date`` and includes the configured ticker filter when
        provided.
        """

        if self._frame is None:
            raise RuntimeError("ExternalFeaturesLoader.load() was not called")
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = self._frame
        if df.empty:
            return df.iloc[0:0]
        out = df.loc[df["feature_name"] == feature_name].copy()
        if out.empty:
            return out
        out["release_date"] = pd.to_datetime(out["release_date"])
        out = out.loc[(out["release_date"] >= start_ts) & (out["release_date"] <= end_ts)]
        if tickers is not None:
            wanted = {str(t).upper() for t in tickers}
            out = out.loc[out["ticker"].astype(str).str.upper().isin(wanted)]
        return out.sort_values(["release_date", "ticker"]).reset_index(drop=True)

    def _expired_snapshot_paths(self, paths: List[Path]) -> set[Path]:
        """Resolve set of snapshot paths that have an expired
        ``fresh_until`` provenance entry."""

        from backtest.provenance import ManualDataProvenanceRegistry

        registry = ManualDataProvenanceRegistry(path=self.provenance_registry_path)
        entries = registry.list_entries(dataset=self.dataset)
        if not entries:
            return set()
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
        expired: set[Path] = set()
        for entry in entries:
            if not entry.fresh_until:
                continue
            if entry.fresh_until >= now_iso:
                continue
            resolved = self._resolve_registered_path(entry.file_path)
            expired.add(resolved)
        return expired

    def snapshot(
        self,
        as_of: date,
        tickers: Optional[Iterable[str]] = None,
    ) -> ExternalFeatureSnapshot:
        """Return PIT snapshot.

        Dedup-Regel: latest row per (ticker, feature_name, source, dataset)
        with release_date <= as_of; tiebreak by snapshot_ts.
        """

        if self._frame is None:
            raise RuntimeError("ExternalFeaturesLoader.load() was not called")

        as_of_ts = pd.Timestamp(as_of)
        df = self._frame
        if df.empty:
            return ExternalFeatureSnapshot(
                as_of=as_of,
                dataset=self.dataset or "",
                data=pd.DataFrame(columns=list(REQUIRED_COLUMNS)),
            )

        eligible = df.loc[pd.to_datetime(df["release_date"]) <= as_of_ts]
        if tickers is not None:
            wanted = {str(t).upper() for t in tickers}
            eligible = eligible.loc[eligible["ticker"].str.upper().isin(wanted)]
        if eligible.empty:
            return ExternalFeatureSnapshot(
                as_of=as_of,
                dataset=self.dataset or "",
                data=pd.DataFrame(columns=list(REQUIRED_COLUMNS)),
            )

        eligible = eligible.sort_values(["release_date", "snapshot_ts"])
        latest = eligible.groupby(
            ["ticker", "feature_name", "source", "dataset"], as_index=False
        ).tail(1)
        latest = latest.reset_index(drop=True)
        return ExternalFeatureSnapshot(
            as_of=as_of,
            dataset=self.dataset or "",
            data=latest,
        )

    @property
    def loaded_paths(self) -> List[Path]:
        return list(self._loaded_paths)

    def _validate_provenance(self, paths: List[Path]) -> None:
        self.provenance_issues = []
        if self.provenance_mode == "off" or not paths:
            return

        from backtest.provenance import ManualDataProvenanceRegistry

        registry = ManualDataProvenanceRegistry(path=self.provenance_registry_path)
        entries = registry.list_entries(dataset=self.dataset)

        by_path: Dict[Path, list] = {}
        for entry in entries:
            resolved = self._resolve_registered_path(entry.file_path)
            by_path.setdefault(resolved, []).append(entry)

        for snapshot_path in paths:
            resolved = snapshot_path.resolve()
            matching = by_path.get(resolved, [])
            if not matching:
                self.provenance_issues.append(
                    f"Missing provenance entry for snapshot: {snapshot_path}"
                )
                continue

            if self.provenance_verify_hash:
                current_hash = _sha256(snapshot_path)
                if not any(
                    entry.checksum_sha256 and entry.checksum_sha256 == current_hash
                    for entry in matching
                ):
                    self.provenance_issues.append(
                        f"Checksum mismatch or missing checksum for: {snapshot_path}"
                    )

        if not self.provenance_issues:
            return

        message = (
            "External features provenance validation found issues:\n- "
            + "\n- ".join(self.provenance_issues)
        )
        if self.provenance_mode == "strict":
            raise ValueError(message)
        warnings.warn(message, UserWarning)

    @staticmethod
    def _resolve_registered_path(path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
