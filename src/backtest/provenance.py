"""
Manual data provenance registry.

Tracks manually imported datasets (including SeekingAlpha exports) with
source, import method, and integrity metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd

QUALITY_TAGS = {"official", "proxy", "community", "manual"}
DEFAULT_PROVENANCE_PATH = "data/manual/provenance.json"
PROVENANCE_PATH_ENV = "BACKTEST_PROVENANCE_PATH"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_iso_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    parsed = str(value).strip()
    if not parsed:
        return None
    date.fromisoformat(parsed)
    return parsed


def _slugify(value: str) -> str:
    chars = []
    for ch in value.strip().lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "manual-data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tabular_shape(path: Path) -> tuple[Optional[int], Optional[int]]:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".tsv", ".txt"}:
        return None, None

    sep = ","
    if suffix in {".tsv", ".txt"}:
        sep = "\t"

    try:
        frame = pd.read_csv(path, sep=sep)
    except Exception:
        return None, None
    return int(len(frame)), int(len(frame.columns))


def _resolve_registry_path(path: Optional[str | Path] = None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.getenv(PROVENANCE_PATH_ENV)
    return Path(env_path) if env_path else Path(DEFAULT_PROVENANCE_PATH)


@dataclass
class ManualDataProvenanceEntry:
    """Single manual data provenance record."""

    entry_id: str
    dataset: str
    file_path: str
    source: str
    quality_tag: str
    as_of_date: Optional[str]
    import_method: str
    license_tos_note: str
    source_url: Optional[str]
    imported_at: str
    checksum_sha256: Optional[str]
    file_size_bytes: Optional[int]
    row_count: Optional[int]
    column_count: Optional[int]
    notes: Optional[str] = None
    # Phase C T-0103c: optional metadata used by news adapters. All four
    # default to None so older JSON payloads round-trip unchanged.
    plan_policy: Optional[str] = None
    fresh_until: Optional[str] = None
    max_age_hours: Optional[int] = None
    cutoff_ts_utc: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "dataset": self.dataset,
            "file_path": self.file_path,
            "source": self.source,
            "quality_tag": self.quality_tag,
            "as_of_date": self.as_of_date,
            "import_method": self.import_method,
            "license_tos_note": self.license_tos_note,
            "source_url": self.source_url,
            "imported_at": self.imported_at,
            "checksum_sha256": self.checksum_sha256,
            "file_size_bytes": self.file_size_bytes,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "notes": self.notes,
            "plan_policy": self.plan_policy,
            "fresh_until": self.fresh_until,
            "max_age_hours": self.max_age_hours,
            "cutoff_ts_utc": self.cutoff_ts_utc,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "ManualDataProvenanceEntry":
        return cls(
            entry_id=str(payload.get("entry_id", "")),
            dataset=str(payload.get("dataset", "")),
            file_path=str(payload.get("file_path", "")),
            source=str(payload.get("source", "")),
            quality_tag=str(payload.get("quality_tag", "manual")),
            as_of_date=(str(payload["as_of_date"]) if payload.get("as_of_date") else None),
            import_method=str(payload.get("import_method", "manual_upload")),
            license_tos_note=str(payload.get("license_tos_note", "")),
            source_url=(str(payload["source_url"]) if payload.get("source_url") else None),
            imported_at=str(payload.get("imported_at", _utc_now_iso())),
            checksum_sha256=(str(payload["checksum_sha256"]) if payload.get("checksum_sha256") else None),
            file_size_bytes=(int(payload["file_size_bytes"]) if payload.get("file_size_bytes") is not None else None),
            row_count=(int(payload["row_count"]) if payload.get("row_count") is not None else None),
            column_count=(int(payload["column_count"]) if payload.get("column_count") is not None else None),
            notes=(str(payload["notes"]) if payload.get("notes") is not None else None),
            plan_policy=(str(payload["plan_policy"]) if payload.get("plan_policy") else None),
            fresh_until=(str(payload["fresh_until"]) if payload.get("fresh_until") else None),
            max_age_hours=(int(payload["max_age_hours"]) if payload.get("max_age_hours") is not None else None),
            cutoff_ts_utc=(str(payload["cutoff_ts_utc"]) if payload.get("cutoff_ts_utc") else None),
        )


class ManualDataProvenanceRegistry:
    """File-backed registry for manual data provenance entries."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = _resolve_registry_path(path)

    def list_entries(
        self,
        dataset: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[ManualDataProvenanceEntry]:
        entries = self._load_entries()
        if dataset:
            dataset_lc = dataset.strip().lower()
            entries = [entry for entry in entries if entry.dataset.lower() == dataset_lc]
        if source:
            source_lc = source.strip().lower()
            entries = [entry for entry in entries if entry.source.lower() == source_lc]
        return sorted(entries, key=lambda entry: entry.imported_at, reverse=True)

    def get_entry(self, entry_id: str) -> Optional[ManualDataProvenanceEntry]:
        for entry in self._load_entries():
            if entry.entry_id == entry_id:
                return entry
        return None

    def find_entry_by_dedup_key(
        self,
        *,
        dataset: str,
        source: str,
        as_of_date: Optional[str],
        checksum_sha256: Optional[str],
    ) -> Optional[ManualDataProvenanceEntry]:
        """Return existing entry that matches the dedup tuple, else None.

        Used by automated adapters (api_pull) to keep registration
        idempotent: a second pull of the same snapshot must not produce
        a second entry.
        """

        dataset_lc = str(dataset).strip().lower()
        source_lc = str(source).strip().lower()
        as_of_norm = (str(as_of_date).strip() if as_of_date else None) or None
        checksum_norm = (str(checksum_sha256).strip() if checksum_sha256 else None) or None
        for entry in self._load_entries():
            if entry.dataset.strip().lower() != dataset_lc:
                continue
            if entry.source.strip().lower() != source_lc:
                continue
            if (entry.as_of_date or None) != as_of_norm:
                continue
            if (entry.checksum_sha256 or None) != checksum_norm:
                continue
            return entry
        return None

    def register_entry(
        self,
        file_path: str | Path,
        dataset: str,
        source: str,
        quality_tag: str = "manual",
        as_of_date: Optional[str] = None,
        import_method: str = "manual_upload",
        license_tos_note: str = "",
        source_url: Optional[str] = None,
        notes: Optional[str] = None,
        entry_id: Optional[str] = None,
        plan_policy: Optional[str] = None,
        fresh_until: Optional[str] = None,
        max_age_hours: Optional[int] = None,
        cutoff_ts_utc: Optional[str] = None,
    ) -> ManualDataProvenanceEntry:
        normalized_dataset = str(dataset).strip()
        normalized_source = str(source).strip()
        normalized_tag = str(quality_tag).strip().lower() or "manual"

        if not normalized_dataset:
            raise ValueError("dataset must not be empty")
        if not normalized_source:
            raise ValueError("source must not be empty")
        if normalized_tag not in QUALITY_TAGS:
            raise ValueError(
                f"quality_tag must be one of: {', '.join(sorted(QUALITY_TAGS))}"
            )

        parsed_as_of = _validate_iso_date(as_of_date)

        target_file = Path(file_path)
        if not target_file.exists() or not target_file.is_file():
            raise FileNotFoundError(f"manual data file not found: {target_file}")

        entries = self._load_entries()

        if entry_id is None:
            file_slug = _slugify(target_file.stem)
            dataset_slug = _slugify(normalized_dataset)
            entry_id = f"{dataset_slug}-{file_slug}-{uuid4().hex[:8]}"
        elif any(existing.entry_id == entry_id for existing in entries):
            raise ValueError(f"entry_id already exists: {entry_id}")

        row_count, column_count = _tabular_shape(target_file)
        rel_or_abs_path = str(target_file)
        try:
            rel_or_abs_path = str(target_file.resolve().relative_to(Path.cwd().resolve()))
        except Exception:
            rel_or_abs_path = str(target_file.resolve())

        entry = ManualDataProvenanceEntry(
            entry_id=entry_id,
            dataset=normalized_dataset,
            file_path=rel_or_abs_path,
            source=normalized_source,
            quality_tag=normalized_tag,
            as_of_date=parsed_as_of,
            import_method=str(import_method).strip() or "manual_upload",
            license_tos_note=str(license_tos_note).strip(),
            source_url=(str(source_url).strip() if source_url else None),
            imported_at=_utc_now_iso(),
            checksum_sha256=_sha256(target_file),
            file_size_bytes=int(target_file.stat().st_size),
            row_count=row_count,
            column_count=column_count,
            notes=(str(notes).strip() if notes else None),
            plan_policy=(str(plan_policy).strip() if plan_policy else None),
            fresh_until=(str(fresh_until).strip() if fresh_until else None),
            max_age_hours=(int(max_age_hours) if max_age_hours is not None else None),
            cutoff_ts_utc=(str(cutoff_ts_utc).strip() if cutoff_ts_utc else None),
        )
        entries.append(entry)
        self._save_entries(entries)
        return entry

    def verify_entries(
        self,
        check_hash: bool = True,
        check_freshness: bool = False,
    ) -> Dict[str, object]:
        entries = self._load_entries()
        issues: List[Dict[str, str]] = []
        ok_count = 0
        now_iso = _utc_now_iso()

        for entry in entries:
            path = Path(entry.file_path)
            if not path.exists() or not path.is_file():
                issues.append(
                    {
                        "entry_id": entry.entry_id,
                        "status": "missing_file",
                        "message": f"File missing: {entry.file_path}",
                    }
                )
                continue

            if check_hash and entry.checksum_sha256:
                current_hash = _sha256(path)
                if current_hash != entry.checksum_sha256:
                    issues.append(
                        {
                            "entry_id": entry.entry_id,
                            "status": "checksum_mismatch",
                            "message": "File hash no longer matches registered checksum.",
                        }
                    )
                    continue

            if check_freshness and entry.fresh_until:
                if entry.fresh_until < now_iso:
                    issues.append(
                        {
                            "entry_id": entry.entry_id,
                            "status": "stale",
                            "message": (
                                f"Snapshot fresh_until={entry.fresh_until} has passed"
                            ),
                        }
                    )
                    continue

            ok_count += 1

        return {
            "registry_path": str(self.path),
            "total_entries": len(entries),
            "ok_entries": ok_count,
            "issue_count": len(issues),
            "issues": issues,
        }

    def _load_entries(self) -> List[ManualDataProvenanceEntry]:
        payload = self._load_payload()
        raw_entries = payload.get("entries", [])
        if not isinstance(raw_entries, list):
            raise ValueError(f"Invalid provenance registry format: {self.path}")
        return [ManualDataProvenanceEntry.from_dict(item) for item in raw_entries]

    def _load_payload(self) -> Dict[str, object]:
        if not self.path.exists():
            return {"schema_version": 1, "entries": []}

        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError(f"Invalid provenance registry payload: {self.path}")
        return payload

    def _save_entries(self, entries: List[ManualDataProvenanceEntry]) -> None:
        payload = {
            "schema_version": 1,
            "entries": [entry.to_dict() for entry in entries],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

