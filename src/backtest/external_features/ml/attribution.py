"""Phase D sidecar helpers for row-level ML attributions (T-0213).

Mirrors the pattern from :mod:`backtest.external_features.news_schema`.
The NDJSON file is written deterministically (sorted keys, ``\n``
terminator) so SHA256 is stable across pandas/json-library versions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Mapping


def _canonical_payload(rows: Iterable[Mapping]) -> str:
    serialised: List[str] = []
    for row in rows:
        serialised.append(json.dumps(dict(row), sort_keys=True, ensure_ascii=False))
    payload = "\n".join(serialised)
    if payload:
        payload += "\n"
    return payload


def write_ml_attribution_ndjson(path: Path, rows: Iterable[Mapping]) -> Path:
    """Persist a sidecar with row-level model attributions."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_payload(rows), encoding="utf-8")
    return path


def read_ml_attribution_ndjson(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def hash_ml_attribution_ndjson(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_ml_attribution_rows(rows: Iterable[Mapping]) -> str:
    return hashlib.sha256(_canonical_payload(rows).encode("utf-8")).hexdigest()


__all__ = [
    "hash_ml_attribution_ndjson",
    "hash_ml_attribution_rows",
    "read_ml_attribution_ndjson",
    "write_ml_attribution_ndjson",
]
