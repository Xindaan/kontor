"""Phase E2 — SignalReport canonical hash + IO (T-0362, Codex R3.3+R4.1).

Important: floats are recursively normalized BEFORE `json.dumps`
(Codex R4.1) — `json.dumps(default=...)` doesn't kick in for
already-serializable floats.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict


def _normalize_for_hash(value: Any) -> Any:
    """Codex R4.1: recursive pre-normalization for the canonical hash.

    - Floats → ``f"{value:.10g}"`` (stable representation).
    - Datetimes → ISO-8601.
    - Dicts → recursively key-sorted (json.dumps does this via
      `sort_keys=True`).
    - Lists/Tuples → normalized element-wise.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _normalize_for_hash(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    return value


def canonical_signal_report_hash(report: Any) -> str:
    """Stable sha256 hash of a SignalReport.

    Accepts a dict-like object (with a ``to_dict()`` method) or
    a dict directly. Floats are stabilized via
    :func:`_normalize_for_hash`.
    """

    if hasattr(report, "to_dict"):
        payload = report.to_dict()
    elif isinstance(report, dict):
        payload = report
    else:
        raise TypeError(
            f"canonical_signal_report_hash expects a SignalReport-like "
            f"object with `to_dict()` or a dict, got {type(report).__name__}"
        )
    normalized = _normalize_for_hash(payload)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_signal_report(path: str | Path) -> Dict[str, Any]:
    """Loads a SignalReport from a JSON file.

    Returns the `dict` (no `SignalReport` reconstruction — the Phase E
    live plan path only needs the fields, not the whole class).
    """

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"SignalReport not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "canonical_signal_report_hash",
    "load_signal_report",
    "_normalize_for_hash",
]
