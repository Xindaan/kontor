"""Offline integrity check for the data/ cache.

Checks the MANIFEST STATE on disk -- not loader behavior (that's covered
by the regression tests in test_data_loader_cache.py): a future manifest
end date once came close to inverting a live signal (a risky target of
31.55% instead of 21.21% on stale SOXL data).

Deliberately NOT covered here: spike/fake-print detection against index
counterparts -- that's the job of a separate script that compares price
sources against each other.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_DATA_DIR = Path("data")
MANIFEST_NAME = "_cache_manifest.json"


def manifest_problems(
    manifest: Dict[str, Dict[str, str]],
    data_dir: Path,
    today: datetime.date,
) -> List[str]:
    """Finds poisoned states in the manifest.

    The loader caps every recorded end date at tomorrow (today+1); anything
    beyond that is a poisoned entry that suppresses refetches. Also checks
    for broken ranges (start > end, unreadable data) and entries whose
    cache file is missing.
    """
    problems: List[str] = []
    horizon = today + datetime.timedelta(days=1)
    for filename, entry in sorted(manifest.items()):
        if not isinstance(entry, dict):
            problems.append(f"{filename}: manifest entry is not an object")
            continue
        try:
            start = datetime.date.fromisoformat(str(entry.get("start")))
            end = datetime.date.fromisoformat(str(entry.get("end")))
        except (TypeError, ValueError):
            problems.append(
                f"{filename}: unreadable range {entry.get('start')}..{entry.get('end')}"
            )
            continue
        if end > horizon:
            problems.append(
                f"{filename}: manifest end {end} is in the future "
                f"(poisoned; loader cap is {horizon})"
            )
        if start > end:
            problems.append(f"{filename}: start {start} > end {end}")
        if not (data_dir / filename).exists():
            problems.append(f"{filename}: manifest entry without a cache file")
    return problems


def check_cache(
    data_dir: Path = DEFAULT_DATA_DIR,
    today: Optional[datetime.date] = None,
) -> List[str]:
    """Checks the cache under data_dir; [] if no manifest exists."""
    manifest_file = data_dir / MANIFEST_NAME
    if not manifest_file.exists():
        return []
    manifest = json.loads(manifest_file.read_text())
    if today is None:
        today = datetime.date.today()
    return manifest_problems(manifest, data_dir, today)


if __name__ == "__main__":
    found = check_cache()
    if found:
        print(f"CACHE INTEGRITY: {len(found)} Problem(e)")
        for line in found:
            print(f"  {line}")
        raise SystemExit(1)
    print("CACHE INTEGRITY: OK")
