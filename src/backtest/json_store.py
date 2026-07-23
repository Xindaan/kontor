"""Atomic JSON writes with rotating backups for the money records.

`data/manual/portfolio_*.json` carries real positions and is deliberately
gitignored (`.gitignore: data/manual/portfolio_*.json`) -- so there is NO
git safety net. The previous write path was a plain `open(path, 'w')` +
`json.dump`: a crash mid-write, a faulty auto-ratchet, or a bug in the rule
module would irrecoverably destroy share counts and highs.

Two layers of protection:

1. ATOMIC: fully serialize first (a non-serializable object raises BEFORE
   anything touches disk), then write to a temp file in the same directory
   (fsync) and swap it in via ``os.replace``. The target file is at every
   point in time either the old or the new state, never a half-written one.
2. GENERATIONS: before each change, the existing state is copied as
   ``<stem>.bak-YYYYMMDD-HHMMSS.json``; only the most recent ``keep`` are
   retained. The backup name DELIBERATELY keeps the ``portfolio_`` prefix
   and ends in ``.json``: that way it falls under the same .gitignore entry
   as the original (``data/manual/portfolio_*.json``) and can never be
   accidentally tracked.
"""

import datetime as _dt
import glob
import json
import os
import shutil
from typing import Any, Optional

BACKUP_KEEP = 7


def _prune_backups(directory: str, stem: str, keep: int) -> None:
    backups = sorted(glob.glob(os.path.join(directory, "%s.bak-*.json" % stem)))
    for old in backups[:-keep] if keep > 0 else backups:
        os.remove(old)


def save_json_mit_backup(
    path: str,
    data: Any,
    keep: int = BACKUP_KEEP,
    now: Optional[_dt.datetime] = None,
) -> Optional[str]:
    """Overwrite the file atomically, first backing up the old state as a generation.

    Args:
        path: Target file (JSON).
        data: JSON-serializable content.
        keep: how many backup generations to retain.
        now:  timestamp for the backup name (default: now). Test use only.

    Returns:
        Path of the created backup, or None if the target file is new.

    Raises:
        TypeError: if ``data`` is not serializable -- in that case neither the
        target file nor a backup is touched.
    """
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"

    target = os.path.abspath(path)
    directory = os.path.dirname(target)
    base = os.path.basename(target)
    stem = base[: -len(".json")] if base.endswith(".json") else base

    backup: Optional[str] = None
    if os.path.exists(target):
        stamp = (now or _dt.datetime.now()).strftime("%Y%m%d-%H%M%S")
        backup = os.path.join(directory, "%s.bak-%s.json" % (stem, stamp))
        suffix = 0
        while os.path.exists(backup):
            suffix += 1
            backup = os.path.join(directory, "%s.bak-%s-%d.json" % (stem, stamp, suffix))
        shutil.copy2(target, backup)
        _prune_backups(directory, stem, keep)

    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)
    return backup
