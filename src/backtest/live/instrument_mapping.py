"""Phase E2 — instrument mapping (T-0365, Codex R2.14).

Loads broker-specific ``ticker -> ISIN/WKN/venue`` mappings from
CSV files under ``data/live/instrument_mapping/<broker>.csv``.
TR/Maxblue: mapping is mandatory; no entry -> skip.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class InstrumentMapping:
    """Broker-specific asset mapping row."""

    ticker: str
    isin: Optional[str] = None
    wkn: Optional[str] = None
    name: Optional[str] = None
    trading_venue: Optional[str] = None
    broker_specific_id: Optional[str] = None


DEFAULT_MAPPING_ROOT = Path("data/live/instrument_mapping")


def load_mapping(
    broker: str,
    *,
    root: Path | str = DEFAULT_MAPPING_ROOT,
) -> Dict[str, InstrumentMapping]:
    """Loads the CSV under `root/<broker>.csv` and returns
    ``{TICKER: InstrumentMapping}``.

    CSV columns: ``ticker, isin, wkn, name, trading_venue,
    broker_specific_id``. A header row is mandatory.

    Returns an empty dict if the file doesn't exist (Codex R2.14:
    broker adapters that don't need this tolerate it).
    """

    root = Path(root)
    path = root / f"{broker}.csv"
    if not path.exists():
        return {}
    out: Dict[str, InstrumentMapping] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            out[ticker] = InstrumentMapping(
                ticker=ticker,
                isin=(row.get("isin") or "").strip() or None,
                wkn=(row.get("wkn") or "").strip() or None,
                name=(row.get("name") or "").strip() or None,
                trading_venue=(row.get("trading_venue") or "").strip() or None,
                broker_specific_id=(row.get("broker_specific_id") or "").strip() or None,
            )
    return out


__all__ = ["DEFAULT_MAPPING_ROOT", "InstrumentMapping", "load_mapping"]
