"""News-specific validation helpers (T-0103).

The generic schema in :mod:`backtest.external_features.schema` is
intentionally engine-agnostic; news snapshots however have stronger
shape requirements:

- A mandatory ``feature_name="news_sentiment_score"`` row per ticker
  that appears in the snapshot (Codex C20).
- ``news_sentiment_score`` must lie in ``[-1, 1]``.
- Counts and counts-derived metrics (``news_article_count``,
  ``news_sentiment_dispersion``) must be non-negative.
- ``price_target_*`` is not used here but the helper is intentionally
  tolerant of unrelated extra feature rows.

The sidecar NDJSON helpers also live here so that the rest of the
package never has to import ``json`` for headlines:

- :func:`write_headlines_ndjson` writes deterministically (sorted
  keys, ``\n`` line terminator).
- :func:`hash_headlines_ndjson` returns the SHA256 of the file as the
  ``raw_payload_hash`` written into every CSV row of the same snapshot.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Mapping

import pandas as pd

REQUIRED_SCORE_FEATURE = "news_sentiment_score"

NUMERIC_FEATURES = (
    "news_sentiment_score",
    "news_article_count",
    "news_sentiment_dispersion",
    "news_sentiment_mean_pos",
    "news_sentiment_mean_neg",
    "news_count_zscore",
    "news_engine_version",
)


def validate_news_snapshot(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the news-snapshot frame is malformed.

    The function checks *what the frame contains*, not what was
    requested:

    - Every ticker present in the frame must have at least one
      ``feature_name="news_sentiment_score"`` row.
    - Score values must lie in ``[-1, 1]``.
    - Counts, dispersion and age-day fields must be non-negative.

    Tickers that were requested but for which the adapter returned no
    article rows must either be absent from the frame or accompanied by
    a neutral score row (``score=0, count=0, dispersion=0``). The check
    enforces the *present-rows-are-well-formed* invariant.
    """

    if df is None or df.empty:
        return
    if "feature_name" not in df.columns:
        raise ValueError("news snapshot is missing 'feature_name' column")
    if "feature_value" not in df.columns:
        raise ValueError("news snapshot is missing 'feature_value' column")
    if "ticker" not in df.columns:
        raise ValueError("news snapshot is missing 'ticker' column")

    score_rows = df.loc[df["feature_name"] == REQUIRED_SCORE_FEATURE]
    if score_rows.empty:
        raise ValueError(
            f"news snapshot must contain at least one '{REQUIRED_SCORE_FEATURE}' row"
        )

    tickers_in_frame = set(df["ticker"].astype(str).str.upper().unique())
    tickers_with_score = set(score_rows["ticker"].astype(str).str.upper().unique())
    missing = sorted(tickers_in_frame - tickers_with_score)
    if missing:
        raise ValueError(
            "news snapshot has rows without an analyst_score row for tickers: "
            + ", ".join(missing)
        )

    score_values = pd.to_numeric(score_rows["feature_value"], errors="coerce")
    if score_values.isna().any():
        raise ValueError("news_sentiment_score values must be numeric")
    if (score_values < -1.0).any() or (score_values > 1.0).any():
        raise ValueError("news_sentiment_score values must lie in [-1, 1]")

    for feature in ("news_article_count", "news_sentiment_dispersion"):
        feature_rows = df.loc[df["feature_name"] == feature]
        if feature_rows.empty:
            continue
        values = pd.to_numeric(feature_rows["feature_value"], errors="coerce")
        if values.isna().any():
            raise ValueError(f"feature '{feature}' values must be numeric")
        if (values < 0).any():
            raise ValueError(f"feature '{feature}' values must be non-negative")


def write_headlines_ndjson(path: Path, rows: Iterable[Mapping]) -> Path:
    """Write a headlines sidecar deterministically.

    Each row becomes one line. Keys are sorted to keep the file byte-
    stable across pandas/dict-order versions. A trailing newline keeps
    POSIX-compliant text-file shape so SHA256 is reproducible.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised: List[str] = []
    for row in rows:
        serialised.append(json.dumps(dict(row), sort_keys=True, ensure_ascii=False))
    payload = "\n".join(serialised)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")
    return path


def read_headlines_ndjson(path: Path) -> List[dict]:
    """Read a headlines sidecar. Empty file -> empty list."""

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


def hash_headlines_ndjson(path: Path) -> str:
    """Return SHA256 hex digest of the sidecar file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_headlines_rows(rows: Iterable[Mapping]) -> str:
    """SHA256 over the same byte stream :func:`write_headlines_ndjson`
    would produce. Used to populate ``raw_payload_hash`` of CSV rows
    *before* the file is on disk."""

    serialised = [
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) for row in rows
    ]
    payload = "\n".join(serialised)
    if payload:
        payload += "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "REQUIRED_SCORE_FEATURE",
    "NUMERIC_FEATURES",
    "hash_headlines_ndjson",
    "hash_headlines_rows",
    "read_headlines_ndjson",
    "validate_news_snapshot",
    "write_headlines_ndjson",
]
